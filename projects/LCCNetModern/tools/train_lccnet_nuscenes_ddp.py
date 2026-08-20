# ============================================================
# projects/LCCNetModern/tools/train_lccnet_nuscenes_ddp.py
#
# O-5C-7 DDP Training Driver
#
# Strategy A
# ------------------------------------------------------------
# BEVFusion : 256 x 704 frozen
# LCCNet    : 256 x 704
#
# torchrun example:
#
# torchrun \
#   --standalone \
#   --nnodes=1 \
#   --nproc_per_node=2 \
#   projects/LCCNetModern/tools/train_lccnet_nuscenes_ddp.py \
#   ...
#
# DDP strategy:
#
#   GPU0 -> frame A -> six cameras
#   GPU1 -> frame B -> six cameras
#
# Each GPU:
#
#   cam0 forward/backward : no_sync
#   cam1 forward/backward : no_sync
#   ...
#   cam5 forward/backward : DDP sync
#   optimizer.step()
#
# Therefore:
#   only ONE gradient all-reduce per frame,
#   not six.
#
# Existing validated geometry functions are imported from:
#
#   train_lccnet_nuscenes.py
#
# ============================================================

from __future__ import annotations

import argparse
import json
import math
import os
import time

from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List

import numpy as np

import torch
import torch.distributed as dist
import torch.nn.functional as F

from torch.nn.parallel import (
    DistributedDataParallel as DDP,
)

from mmengine.config import Config
from mmengine.registry import init_default_scope

from mmdet3d.utils import register_all_modules

from projects.LCCNetModern.lie import se3

from projects.LCCNetModern.tools.train_lccnet_nuscenes import (
    NUSC_CAMERA_NAMES,
    set_seed,
    build_one_dataloader,
    build_data_preprocessor,
    prepare_lccnet_batch,
    build_lccnet,
    compute_calibration_residual,
    balanced_twist_l1,
    balanced_twist_l1_chunk
)


# ============================================================
# Defaults
# ============================================================

DEFAULT_CONFIG = (
    'projects/LCCNetModern/configs/'
    'lccnet_nuscenes_v110.py'
)

DEFAULT_WORK_DIR = (
    'data/work_dirs/'
    'lccnet_nuscenes_v110_256x704_ddp'
)


# ============================================================
# Distributed helpers
# ============================================================

def setup_distributed(args):

    local_rank = int(
        os.environ.get(
            'LOCAL_RANK',
            args.local_rank,
        )
    )

    rank = int(
        os.environ.get(
            'RANK',
            0,
        )
    )

    world_size = int(
        os.environ.get(
            'WORLD_SIZE',
            1,
        )
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            'CUDA is required for this DDP training.'
        )

    torch.cuda.set_device(
        local_rank
    )

    dist.init_process_group(
        backend=args.dist_backend,
        init_method='env://',
    )

    device = torch.device(
        f'cuda:{local_rank}'
    )

    return (
        rank,
        local_rank,
        world_size,
        device,
    )


def cleanup_distributed():

    if not (
        dist.is_available()
        and dist.is_initialized()
    ):
        return

    try:
        dist.barrier()
    except Exception as exc:
        print(
            f"[DDP CLEANUP WARNING] barrier failed: {exc}",
            flush=True,
        )

    try:
        dist.destroy_process_group()
    except Exception as exc:
        print(
            f"[DDP CLEANUP WARNING] destroy_process_group failed: {exc}",
            flush=True,
        )


def is_main_process():

    return (
        not dist.is_initialized()
        or dist.get_rank() == 0
    )


def rank_print(
    *args,
    **kwargs,
):

    if is_main_process():

        print(
            *args,
            **kwargs,
            flush=True,
        )


def unwrap_model(model):

    if isinstance(
        model,
        DDP,
    ):

        return model.module

    return model


# ============================================================
# For DDP training all ranks MUST execute same number of
# optimizer steps.
#
# train-max-frames means GLOBAL requested frames.
#
# world_size=2:
#
#   1024 global
#     ->
#   512 rank0
#   512 rank1
#
# If not divisible by world_size, remainder is dropped.
# ============================================================

def get_equal_local_limit(
    global_max_frames: int,
    world_size: int,
):

    if global_max_frames <= 0:

        return 0

    local_limit = (
        global_max_frames
        // world_size
    )

    if local_limit <= 0:

        raise ValueError(
            'train/val max frames must be '
            '>= world_size.'
        )

    return local_limit


# ============================================================
# Scalar distributed reduction
# ============================================================

def global_sum(
    value: float,
    device,
):

    tensor = torch.tensor(
        float(value),
        device=device,
        dtype=torch.float64,
    )

    dist.all_reduce(
        tensor,
        op=dist.ReduceOp.SUM,
    )

    return float(
        tensor.item()
    )


def global_max(
    value: float,
    device,
):

    tensor = torch.tensor(
        float(value),
        device=device,
        dtype=torch.float64,
    )

    dist.all_reduce(
        tensor,
        op=dist.ReduceOp.MAX,
    )

    return float(
        tensor.item()
    )


# ============================================================
# Metric summaries
# ============================================================

def summarize_values(
    values: List[float],
):

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.size == 0:

        raise RuntimeError(
            'Metric accumulator is empty.'
        )

    return dict(

        mean=float(
            np.mean(array)
        ),

        median=float(
            np.median(array)
        ),

        p90=float(
            np.quantile(
                array,
                0.90,
            )
        ),

        max=float(
            np.max(array)
        ),
    )


# ============================================================
# Chunk prediction
# ============================================================

@torch.no_grad()
def predict_in_chunks_ddp(
    model,
    rgb,
    depth,
    chunk_size: int,
):

    outputs = []

    M = rgb.shape[0]

    for start in range(
        0,
        M,
        chunk_size,
    ):

        end = min(
            start + chunk_size,
            M,
        )

        pred = model(

            rgb[
                start:end
            ],

            depth[
                start:end
            ],
        )

        outputs.append(
            pred
        )

    return torch.cat(
        outputs,
        dim=0,
    )


# ============================================================
# DDP validation
#
# Each rank evaluates its local subset.
#
# At the end:
#
#   all_gather_object()
#
# combines physical residual values from every GPU.
#
# This is needed because mean alone is not enough:
# we also need median / P90 / per-camera statistics.
# ============================================================

@torch.no_grad()
def validate_ddp(
    model,
    val_loader,
    data_preprocessor,
    device,
    rank,
    world_size,
    args,
):

    model.eval()

    local_frame_limit = (
        get_equal_local_limit(

            args.val_max_frames,

            world_size,
        )
        if args.val_max_frames > 0
        else 0
    )

    local_broken_rot = []
    local_broken_trans = []

    local_pred_rot = []
    local_pred_trans = []

    local_loss_sum = 0.0
    local_loss_count = 0

    local_per_camera = {}

    local_frame_count = 0

    first_debug_printed = False

    start_time = time.time()

    for batch_idx, raw_batch in enumerate(
        val_loader
    ):

        if (
            local_frame_limit > 0
            and local_frame_count
            >= local_frame_limit
        ):

            break

        batch = prepare_lccnet_batch(

            raw_batch,

            data_preprocessor,

            device,

            max_rot_deg=(
                args.val_max_rot_deg
            ),

            max_trans_m=(
                args.val_max_trans_m
            ),

            depth_scale=(
                args.depth_scale
            ),

            input_color_order=(
                args.input_color_order
            ),

            fixed_validation=True,

            validation_seed=(
                args.val_seed
            ),
        )

        rgb = batch[
            'rgb'
        ]

        depth = batch[
            'depth'
        ]

        target = batch[
            'target'
        ]

        broken = batch[
            'broken_c2l'
        ]

        gt = batch[
            'gt_c2l'
        ]

        M = rgb.shape[0]

        # ----------------------------------------------------
        # Show one deterministic perturbation from rank0.
        # ----------------------------------------------------

        if (
            rank == 0
            and not first_debug_printed
        ):

            debug = batch[
                'target_info'
            ][
                'debug'
            ]

            rank_print()
            rank_print(
                '[VAL FIXED PERTURBATION]'
            )

            rank_print(
                'First camera =',
                {
                    'rx_deg':
                        debug[
                            'rx_deg'
                        ][0, 0].item(),

                    'ry_deg':
                        debug[
                            'ry_deg'
                        ][0, 0].item(),

                    'rz_deg':
                        debug[
                            'rz_deg'
                        ][0, 0].item(),

                    'tx_m':
                        debug[
                            'tx_m'
                        ][0, 0].item(),

                    'ty_m':
                        debug[
                            'ty_m'
                        ][0, 0].item(),

                    'tz_m':
                        debug[
                            'tz_m'
                        ][0, 0].item(),
                }
            )

            first_debug_printed = True

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        pred_x = predict_in_chunks_ddp(

            model,

            rgb,

            depth,

            chunk_size=(
                args.cam_chunk
            ),
        )

        if not torch.isfinite(
            pred_x
        ).all():

            raise RuntimeError(
                f'Rank {rank}: '
                'non-finite validation prediction.'
            )

        # loss = F.l1_loss(

        #     pred_x,

        #     target,

        #     reduction='mean',
        # )

        (
            loss,
            rot_loss,
            trans_loss,
        ) = balanced_twist_l1(

            pred=pred_x,

            target=target,

            max_rot_deg=(
                args.val_max_rot_deg
            ),

            max_trans_m=(
                args.val_max_trans_m
            ),

            rot_weight=(
                args.rot_loss_weight
            ),

            trans_weight=(
                args.trans_loss_weight
            ),
        )

        local_loss_sum += float(
            loss.item()
        )

        local_loss_count += 1

        correction_pred = se3.exp(
            pred_x
        )

        identity = (

            torch.eye(
                4,
                device=device,
                dtype=broken.dtype,
            )
            .unsqueeze(0)
            .repeat(
                M,
                1,
                1,
            )
        )

        (
            broken_rot,
            broken_trans,
        ) = compute_calibration_residual(

            identity,

            broken,

            gt,
        )

        (
            pred_rot,
            pred_trans,
        ) = compute_calibration_residual(

            correction_pred,

            broken,

            gt,
        )

        local_broken_rot.extend(

            broken_rot
            .detach()
            .cpu()
            .tolist()
        )

        local_broken_trans.extend(

            broken_trans
            .detach()
            .cpu()
            .tolist()
        )

        local_pred_rot.extend(

            pred_rot
            .detach()
            .cpu()
            .tolist()
        )

        local_pred_trans.extend(

            pred_trans
            .detach()
            .cpu()
            .tolist()
        )

        # ----------------------------------------------------
        # Per camera
        # ----------------------------------------------------

        names = batch[
            'camera_names'
        ]

        for cam_idx, name in enumerate(
            names
        ):

            if name not in local_per_camera:

                local_per_camera[
                    name
                ] = dict(

                    broken_rot=[],

                    broken_trans=[],

                    pred_rot=[],

                    pred_trans=[],
                )

            local_per_camera[
                name
            ][
                'broken_rot'
            ].append(

                float(
                    broken_rot[
                        cam_idx
                    ].item()
                )
            )

            local_per_camera[
                name
            ][
                'broken_trans'
            ].append(

                float(
                    broken_trans[
                        cam_idx
                    ].item()
                )
            )

            local_per_camera[
                name
            ][
                'pred_rot'
            ].append(

                float(
                    pred_rot[
                        cam_idx
                    ].item()
                )
            )

            local_per_camera[
                name
            ][
                'pred_trans'
            ].append(

                float(
                    pred_trans[
                        cam_idx
                    ].item()
                )
            )

        local_frame_count += int(
            batch[
                'B'
            ]
        )

        if (
            args.val_log_interval > 0
            and local_frame_count
            % args.val_log_interval
            == 0
        ):

            if rank == 0:

                approx_global = (
                    local_frame_count
                    * world_size
                )

                rank_print(
                    f'[VAL] '
                    f'approx_global_frames='
                    f'{approx_global}'
                )

    # ========================================================
    # Gather every rank's metrics
    # ========================================================

    local_payload = dict(

        frames=(
            local_frame_count
        ),

        loss_sum=(
            local_loss_sum
        ),

        loss_count=(
            local_loss_count
        ),

        broken_rot=(
            local_broken_rot
        ),

        broken_trans=(
            local_broken_trans
        ),

        pred_rot=(
            local_pred_rot
        ),

        pred_trans=(
            local_pred_trans
        ),

        per_camera=(
            local_per_camera
        ),
    )

    gathered = [
        None
        for _ in range(
            world_size
        )
    ]

    dist.all_gather_object(

        gathered,

        local_payload,
    )

    # ========================================================
    # Merge
    # ========================================================

    all_broken_rot = []
    all_broken_trans = []

    all_pred_rot = []
    all_pred_trans = []

    total_frames = 0
    total_loss_sum = 0.0
    total_loss_count = 0

    merged_per_camera = {}

    for payload in gathered:

        total_frames += int(
            payload[
                'frames'
            ]
        )

        total_loss_sum += float(
            payload[
                'loss_sum'
            ]
        )

        total_loss_count += int(
            payload[
                'loss_count'
            ]
        )

        all_broken_rot.extend(
            payload[
                'broken_rot'
            ]
        )

        all_broken_trans.extend(
            payload[
                'broken_trans'
            ]
        )

        all_pred_rot.extend(
            payload[
                'pred_rot'
            ]
        )

        all_pred_trans.extend(
            payload[
                'pred_trans'
            ]
        )

        for name, values in (
            payload[
                'per_camera'
            ].items()
        ):

            if name not in merged_per_camera:

                merged_per_camera[
                    name
                ] = dict(

                    broken_rot=[],

                    broken_trans=[],

                    pred_rot=[],

                    pred_trans=[],
                )

            for key in [
                'broken_rot',
                'broken_trans',
                'pred_rot',
                'pred_trans',
            ]:

                merged_per_camera[
                    name
                ][
                    key
                ].extend(

                    values[
                        key
                    ]
                )

    # ========================================================
    # Global summaries
    # ========================================================

    broken_rot_summary = (
        summarize_values(
            all_broken_rot
        )
    )

    broken_trans_summary = (
        summarize_values(
            all_broken_trans
        )
    )

    pred_rot_summary = (
        summarize_values(
            all_pred_rot
        )
    )

    pred_trans_summary = (
        summarize_values(
            all_pred_trans
        )
    )

    rot_recovery_pct = (

        100.0
        * (
            1.0
            -
            pred_rot_summary[
                'mean'
            ]
            /
            max(
                broken_rot_summary[
                    'mean'
                ],
                1e-12,
            )
        )
    )

    trans_recovery_pct = (

        100.0
        * (
            1.0
            -
            pred_trans_summary[
                'mean'
            ]
            /
            max(
                broken_trans_summary[
                    'mean'
                ],
                1e-12,
            )
        )
    )

    joint_score = (

        pred_rot_summary[
            'mean'
        ]
        /
        max(
            broken_rot_summary[
                'mean'
            ],
            1e-12,
        )

        +

        pred_trans_summary[
            'mean'
        ]
        /
        max(
            broken_trans_summary[
                'mean'
            ],
            1e-12,
        )
    )

    per_camera_summary = {}

    for name, values in (
        merged_per_camera.items()
    ):

        per_camera_summary[
            name
        ] = dict(

            broken_rot_mean_deg=float(
                np.mean(
                    values[
                        'broken_rot'
                    ]
                )
            ),

            pred_rot_mean_deg=float(
                np.mean(
                    values[
                        'pred_rot'
                    ]
                )
            ),

            broken_trans_mean_m=float(
                np.mean(
                    values[
                        'broken_trans'
                    ]
                )
            ),

            pred_trans_mean_m=float(
                np.mean(
                    values[
                        'pred_trans'
                    ]
                )
            ),
        )

    metrics = dict(

        frames=int(
            total_frames
        ),

        l1_loss=float(
            total_loss_sum
            /
            max(
                total_loss_count,
                1,
            )
        ),

        broken=dict(

            rotation_deg=(
                broken_rot_summary
            ),

            translation_m=(
                broken_trans_summary
            ),
        ),

        predicted=dict(

            rotation_deg=(
                pred_rot_summary
            ),

            translation_m=(
                pred_trans_summary
            ),
        ),

        rotation_recovery_pct=float(
            rot_recovery_pct
        ),

        translation_recovery_pct=float(
            trans_recovery_pct
        ),

        joint_score=float(
            joint_score
        ),

        per_camera=(
            per_camera_summary
        ),

        elapsed_sec=float(
            time.time()
            - start_time
        ),
    )

    return metrics


# ============================================================
# DDP training one epoch
# ============================================================

def train_one_epoch_ddp(
    model,
    train_loader,
    data_preprocessor,
    optimizer,
    device,
    rank,
    world_size,
    epoch,
    args,
):

    sampler = getattr(
        train_loader,
        'sampler',
        None,
    )

    if (
        sampler is not None
        and hasattr(
            sampler,
            'set_epoch'
        )
    ):

        sampler.set_epoch(
            epoch
        )

    model.train()

    local_frame_limit = (
        get_equal_local_limit(

            args.train_max_frames,

            world_size,
        )
        if args.train_max_frames > 0
        else 0
    )

    local_running_loss = 0.0
    local_frame_count = 0

    last_grad_norm = 0.0

    start_time = time.time()

    for batch_idx, raw_batch in enumerate(
        train_loader
    ):

        if (
            local_frame_limit > 0
            and local_frame_count
            >= local_frame_limit
        ):

            break

        # ====================================================
        # New random Delta every frame visit
        # ====================================================

        batch = prepare_lccnet_batch(

            raw_batch,

            data_preprocessor,

            device,

            max_rot_deg=(
                args.train_max_rot_deg
            ),

            max_trans_m=(
                args.train_max_trans_m
            ),

            depth_scale=(
                args.depth_scale
            ),

            input_color_order=(
                args.input_color_order
            ),

            fixed_validation=False,
        )

        rgb = batch[
            'rgb'
        ]

        depth = batch[
            'depth'
        ]

        target = batch[
            'target'
        ]

        M = rgb.shape[0]

        optimizer.zero_grad(
            set_to_none=True
        )

        # total_target_elements = int(
        #     target.numel()
        # )

        total_samples = int(
            target.shape[0]
        )

        total_loss = 0.0

        # ====================================================
        # Critical DDP optimization:
        #
        # Do NOT synchronize gradients after every camera.
        #
        # cam 0~4:
        #     no_sync()
        #
        # last camera:
        #     normal backward
        #     -> gradient all-reduce ONCE
        # ====================================================

        for start in range(
            0,
            M,
            args.cam_chunk,
        ):

            end = min(
                start
                + args.cam_chunk,
                M,
            )

            is_last_chunk = (
                end >= M
            )

            sync_context = (

                nullcontext()

                if is_last_chunk

                else model.no_sync()
            )

            with sync_context:

                pred = model(

                    rgb[
                        start:end
                    ],

                    depth[
                        start:end
                    ],
                )

                target_chunk = (
                    target[
                        start:end
                    ]
                )

                # chunk_loss = (

                #     F.l1_loss(

                #         pred,

                #         target_chunk,

                #         reduction='sum',
                #     )

                #     /

                #     float(
                #         total_target_elements
                #     )
                # )

                (
                    chunk_loss,
                    chunk_rot_loss,
                    chunk_trans_loss,
                ) = balanced_twist_l1_chunk(

                    pred_chunk=pred,

                    target_chunk=target_chunk,

                    total_samples=total_samples,

                    max_rot_deg=(
                        args.train_max_rot_deg
                    ),

                    max_trans_m=(
                        args.train_max_trans_m
                    ),

                    rot_weight=(
                        args.rot_loss_weight
                    ),

                    trans_weight=(
                        args.trans_loss_weight
                    ),
                )

                if not torch.isfinite(
                    chunk_loss
                ):

                    raise RuntimeError(

                        f'Rank {rank}: '
                        f'non-finite loss '
                        f'epoch={epoch}, '
                        f'batch={batch_idx}'
                    )

                chunk_loss.backward()

            total_loss += float(
                chunk_loss
                .detach()
                .item()
            )

        # ====================================================
        # Check gradients
        # ====================================================

        for name, param in (
            unwrap_model(
                model
            ).named_parameters()
        ):

            if param.grad is None:

                continue

            if not torch.isfinite(
                param.grad
            ).all():

                raise RuntimeError(

                    f'Rank {rank}: '
                    f'non-finite gradient '
                    f'in {name}'
                )

        # ====================================================
        # Gradients are already globally averaged by DDP
        # after the final chunk.
        # ====================================================

        grad_norm = (
            torch.nn.utils
            .clip_grad_norm_(

                model.parameters(),

                max_norm=(
                    args.grad_clip
                ),

                error_if_nonfinite=True,
            )
        )

        last_grad_norm = float(
            grad_norm
        )

        optimizer.step()

        local_running_loss += (
            total_loss
        )

        local_frame_count += int(
            batch[
                'B'
            ]
        )

        # ====================================================
        # Logging
        #
        # Since every rank executes the same number of steps,
        # all_reduce here is safe.
        # ====================================================

        if (
            args.log_interval > 0
            and local_frame_count
            % args.log_interval
            == 0
        ):

            global_loss_sum = global_sum(

                local_running_loss,

                device,
            )

            global_frames = global_sum(

                local_frame_count,

                device,
            )

            global_grad_max = global_max(

                last_grad_norm,

                device,
            )

            if rank == 0:

                avg_loss = (

                    global_loss_sum
                    /
                    max(
                        global_frames,
                        1.0,
                    )
                )

                lr = (
                    optimizer
                    .param_groups[0][
                        'lr'
                    ]
                )

                elapsed = (
                    time.time()
                    - start_time
                )

                rank_print(

                    f'[TRAIN-DDP] '
                    f'epoch={epoch:02d} '
                    f'global_frames='
                    f'{int(global_frames)} '
                    f'loss={avg_loss:.8f} '
                    f'lr={lr:.8e} '
                    f'grad_norm_max='
                    f'{global_grad_max:.6f} '
                    f'time={elapsed:.1f}s'
                )

    # ========================================================
    # Epoch summary
    # ========================================================

    global_loss_sum = global_sum(

        local_running_loss,

        device,
    )

    global_frames = global_sum(

        local_frame_count,

        device,
    )

    global_grad_max = global_max(

        last_grad_norm,

        device,
    )

    avg_loss = (

        global_loss_sum
        /
        max(
            global_frames,
            1.0,
        )
    )

    return dict(

        frames=int(
            global_frames
        ),

        loss=float(
            avg_loss
        ),

        last_grad_norm=float(
            global_grad_max
        ),

        elapsed_sec=float(
            time.time()
            - start_time
        ),
    )


# ============================================================
# Validation print
# ============================================================

def print_validation_metrics(
    epoch,
    metrics,
):

    if not is_main_process():

        return

    b_rot = (
        metrics[
            'broken'
        ][
            'rotation_deg'
        ]
    )

    b_trans = (
        metrics[
            'broken'
        ][
            'translation_m'
        ]
    )

    p_rot = (
        metrics[
            'predicted'
        ][
            'rotation_deg'
        ]
    )

    p_trans = (
        metrics[
            'predicted'
        ][
            'translation_m'
        ]
    )

    print()
    print(
        '===================================================='
    )

    print(
        f'[VALIDATION-DDP] epoch={epoch}'
    )

    print(
        '===================================================='
    )

    print(
        f"Frames = "
        f"{metrics['frames']}"
    )

    # print(
    #     f"L1 loss = "
    #     f"{metrics['l1_loss']:.8f}"
    # )

    print(
        f"Balanced SE3 loss = "
        f"{metrics['l1_loss']:.8f}"
    )

    print()

    print(
        '[BROKEN]'
    )

    print(

        f"Rot   mean="
        f"{b_rot['mean']:.6f} deg  "

        f"median="
        f"{b_rot['median']:.6f} deg  "

        f"P90="
        f"{b_rot['p90']:.6f} deg"
    )

    print(

        f"Trans mean="
        f"{b_trans['mean']:.6f} m    "

        f"median="
        f"{b_trans['median']:.6f} m    "

        f"P90="
        f"{b_trans['p90']:.6f} m"
    )

    print()
    print(
        '[LCCNET]'
    )

    print(

        f"Rot   mean="
        f"{p_rot['mean']:.6f} deg  "

        f"median="
        f"{p_rot['median']:.6f} deg  "

        f"P90="
        f"{p_rot['p90']:.6f} deg"
    )

    print(

        f"Trans mean="
        f"{p_trans['mean']:.6f} m    "

        f"median="
        f"{p_trans['median']:.6f} m    "

        f"P90="
        f"{p_trans['p90']:.6f} m"
    )

    print()

    print(

        f"Rotation recovery    = "
        f"{metrics['rotation_recovery_pct']:.2f}%"
    )

    print(

        f"Translation recovery = "
        f"{metrics['translation_recovery_pct']:.2f}%"
    )

    print(

        f"Joint score          = "
        f"{metrics['joint_score']:.6f}"
    )

    print()
    print(
        '[PER CAMERA]'
    )

    preferred_order = (
        NUSC_CAMERA_NAMES
        +
        sorted(
            [
                name
                for name
                in metrics[
                    'per_camera'
                ].keys()
                if name
                not in NUSC_CAMERA_NAMES
            ]
        )
    )

    for name in preferred_order:

        if name not in (
            metrics[
                'per_camera'
            ]
        ):

            continue

        cm = (
            metrics[
                'per_camera'
            ][
                name
            ]
        )

        print(

            f'{name:<18} '

            f"Rot "
            f"{cm['broken_rot_mean_deg']:.3f}"
            f" -> "
            f"{cm['pred_rot_mean_deg']:.3f} deg   "

            f"Trans "
            f"{cm['broken_trans_mean_m']:.3f}"
            f" -> "
            f"{cm['pred_trans_mean_m']:.3f} m"
        )

    print(
        '===================================================='
    )


# ============================================================
# Checkpoints
# ============================================================

def save_model_checkpoint(
    path,
    epoch,
    model,
    train_metrics,
    val_metrics,
    args,
    world_size,
):

    if not is_main_process():

        return

    base_model = unwrap_model(
        model
    )

    state = dict(

        gate='O-5C-7-DDP',

        epoch=epoch,

        world_size=(
            world_size
        ),

        model=(
            base_model.state_dict()
        ),

        train_metrics=(
            train_metrics
        ),

        val_metrics=(
            val_metrics
        ),

        resolution=[
            256,
            704,
        ],

        imagenet_pretrained=(
            not args.no_imagenet_pretrained
        ),

        calibration_pretrained=False,

        train_max_rot_deg=(
            args.train_max_rot_deg
        ),

        train_max_trans_m=(
            args.train_max_trans_m
        ),

        val_max_rot_deg=(
            args.val_max_rot_deg
        ),

        val_max_trans_m=(
            args.val_max_trans_m
        ),

        val_seed=(
            args.val_seed
        ),

        args=vars(
            args
        ),
    )

    torch.save(
        state,
        path,
    )


def save_training_state(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    best_joint_score,
    args,
    world_size,
):

    if not is_main_process():

        return

    state = dict(

        epoch=epoch,

        world_size=(
            world_size
        ),

        model=(
            unwrap_model(
                model
            ).state_dict()
        ),

        optimizer=(
            optimizer.state_dict()
        ),

        scheduler=(
            scheduler.state_dict()
            if scheduler
            is not None
            else None
        ),

        best_joint_score=(
            best_joint_score
        ),

        args=vars(
            args
        ),
    )

    torch.save(
        state,
        path,
    )


# ============================================================
# Resume
# ============================================================

def resume_training(
    path,
    model,
    optimizer,
    scheduler,
):

    checkpoint = torch.load(

        path,

        map_location='cpu',

        weights_only=False,
    )

    unwrap_model(
        model
    ).load_state_dict(

        checkpoint[
            'model'
        ],

        strict=True,
    )

    if checkpoint.get(
        'optimizer',
        None,
    ) is not None:

        optimizer.load_state_dict(

            checkpoint[
                'optimizer'
            ]
        )

    if (
        scheduler is not None
        and checkpoint.get(
            'scheduler',
            None,
        ) is not None
    ):

        scheduler.load_state_dict(

            checkpoint[
                'scheduler'
            ]
        )

    start_epoch = (

        int(
            checkpoint.get(
                'epoch',
                0,
            )
        )
        + 1
    )

    best_joint_score = float(

        checkpoint.get(
            'best_joint_score',
            float('inf'),
        )
    )

    return (
        start_epoch,
        best_joint_score,
    )


# ============================================================
# Main
# ============================================================

def main(
    args,
):

    (
        rank,
        local_rank,
        world_size,
        device,
    ) = setup_distributed(
        args
    )

    try:

        # ====================================================
        # Rank-specific RNG
        #
        # Different ranks should not generate identical
        # training perturbations.
        # ====================================================

        set_seed(
            args.seed
            + rank
        )

        register_all_modules(
            init_default_scope=True
        )

        cfg = Config.fromfile(
            args.config
        )

        init_default_scope(

            cfg.get(
                'default_scope',
                'mmdet3d',
            )
        )

        rank_print()
        rank_print(
            '===================================================='
        )

        rank_print(
            'O-5C-7 LCCNet DDP Training'
        )

        rank_print(
            'Strategy A: 256 x 704'
        )

        rank_print(
            '===================================================='
        )

        rank_print(
            f'World size = {world_size}'
        )

        rank_print(
            'Global frame batch =',
            world_size
        )

        rank_print(
            'Camera chunk =',
            args.cam_chunk
        )

        rank_print(
            'Gradient synchronization = '
            'once per frame'
        )

        rank_print(
            'Train requested global frames =',
            args.train_max_frames
        )

        rank_print(
            'Val requested global frames =',
            args.val_max_frames
        )

        if (
            args.train_max_frames > 0
            and args.train_max_frames
            % world_size
            != 0
        ):

            effective = (

                (
                    args.train_max_frames
                    // world_size
                )
                * world_size
            )

            rank_print(
                '[WARNING] '
                'train-max-frames is not divisible '
                'by world_size.'
            )

            rank_print(
                'Effective global train frames =',
                effective
            )

        # ====================================================
        # Dataloaders
        #
        # IMPORTANT:
        # DDP has already been initialized.
        #
        # MMEngine DefaultSampler therefore sees
        # rank/world_size and distributes samples.
        # ====================================================

        train_loader = build_one_dataloader(

            cfg,

            split='train',

            batch_size=1,

            num_workers=(
                args.num_workers
            ),

            seed=(
                args.seed
            ),
        )

        val_loader = build_one_dataloader(

            cfg,

            split='val',

            batch_size=1,

            num_workers=(
                args.num_workers
            ),

            seed=(
                args.val_seed
            ),
        )

        rank_print()
        rank_print(
            '[DATASET]'
        )

        rank_print(
            'Full train dataset length =',
            len(
                train_loader.dataset
            )
        )

        rank_print(
            'Local train loader length =',
            len(
                train_loader
            )
        )

        rank_print(
            'Full val dataset length =',
            len(
                val_loader.dataset
            )
        )

        rank_print(
            'Local val loader length =',
            len(
                val_loader
            )
        )

        # ====================================================
        # Preprocessor
        # ====================================================

        data_preprocessor = (
            build_data_preprocessor(

                cfg,

                device,
            )
        )

        # ====================================================
        # Input gate
        # ====================================================

        first_raw_batch = next(
            iter(
                train_loader
            )
        )

        first_batch = (
            prepare_lccnet_batch(

                first_raw_batch,

                data_preprocessor,

                device,

                max_rot_deg=(
                    args.train_max_rot_deg
                ),

                max_trans_m=(
                    args.train_max_trans_m
                ),

                depth_scale=(
                    args.depth_scale
                ),

                input_color_order=(
                    args.input_color_order
                ),

                fixed_validation=False,
            )
        )

        H = first_batch[
            'H'
        ]

        W = first_batch[
            'W'
        ]

        rank_print()
        rank_print(
            '[INPUT GATE]'
        )

        rank_print(
            'RGB =',
            tuple(
                first_batch[
                    'rgb'
                ].shape
            )
        )

        rank_print(
            'Depth =',
            tuple(
                first_batch[
                    'depth'
                ].shape
            )
        )

        rank_print(
            'Target =',
            tuple(
                first_batch[
                    'target'
                ].shape
            )
        )

        rank_print(
            'Resolution =',
            (
                H,
                W,
            )
        )

        if (
            H != 256
            or W != 704
        ):

            raise RuntimeError(

                'DDP Strategy A expects '
                f'256x704, got {H}x{W}'
            )

        # ====================================================
        # Build base model on every GPU
        # ====================================================

        base_model = build_lccnet(

            device=device,

            image_h=H,

            image_w=W,

            pretrained=(
                not args.no_imagenet_pretrained
            ),
                
            use_feat_from=(
                args.use_feat_from
            ),
        )

        total_params = sum(

            p.numel()

            for p in (
                base_model.parameters()
            )
        )

        rank_print()
        rank_print(
            '[MODEL]'
        )

        rank_print(
            'Total parameters =',
            total_params
        )

        # ====================================================
        # DDP wrapper
        #
        # find_unused_parameters=False:
        # faster and expected for current verified graph.
        #
        # gradient_as_bucket_view=True:
        # lowers gradient bucket memory overhead.
        # ====================================================

        model = DDP(

            base_model,

            device_ids=[
                local_rank
            ],

            output_device=(
                local_rank
            ),

            broadcast_buffers=True,

            find_unused_parameters=False,

            gradient_as_bucket_view=True,
        )

        # ====================================================
        # Forward sanity
        # ====================================================

        model.eval()

        with torch.no_grad():

            sanity_pred = model(

                first_batch[
                    'rgb'
                ][0:1],

                first_batch[
                    'depth'
                ][0:1],
            )

        if not torch.isfinite(
            sanity_pred
        ).all():

            raise RuntimeError(
                f'Rank {rank}: '
                'forward sanity failed.'
            )

        rank_print(
            'Forward sanity = PASS'
        )

        # Synchronize after build/sanity
        dist.barrier()

        # ====================================================
        # Optimizer
        #
        # Keep lr=1e-4 even though global batch becomes 2.
        #
        # Do NOT scale to 2e-4 yet.
        # Stability/generalization first.
        # ====================================================

        optimizer = torch.optim.AdamW(

            (
                p
                for p in model.parameters()
                if p.requires_grad
            ),

            lr=(
                args.lr
            ),

            weight_decay=(
                args.weight_decay
            ),

            betas=(
                0.9,
                0.999,
            ),
        )

        # ====================================================
        # Scheduler
        # ====================================================

        scheduler = None

        if args.scheduler == 'poly':

            scheduler = (
                torch.optim.lr_scheduler
                .PolynomialLR(

                    optimizer,

                    total_iters=(
                        args.epochs
                    ),

                    power=(
                        args.poly_power
                    ),
                )
            )

        elif args.scheduler == 'cosine':

            scheduler = (
                torch.optim.lr_scheduler
                .CosineAnnealingLR(

                    optimizer,

                    T_max=(
                        args.epochs
                    ),
                )
            )

        elif args.scheduler == 'none':

            scheduler = None

        else:

            raise ValueError(
                f'Unknown scheduler: '
                f'{args.scheduler}'
            )

        # ====================================================
        # Work directory
        # ====================================================

        work_dir = Path(
            args.work_dir
        )

        if rank == 0:

            work_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        dist.barrier()

        metrics_file = (
            work_dir
            / 'metrics.jsonl'
        )

        # ====================================================
        # Resume
        # ====================================================

        start_epoch = 1
        best_joint_score = float(
            'inf'
        )

        if args.resume is not None:

            (
                start_epoch,
                best_joint_score,
            ) = resume_training(

                args.resume,

                model,

                optimizer,

                scheduler,
            )

            rank_print()
            rank_print(
                '[RESUME]'
            )

            rank_print(
                'start_epoch =',
                start_epoch
            )

            rank_print(
                'best_joint_score =',
                best_joint_score
            )

        # ====================================================
        # Initial validation
        # ====================================================

        if not args.skip_initial_val:

            rank_print()
            rank_print(
                'Running DDP initial validation...'
            )

            initial_val = validate_ddp(

                model,

                val_loader,

                data_preprocessor,

                device,

                rank,

                world_size,

                args,
            )

            print_validation_metrics(

                epoch=0,

                metrics=(
                    initial_val
                ),
            )

        # ====================================================
        # Training
        # ====================================================

        for epoch in range(
            start_epoch,
            args.epochs + 1,
        ):

            rank_print()
            rank_print(
                '####################################################'
            )

            rank_print(
                f'EPOCH '
                f'{epoch}/{args.epochs}'
            )

            rank_print(
                '####################################################'
            )

            train_metrics = (
                train_one_epoch_ddp(

                    model,

                    train_loader,

                    data_preprocessor,

                    optimizer,

                    device,

                    rank,

                    world_size,

                    epoch,

                    args,
                )
            )

            rank_print()
            rank_print(
                '[TRAIN SUMMARY]'
            )

            rank_print(
                'Global frames =',
                train_metrics[
                    'frames'
                ]
            )

            rank_print(
                'Loss =',
                f"{train_metrics['loss']:.8f}"
            )

            rank_print(
                'Max grad norm =',
                f"{train_metrics['last_grad_norm']:.6f}"
            )

            # ================================================
            # Validation
            # ================================================

            val_metrics = validate_ddp(

                model,

                val_loader,

                data_preprocessor,

                device,

                rank,

                world_size,

                args,
            )

            print_validation_metrics(

                epoch,

                val_metrics,
            )

            # ================================================
            # Scheduler
            # ================================================

            if scheduler is not None:

                scheduler.step()

            # ================================================
            # Rank 0 only:
            # metrics/checkpoints
            # ================================================

            if rank == 0:

                record = dict(

                    epoch=epoch,

                    world_size=(
                        world_size
                    ),

                    train=(
                        train_metrics
                    ),

                    val=(
                        val_metrics
                    ),

                    lr=float(

                        optimizer
                        .param_groups[0][
                            'lr'
                        ]
                    ),
                )

                with open(

                    metrics_file,

                    'a',

                    encoding='utf-8',

                ) as f:

                    f.write(

                        json.dumps(

                            record,

                            ensure_ascii=False,
                        )
                    )

                    f.write(
                        '\n'
                    )

                # --------------------------------------------
                # Last model
                # --------------------------------------------

                save_model_checkpoint(

                    work_dir
                    / 'last_model.pth',

                    epoch,

                    model,

                    train_metrics,

                    val_metrics,

                    args,

                    world_size,
                )

                current_joint_score = float(

                    val_metrics[
                        'joint_score'
                    ]
                )

                # --------------------------------------------
                # Best
                # --------------------------------------------

                if (
                    current_joint_score
                    < best_joint_score
                ):

                    best_joint_score = (
                        current_joint_score
                    )

                    save_model_checkpoint(

                        work_dir
                        / 'best_joint_model.pth',

                        epoch,

                        model,

                        train_metrics,

                        val_metrics,

                        args,

                        world_size,
                    )

                    rank_print(
                        'NEW BEST JOINT MODEL'
                    )

                    rank_print(
                        'Joint score =',
                        f'{best_joint_score:.6f}'
                    )

                # --------------------------------------------
                # Optional huge resume state
                # --------------------------------------------

                if args.save_training_state:

                    save_training_state(

                        work_dir
                        / 'last_training_state.pth',

                        epoch,

                        model,

                        optimizer,

                        scheduler,

                        best_joint_score,

                        args,

                        world_size,
                    )

            # Other ranks wait until rank0 has finished I/O.
            dist.barrier()

        rank_print()
        rank_print(
            '===================================================='
        )

        rank_print(
            '[O-5C-7 DDP TRAINING COMPLETE]'
        )

        rank_print(
            'Best joint score =',
            best_joint_score
        )

        rank_print(
            'Work dir =',
            work_dir
        )

        rank_print(
            '===================================================='
        )

    finally:

        cleanup_distributed()


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(

        description=(
            'O-5C-7 LCCNet '
            'nuScenes 2-GPU DDP training'
        )
    )

    # torchrun compatibility
    parser.add_argument(

        '--local-rank',
        '--local_rank',

        dest='local_rank',

        type=int,

        default=int(
            os.environ.get(
                'LOCAL_RANK',
                0,
            )
        ),
    )

    parser.add_argument(

        '--dist-backend',

        type=str,

        default='nccl',
    )

    parser.add_argument(

        '--config',

        type=str,

        default=DEFAULT_CONFIG,
    )

    parser.add_argument(

        '--work-dir',

        type=str,

        default=DEFAULT_WORK_DIR,
    )

    parser.add_argument(

        '--epochs',

        type=int,

        default=10,
    )

    # GLOBAL frame counts
    parser.add_argument(

        '--train-max-frames',

        type=int,

        default=0,

        help=(
            'GLOBAL training frames across all ranks. '
            '0 = full dataset.'
        ),
    )

    parser.add_argument(

        '--val-max-frames',

        type=int,

        default=512,

        help=(
            'GLOBAL validation frames across all ranks. '
            '0 = full validation dataset.'
        ),
    )

    # PER-RANK workers
    parser.add_argument(

        '--num-workers',

        type=int,

        default=2,
    )

    parser.add_argument(

        '--seed',

        type=int,

        default=20260819,
    )

    parser.add_argument(

        '--val-seed',

        type=int,

        default=20260811,
    )

    # --------------------------------------------------------
    # Perturbation
    # --------------------------------------------------------

    parser.add_argument(

        '--train-max-rot-deg',

        type=float,

        default=10.0,
    )

    parser.add_argument(

        '--train-max-trans-m',

        type=float,

        default=0.75,
    )

    parser.add_argument(

        '--val-max-rot-deg',

        type=float,

        default=10.0,
    )

    parser.add_argument(

        '--val-max-trans-m',

        type=float,

        default=0.75,
    )

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    parser.add_argument(

        '--depth-scale',

        type=float,

        default=50.0,
    )

    parser.add_argument(

        '--input-color-order',

        type=str,

        choices=[
            'rgb',
            'bgr',
        ],

        default='rgb',
    )

    parser.add_argument(

        '--cam-chunk',

        type=int,

        default=1,
    )

    parser.add_argument(

        '--use-feat-from',

        type=int,

        default=2,

        choices=[
            1,
            2,
            3,
            4,
            5,
            6,
        ],

        help=(
            'LCCNet feature level. '
            'O-5C-7 baseline uses 2.'
        ),
    )

    parser.add_argument(
        '--rot-loss-weight',
        type=float,
        default=2.0,
    )

    parser.add_argument(
        '--trans-loss-weight',
        type=float,
        default=1.0,
    )

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    parser.add_argument(

        '--lr',

        type=float,

        default=1e-4,
    )

    parser.add_argument(

        '--weight-decay',

        type=float,

        default=1e-4,
    )

    parser.add_argument(

        '--grad-clip',

        type=float,

        default=1.0,
    )

    parser.add_argument(

        '--scheduler',

        type=str,

        choices=[
            'none',
            'poly',
            'cosine',
        ],

        default='poly',
    )

    parser.add_argument(

        '--poly-power',

        type=float,

        default=0.9,
    )

    # --------------------------------------------------------
    # Initialization
    # --------------------------------------------------------

    parser.add_argument(

        '--no-imagenet-pretrained',

        action='store_true',
    )

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    parser.add_argument(

        '--log-interval',

        type=int,

        default=25,

        help=(
            'Per-rank frame interval. '
            'With 2 GPUs, 25 means about '
            '50 global frames.'
        ),
    )

    parser.add_argument(

        '--val-log-interval',

        type=int,

        default=50,
    )

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    parser.add_argument(

        '--resume',

        type=str,

        default=None,
    )

    parser.add_argument(

        '--save-training-state',

        action='store_true',
    )

    parser.add_argument(

        '--skip-initial-val',

        action='store_true',
    )

    return parser.parse_args()


# ============================================================
# Entry
# ============================================================

if __name__ == '__main__':

    args = parse_args()

    rank_print(
        'LCCNet use_feat_from =',
        args.use_feat_from
    )

    rank_print(
        'Camera chunk =',
        args.cam_chunk
    )

    rank_print(
        'LCCNet use_feat_from =',
        args.use_feat_from
    )

    main(
        args
    )