# ============================================================
# projects/LCCNetModern/tools/quick_overfit_lccnet_nuscenes.py
#
# Gate O-5C-6
#
# Purpose:
#
#   ONE nuScenes frame
#   × SIX cameras
#   × ONE FIXED perturbation Delta
#   × FIXED RGB / broken depth / correction target
#
#   Repeat training for ~200 iterations.
#
# PASS condition:
#
#   L1 loss decreases
#   rotation residual decreases
#   translation residual decreases
#
# IMPORTANT:
#
#   Delta is generated ONCE before the training loop.
#   It MUST NOT be regenerated every iteration.
#
# ============================================================

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F


# ============================================================
# Reuse the already PASSed O-5C-2 ~ O-5C-5 implementation.
#
# Because this script is in the same tools/ directory,
# train_lccnet_nuscenes.py can be imported directly.
# ============================================================

from train_lccnet_nuscenes import (
    set_seed,
    build_dataloader_and_preprocessor,
    extract_batch_geometry,
    build_training_target,
    bevfusion_img_to_lccnet,
    build_lccnet,
)

from projects.LCCNetModern.calib_geometry import (
    build_calib_from_cam2lidar,
    project_lidar_to_sparse_depth,
)

from projects.LCCNetModern.lie import se3


# ============================================================
# Default config
# ============================================================

DEFAULT_CONFIG = (
    'projects/LCCNetModern/configs/'
    'lccnet_nuscenes_v110.py'
)


# ============================================================
# Helper:
# se3.exp for [M,6]
#
# Output:
#     [M,4,4]
# ============================================================

def prediction_to_matrix(
    pred_x: torch.Tensor,
) -> torch.Tensor:

    if pred_x.ndim != 2:

        raise ValueError(
            f"pred_x must be [M,6], "
            f"got {tuple(pred_x.shape)}"
        )

    if pred_x.shape[-1] != 6:

        raise ValueError(
            f"pred_x last dimension must be 6, "
            f"got {tuple(pred_x.shape)}"
        )

    return se3.exp(
        pred_x
    )


# ============================================================
# Calibration residual
#
# Corrected:
#
#   T_corrected = C_pred @ T_broken
#
# Residual:
#
#   E = T_corrected @ inv(T_GT)
#
# If prediction is perfect:
#
#   E = Identity
#
# Returns:
#
#   rot_error_deg : [M]
#   trans_error_m : [M]
# ============================================================

def compute_calibration_residual(
    correction: torch.Tensor,
    broken_c2l: torch.Tensor,
    gt_c2l: torch.Tensor,
):

    if correction.shape[-2:] != (
        4,
        4,
    ):
        raise ValueError(
            "correction must be [...,4,4]"
        )

    if broken_c2l.shape[-2:] != (
        4,
        4,
    ):
        raise ValueError(
            "broken_c2l must be [...,4,4]"
        )

    if gt_c2l.shape[-2:] != (
        4,
        4,
    ):
        raise ValueError(
            "gt_c2l must be [...,4,4]"
        )

    # --------------------------------------------------------
    # Apply predicted correction
    # --------------------------------------------------------

    corrected_c2l = (
        correction
        @ broken_c2l
    )

    # --------------------------------------------------------
    # Residual transform
    #
    # Perfect:
    # corrected @ inv(gt) = Identity
    # --------------------------------------------------------

    residual = (
        corrected_c2l
        @ torch.linalg.inv(
            gt_c2l
        )
    )

    # --------------------------------------------------------
    # Translation error
    # --------------------------------------------------------

    trans_error_m = (
        torch.linalg.norm(
            residual[
                ...,
                :3,
                3
            ],
            dim=-1,
        )
    )

    # --------------------------------------------------------
    # Rotation geodesic error
    #
    # theta =
    # acos((trace(R)-1)/2)
    # --------------------------------------------------------

    R = residual[
        ...,
        :3,
        :3
    ]

    trace = (
        R[..., 0, 0]
        + R[..., 1, 1]
        + R[..., 2, 2]
    )

    cos_theta = (
        (trace - 1.0)
        / 2.0
    )

    cos_theta = torch.clamp(
        cos_theta,
        min=-1.0,
        max=1.0,
    )

    rot_error_rad = (
        torch.acos(
            cos_theta
        )
    )

    rot_error_deg = (
        torch.rad2deg(
            rot_error_rad
        )
    )

    return (
        rot_error_deg,
        trans_error_m,
    )


# ============================================================
# Metrics summarizer
# ============================================================

def summarize_errors(
    rot_error_deg: torch.Tensor,
    trans_error_m: torch.Tensor,
):

    return {

        'rot_mean_deg':
            rot_error_deg.mean().item(),

        'rot_median_deg':
            rot_error_deg.median().item(),

        'rot_max_deg':
            rot_error_deg.max().item(),

        'trans_mean_m':
            trans_error_m.mean().item(),

        'trans_median_m':
            trans_error_m.median().item(),

        'trans_max_m':
            trans_error_m.max().item(),
    }


# ============================================================
# Forward in camera chunks
#
# Used for metric evaluation.
#
# model.eval()
# torch.no_grad()
#
# Input:
#     RGB   [M,3,H,W]
#     depth [M,1,H,W]
#
# Output:
#     pred  [M,6]
# ============================================================

@torch.no_grad()
def evaluate_model(
    model,
    rgb,
    depth,
    target,
    broken_c2l,
    gt_c2l,
    chunk_size,
):

    model.eval()

    predictions = []

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
            rgb[start:end],
            depth[start:end],
        )

        predictions.append(
            pred
        )

    pred_x = torch.cat(
        predictions,
        dim=0,
    )

    # --------------------------------------------------------
    # L1
    # --------------------------------------------------------

    loss = F.l1_loss(
        pred_x,
        target,
        reduction='mean',
    )

    # --------------------------------------------------------
    # Prediction -> SE(3)
    # --------------------------------------------------------

    correction_pred = (
        prediction_to_matrix(
            pred_x
        )
    )

    # --------------------------------------------------------
    # Calibration residual
    # --------------------------------------------------------

    (
        rot_error_deg,
        trans_error_m,
    ) = compute_calibration_residual(

        correction_pred,

        broken_c2l,

        gt_c2l,
    )

    metrics = summarize_errors(
        rot_error_deg,
        trans_error_m,
    )

    metrics[
        'loss'
    ] = loss.item()

    return (
        pred_x,
        metrics,
    )


# ============================================================
# Print metrics
# ============================================================

def print_metrics(
    iteration,
    metrics,
):

    print(
        f"\n[QUICK OVERFIT] "
        f"iter={iteration}"
    )

    print(
        f"  L1 loss          = "
        f"{metrics['loss']:.8f}"
    )

    print(
        f"  Rot mean         = "
        f"{metrics['rot_mean_deg']:.6f} deg"
    )

    print(
        f"  Rot median       = "
        f"{metrics['rot_median_deg']:.6f} deg"
    )

    print(
        f"  Rot max          = "
        f"{metrics['rot_max_deg']:.6f} deg"
    )

    print(
        f"  Trans mean       = "
        f"{metrics['trans_mean_m']:.6f} m"
    )

    print(
        f"  Trans median     = "
        f"{metrics['trans_median_m']:.6f} m"
    )

    print(
        f"  Trans max        = "
        f"{metrics['trans_max_m']:.6f} m"
    )


# ============================================================
# MAIN
# ============================================================

def main(
    args,
):

    # ========================================================
    # Seed
    #
    # This makes the ONE fixed training sample and Delta
    # reproducible for this quick-overfit experiment.
    # ========================================================

    set_seed(
        args.seed
    )

    device = torch.device(
        args.device
    )

    print(
        "\n===================================================="
    )

    print(
        "Gate O-5C-6"
    )

    print(
        "LCCNet Single-Batch Quick Overfit"
    )

    print(
        "===================================================="
    )

    print(
        "Config =",
        args.config
    )

    print(
        "Device =",
        device
    )

    print(
        "Seed =",
        args.seed
    )

    print(
        "Iterations =",
        args.iterations
    )

    print(
        "LR =",
        args.lr
    )

    print(
        "max_rot_deg =",
        args.max_rot_deg
    )

    print(
        "max_trans_m =",
        args.max_trans_m
    )

    print(
        "cam_chunk =",
        args.cam_chunk
    )

    # ========================================================
    # 1.
    # Build nuScenes dataloader
    # ========================================================

    (
        cfg,
        train_loader,
        data_preprocessor,
    ) = build_dataloader_and_preprocessor(

        config_path=(
            args.config
        ),

        device=device,

        batch_size=1,

        num_workers=(
            args.num_workers
        ),

        seed=args.seed,
    )

    # ========================================================
    # 2.
    # Fetch EXACTLY ONE batch
    #
    # This batch is NEVER changed during quick-overfit.
    # ========================================================

    raw_batch = next(
        iter(
            train_loader
        )
    )

    (
        imgs,
        points,
        camera2lidar_gt,
        camera_intrinsics,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
    ) = extract_batch_geometry(

        raw_batch,

        data_preprocessor,

        device,
    )

    B = imgs.shape[0]
    N = imgs.shape[1]

    H = imgs.shape[-2]
    W = imgs.shape[-1]

    print(
        "\n[FIXED INPUT]"
    )

    print(
        "imgs =",
        tuple(
            imgs.shape
        )
    )

    print(
        "points =",
        tuple(
            points[0].shape
        )
    )

    print(
        "GT C2L =",
        tuple(
            camera2lidar_gt.shape
        )
    )

    # ========================================================
    # 3.
    # Generate ONE FIXED Delta
    #
    # CRITICAL:
    #
    # build_training_target() is called ONCE here,
    # OUTSIDE the training loop.
    #
    # Therefore:
    #
    #   Delta
    #   Broken C2L
    #   GT correction
    #   SE3 target
    #
    # remain FIXED during all iterations.
    # ========================================================

    target_info = (
        build_training_target(

            camera2lidar_gt,

            max_rot_deg=(
                args.max_rot_deg
            ),

            max_trans_m=(
                args.max_trans_m
            ),
        )
    )

    broken_c2l = (
        target_info[
            'broken_c2l'
        ]
    )

    correction_gt = (
        target_info[
            'correction_gt'
        ]
    )

    target_x = (
        target_info[
            'target_x'
        ]
    )

    delta_gt = (
        target_info[
            'delta_gt'
        ]
    )

    restore_err = (
        target_info[
            'restore_err'
        ]
    )

    se3_err = (
        target_info[
            'se3_err'
        ]
    )

    debug = (
        target_info[
            'debug'
        ]
    )

    print(
        "\n[FIXED PERTURBATION]"
    )

    print(
        "Oracle restore error =",
        f"{restore_err.item():.8e}"
    )

    print(
        "SE3 log-exp error =",
        f"{se3_err.item():.8e}"
    )

    print(
        "First camera perturb =",
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

    assert (
        restore_err.item()
        < 1e-5
    )

    assert (
        se3_err.item()
        < 1e-5
    )

    # ========================================================
    # 4.
    # Generate ONE FIXED broken sparse depth
    # ========================================================

    broken_calib = (
        build_calib_from_cam2lidar(

            broken_c2l,

            camera_intrinsics,
        )
    )

    broken_depth = (
        project_lidar_to_sparse_depth(

            points,

            broken_calib[
                'lidar2img'
            ],

            img_aug_matrix,

            lidar_aug_matrix,

            image_hw=(
                H,
                W,
            ),
        )
    )

    depth_lcc = (
        broken_depth
        / args.depth_scale
    )

    # ========================================================
    # 5.
    # Fixed RGB
    # ========================================================

    rgb_lcc = (
        bevfusion_img_to_lccnet(

            imgs,

            input_color_order=(
                args.input_color_order
            ),
        )
    )

    # ========================================================
    # Flatten:
    #
    # [B,N,...]
    #
    # ->
    #
    # [B*N,...]
    # ========================================================

    rgb_lcc = (
        rgb_lcc.reshape(
            B * N,
            3,
            H,
            W,
        )
        .detach()
    )

    depth_lcc = (
        depth_lcc.reshape(
            B * N,
            1,
            H,
            W,
        )
        .detach()
    )

    target_lcc = (
        target_x.reshape(
            B * N,
            6,
        )
        .detach()
    )

    broken_flat = (
        broken_c2l.reshape(
            B * N,
            4,
            4,
        )
        .detach()
    )

    gt_flat = (
        camera2lidar_gt.reshape(
            B * N,
            4,
            4,
        )
        .detach()
    )

    delta_flat = (
        delta_gt.reshape(
            B * N,
            4,
            4,
        )
        .detach()
    )

    correction_gt_flat = (
        correction_gt.reshape(
            B * N,
            4,
            4,
        )
        .detach()
    )

    print(
        "\n[FIXED LCCNET INPUT]"
    )

    print(
        "RGB =",
        tuple(
            rgb_lcc.shape
        )
    )

    print(
        "Depth =",
        tuple(
            depth_lcc.shape
        )
    )

    print(
        "Target =",
        tuple(
            target_lcc.shape
        )
    )

    print(
        "Depth nonzero =",
        (
            depth_lcc > 0
        ).sum().item()
    )

    # ========================================================
    # 6.
    # Calculate BROKEN baseline residual
    #
    # Before LCCNet:
    #
    # correction = Identity
    # ========================================================

    identity_correction = (
        torch.eye(
            4,
            device=device,
            dtype=broken_flat.dtype,
        )
        .unsqueeze(0)
        .repeat(
            B * N,
            1,
            1,
        )
    )

    (
        broken_rot,
        broken_trans,
    ) = compute_calibration_residual(

        identity_correction,

        broken_flat,

        gt_flat,
    )

    broken_metrics = (
        summarize_errors(
            broken_rot,
            broken_trans,
        )
    )

    print(
        "\n===================================================="
    )

    print(
        "[BROKEN BASELINE]"
    )

    print(
        "===================================================="
    )

    print(
        f"Rotation mean   = "
        f"{broken_metrics['rot_mean_deg']:.6f} deg"
    )

    print(
        f"Rotation median = "
        f"{broken_metrics['rot_median_deg']:.6f} deg"
    )

    print(
        f"Translation mean   = "
        f"{broken_metrics['trans_mean_m']:.6f} m"
    )

    print(
        f"Translation median = "
        f"{broken_metrics['trans_median_m']:.6f} m"
    )

    # ========================================================
    # 7.
    # Oracle residual sanity check
    #
    # Must be approximately zero.
    # ========================================================

    (
        oracle_rot,
        oracle_trans,
    ) = compute_calibration_residual(

        correction_gt_flat,

        broken_flat,

        gt_flat,
    )

    oracle_metrics = (
        summarize_errors(
            oracle_rot,
            oracle_trans,
        )
    )

    print(
        "\n===================================================="
    )

    print(
        "[ORACLE SANITY CHECK]"
    )

    print(
        "===================================================="
    )

    print(
        f"Rotation mean   = "
        f"{oracle_metrics['rot_mean_deg']:.8f} deg"
    )

    print(
        f"Translation mean = "
        f"{oracle_metrics['trans_mean_m']:.8e} m"
    )

    # Floating-point acos can produce a very small
    # non-zero rotation value. Do not require exactly zero.
    assert (
        oracle_metrics[
            'trans_mean_m'
        ]
        < 1e-5
    )

    # ========================================================
    # 8.
    # Build LCCNet
    # ========================================================

    model = build_lccnet(

        device=device,

        image_h=H,

        image_w=W,

        pretrained=(
            args.pretrained
        ),
    )

    total_parameters = sum(
        p.numel()
        for p
        in model.parameters()
    )

    print(
        "\n[LCCNET]"
    )

    print(
        "Total parameters =",
        total_parameters
    )

    # ========================================================
    # 9.
    # AdamW
    # ========================================================

    optimizer = (
        torch.optim.AdamW(

            model.parameters(),

            lr=args.lr,

            betas=(
                0.9,
                0.999,
            ),
        )
    )

    # ========================================================
    # 10.
    # BEFORE TRAINING evaluation
    # ========================================================

    (
        _,
        initial_metrics,
    ) = evaluate_model(

        model,

        rgb_lcc,

        depth_lcc,

        target_lcc,

        broken_flat,

        gt_flat,

        chunk_size=(
            args.cam_chunk
        ),
    )

    print(
        "\n===================================================="
    )

    print(
        "[INITIAL LCCNET]"
    )

    print(
        "===================================================="
    )

    print_metrics(
        0,
        initial_metrics,
    )

    # ========================================================
    # Save initial metrics
    # ========================================================

    initial_loss = (
        initial_metrics[
            'loss'
        ]
    )

    initial_rot = (
        initial_metrics[
            'rot_mean_deg'
        ]
    )

    initial_trans = (
        initial_metrics[
            'trans_mean_m'
        ]
    )

    # ========================================================
    # 11.
    # QUICK OVERFIT LOOP
    #
    # FIXED:
    #
    #   rgb_lcc
    #   depth_lcc
    #   target_lcc
    #   broken_flat
    #
    # NOTHING is regenerated.
    # ========================================================

    M = rgb_lcc.shape[0]

    total_target_elements = (
        target_lcc.numel()
    )

    for iteration in range(
        1,
        args.iterations + 1,
    ):

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        training_loss = 0.0

        # ====================================================
        # Camera chunk gradient accumulation
        #
        # For M=6 and chunk=1:
        #
        #   camera0 forward/backward
        #   camera1 forward/backward
        #   ...
        #   camera5 forward/backward
        #
        # then one optimizer.step()
        #
        # Equivalent to a global mean L1 across six cameras.
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

            pred_chunk = model(

                rgb_lcc[
                    start:end
                ],

                depth_lcc[
                    start:end
                ],
            )

            target_chunk = (
                target_lcc[
                    start:end
                ]
            )

            chunk_loss = (
                F.l1_loss(

                    pred_chunk,

                    target_chunk,

                    reduction='sum',
                )
                / float(
                    total_target_elements
                )
            )

            if not torch.isfinite(
                chunk_loss
            ):

                raise RuntimeError(
                    "Non-finite training loss "
                    f"at iteration {iteration}"
                )

            chunk_loss.backward()

            training_loss += (
                chunk_loss
                .detach()
                .item()
            )

        # ====================================================
        # Gradient finite check
        # ====================================================

        grad_ok = True

        for name, p in (
            model.named_parameters()
        ):

            if p.grad is None:
                continue

            if not torch.isfinite(
                p.grad
            ).all():

                print(
                    "NON-FINITE GRAD:",
                    name
                )

                grad_ok = False

        if not grad_ok:

            raise RuntimeError(
                "Non-finite gradient detected."
            )

        # ====================================================
        # Gradient clipping
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

        # ====================================================
        # Weight update
        # ====================================================

        optimizer.step()

        # ====================================================
        # Evaluation
        # ====================================================

        should_log = (

            iteration == 1

            or iteration
            % args.log_interval
            == 0

            or iteration
            == args.iterations
        )

        if should_log:

            (
                _,
                current_metrics,
            ) = evaluate_model(

                model,

                rgb_lcc,

                depth_lcc,

                target_lcc,

                broken_flat,

                gt_flat,

                chunk_size=(
                    args.cam_chunk
                ),
            )

            print_metrics(
                iteration,
                current_metrics,
            )

            print(
                f"  Train grad norm  = "
                f"{float(grad_norm):.6f}"
            )

    # ========================================================
    # 12.
    # FINAL METRICS
    # ========================================================

    (
        final_pred,
        final_metrics,
    ) = evaluate_model(

        model,

        rgb_lcc,

        depth_lcc,

        target_lcc,

        broken_flat,

        gt_flat,

        chunk_size=(
            args.cam_chunk
        ),
    )

    final_loss = (
        final_metrics[
            'loss'
        ]
    )

    final_rot = (
        final_metrics[
            'rot_mean_deg'
        ]
    )

    final_trans = (
        final_metrics[
            'trans_mean_m'
        ]
    )

    # ========================================================
    # Improvement ratios
    # ========================================================

    loss_reduction = (
        1.0
        - final_loss
        / max(
            initial_loss,
            1e-12,
        )
    )

    rot_reduction = (
        1.0
        - final_rot
        / max(
            initial_rot,
            1e-12,
        )
    )

    trans_reduction = (
        1.0
        - final_trans
        / max(
            initial_trans,
            1e-12,
        )
    )

    print(
        "\n===================================================="
    )

    print(
        "[O-5C-6 FINAL RESULT]"
    )

    print(
        "===================================================="
    )

    print(
        "Initial L1 =",
        initial_loss
    )

    print(
        "Final L1   =",
        final_loss
    )

    print(
        "Loss reduction =",
        f"{loss_reduction * 100.0:.2f}%"
    )

    print()

    print(
        "Initial rotation mean =",
        f"{initial_rot:.6f} deg"
    )

    print(
        "Final rotation mean   =",
        f"{final_rot:.6f} deg"
    )

    print(
        "Rotation reduction =",
        f"{rot_reduction * 100.0:.2f}%"
    )

    print()

    print(
        "Initial translation mean =",
        f"{initial_trans:.6f} m"
    )

    print(
        "Final translation mean   =",
        f"{final_trans:.6f} m"
    )

    print(
        "Translation reduction =",
        f"{trans_reduction * 100.0:.2f}%"
    )

    # ========================================================
    # 13.
    # PASS / FAIL
    #
    # Quick-overfit purpose is simply proving that all THREE:
    #
    #     loss
    #     rotation residual
    #     translation residual
    #
    # decrease on the fixed sample.
    # ========================================================

    pass_loss = (
        final_loss
        < initial_loss
    )

    pass_rot = (
        final_rot
        < initial_rot
    )

    pass_trans = (
        final_trans
        < initial_trans
    )

    print(
        "\nPASS loss  =",
        pass_loss
    )

    print(
        "PASS rot   =",
        pass_rot
    )

    print(
        "PASS trans =",
        pass_trans
    )

    quick_overfit_pass = (
        pass_loss
        and pass_rot
        and pass_trans
    )

    # ========================================================
    # 14.
    # Save debugging checkpoint
    # ========================================================

    work_dir = Path(
        args.work_dir
    )

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        work_dir
        / (
            f'quick_overfit_'
            f'{args.iterations}iter.pth'
        )
    )

    torch.save(

        {
            'gate':
                'O-5C-6',

            'iterations':
                args.iterations,

            'seed':
                args.seed,

            'max_rot_deg':
                args.max_rot_deg,

            'max_trans_m':
                args.max_trans_m,

            'depth_scale':
                args.depth_scale,

            'model':
                model.state_dict(),

            'optimizer':
                optimizer.state_dict(),

            'initial_metrics':
                initial_metrics,

            'final_metrics':
                final_metrics,

            'broken_metrics':
                broken_metrics,

            'oracle_metrics':
                oracle_metrics,

            'quick_overfit_pass':
                quick_overfit_pass,
        },

        checkpoint_path,
    )

    print(
        "\nCheckpoint saved:"
    )

    print(
        checkpoint_path
    )

    # ========================================================
    # FINAL GATE
    # ========================================================

    print(
        "\n===================================================="
    )

    if quick_overfit_pass:

        print(
            "[GATE O-5C-6] PASS"
        )

        print(
            "Loss / rotation / translation "
            "all decreased."
        )

        print(
            "Next step: O-5C-7 full training."
        )

    else:

        print(
            "[GATE O-5C-6] FAIL"
        )

        print(
            "DO NOT start full training yet."
        )

        print(
            "Inspect loss / rotation / "
            "translation trends first."
        )

    print(
        "====================================================\n"
    )


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(

        description=(
            "Gate O-5C-6: "
            "LCCNet single-batch quick-overfit"
        )
    )

    parser.add_argument(

        '--config',

        type=str,

        default=DEFAULT_CONFIG,
    )

    parser.add_argument(

        '--device',

        type=str,

        default='cuda:0',
    )

    parser.add_argument(

        '--num-workers',

        type=int,

        default=0,
    )

    parser.add_argument(

        '--seed',

        type=int,

        default=20260811,
    )

    parser.add_argument(

        '--iterations',

        type=int,

        default=200,
    )

    parser.add_argument(

        '--log-interval',

        type=int,

        default=10,
    )

    parser.add_argument(

        '--max-rot-deg',

        type=float,

        default=10.0,
    )

    parser.add_argument(

        '--max-trans-m',

        type=float,

        default=0.75,
    )

    parser.add_argument(

        '--depth-scale',

        type=float,

        default=50.0,
    )

    parser.add_argument(

        '--lr',

        type=float,

        default=1e-4,
    )

    parser.add_argument(

        '--grad-clip',

        type=float,

        default=1.0,
    )

    parser.add_argument(

        '--cam-chunk',

        type=int,

        default=1,
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

        '--pretrained',

        action=argparse.BooleanOptionalAction,

        default=True,
    )

    parser.add_argument(

        '--work-dir',

        type=str,

        default=(
            'work_dirs/'
            'lccnet_nuscenes_v110'
        ),
    )

    return parser.parse_args()


# ============================================================
# Entry
# ============================================================

if __name__ == '__main__':

    args = parse_args()

    main(
        args
    )