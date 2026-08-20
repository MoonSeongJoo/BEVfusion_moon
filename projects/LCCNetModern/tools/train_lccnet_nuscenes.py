# ============================================================
# projects/LCCNetModern/tools/train_lccnet_nuscenes.py
#
# O-5C-7
# LCCNet nuScenes Training
#
# Strategy A:
#
#   BEVFusion input : 256 x 704  (frozen / unchanged)
#   LCCNet input    : 256 x 704
#
#   Calibration-specific pretrained checkpoint:
#       NOT USED
#
#   RGB ResNet-18:
#       ImageNet initialization
#
# Training:
#
#   Every training frame:
#       new random Delta
#
#   T_broken = Delta @ T_GT
#   C_GT     = inv(Delta)
#   target   = log(C_GT)
#
#   LCCNet:
#       RGB + physically-consistent broken sparse depth
#       -> pred_x [6]
#
#   C_pred = exp(pred_x)
#   T_corrected = C_pred @ T_broken
#
# Validation:
#
#   fixed deterministic perturbation
#   same perturbation for each sample/camera across epochs
#
# Output:
#
#   last_model.pth
#   best_joint_model.pth
#   metrics.jsonl
#
# Optional:
#
#   last_training_state.pth
#   (model + optimizer + scheduler for resume)
#
# Compatibility:
#
#   The following functions are intentionally kept because
#   quick_overfit_lccnet_nuscenes.py and
#   diagnose_quick_overfit_per_camera.py import them:
#
#       set_seed
#       build_dataloader_and_preprocessor
#       extract_batch_geometry
#       build_training_target
#       bevfusion_img_to_lccnet
#       build_lccnet
#
# ============================================================

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner

from mmdet3d.registry import MODELS
from mmdet3d.utils import register_all_modules

from projects.LCCNetModern.lccnet import LCCNet

from projects.LCCNetModern.calib_geometry import (
    build_calib_from_cam2lidar,
    project_lidar_to_sparse_depth,
)

from projects.LCCNetModern.lie import se3


# ============================================================
# Defaults
# ============================================================

DEFAULT_CONFIG = (
    'projects/LCCNetModern/configs/'
    'lccnet_nuscenes_v110.py'
)

DEFAULT_WORK_DIR = (
    'work_dirs/'
    'lccnet_nuscenes_v110_256x704'
)

NUSC_CAMERA_NAMES = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Data preprocessor
# ============================================================

def build_data_preprocessor(
    cfg: Config,
    device: torch.device,
):

    dp_cfg = None

    # --------------------------------------------------------
    # LCCNet-specific top-level data_preprocessor
    # --------------------------------------------------------

    if cfg.get(
        'data_preprocessor',
        None,
    ) is not None:

        dp_cfg = copy.deepcopy(
            cfg.data_preprocessor
        )

    # --------------------------------------------------------
    # Fallback:
    # model.data_preprocessor
    # --------------------------------------------------------

    elif (
        cfg.get(
            'model',
            None,
        ) is not None
        and isinstance(
            cfg.model,
            dict,
        )
        and cfg.model.get(
            'data_preprocessor',
            None,
        ) is not None
    ):

        dp_cfg = copy.deepcopy(
            cfg.model[
                'data_preprocessor'
            ]
        )

    # --------------------------------------------------------
    # Minimal default
    # --------------------------------------------------------

    else:

        dp_cfg = dict(
            type='Det3DDataPreprocessor',
        )

    data_preprocessor = (
        MODELS.build(
            dp_cfg
        )
    )

    data_preprocessor = (
        data_preprocessor.to(
            device
        )
    )

    data_preprocessor.eval()

    return data_preprocessor


# ============================================================
# Build dataloader
# ============================================================

def build_one_dataloader(
    cfg: Config,
    split: str,
    batch_size: int,
    num_workers: int,
    seed: int,
):

    if split == 'train':

        if cfg.get(
            'train_dataloader',
            None,
        ) is None:

            raise KeyError(
                'Config does not contain train_dataloader'
            )

        dataloader_cfg = copy.deepcopy(
            cfg.train_dataloader
        )

        shuffle = True

    elif split == 'val':

        if cfg.get(
            'val_dataloader',
            None,
        ) is None:

            raise KeyError(
                'Config does not contain val_dataloader'
            )

        dataloader_cfg = copy.deepcopy(
            cfg.val_dataloader
        )

        shuffle = False

    else:

        raise ValueError(
            f'Unknown split: {split}'
        )

    # --------------------------------------------------------
    # Force frame batch = user argument
    # --------------------------------------------------------

    dataloader_cfg[
        'batch_size'
    ] = batch_size

    dataloader_cfg[
        'num_workers'
    ] = num_workers

    # --------------------------------------------------------
    # persistent_workers cannot be True with num_workers=0
    # --------------------------------------------------------

    if num_workers == 0:

        dataloader_cfg[
            'persistent_workers'
        ] = False

    # --------------------------------------------------------
    # Ensure sampler exists
    # --------------------------------------------------------

    if dataloader_cfg.get(
        'sampler',
        None,
    ) is None:

        dataloader_cfg[
            'sampler'
        ] = dict(
            type='DefaultSampler',
            shuffle=shuffle,
        )

    elif isinstance(
        dataloader_cfg[
            'sampler'
        ],
        dict,
    ):

        dataloader_cfg[
            'sampler'
        ][
            'shuffle'
        ] = shuffle

    dataloader = (
        Runner.build_dataloader(

            dataloader_cfg,

            seed=seed,
        )
    )

    return dataloader


# ============================================================
# Backward-compatible helper used by O-5C-6
# ============================================================

def build_dataloader_and_preprocessor(
    config_path: str,
    device: torch.device,
    batch_size: int = 1,
    num_workers: int = 0,
    seed: int = 20260811,
):

    register_all_modules(
        init_default_scope=True
    )

    cfg = Config.fromfile(
        config_path
    )

    default_scope = cfg.get(
        'default_scope',
        'mmdet3d',
    )

    init_default_scope(
        default_scope
    )

    train_loader = (
        build_one_dataloader(

            cfg,

            split='train',

            batch_size=batch_size,

            num_workers=num_workers,

            seed=seed,
        )
    )

    data_preprocessor = (
        build_data_preprocessor(

            cfg,

            device,
        )
    )

    return (
        cfg,
        train_loader,
        data_preprocessor,
    )


# ============================================================
# Utility:
# Tensor conversion
# ============================================================

def _as_tensor(
    value,
    device,
    dtype=torch.float32,
):

    if torch.is_tensor(
        value
    ):

        return value.to(
            device=device,
            dtype=dtype,
        )

    if isinstance(
        value,
        (list, tuple),
    ):

        if (
            len(value) > 0
            and torch.is_tensor(
                value[0]
            )
        ):

            return torch.stack(
                [
                    x.to(
                        device=device,
                        dtype=dtype,
                    )
                    for x in value
                ],
                dim=0,
            )

    array = np.asarray(
        value
    )

    return torch.as_tensor(
        array,
        device=device,
        dtype=dtype,
    )


# ============================================================
# Promote matrix to 4x4 homogeneous
# ============================================================

def _promote_to_4x4(
    matrix: torch.Tensor,
):

    if matrix.shape[-2:] == (
        4,
        4,
    ):

        return matrix

    shape_prefix = matrix.shape[:-2]

    output = torch.eye(
        4,
        device=matrix.device,
        dtype=matrix.dtype,
    )

    output = output.reshape(
        *((1,) * len(shape_prefix)),
        4,
        4,
    )

    output = output.expand(
        *shape_prefix,
        4,
        4,
    ).clone()

    if matrix.shape[-2:] == (
        3,
        3,
    ):

        output[
            ...,
            :3,
            :3
        ] = matrix

        return output

    if matrix.shape[-2:] == (
        3,
        4,
    ):

        output[
            ...,
            :3,
            :4
        ] = matrix

        return output

    raise ValueError(
        'Unsupported matrix shape: '
        f'{tuple(matrix.shape)}'
    )


# ============================================================
# Stack camera metadata
#
# Output:
#     [B,N,4,4]
# ============================================================

def _stack_camera_meta(
    metas: List[dict],
    key: str,
    device: torch.device,
    num_cams: int,
    required: bool = True,
):

    batch = []

    for meta in metas:

        value = meta.get(
            key,
            None,
        )

        if value is None:

            if required:

                raise KeyError(
                    f'Metadata does not contain "{key}"'
                )

            identity = (
                torch.eye(
                    4,
                    device=device,
                    dtype=torch.float32,
                )
                .unsqueeze(0)
                .repeat(
                    num_cams,
                    1,
                    1,
                )
            )

            batch.append(
                identity
            )

            continue

        tensor = _as_tensor(
            value,
            device,
        )

        tensor = (
            _promote_to_4x4(
                tensor
            )
        )

        if tensor.ndim == 2:

            tensor = (
                tensor
                .unsqueeze(0)
            )

        if (
            tensor.shape[0] == 1
            and num_cams > 1
        ):

            tensor = tensor.repeat(
                num_cams,
                1,
                1,
            )

        if tensor.shape[0] != num_cams:

            raise RuntimeError(
                f'{key}: expected {num_cams} cameras, '
                f'got {tuple(tensor.shape)}'
            )

        batch.append(
            tensor
        )

    return torch.stack(
        batch,
        dim=0,
    )


# ============================================================
# Stack lidar augmentation metadata
#
# Output:
#     [B,4,4]
# ============================================================

def _stack_lidar_aug(
    metas: List[dict],
    device: torch.device,
):

    matrices = []

    for meta in metas:

        value = meta.get(
            'lidar_aug_matrix',
            None,
        )

        if value is None:

            tensor = torch.eye(
                4,
                device=device,
                dtype=torch.float32,
            )

        else:

            tensor = _as_tensor(
                value,
                device,
            )

            tensor = (
                _promote_to_4x4(
                    tensor
                )
            )

            if tensor.ndim == 3:

                if tensor.shape[0] != 1:

                    raise RuntimeError(
                        'Unexpected lidar_aug_matrix shape: '
                        f'{tuple(tensor.shape)}'
                    )

                tensor = tensor[0]

        matrices.append(
            tensor
        )

    return torch.stack(
        matrices,
        dim=0,
    )


# ============================================================
# Extract geometry from MMDetection3D batch
#
# Returns:
#
#   imgs                [B,N,3,H,W]
#   points              list[B] of [P,C]
#   cam2lidar            [B,N,4,4]
#   cam2img              [B,N,4,4]
#   img_aug_matrix       [B,N,4,4]
#   lidar_aug_matrix     [B,4,4]
#   metas                list[dict]
#
# ============================================================

@torch.no_grad()
def extract_batch_geometry(
    raw_batch,
    data_preprocessor,
    device,
):

    processed = data_preprocessor(
        raw_batch,
        training=False,
    )

    if 'inputs' not in processed:

        raise KeyError(
            'Processed batch has no "inputs"'
        )

    inputs = processed[
        'inputs'
    ]

    data_samples = processed.get(
        'data_samples',
        None,
    )

    if data_samples is None:

        raise KeyError(
            'Processed batch has no "data_samples"'
        )

    # --------------------------------------------------------
    # Images
    # --------------------------------------------------------

    imgs = inputs.get(
        'imgs',
        None,
    )

    if imgs is None:

        imgs = inputs.get(
            'img',
            None,
        )

    if imgs is None:

        raise KeyError(
            'Neither inputs["imgs"] nor inputs["img"] exists.'
        )

    if isinstance(
        imgs,
        (list, tuple),
    ):

        imgs = torch.stack(
            imgs,
            dim=0,
        )

    imgs = imgs.to(
        device=device,
        dtype=torch.float32,
    )

    # One frame with six views may occasionally be [6,3,H,W].
    if imgs.ndim == 4:

        if len(
            data_samples
        ) == 1:

            imgs = (
                imgs.unsqueeze(0)
            )

    if imgs.ndim != 5:

        raise RuntimeError(
            'Expected imgs [B,N,3,H,W], '
            f'got {tuple(imgs.shape)}'
        )

    B = imgs.shape[0]
    N = imgs.shape[1]

    # --------------------------------------------------------
    # Points
    # --------------------------------------------------------

    points = inputs.get(
        'points',
        None,
    )

    if points is None:

        raise KeyError(
            'inputs["points"] does not exist.'
        )

    if torch.is_tensor(
        points
    ):

        if points.ndim == 2:

            if B != 1:

                raise RuntimeError(
                    '2-D points tensor but batch size != 1'
                )

            points = [
                points
            ]

        elif points.ndim == 3:

            points = [
                points[i]
                for i in range(
                    points.shape[0]
                )
            ]

    points = [
        p.to(
            device=device,
            dtype=torch.float32,
        )
        for p in points
    ]

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metas = []

    for sample in data_samples:

        if hasattr(
            sample,
            'metainfo',
        ):

            meta = sample.metainfo

        elif isinstance(
            sample,
            dict,
        ):

            meta = sample

        else:

            raise TypeError(
                'Unsupported data_sample type: '
                f'{type(sample)}'
            )

        metas.append(
            dict(meta)
        )

    if len(
        metas
    ) != B:

        raise RuntimeError(
            f'Metadata batch mismatch: '
            f'B={B}, metas={len(metas)}'
        )

    # --------------------------------------------------------
    # Calibration matrices
    # --------------------------------------------------------

    cam2lidar = (
        _stack_camera_meta(

            metas,

            'cam2lidar',

            device,

            N,

            required=True,
        )
    )

    cam2img = (
        _stack_camera_meta(

            metas,

            'cam2img',

            device,

            N,

            required=True,
        )
    )

    img_aug_matrix = (
        _stack_camera_meta(

            metas,

            'img_aug_matrix',

            device,

            N,

            required=False,
        )
    )

    lidar_aug_matrix = (
        _stack_lidar_aug(

            metas,

            device,
        )
    )

    return (
        imgs,
        points,
        cam2lidar,
        cam2img,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
    )


# ============================================================
# RGB preprocessing for LCCNet
#
# Input:
#     BEVFusion image tensor [B,N,3,H,W]
#
# Output:
#     ImageNet-normalized RGB tensor
#
# ============================================================

def bevfusion_img_to_lccnet(
    imgs: torch.Tensor,
    input_color_order: str = 'rgb',
):

    if imgs.ndim != 5:

        raise ValueError(
            'imgs must be [B,N,3,H,W]'
        )

    x = imgs.float()

    # --------------------------------------------------------
    # BEVLoadMultiViewImageFromFiles returns RGB in our
    # verified pipeline.
    #
    # Keep bgr option only for explicit debugging.
    # --------------------------------------------------------

    if input_color_order == 'bgr':

        x = x[
            ...,
            [2, 1, 0],
            :,
            :
        ]

    elif input_color_order != 'rgb':

        raise ValueError(
            f'Unknown color order: {input_color_order}'
        )

    # --------------------------------------------------------
    # Convert 0~255 to 0~1 when necessary.
    # --------------------------------------------------------

    max_value = (
        x.detach()
        .max()
        .item()
    )

    min_value = (
        x.detach()
        .min()
        .item()
    )

    if (
        min_value >= 0.0
        and max_value > 2.0
    ):

        x = (
            x / 255.0
        )

    # --------------------------------------------------------
    # ImageNet normalization
    # --------------------------------------------------------

    mean = torch.tensor(
        [
            0.485,
            0.456,
            0.406,
        ],
        device=x.device,
        dtype=x.dtype,
    ).view(
        1,
        1,
        3,
        1,
        1,
    )

    std = torch.tensor(
        [
            0.229,
            0.224,
            0.225,
        ],
        device=x.device,
        dtype=x.dtype,
    ).view(
        1,
        1,
        3,
        1,
        1,
    )

    x = (
        x - mean
    ) / std

    return x


# ============================================================
# Rotation matrices
# ============================================================

def _rotation_x(
    angle: torch.Tensor,
):

    c = torch.cos(
        angle
    )

    s = torch.sin(
        angle
    )

    R = torch.zeros(
        *angle.shape,
        3,
        3,
        device=angle.device,
        dtype=angle.dtype,
    )

    R[
        ...,
        0,
        0
    ] = 1.0

    R[
        ...,
        1,
        1
    ] = c

    R[
        ...,
        1,
        2
    ] = -s

    R[
        ...,
        2,
        1
    ] = s

    R[
        ...,
        2,
        2
    ] = c

    return R


def _rotation_y(
    angle: torch.Tensor,
):

    c = torch.cos(
        angle
    )

    s = torch.sin(
        angle
    )

    R = torch.zeros(
        *angle.shape,
        3,
        3,
        device=angle.device,
        dtype=angle.dtype,
    )

    R[
        ...,
        1,
        1
    ] = 1.0

    R[
        ...,
        0,
        0
    ] = c

    R[
        ...,
        0,
        2
    ] = s

    R[
        ...,
        2,
        0
    ] = -s

    R[
        ...,
        2,
        2
    ] = c

    return R


def _rotation_z(
    angle: torch.Tensor,
):

    c = torch.cos(
        angle
    )

    s = torch.sin(
        angle
    )

    R = torch.zeros(
        *angle.shape,
        3,
        3,
        device=angle.device,
        dtype=angle.dtype,
    )

    R[
        ...,
        2,
        2
    ] = 1.0

    R[
        ...,
        0,
        0
    ] = c

    R[
        ...,
        0,
        1
    ] = -s

    R[
        ...,
        1,
        0
    ] = s

    R[
        ...,
        1,
        1
    ] = c

    return R


# ============================================================
# Build target from explicit perturbation values
#
# Convention:
#
#   R = Rz @ Ry @ Rx
#
#   T_broken = Delta @ T_GT
#
#   C_GT = inv(Delta)
#
# ============================================================

def _build_target_from_components(
    camera2lidar_gt: torch.Tensor,
    rx_deg: torch.Tensor,
    ry_deg: torch.Tensor,
    rz_deg: torch.Tensor,
    tx_m: torch.Tensor,
    ty_m: torch.Tensor,
    tz_m: torch.Tensor,
):

    device = (
        camera2lidar_gt.device
    )

    dtype = (
        camera2lidar_gt.dtype
    )

    deg2rad = (
        math.pi
        / 180.0
    )

    rx = (
        rx_deg
        * deg2rad
    )

    ry = (
        ry_deg
        * deg2rad
    )

    rz = (
        rz_deg
        * deg2rad
    )

    Rx = _rotation_x(
        rx
    )

    Ry = _rotation_y(
        ry
    )

    Rz = _rotation_z(
        rz
    )

    rotation = (
        Rz
        @ Ry
        @ Rx
    )

    B = camera2lidar_gt.shape[0]
    N = camera2lidar_gt.shape[1]

    delta_gt = (
        torch.eye(
            4,
            device=device,
            dtype=dtype,
        )
        .view(
            1,
            1,
            4,
            4,
        )
        .repeat(
            B,
            N,
            1,
            1,
        )
    )

    delta_gt[
        ...,
        :3,
        :3
    ] = rotation

    delta_gt[
        ...,
        0,
        3
    ] = tx_m

    delta_gt[
        ...,
        1,
        3
    ] = ty_m

    delta_gt[
        ...,
        2,
        3
    ] = tz_m

    # --------------------------------------------------------
    # Broken C2L
    # --------------------------------------------------------

    broken_c2l = (
        delta_gt
        @ camera2lidar_gt
    )

    # --------------------------------------------------------
    # Exact correction
    # --------------------------------------------------------

    correction_gt = (
        torch.linalg.inv(
            delta_gt
        )
    )

    # --------------------------------------------------------
    # SE(3) Lie target
    # --------------------------------------------------------

    correction_flat = (
        correction_gt.reshape(
            B * N,
            4,
            4,
        )
    )

    target_flat = (
        se3.log(
            correction_flat
        )
    )

    target_x = (
        target_flat.reshape(
            B,
            N,
            6,
        )
    )

    # --------------------------------------------------------
    # Oracle restore check
    # --------------------------------------------------------

    restored = (
        correction_gt
        @ broken_c2l
    )

    restore_err = (
        (
            restored
            - camera2lidar_gt
        )
        .abs()
        .max()
    )

    # --------------------------------------------------------
    # log-exp roundtrip
    # --------------------------------------------------------

    correction_reconstructed = (
        se3.exp(
            target_flat
        )
        .reshape(
            B,
            N,
            4,
            4,
        )
    )

    se3_err = (
        (
            correction_reconstructed
            - correction_gt
        )
        .abs()
        .max()
    )

    debug = dict(

        rx_deg=rx_deg,

        ry_deg=ry_deg,

        rz_deg=rz_deg,

        tx_m=tx_m,

        ty_m=ty_m,

        tz_m=tz_m,
    )

    return dict(

        broken_c2l=broken_c2l,

        correction_gt=correction_gt,

        target_x=target_x,

        delta_gt=delta_gt,

        restore_err=restore_err,

        se3_err=se3_err,

        debug=debug,
    )


# ============================================================
# RANDOM training perturbation
#
# Every call generates a new perturbation.
# ============================================================

def build_training_target(
    camera2lidar_gt: torch.Tensor,
    max_rot_deg: float = 10.0,
    max_trans_m: float = 0.75,
):

    B = camera2lidar_gt.shape[0]
    N = camera2lidar_gt.shape[1]

    device = (
        camera2lidar_gt.device
    )

    dtype = (
        camera2lidar_gt.dtype
    )

    shape = (
        B,
        N,
    )

    def uniform_symmetric(
        magnitude,
    ):

        return (
            (
                torch.rand(
                    shape,
                    device=device,
                    dtype=dtype,
                )
                * 2.0
            )
            - 1.0
        ) * magnitude

    rx_deg = uniform_symmetric(
        max_rot_deg
    )

    ry_deg = uniform_symmetric(
        max_rot_deg
    )

    rz_deg = uniform_symmetric(
        max_rot_deg
    )

    tx_m = uniform_symmetric(
        max_trans_m
    )

    ty_m = uniform_symmetric(
        max_trans_m
    )

    tz_m = uniform_symmetric(
        max_trans_m
    )

    return (
        _build_target_from_components(

            camera2lidar_gt,

            rx_deg,
            ry_deg,
            rz_deg,

            tx_m,
            ty_m,
            tz_m,
        )
    )


# ============================================================
# Stable per-sample validation seed
# ============================================================

def _stable_seed(
    base_seed: int,
    sample_key: str,
    camera_index: int,
):

    text = (
        f'{base_seed}'
        f'|{sample_key}'
        f'|{camera_index}'
    )

    digest = hashlib.sha256(
        text.encode(
            'utf-8'
        )
    ).digest()

    value = int.from_bytes(
        digest[:8],
        byteorder='little',
        signed=False,
    )

    return value % (
        2**32
    )


# ============================================================
# FIXED validation perturbation
#
# This gives the same perturbation every epoch for
# sample + camera.
#
# NOTE:
# O-5C-8 final benchmark must use the exact frozen O-3
# perturbation generator if its implementation differs.
# ============================================================

def build_fixed_validation_target(
    camera2lidar_gt: torch.Tensor,
    metas: List[dict],
    base_seed: int,
    max_rot_deg: float,
    max_trans_m: float,
):

    B = camera2lidar_gt.shape[0]
    N = camera2lidar_gt.shape[1]

    device = (
        camera2lidar_gt.device
    )

    dtype = (
        camera2lidar_gt.dtype
    )

    rx_deg = torch.zeros(
        B,
        N,
        device=device,
        dtype=dtype,
    )

    ry_deg = torch.zeros_like(
        rx_deg
    )

    rz_deg = torch.zeros_like(
        rx_deg
    )

    tx_m = torch.zeros_like(
        rx_deg
    )

    ty_m = torch.zeros_like(
        rx_deg
    )

    tz_m = torch.zeros_like(
        rx_deg
    )

    for b in range(B):

        meta = metas[b]

        sample_key = str(

            meta.get(
                'sample_idx',

                meta.get(
                    'token',

                    meta.get(
                        'lidar_path',
                        f'frame_{b}',
                    )
                )
            )
        )

        for cam_idx in range(N):

            seed = _stable_seed(

                base_seed,

                sample_key,

                cam_idx,
            )

            rng = np.random.default_rng(
                seed
            )

            values = rng.uniform(
                low=-1.0,
                high=1.0,
                size=6,
            )

            rx_deg[
                b,
                cam_idx
            ] = float(
                values[0]
                * max_rot_deg
            )

            ry_deg[
                b,
                cam_idx
            ] = float(
                values[1]
                * max_rot_deg
            )

            rz_deg[
                b,
                cam_idx
            ] = float(
                values[2]
                * max_rot_deg
            )

            tx_m[
                b,
                cam_idx
            ] = float(
                values[3]
                * max_trans_m
            )

            ty_m[
                b,
                cam_idx
            ] = float(
                values[4]
                * max_trans_m
            )

            tz_m[
                b,
                cam_idx
            ] = float(
                values[5]
                * max_trans_m
            )

    return (
        _build_target_from_components(

            camera2lidar_gt,

            rx_deg,
            ry_deg,
            rz_deg,

            tx_m,
            ty_m,
            tz_m,
        )
    )


# ============================================================
# LCCNet model
# ============================================================

def build_lccnet(
    device: torch.device,
    image_h: int,
    image_w: int,
    pretrained: bool = True,
    use_feat_from: int = 1,
):

    model = LCCNet(

        resnet_argv=dict(

            num_layers=18,

            pretrained=pretrained,

            frozen=False,
        ),

        image_size=(
            image_h,
            image_w,
        ),

        use_feat_from=use_feat_from,

        md=4,

        use_reflectance=False,

        dropout=0.0,

        Action_Func='leakyrelu',

        attention=False,
    )

    model = model.to(
        device
    )

    return model


# ============================================================
# Camera names
# ============================================================

def extract_camera_names(
    metas,
    num_cams: int,
):

    if (
        metas is None
        or len(metas) == 0
    ):

        return [
            f'CAMERA_INDEX_{i}'
            for i in range(
                num_cams
            )
        ]

    meta = metas[0]

    img_paths = meta.get(
        'img_path',
        None,
    )

    if img_paths is None:

        return [
            f'CAMERA_INDEX_{i}'
            for i in range(
                num_cams
            )
        ]

    if isinstance(
        img_paths,
        str,
    ):

        img_paths = [
            img_paths
        ]

    output = []

    for i in range(
        num_cams
    ):

        if i >= len(
            img_paths
        ):

            output.append(
                f'CAMERA_INDEX_{i}'
            )

            continue

        text = str(
            img_paths[i]
        ).replace(
            '\\',
            '/',
        )

        detected = None

        for cam_name in (
            NUSC_CAMERA_NAMES
        ):

            if (
                f'/{cam_name}/'
                in text
            ):

                detected = (
                    cam_name
                )

                break

        if detected is None:

            detected = (
                f'CAMERA_INDEX_{i}'
            )

        output.append(
            detected
        )

    return output


# ============================================================
# Prepare LCCNet input
#
# This is the core pipeline shared by training and validation.
# ============================================================

def prepare_lccnet_batch(
    raw_batch,
    data_preprocessor,
    device,
    max_rot_deg: float,
    max_trans_m: float,
    depth_scale: float,
    input_color_order: str,
    fixed_validation: bool = False,
    validation_seed: int = 20260811,
):

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

    # --------------------------------------------------------
    # This training strategy intentionally uses frame batch 1.
    # --------------------------------------------------------

    if B != 1:

        raise RuntimeError(
            'O-5C-7 currently expects frame batch size = 1. '
            f'Got B={B}'
        )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    if fixed_validation:

        target_info = (
            build_fixed_validation_target(

                camera2lidar_gt,

                metas,

                base_seed=(
                    validation_seed
                ),

                max_rot_deg=(
                    max_rot_deg
                ),

                max_trans_m=(
                    max_trans_m
                ),
            )
        )

    else:

        target_info = (
            build_training_target(

                camera2lidar_gt,

                max_rot_deg=(
                    max_rot_deg
                ),

                max_trans_m=(
                    max_trans_m
                ),
            )
        )

    broken_c2l = (
        target_info[
            'broken_c2l'
        ]
    )

    target_x = (
        target_info[
            'target_x'
        ]
    )

    # --------------------------------------------------------
    # Geometry gate
    # --------------------------------------------------------

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

    if (
        restore_err.item()
        >= 1e-5
    ):

        raise RuntimeError(
            'Oracle restore error too large: '
            f'{restore_err.item():.8e}'
        )

    if (
        se3_err.item()
        >= 1e-5
    ):

        raise RuntimeError(
            'SE3 log-exp error too large: '
            f'{se3_err.item():.8e}'
        )

    # --------------------------------------------------------
    # Same broken C2L generates sparse depth
    # --------------------------------------------------------

    broken_calib = (
        build_calib_from_cam2lidar(

            broken_c2l,

            camera_intrinsics,
        )
    )

    broken_depth_m = (
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
        broken_depth_m
        / depth_scale
    )

    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    rgb_lcc = (
        bevfusion_img_to_lccnet(

            imgs,

            input_color_order=(
                input_color_order
            ),
        )
    )

    # --------------------------------------------------------
    # Flatten six cameras
    # --------------------------------------------------------

    rgb_flat = (
        rgb_lcc.reshape(
            B * N,
            3,
            H,
            W,
        )
    )

    depth_flat = (
        depth_lcc.reshape(
            B * N,
            1,
            H,
            W,
        )
    )

    target_flat = (
        target_x.reshape(
            B * N,
            6,
        )
    )

    broken_flat = (
        broken_c2l.reshape(
            B * N,
            4,
            4,
        )
    )

    gt_flat = (
        camera2lidar_gt.reshape(
            B * N,
            4,
            4,
        )
    )

    camera_names = (
        extract_camera_names(

            metas,

            N,
        )
    )

    return dict(

        rgb=rgb_flat,

        depth=depth_flat,

        target=target_flat,

        broken_c2l=broken_flat,

        gt_c2l=gt_flat,

        broken_depth_m=(
            broken_depth_m
        ),

        camera_names=(
            camera_names
        ),

        target_info=(
            target_info
        ),

        H=H,

        W=W,

        B=B,

        N=N,

        metas=metas,
    )


# ============================================================
# SE(3) physical residual
#
# E =
#   (C_pred @ T_broken) @ inv(T_GT)
#
# Perfect:
#   Identity
# ============================================================

def compute_calibration_residual(
    correction: torch.Tensor,
    broken_c2l: torch.Tensor,
    gt_c2l: torch.Tensor,
):

    corrected_c2l = (
        correction
        @ broken_c2l
    )

    residual = (
        corrected_c2l
        @ torch.linalg.inv(
            gt_c2l
        )
    )

    # --------------------------------------------------------
    # Translation
    # --------------------------------------------------------

    trans_m = (
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
    # Rotation geodesic
    # --------------------------------------------------------

    R = residual[
        ...,
        :3,
        :3
    ]

    trace = (
        R[
            ...,
            0,
            0
        ]
        +
        R[
            ...,
            1,
            1
        ]
        +
        R[
            ...,
            2,
            2
        ]
    )

    cosine = (
        trace - 1.0
    ) / 2.0

    cosine = torch.clamp(
        cosine,
        -1.0,
        1.0,
    )

    rot_deg = torch.rad2deg(
        torch.acos(
            cosine
        )
    )

    return (
        rot_deg,
        trans_m,
    )

# ============================================================
# Balanced SE(3) twist L1 loss
#
# target / prediction:
#
#   [wx, wy, wz, vx, vy, vz]
#
# rotation:
#   radian
#
# translation:
#   meter
#
# Problem with raw L1:
#
#   rotation range:
#       +/-10 deg ~= +/-0.1745 rad
#
#   translation range:
#       +/-0.75 m
#
# Therefore raw translation values dominate the loss.
#
# Solution:
#
#   normalized rotation loss
#       = MAE(rotation) / max_rotation_rad
#
#   normalized translation loss
#       = MAE(translation) / max_translation_m
#
#   total
#       = 0.5 * rotation
#       + 0.5 * translation
#
# ============================================================

def balanced_twist_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_rot_deg: float,
    max_trans_m: float,
    rot_weight=2.0,
    trans_weight=1.0,
):

    if pred.shape != target.shape:

        raise ValueError(
            f'Prediction/target shape mismatch: '
            f'{tuple(pred.shape)} vs '
            f'{tuple(target.shape)}'
        )

    if pred.shape[-1] != 6:

        raise ValueError(
            'Balanced twist loss expects '
            'last dimension = 6.'
        )

    if max_rot_deg <= 0.0:

        raise ValueError(
            'max_rot_deg must be > 0.'
        )

    if max_trans_m <= 0.0:

        raise ValueError(
            'max_trans_m must be > 0.'
        )

    # --------------------------------------------------------
    # Unit scales
    # --------------------------------------------------------

    rot_scale = math.radians(
        max_rot_deg
    )

    trans_scale = float(
        max_trans_m
    )

    # --------------------------------------------------------
    # Rotation:
    # [wx, wy, wz]
    # --------------------------------------------------------

    rot_error = (

        (
            pred[
                ...,
                0:3
            ]
            -
            target[
                ...,
                0:3
            ]
        )
        .abs()

        /

        rot_scale
    )

    # --------------------------------------------------------
    # Translation:
    # [vx, vy, vz]
    # --------------------------------------------------------

    trans_error = (

        (
            pred[
                ...,
                3:6
            ]
            -
            target[
                ...,
                3:6
            ]
        )
        .abs()

        /

        trans_scale
    )

    rot_loss = (
        rot_error.mean()
    )

    trans_loss = (
        trans_error.mean()
    )
    
    total_weight = rot_weight + trans_weight

    loss = (
        rot_weight * rot_loss
        +
        trans_weight * trans_loss
    ) / total_weight

    return (
        loss,
        rot_loss,
        trans_loss,
    )


# ============================================================
# Chunked version for camera gradient accumulation
#
# IMPORTANT:
#
# Suppose:
#
#   total cameras = 6
#   cam_chunk = 1
#
# We perform:
#
#   camera0.backward()
#   camera1.backward()
#   ...
#   camera5.backward()
#
# Each chunk must therefore be normalized against ALL six
# cameras, not against the current chunk only.
#
# Summing all chunk losses then gives exactly the same loss as:
#
#   balanced_twist_l1(pred_all_6_cameras, target_all_6_cameras)
#
# ============================================================

def balanced_twist_l1_chunk(
    pred_chunk: torch.Tensor,
    target_chunk: torch.Tensor,
    total_samples: int,
    max_rot_deg: float,
    max_trans_m: float,
    rot_weight=2.0,
    trans_weight=1.0,
):

    if pred_chunk.shape != target_chunk.shape:

        raise ValueError(
            f'Chunk prediction/target shape mismatch: '
            f'{tuple(pred_chunk.shape)} vs '
            f'{tuple(target_chunk.shape)}'
        )

    if pred_chunk.shape[-1] != 6:

        raise ValueError(
            'Balanced chunk loss expects '
            'last dimension = 6.'
        )

    if total_samples <= 0:

        raise ValueError(
            'total_samples must be > 0.'
        )

    rot_scale = math.radians(
        max_rot_deg
    )

    trans_scale = float(
        max_trans_m
    )

    # There are 3 rotational and 3 translational
    # elements per camera/sample.
    total_rot_elements = (
        total_samples
        * 3
    )

    total_trans_elements = (
        total_samples
        * 3
    )

    # --------------------------------------------------------
    # Sum error of CURRENT chunk, but normalize by
    # ALL cameras in this optimizer step.
    # --------------------------------------------------------

    rot_loss_chunk = (

        (
            pred_chunk[
                ...,
                0:3
            ]
            -
            target_chunk[
                ...,
                0:3
            ]
        )
        .abs()
        .sum()

        /

        (
            float(
                total_rot_elements
            )
            *
            rot_scale
        )
    )

    trans_loss_chunk = (

        (
            pred_chunk[
                ...,
                3:6
            ]
            -
            target_chunk[
                ...,
                3:6
            ]
        )
        .abs()
        .sum()

        /

        (
            float(
                total_trans_elements
            )
            *
            trans_scale
        )
    )

    # total_loss_chunk = (

    #     0.5
    #     * rot_loss_chunk

    #     +

    #     0.5
    #     * trans_loss_chunk
    # )
    total_loss_chunk = (
            rot_weight * rot_loss_chunk
            +
            trans_weight * trans_loss_chunk
        ) / (
            rot_weight + trans_weight
        )

    return (
        total_loss_chunk,
        rot_loss_chunk,
        trans_loss_chunk,
    )

# ============================================================
# Chunked prediction
# ============================================================

@torch.no_grad()
def predict_in_chunks(
    model,
    rgb,
    depth,
    chunk_size: int,
):

    model.eval()

    outputs = []

    M = rgb.shape[0]

    for start in range(
        0,
        M,
        chunk_size,
    ):

        end = min(
            start
            + chunk_size,
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
# Metric helpers
# ============================================================

def summarize_tensor(
    tensor: torch.Tensor,
):

    tensor = (
        tensor.detach()
        .float()
        .cpu()
    )

    return dict(

        mean=float(
            tensor.mean()
        ),

        median=float(
            tensor.median()
        ),

        p90=float(
            torch.quantile(
                tensor,
                0.90,
            )
        ),

        max=float(
            tensor.max()
        ),
    )


def _concat(
    values,
):

    if len(
        values
    ) == 0:

        raise RuntimeError(
            'Metric accumulator is empty.'
        )

    return torch.cat(
        values,
        dim=0,
    )


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(
    model,
    val_loader,
    data_preprocessor,
    device,
    args,
):

    model.eval()

    broken_rot_all = []
    broken_trans_all = []

    pred_rot_all = []
    pred_trans_all = []

    loss_values = []

    per_camera = {}

    frame_count = 0

    first_debug_printed = False

    start_time = time.time()

    for batch_idx, raw_batch in enumerate(
        val_loader
    ):

        if (
            args.val_max_frames > 0
            and frame_count
            >= args.val_max_frames
        ):

            break

        batch = (
            prepare_lccnet_batch(

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
        # Print first fixed perturbation
        # ----------------------------------------------------

        if not first_debug_printed:

            debug = (
                batch[
                    'target_info'
                ][
                    'debug'
                ]
            )

            print()
            print(
                "[VAL FIXED PERTURBATION]"
            )

            print(
                "First camera =",
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

        pred_x = predict_in_chunks(

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
                'Non-finite validation prediction.'
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
        )

        loss_values.append(
            loss.item()
        )

        correction_pred = (
            se3.exp(
                pred_x
            )
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

        # ----------------------------------------------------
        # Broken residual
        # ----------------------------------------------------

        (
            broken_rot,
            broken_trans,
        ) = compute_calibration_residual(

            identity,

            broken,

            gt,
        )

        # ----------------------------------------------------
        # Pred residual
        # ----------------------------------------------------

        (
            pred_rot,
            pred_trans,
        ) = compute_calibration_residual(

            correction_pred,

            broken,

            gt,
        )

        broken_rot_all.append(
            broken_rot.detach().cpu()
        )

        broken_trans_all.append(
            broken_trans.detach().cpu()
        )

        pred_rot_all.append(
            pred_rot.detach().cpu()
        )

        pred_trans_all.append(
            pred_trans.detach().cpu()
        )

        # ----------------------------------------------------
        # Per-camera
        # ----------------------------------------------------

        names = batch[
            'camera_names'
        ]

        for cam_idx in range(
            len(names)
        ):

            name = names[
                cam_idx
            ]

            if name not in per_camera:

                per_camera[
                    name
                ] = dict(

                    broken_rot=[],

                    broken_trans=[],

                    pred_rot=[],

                    pred_trans=[],
                )

            per_camera[
                name
            ][
                'broken_rot'
            ].append(

                broken_rot[
                    cam_idx
                ]
                .detach()
                .cpu()
                .view(1)
            )

            per_camera[
                name
            ][
                'broken_trans'
            ].append(

                broken_trans[
                    cam_idx
                ]
                .detach()
                .cpu()
                .view(1)
            )

            per_camera[
                name
            ][
                'pred_rot'
            ].append(

                pred_rot[
                    cam_idx
                ]
                .detach()
                .cpu()
                .view(1)
            )

            per_camera[
                name
            ][
                'pred_trans'
            ].append(

                pred_trans[
                    cam_idx
                ]
                .detach()
                .cpu()
                .view(1)
            )

        frame_count += (
            batch[
                'B'
            ]
        )

        if (
            args.val_log_interval > 0
            and frame_count
            % args.val_log_interval
            == 0
        ):

            print(
                f"[VAL] "
                f"frames={frame_count}"
            )

    # ========================================================
    # Aggregate
    # ========================================================

    broken_rot = _concat(
        broken_rot_all
    )

    broken_trans = _concat(
        broken_trans_all
    )

    pred_rot = _concat(
        pred_rot_all
    )

    pred_trans = _concat(
        pred_trans_all
    )

    broken_rot_summary = (
        summarize_tensor(
            broken_rot
        )
    )

    broken_trans_summary = (
        summarize_tensor(
            broken_trans
        )
    )

    pred_rot_summary = (
        summarize_tensor(
            pred_rot
        )
    )

    pred_trans_summary = (
        summarize_tensor(
            pred_trans
        )
    )

    # --------------------------------------------------------
    # Recovery
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Physical joint score
    #
    # smaller = better
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Per-camera summaries
    # --------------------------------------------------------

    per_camera_summary = {}

    for name, values in (
        per_camera.items()
    ):

        b_rot = _concat(
            values[
                'broken_rot'
            ]
        )

        b_trans = _concat(
            values[
                'broken_trans'
            ]
        )

        p_rot = _concat(
            values[
                'pred_rot'
            ]
        )

        p_trans = _concat(
            values[
                'pred_trans'
            ]
        )

        per_camera_summary[
            name
        ] = dict(

            broken_rot_mean_deg=float(
                b_rot.mean()
            ),

            pred_rot_mean_deg=float(
                p_rot.mean()
            ),

            broken_trans_mean_m=float(
                b_trans.mean()
            ),

            pred_trans_mean_m=float(
                p_trans.mean()
            ),
        )

    metrics = dict(

        frames=frame_count,

        l1_loss=float(
            np.mean(
                loss_values
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
# Training one epoch
# ============================================================

def train_one_epoch(
    model,
    train_loader,
    data_preprocessor,
    optimizer,
    device,
    epoch,
    args,
):

    # --------------------------------------------------------
    # Make DefaultSampler shuffle differently each epoch.
    # --------------------------------------------------------

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

    running_loss = 0.0
    frame_count = 0

    last_grad_norm = 0.0

    start_time = time.time()

    for batch_idx, raw_batch in enumerate(
        train_loader
    ):

        if (
            args.train_max_frames > 0
            and frame_count
            >= args.train_max_frames
        ):

            break

        # ====================================================
        # IMPORTANT:
        # build_training_target happens HERE,
        # inside the training loop.
        #
        # Therefore every visit receives a NEW Delta.
        # ====================================================

        batch = (
            prepare_lccnet_batch(

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

        # total_target_elements = (
        #     target.numel()
        # )

        total_samples = (
            target.shape[0]
        )

        total_loss = 0.0

        # ====================================================
        # Camera chunk gradient accumulation
        #
        # cam_chunk=1:
        #
        # camera 0 forward/backward
        # camera 1 forward/backward
        # ...
        # camera 5 forward/backward
        #
        # one optimizer.step()
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
            )

            if not torch.isfinite(
                chunk_loss
            ):

                raise RuntimeError(
                    'Non-finite train loss '
                    f'at epoch={epoch}, '
                    f'batch={batch_idx}'
                )

            chunk_loss.backward()

            total_loss += (
                chunk_loss
                .detach()
                .item()
            )

        # ====================================================
        # Gradient finite check
        # ====================================================

        for name, param in (
            model.named_parameters()
        ):

            if param.grad is None:

                continue

            if not torch.isfinite(
                param.grad
            ).all():

                raise RuntimeError(
                    'Non-finite gradient: '
                    f'{name}'
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

        last_grad_norm = float(
            grad_norm
        )

        optimizer.step()

        running_loss += (
            total_loss
        )

        frame_count += (
            batch[
                'B'
            ]
        )

        # ====================================================
        # Train logging
        # ====================================================

        if (
            args.log_interval > 0
            and (
                frame_count
                % args.log_interval
                == 0
            )
        ):

            avg_loss = (
                running_loss
                /
                max(
                    frame_count,
                    1,
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

            print(

                f"[TRAIN] "
                f"epoch={epoch:02d} "
                f"frame={frame_count} "
                f"loss={avg_loss:.8f} "
                f"lr={lr:.8e} "
                f"grad_norm={last_grad_norm:.6f} "
                f"time={elapsed:.1f}s"
            )

    avg_loss = (
        running_loss
        /
        max(
            frame_count,
            1,
        )
    )

    return dict(

        frames=frame_count,

        loss=float(
            avg_loss
        ),

        last_grad_norm=float(
            last_grad_norm
        ),

        elapsed_sec=float(
            time.time()
            - start_time
        ),
    )


# ============================================================
# Pretty validation print
# ============================================================

def print_validation_metrics(
    epoch,
    metrics,
):

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
        "===================================================="
    )

    print(
        f"[VALIDATION] epoch={epoch}"
    )

    print(
        "===================================================="
    )

    print(
        f"Frames = {metrics['frames']}"
    )

    print(
        f"L1 loss = "
        f"{metrics['l1_loss']:.8f}"
    )

    print()

    print(
        "[BROKEN]"
    )

    print(
        f"Rot   mean={b_rot['mean']:.6f} deg  "
        f"median={b_rot['median']:.6f} deg  "
        f"P90={b_rot['p90']:.6f} deg"
    )

    print(
        f"Trans mean={b_trans['mean']:.6f} m    "
        f"median={b_trans['median']:.6f} m    "
        f"P90={b_trans['p90']:.6f} m"
    )

    print()

    print(
        "[LCCNET]"
    )

    print(
        f"Rot   mean={p_rot['mean']:.6f} deg  "
        f"median={p_rot['median']:.6f} deg  "
        f"P90={p_rot['p90']:.6f} deg"
    )

    print(
        f"Trans mean={p_trans['mean']:.6f} m    "
        f"median={p_trans['median']:.6f} m    "
        f"P90={p_trans['p90']:.6f} m"
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
        "[PER CAMERA]"
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

    for name in (
        preferred_order
    ):

        if name not in metrics[
            'per_camera'
        ]:

            continue

        cm = metrics[
            'per_camera'
        ][
            name
        ]

        print(

            f"{name:<18} "
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
        "===================================================="
    )


# ============================================================
# Save model-only checkpoint
# ============================================================

def save_model_checkpoint(
    path: Path,
    epoch: int,
    model,
    train_metrics,
    val_metrics,
    args,
):

    state = dict(

        gate='O-5C-7',

        epoch=epoch,

        model=(
            model.state_dict()
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


# ============================================================
# Optional large training-state checkpoint
# ============================================================

def save_training_state(
    path: Path,
    epoch: int,
    model,
    optimizer,
    scheduler,
    best_joint_score,
    args,
):

    state = dict(

        epoch=epoch,

        model=(
            model.state_dict()
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
    device,
):

    print(
        "\n[RESUME]"
    )

    print(
        "Loading:",
        path
    )

    checkpoint = torch.load(
        path,
        map_location='cpu',
    )

    model.load_state_dict(
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

    print(
        "start_epoch =",
        start_epoch
    )

    print(
        "best_joint_score =",
        best_joint_score
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

    # ========================================================
    # Environment
    # ========================================================

    set_seed(
        args.seed
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

    device = torch.device(
        args.device
    )

    print()
    print(
        "===================================================="
    )

    print(
        "O-5C-7 LCCNet nuScenes Training"
    )

    print(
        "Strategy A: 256 x 704"
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
        "Epochs =",
        args.epochs
    )

    print(
        "Train max frames =",
        args.train_max_frames
    )

    print(
        "Val max frames =",
        args.val_max_frames
    )

    print(
        "Train perturbation = "
        f"+/-{args.train_max_rot_deg} deg, "
        f"+/-{args.train_max_trans_m} m"
    )

    print(
        "Val perturbation = "
        f"+/-{args.val_max_rot_deg} deg, "
        f"+/-{args.val_max_trans_m} m"
    )

    print(
        "Val seed =",
        args.val_seed
    )

    print(
        "ImageNet RGB backbone =",
        not args.no_imagenet_pretrained
    )

    print(
        "Calibration-specific pretrained = False"
    )

    # ========================================================
    # Dataloaders
    # ========================================================

    train_loader = (
        build_one_dataloader(

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
    )

    val_loader = (
        build_one_dataloader(

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
    )

    print()
    print(
        "[DATASET]"
    )

    print(
        "Train dataset length =",
        len(
            train_loader.dataset
        )
    )

    print(
        "Val dataset length =",
        len(
            val_loader.dataset
        )
    )

    # ========================================================
    # Data preprocessor
    # ========================================================

    data_preprocessor = (
        build_data_preprocessor(

            cfg,

            device,
        )
    )

    # ========================================================
    # First batch geometry gate
    # ========================================================

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

    print()
    print(
        "[INPUT GATE]"
    )

    print(
        "RGB =",
        tuple(
            first_batch[
                'rgb'
            ].shape
        )
    )

    print(
        "Depth =",
        tuple(
            first_batch[
                'depth'
            ].shape
        )
    )

    print(
        "Target =",
        tuple(
            first_batch[
                'target'
            ].shape
        )
    )

    print(
        "Depth nonzero =",
        (
            first_batch[
                'depth'
            ]
            > 0
        )
        .sum()
        .item()
    )

    print(
        "Resolution =",
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

            "Strategy A expects "
            "LCCNet input 256x704, "
            f"got {H}x{W}"
        )

    # ========================================================
    # Model
    # ========================================================

    model = build_lccnet(

        device=device,

        image_h=H,

        image_w=W,

        pretrained=(
            not args.no_imagenet_pretrained
        ),
    )

    total_params = sum(
        p.numel()
        for p
        in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p
        in model.parameters()
        if p.requires_grad
    )

    print()
    print(
        "[MODEL]"
    )

    print(
        "Total parameters =",
        total_params
    )

    print(
        "Trainable parameters =",
        trainable_params
    )

    # ========================================================
    # Forward sanity
    # ========================================================

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

    print(
        "Forward sanity shape =",
        tuple(
            sanity_pred.shape
        )
    )

    print(
        "Forward finite =",
        torch.isfinite(
            sanity_pred
        ).all().item()
    )

    if not torch.isfinite(
        sanity_pred
    ).all():

        raise RuntimeError(
            'Initial model forward is non-finite.'
        )

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = torch.optim.AdamW(

        model.parameters(),

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

    # ========================================================
    # Scheduler
    # ========================================================

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
            f'Unknown scheduler: {args.scheduler}'
        )

    # ========================================================
    # Work dir
    # ========================================================

    work_dir = Path(
        args.work_dir
    )

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_file = (
        work_dir
        / 'metrics.jsonl'
    )

    # ========================================================
    # Resume
    # ========================================================

    start_epoch = 1

    best_joint_score = (
        float('inf')
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

            device,
        )

    # ========================================================
    # Initial validation
    #
    # Useful to know the starting point before nuScenes
    # fine-tuning.
    # ========================================================

    if not args.skip_initial_val:

        print()
        print(
            "Running initial validation..."
        )

        initial_val = validate(

            model,

            val_loader,

            data_preprocessor,

            device,

            args,
        )

        print_validation_metrics(

            epoch=0,

            metrics=(
                initial_val
            ),
        )

    # ========================================================
    # Training
    # ========================================================

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):

        print()
        print(
            "####################################################"
        )

        print(
            f"EPOCH {epoch}/{args.epochs}"
        )

        print(
            "####################################################"
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_metrics = (
            train_one_epoch(

                model,

                train_loader,

                data_preprocessor,

                optimizer,

                device,

                epoch,

                args,
            )
        )

        print()
        print(
            "[TRAIN SUMMARY]"
        )

        print(
            "frames =",
            train_metrics[
                'frames'
            ]
        )

        print(
            "loss =",
            f"{train_metrics['loss']:.8f}"
        )

        print(
            "last grad norm =",
            f"{train_metrics['last_grad_norm']:.6f}"
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_metrics = validate(

            model,

            val_loader,

            data_preprocessor,

            device,

            args,
        )

        print_validation_metrics(

            epoch,

            val_metrics,
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        if scheduler is not None:

            scheduler.step()

        # ----------------------------------------------------
        # Metrics JSONL
        # ----------------------------------------------------

        record = dict(

            epoch=epoch,

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

        # ----------------------------------------------------
        # Last model
        # ----------------------------------------------------

        last_path = (
            work_dir
            / 'last_model.pth'
        )

        save_model_checkpoint(

            last_path,

            epoch,

            model,

            train_metrics,

            val_metrics,

            args,
        )

        print(
            "\nSaved:",
            last_path
        )

        # ----------------------------------------------------
        # Best physical joint model
        # ----------------------------------------------------

        current_joint_score = float(
            val_metrics[
                'joint_score'
            ]
        )

        if (
            current_joint_score
            < best_joint_score
        ):

            best_joint_score = (
                current_joint_score
            )

            best_path = (
                work_dir
                / 'best_joint_model.pth'
            )

            save_model_checkpoint(

                best_path,

                epoch,

                model,

                train_metrics,

                val_metrics,

                args,
            )

            print(
                "NEW BEST JOINT MODEL"
            )

            print(
                "Joint score =",
                f"{best_joint_score:.6f}"
            )

            print(
                "Saved:",
                best_path
            )

        # ----------------------------------------------------
        # Optional optimizer state
        #
        # WARNING:
        # AdamW state for 310M parameters is very large.
        # ----------------------------------------------------

        if args.save_training_state:

            training_state_path = (
                work_dir
                / 'last_training_state.pth'
            )

            save_training_state(

                training_state_path,

                epoch,

                model,

                optimizer,

                scheduler,

                best_joint_score,

                args,
            )

            print(
                "Saved training state:",
                training_state_path
            )

    # ========================================================
    # Finished
    # ========================================================

    print()
    print(
        "===================================================="
    )

    print(
        "[O-5C-7 TRAINING COMPLETE]"
    )

    print(
        "Best joint score =",
        best_joint_score
    )

    print(
        "Work dir =",
        work_dir
    )

    print(
        "===================================================="
    )


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(

        description=(
            'O-5C-7 LCCNet '
            'nuScenes training '
            '(Strategy A: 256x704)'
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

        '--work-dir',

        type=str,

        default=DEFAULT_WORK_DIR,
    )

    parser.add_argument(

        '--epochs',

        type=int,

        default=10,
    )

    parser.add_argument(

        '--train-max-frames',

        type=int,

        default=0,

        help=(
            '0 = full train dataset. '
            'For pilot use 1024.'
        ),
    )

    parser.add_argument(

        '--val-max-frames',

        type=int,

        default=512,

        help=(
            '0 = full val dataset.'
        ),
    )

    parser.add_argument(

        '--num-workers',

        type=int,

        default=0,
    )

    parser.add_argument(

        '--seed',

        type=int,

        default=20260818,
    )

    parser.add_argument(

        '--val-seed',

        type=int,

        default=20260811,
    )

    # --------------------------------------------------------
    # Training perturbation
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

    # --------------------------------------------------------
    # Validation perturbation
    # --------------------------------------------------------

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
    # LCCNet input
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

        help=(
            'Disable ImageNet initialization '
            'of RGB ResNet18.'
        ),
    )

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    parser.add_argument(

        '--log-interval',

        type=int,

        default=50,
    )

    parser.add_argument(

        '--val-log-interval',

        type=int,

        default=100,
    )

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    parser.add_argument(

        '--resume',

        type=str,

        default=None,

        help=(
            'Path to last_training_state.pth'
        ),
    )

    parser.add_argument(

        '--save-training-state',

        action='store_true',

        help=(
            'Also save optimizer and scheduler state. '
            'This checkpoint can be several GB.'
        ),
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

    main(
        args
    )