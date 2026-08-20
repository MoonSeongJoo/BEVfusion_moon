# ============================================================
# projects/LCCNetModern/tools/
# diagnose_quick_overfit_per_camera.py
#
# Gate O-5C-6B
#
# Purpose:
#   Diagnose the already-trained O-5C-6 quick-overfit model
#   camera by camera.
#
# Metrics:
#   - Broken rotation error [deg]
#   - LCCNet corrected rotation error [deg]
#   - Broken translation error [m]
#   - LCCNet corrected translation error [m]
#   - Broken sparse-depth nonzero pixels
#   - Broken sparse-depth density [%]
#
# IMPORTANT:
#   This script reconstructs the SAME first batch and SAME
#   perturbation by using the same seed as O-5C-6.
# ============================================================

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from train_lccnet_nuscenes import (
    set_seed,
    build_dataloader_and_preprocessor,
    extract_batch_geometry,
    build_training_target,
    bevfusion_img_to_lccnet,
    build_lccnet,
)

from quick_overfit_lccnet_nuscenes import (
    compute_calibration_residual,
)

from projects.LCCNetModern.calib_geometry import (
    build_calib_from_cam2lidar,
    project_lidar_to_sparse_depth,
)

from projects.LCCNetModern.lie import se3


DEFAULT_CONFIG = (
    'projects/LCCNetModern/configs/'
    'lccnet_nuscenes_v110.py'
)

DEFAULT_CHECKPOINT = (
    'work_dirs/lccnet_nuscenes_v110/'
    'quick_overfit_200iter.pth'
)


# ============================================================
# Camera names
# ============================================================

NUSC_CAMERA_NAMES = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]


def extract_camera_names(
    metas,
    num_cams: int,
):
    """Try to recover camera names from img_path.

    If metadata ordering cannot be inferred safely,
    return CAMERA_INDEX_i instead of guessing.
    """

    names = []

    if metas is None or len(metas) == 0:
        return [
            f'CAMERA_INDEX_{i}'
            for i in range(num_cams)
        ]

    meta = metas[0]

    if hasattr(meta, 'metainfo'):
        meta = meta.metainfo

    if not isinstance(meta, dict):
        return [
            f'CAMERA_INDEX_{i}'
            for i in range(num_cams)
        ]

    img_paths = meta.get(
        'img_path',
        None,
    )

    if img_paths is None:
        return [
            f'CAMERA_INDEX_{i}'
            for i in range(num_cams)
        ]

    if isinstance(
        img_paths,
        str,
    ):
        img_paths = [
            img_paths
        ]

    for i in range(num_cams):

        if i >= len(img_paths):
            names.append(
                f'CAMERA_INDEX_{i}'
            )
            continue

        path_text = str(
            img_paths[i]
        ).replace(
            '\\',
            '/',
        )

        detected = None

        for cam_name in NUSC_CAMERA_NAMES:

            token = (
                f'/{cam_name}/'
            )

            if token in path_text:
                detected = cam_name
                break

        if detected is None:
            detected = (
                f'CAMERA_INDEX_{i}'
            )

        names.append(
            detected
        )

    return names


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

    total = rgb.shape[0]

    for start in range(
        0,
        total,
        chunk_size,
    ):

        end = min(
            start + chunk_size,
            total,
        )

        pred = model(
            rgb[start:end],
            depth[start:end],
        )

        outputs.append(
            pred
        )

    return torch.cat(
        outputs,
        dim=0,
    )


# ============================================================
# Main
# ============================================================

def main(args):

    set_seed(
        args.seed
    )

    device = torch.device(
        args.device
    )

    print()
    print(
        "===================================================="
    )
    print(
        "Gate O-5C-6B"
    )
    print(
        "Per-Camera Quick-Overfit Diagnostic"
    )
    print(
        "===================================================="
    )

    print(
        "Config      =",
        args.config,
    )

    print(
        "Checkpoint  =",
        args.checkpoint,
    )

    print(
        "Seed        =",
        args.seed,
    )

    # ========================================================
    # Build fixed first batch
    # ========================================================

    (
        cfg,
        train_loader,
        data_preprocessor,
    ) = build_dataloader_and_preprocessor(

        config_path=args.config,

        device=device,

        batch_size=1,

        num_workers=(
            args.num_workers
        ),

        seed=args.seed,
    )

    raw_batch = next(
        iter(train_loader)
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

    assert B == 1

    print()
    print(
        "[FIXED INPUT]"
    )

    print(
        "imgs =",
        tuple(imgs.shape),
    )

    print(
        "points =",
        tuple(points[0].shape),
    )

    # ========================================================
    # Reconstruct same fixed perturbation
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

    restore_err = (
        target_info[
            'restore_err'
        ]
    )

    print()
    print(
        "[GEOMETRY]"
    )

    print(
        "Oracle restore error =",
        f"{restore_err.item():.8e}"
    )

    assert (
        restore_err.item()
        < 1e-5
    )

    # ========================================================
    # Broken calibration
    # ========================================================

    broken_calib = (
        build_calib_from_cam2lidar(

            broken_c2l,

            camera_intrinsics,
        )
    )

    # ========================================================
    # Sparse depth in meters
    # ========================================================

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
        / args.depth_scale
    )

    # ========================================================
    # RGB
    # ========================================================

    rgb_lcc = (
        bevfusion_img_to_lccnet(

            imgs,

            input_color_order=(
                args.input_color_order
            ),
        )
    )

    rgb_lcc = (
        rgb_lcc.reshape(
            B * N,
            3,
            H,
            W,
        )
    )

    depth_lcc = (
        depth_lcc.reshape(
            B * N,
            1,
            H,
            W,
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

    correction_gt_flat = (
        correction_gt.reshape(
            B * N,
            4,
            4,
        )
    )

    # ========================================================
    # Build current 256x704 model
    #
    # pretrained=False is intentional:
    # checkpoint will overwrite model weights.
    # ========================================================

    model = build_lccnet(

        device=device,

        image_h=H,

        image_w=W,

        pretrained=False,
    )

    # ========================================================
    # Load O-5C-6 checkpoint
    # ========================================================

    checkpoint_path = Path(
        args.checkpoint
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            checkpoint_path
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location='cpu',
        weights_only=False,
    )

    if 'model' not in checkpoint:

        raise KeyError(
            "Checkpoint does not contain key 'model'"
        )

    load_result = (
        model.load_state_dict(
            checkpoint['model'],
            strict=True,
        )
    )

    print()
    print(
        "[CHECKPOINT]"
    )

    print(
        "load_state_dict =",
        load_result,
    )

    print(
        "quick_overfit_pass =",
        checkpoint.get(
            'quick_overfit_pass',
            'UNKNOWN',
        ),
    )

    # ========================================================
    # LCCNet prediction
    # ========================================================

    pred_x = predict_in_chunks(

        model,

        rgb_lcc,

        depth_lcc,

        chunk_size=(
            args.cam_chunk
        ),
    )

    assert pred_x.shape == (
        B * N,
        6,
    )

    assert torch.isfinite(
        pred_x
    ).all()

    correction_pred = (
        se3.exp(
            pred_x
        )
    )

    # ========================================================
    # Broken baseline residual
    # ========================================================

    identity = (
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
        broken_rot_deg,
        broken_trans_m,
    ) = compute_calibration_residual(

        identity,

        broken_flat,

        gt_flat,
    )

    # ========================================================
    # Predicted correction residual
    # ========================================================

    (
        pred_rot_deg,
        pred_trans_m,
    ) = compute_calibration_residual(

        correction_pred,

        broken_flat,

        gt_flat,
    )

    # ========================================================
    # Oracle check
    # ========================================================

    (
        oracle_rot_deg,
        oracle_trans_m,
    ) = compute_calibration_residual(

        correction_gt_flat,

        broken_flat,

        gt_flat,
    )

    # ========================================================
    # Camera names
    # ========================================================

    camera_names = (
        extract_camera_names(
            metas,
            N,
        )
    )

    # ========================================================
    # Header
    # ========================================================

    print()
    print(
        "===================================================="
    )

    print(
        "[PER-CAMERA RESULT]"
    )

    print(
        "===================================================="
    )

    header = (
        f"{'idx':>3} "
        f"{'camera':<18} "
        f"{'depth_nz':>10} "
        f"{'density%':>9} "
        f"{'B_rot':>9} "
        f"{'P_rot':>9} "
        f"{'Rrec%':>8} "
        f"{'B_trans':>10} "
        f"{'P_trans':>10} "
        f"{'Trec%':>8}"
    )

    print(
        header
    )

    print(
        "-" * len(header)
    )

    image_pixels = (
        H * W
    )

    for cam_idx in range(N):

        depth_cam = (
            broken_depth_m[
                0,
                cam_idx,
                0,
            ]
        )

        nonzero_mask = (
            depth_cam > 0
        )

        depth_nonzero = (
            nonzero_mask
            .sum()
            .item()
        )

        density_pct = (
            100.0
            * depth_nonzero
            / image_pixels
        )

        b_rot = (
            broken_rot_deg[
                cam_idx
            ].item()
        )

        p_rot = (
            pred_rot_deg[
                cam_idx
            ].item()
        )

        b_trans = (
            broken_trans_m[
                cam_idx
            ].item()
        )

        p_trans = (
            pred_trans_m[
                cam_idx
            ].item()
        )

        rot_recovery = (
            100.0
            * (
                1.0
                - p_rot
                / max(
                    b_rot,
                    1e-12,
                )
            )
        )

        trans_recovery = (
            100.0
            * (
                1.0
                - p_trans
                / max(
                    b_trans,
                    1e-12,
                )
            )
        )

        print(

            f"{cam_idx:>3d} "

            f"{camera_names[cam_idx]:<18} "

            f"{depth_nonzero:>10d} "

            f"{density_pct:>8.3f}% "

            f"{b_rot:>8.3f}° "

            f"{p_rot:>8.3f}° "

            f"{rot_recovery:>7.2f}% "

            f"{b_trans:>9.4f}m "

            f"{p_trans:>9.4f}m "

            f"{trans_recovery:>7.2f}%"
        )

    # ========================================================
    # Aggregate
    # ========================================================

    print()
    print(
        "===================================================="
    )

    print(
        "[AGGREGATE]"
    )

    print(
        "===================================================="
    )

    print(
        "Broken rotation mean =",
        f"{broken_rot_deg.mean().item():.6f} deg",
    )

    print(
        "Pred rotation mean   =",
        f"{pred_rot_deg.mean().item():.6f} deg",
    )

    print(
        "Broken translation mean =",
        f"{broken_trans_m.mean().item():.6f} m",
    )

    print(
        "Pred translation mean   =",
        f"{pred_trans_m.mean().item():.6f} m",
    )

    print(
        "Oracle rotation mean =",
        f"{oracle_rot_deg.mean().item():.8f} deg",
    )

    print(
        "Oracle translation mean =",
        f"{oracle_trans_m.mean().item():.8e} m",
    )

    # ========================================================
    # Compare against metrics stored in checkpoint
    # ========================================================

    stored = checkpoint.get(
        'final_metrics',
        None,
    )

    if stored is not None:

        print()
        print(
            "[CHECKPOINT REPRODUCTION]"
        )

        print(
            "Stored rot mean =",
            stored.get(
                'rot_mean_deg',
                None,
            ),
        )

        print(
            "Current rot mean =",
            pred_rot_deg.mean().item(),
        )

        print(
            "Stored trans mean =",
            stored.get(
                'trans_mean_m',
                None,
            ),
        )

        print(
            "Current trans mean =",
            pred_trans_m.mean().item(),
        )

    # ========================================================
    # Worst camera
    # ========================================================

    worst_rot_idx = (
        torch.argmax(
            pred_rot_deg
        ).item()
    )

    worst_trans_idx = (
        torch.argmax(
            pred_trans_m
        ).item()
    )

    print()
    print(
        "[WORST CAMERA]"
    )

    print(
        "Worst rotation =",
        camera_names[
            worst_rot_idx
        ],
        f"{pred_rot_deg[worst_rot_idx].item():.6f} deg",
    )

    print(
        "Worst translation =",
        camera_names[
            worst_trans_idx
        ],
        f"{pred_trans_m[worst_trans_idx].item():.6f} m",
    )

    print()
    print(
        "===================================================="
    )

    print(
        "[GATE O-5C-6B] COMPLETE"
    )

    print(
        "Inspect per-camera residual and depth density."
    )

    print(
        "===================================================="
    )


# ============================================================
# Args
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config',
        type=str,
        default=DEFAULT_CONFIG,
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=DEFAULT_CHECKPOINT,
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
        '--cam-chunk',
        type=int,
        default=1,
    )

    parser.add_argument(
        '--input-color-order',
        choices=[
            'rgb',
            'bgr',
        ],
        default='rgb',
    )

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    main(args)