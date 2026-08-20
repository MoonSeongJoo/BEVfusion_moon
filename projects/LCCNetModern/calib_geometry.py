# projects/LCCNetModern/calib_geometry.py

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch


Tensor = torch.Tensor


# ============================================================
# 1. Euler -> Rotation Matrix
#
# IMPORTANT:
# This MUST follow exactly the same convention used in O-3.
#
#     R = Rz @ Ry @ Rx
#
# Input:
#     rx_deg, ry_deg, rz_deg : [...], degree
#
# Output:
#     R : [..., 3, 3]
# ============================================================

def euler_xyz_to_matrix(
    rx_deg: Tensor,
    ry_deg: Tensor,
    rz_deg: Tensor,
) -> Tensor:

    rx = torch.deg2rad(rx_deg)
    ry = torch.deg2rad(ry_deg)
    rz = torch.deg2rad(rz_deg)

    cx = torch.cos(rx)
    sx = torch.sin(rx)

    cy = torch.cos(ry)
    sy = torch.sin(ry)

    cz = torch.cos(rz)
    sz = torch.sin(rz)

    zero = torch.zeros_like(rx)
    one = torch.ones_like(rx)

    # --------------------------------------------------------
    # Rx
    # --------------------------------------------------------

    Rx = torch.stack(
        [
            one,  zero, zero,
            zero, cx,   -sx,
            zero, sx,   cx,
        ],
        dim=-1,
    ).reshape(
        *rx.shape,
        3,
        3,
    )

    # --------------------------------------------------------
    # Ry
    # --------------------------------------------------------

    Ry = torch.stack(
        [
            cy,   zero, sy,
            zero, one,  zero,
            -sy,  zero, cy,
        ],
        dim=-1,
    ).reshape(
        *ry.shape,
        3,
        3,
    )

    # --------------------------------------------------------
    # Rz
    # --------------------------------------------------------

    Rz = torch.stack(
        [
            cz,   -sz, zero,
            sz,   cz,  zero,
            zero, zero, one,
        ],
        dim=-1,
    ).reshape(
        *rz.shape,
        3,
        3,
    )

    # ========================================================
    # Same convention as Gate O-3
    # ========================================================

    R = Rz @ Ry @ Rx

    return R


# ============================================================
# 2. Training Random Perturbation
#
# Each axis independently sampled:
#
#     rx, ry, rz ~ U(-max_rot_deg, +max_rot_deg)
#     tx, ty, tz ~ U(-max_trans_m, +max_trans_m)
#
# Output:
#     delta_gt : [B, N, 4, 4]
#
# IMPORTANT:
#     Broken = Delta_GT @ GT_C2L
# ============================================================

def sample_random_delta(
    batch_size: int,
    num_cams: int,
    device: torch.device,
    dtype: torch.dtype,
    max_rot_deg: float = 10.0,
    max_trans_m: float = 0.75,
) -> Tuple[Tensor, Dict[str, Tensor]]:

    shape = (
        batch_size,
        num_cams,
    )

    def rand_symmetric(
        max_value: float,
    ) -> Tensor:

        return (
            torch.rand(
                shape,
                device=device,
                dtype=dtype,
            )
            * 2.0
            - 1.0
        ) * max_value

    # --------------------------------------------------------
    # Rotation
    # --------------------------------------------------------

    rx = rand_symmetric(
        max_rot_deg
    )

    ry = rand_symmetric(
        max_rot_deg
    )

    rz = rand_symmetric(
        max_rot_deg
    )

    # --------------------------------------------------------
    # Translation
    # --------------------------------------------------------

    tx = rand_symmetric(
        max_trans_m
    )

    ty = rand_symmetric(
        max_trans_m
    )

    tz = rand_symmetric(
        max_trans_m
    )

    # --------------------------------------------------------
    # Rotation matrix
    # --------------------------------------------------------

    R = euler_xyz_to_matrix(
        rx,
        ry,
        rz,
    )

    # --------------------------------------------------------
    # Homogeneous Delta matrix
    # --------------------------------------------------------

    delta = torch.eye(
        4,
        device=device,
        dtype=dtype,
    ).view(
        1,
        1,
        4,
        4,
    ).repeat(
        batch_size,
        num_cams,
        1,
        1,
    )

    delta[
        ...,
        :3,
        :3,
    ] = R

    delta[
        ...,
        0,
        3,
    ] = tx

    delta[
        ...,
        1,
        3,
    ] = ty

    delta[
        ...,
        2,
        3,
    ] = tz

    debug = {
        'rx_deg': rx,
        'ry_deg': ry,
        'rz_deg': rz,
        'tx_m': tx,
        'ty_m': ty,
        'tz_m': tz,
    }

    return (
        delta,
        debug,
    )


# ============================================================
# 3. Build calibration dictionary from Camera -> LiDAR
#
#     C2L = camera2lidar
#
#     L2C = inv(C2L)
#
#     L2I = K @ L2C
#
# This is the same convention validated in O-2/O-3/O-4.
# ============================================================

def build_calib_from_cam2lidar(
    camera2lidar: Tensor,
    camera_intrinsics: Tensor,
) -> Dict[str, Tensor]:

    if camera2lidar.shape[-2:] != (
        4,
        4,
    ):
        raise ValueError(
            "camera2lidar must have "
            "shape [...,4,4], got "
            f"{tuple(camera2lidar.shape)}"
        )

    if camera_intrinsics.shape[-2:] != (
        4,
        4,
    ):
        raise ValueError(
            "camera_intrinsics must have "
            "shape [...,4,4], got "
            f"{tuple(camera_intrinsics.shape)}"
        )

    lidar2camera = torch.linalg.inv(
        camera2lidar
    )

    lidar2image = (
        camera_intrinsics
        @ lidar2camera
    )

    return {
        'cam2lidar':
            camera2lidar,

        'cam2img':
            camera_intrinsics,

        'lidar2img':
            lidar2image,
    }


# ============================================================
# 4. LiDAR -> Sparse Depth Projection
#
# This function intentionally follows the SAME geometry
# implemented and validated in Gate O-5A.
#
# Input:
#     points
#         List[Tensor]
#         B elements
#         each [P, >=3]
#
#     lidar2image
#         [B, N, 4, 4]
#
#     img_aug_matrix
#         [B, N, 4, 4]
#
#     lidar_aug_matrix
#         [B, 4, 4]
#           or
#         [B, N, 4, 4]
#
# Output:
#     depth
#         [B, N, 1, H, W]
# ============================================================

def project_lidar_to_sparse_depth(
    points: Sequence[Tensor],
    lidar2image: Tensor,
    img_aug_matrix: Tensor,
    lidar_aug_matrix: Tensor,
    image_hw: Tuple[int, int],
) -> Tensor:

    H, W = image_hw

    if len(points) == 0:
        raise ValueError(
            "points list is empty."
        )

    device = points[0].device
    dtype = points[0].dtype

    # --------------------------------------------------------
    # Device / dtype normalization
    # --------------------------------------------------------

    lidar2image = lidar2image.to(
        device=device,
        dtype=dtype,
    )

    img_aug_matrix = (
        img_aug_matrix.to(
            device=device,
            dtype=dtype,
        )
    )

    lidar_aug_matrix = (
        lidar_aug_matrix.to(
            device=device,
            dtype=dtype,
        )
    )

    B = len(points)

    if lidar2image.ndim != 4:
        raise ValueError(
            "lidar2image must be "
            "[B,N,4,4], got "
            f"{tuple(lidar2image.shape)}"
        )

    N = lidar2image.shape[1]

    # --------------------------------------------------------
    # Output sparse depth
    # --------------------------------------------------------

    depth = points[0].new_zeros(
        (
            B,
            N,
            1,
            H,
            W,
        )
    )

    # ========================================================
    # Process each batch
    # ========================================================

    for b in range(B):

        # ----------------------------------------------------
        # Current LiDAR XYZ
        # ----------------------------------------------------

        xyz = (
            points[b][
                :,
                :3
            ]
            .clone()
        )

        # ----------------------------------------------------
        # LiDAR augmentation matrix
        # ----------------------------------------------------

        cur_lidar_aug = (
            lidar_aug_matrix[b]
        )

        # In case shape is [N,4,4],
        # LiDAR augmentation is common for cameras.
        if cur_lidar_aug.ndim == 3:
            cur_lidar_aug = (
                cur_lidar_aug[0]
            )

        if cur_lidar_aug.shape != (
            4,
            4,
        ):
            raise ValueError(
                "lidar_aug_matrix[b] "
                "must resolve to [4,4], got "
                f"{tuple(cur_lidar_aug.shape)}"
            )

        # ----------------------------------------------------
        # Undo LiDAR augmentation
        #
        # p_raw =
        #   inv(R_aug) @
        #   (p_aug - t_aug)
        # ----------------------------------------------------

        xyz = (
            xyz
            - cur_lidar_aug[
                :3,
                3
            ]
        )

        xyz = (
            torch.linalg.inv(
                cur_lidar_aug[
                    :3,
                    :3
                ]
            )
            @ xyz.transpose(
                0,
                1,
            )
        )

        # xyz:
        # [3, P]

        # ----------------------------------------------------
        # Current L2I matrices
        # ----------------------------------------------------

        cur_l2i = (
            lidar2image[b]
        )

        # ----------------------------------------------------
        # Project LiDAR -> Camera/Image homogeneous
        # ----------------------------------------------------

        proj = (
            cur_l2i[
                :,
                :3,
                :3
            ]
            @ xyz
        )

        proj = (
            proj
            + cur_l2i[
                :,
                :3,
                3
            ].reshape(
                N,
                3,
                1,
            )
        )

        # ----------------------------------------------------
        # Depth before perspective division
        # ----------------------------------------------------

        dist = (
            proj[
                :,
                2,
                :
            ]
            .clone()
        )

        valid_z = (
            dist > 1e-5
        )

        z_safe = torch.clamp(
            proj[
                :,
                2:3,
                :
            ],
            min=1e-5,
            max=1e5,
        )

        # ----------------------------------------------------
        # Perspective division
        # ----------------------------------------------------

        proj[
            :,
            :2,
            :
        ] = (
            proj[
                :,
                :2,
                :
            ]
            / z_safe
        )

        # ----------------------------------------------------
        # Image augmentation
        # ----------------------------------------------------

        cur_img_aug = (
            img_aug_matrix[b]
        )

        # Support a shared [4,4]
        if cur_img_aug.ndim == 2:

            cur_img_aug = (
                cur_img_aug
                .unsqueeze(0)
                .expand(
                    N,
                    -1,
                    -1,
                )
            )

        if cur_img_aug.shape != (
            N,
            4,
            4,
        ):
            raise ValueError(
                "img_aug_matrix[b] "
                "must be [N,4,4], got "
                f"{tuple(cur_img_aug.shape)}"
            )

        proj = (
            cur_img_aug[
                :,
                :3,
                :3
            ]
            @ proj
        )

        proj = (
            proj
            + cur_img_aug[
                :,
                :3,
                3
            ].reshape(
                N,
                3,
                1,
            )
        )

        # ----------------------------------------------------
        # [N, 2, P] -> [N, P, 2]
        # ----------------------------------------------------

        uv = (
            proj[
                :,
                :2,
                :
            ]
            .transpose(
                1,
                2,
            )
        )

        # ====================================================
        # Rasterize each camera
        # ====================================================

        for cam in range(N):

            x = uv[
                cam,
                :,
                0
            ]

            y = uv[
                cam,
                :,
                1
            ]

            valid = (
                valid_z[cam]
                & (x >= 0)
                & (x < W)
                & (y >= 0)
                & (y < H)
            )

            px = (
                x[valid]
                .long()
            )

            py = (
                y[valid]
                .long()
            )

            d = dist[
                cam,
                valid
            ]

            # ------------------------------------------------
            # Same simple sparse rasterization used in O-5A
            # ------------------------------------------------

            depth[
                b,
                cam,
                0,
                py,
                px,
            ] = d

    return depth