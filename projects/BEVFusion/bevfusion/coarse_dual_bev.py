
import torch
import torch.nn.functional as F


def sample_corrected_camera_bev(
    camera_bev,
    proposal_xyz_norm,
    lidar_aug_matrix,
    pc_range,
    xbound,
    ybound,
):
    """
    Args:
        camera_bev:
            [B, Cbev, H, W]

        proposal_xyz_norm:
            [B, Np, 3]
            Normalized unaugmented LiDAR XYZ

        lidar_aug_matrix:
            [B, 4, 4]

    Returns:
        sampled:
            [B, Np, Cbev]

        valid:
            [B, Np] boolean
    """

    if camera_bev.ndim != 4:
        raise ValueError("Invalid camera_bev shape")

    if proposal_xyz_norm.ndim != 3:
        raise ValueError("Invalid proposal shape")

    B, C, H, W = camera_bev.shape

    device = camera_bev.device

    # ------------------------------------
    # 1. Normalized XYZ -> LiDAR meters
    # ------------------------------------

    xyz = proposal_xyz_norm.detach().to(
        device=device,
        dtype=torch.float32,
    )

    pcr = torch.as_tensor(
        pc_range,
        device=device,
        dtype=torch.float32,
    )

    xyz_m = (
        xyz
        * (pcr[3:] - pcr[:3]).view(1, 1, 3)
        + pcr[:3].view(1, 1, 3)
    )

    # ------------------------------------
    # 2. Apply LiDAR augmentation
    # ------------------------------------

    aug = torch.as_tensor(
        lidar_aug_matrix,
        device=device,
        dtype=torch.float32,
    )

    if aug.shape != (B, 4, 4):
        raise ValueError(
            f"Expected [B,4,4], got {aug.shape}"
        )

    xyz_aug = (
        torch.einsum(
            "bij,bnj->bni",
            aug[:, :3, :3],
            xyz_m,
        )
        + aug[:, None, :3, 3]
    )

    xx = xyz_aug[..., 0]
    yy = xyz_aug[..., 1]

    xb = torch.as_tensor(
        xbound, device=device, dtype=torch.float32
    )

    yb = torch.as_tensor(
        ybound, device=device, dtype=torch.float32
    )

    # ------------------------------------
    # 3. Metric XY -> normalized BEV grid
    # ------------------------------------

    # BEV pool: H=X, W=Y
    # grid_sample: grid[...,0]=W
    #              grid[...,1]=H

    grid_w = (
        2.0 * (yy - yb[0]) / (yb[1] - yb[0])
        - 1.0
    )

    grid_h = (
        2.0 * (xx - xb[0]) / (xb[1] - xb[0])
        - 1.0
    )

    grid = torch.stack(
        [grid_w, grid_h],
        dim=-1,
    )

    # ------------------------------------
    # 4. Validity mask
    # ------------------------------------

    valid = (
        torch.isfinite(xyz).all(-1)
        & torch.isfinite(xyz_aug).all(-1)
        & (xx >= xb[0])
        & (xx < xb[1])
        & (yy >= yb[0])
        & (yy < yb[1])
    )

    grid = torch.nan_to_num(
        grid,
        nan=2.0,
        posinf=2.0,
        neginf=2.0,
    )

    # ------------------------------------
    # 5. Bilinear Sampling
    # ------------------------------------

    sampled = F.grid_sample(
        camera_bev.float(),
        grid[:, None, :, :],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )

    # [B,C,1,Np] -> [B,Np,C]
    sampled = (
        sampled[:, :, 0, :]
        .transpose(1, 2)
        .contiguous()
    )

    sampled = (
        sampled
        * valid[..., None].to(sampled.dtype)
    )

    return sampled.to(camera_bev.dtype), valid
