import torch
from torch import nn
from torch.nn import functional as F


class CorrNetAlignedBEVLifter(nn.Module):

    def __init__(
        self,
        in_channels=312,
        out_channels=80,
    ):
        super().__init__()

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        # 312ch CorrNet -> 80ch BEV feature
        self.adapter = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=1,
            ),
        )
    
    @staticmethod
    def align_rgb_tokens_to_image_plane(
        enc_out,
        frustum,
        img_aug_matrix,
        original_hw,
    ):
        """
        enc_out: [B*N, 312, Henc, Wsbs]
        frustum: [D, fH, fW, 3]
        img_aug_matrix: [B*N, 4, 4]
        """

        BN, C, Henc, Wsbs = enc_out.shape
        fH, fW = frustum.shape[1:3]
        Hraw, Wraw = original_hw

        if Wsbs % 2 != 0:
            raise ValueError(
                'SBS feature width must be even'
            )

        # SBS 왼쪽 RGB 영역만 추출
        rgb_tokens = enc_out[
            :, :, :, :Wsbs // 2
        ].float()

        # View Transform의 이미지 Feature 위치
        uv_aug = frustum[
            0, :, :, :2
        ].to(
            device=enc_out.device,
            dtype=torch.float32,
        )

        uv3 = torch.cat(
            [
                uv_aug,
                torch.zeros_like(
                    uv_aug[..., :1]
                ),
            ],
            dim=-1,
        )

        uv3 = (
            uv3.reshape(1, fH * fW, 3)
            .expand(BN, -1, -1)
        )

        # Image Augmentation 역변환
        post_rot = img_aug_matrix[:, :3, :3]
        post_trans = img_aug_matrix[:, :3, 3]

        uv_orig = torch.linalg.solve(
            post_rot,
            (
                uv3 - post_trans[:, None, :]
            ).transpose(1, 2),
        ).transpose(1, 2)[..., :2]

        u = uv_orig[..., 0]
        v = uv_orig[..., 1]

        # 원본 RGB 픽셀 좌표 -> grid_sample 좌표
        grid_x = (
            2.0 * (u + 0.5) / Wraw - 1.0
        )

        grid_y = (
            2.0 * (v + 0.5) / Hraw - 1.0
        )

        grid = torch.stack(
            [grid_x, grid_y],
            dim=-1,
        ).reshape(
            BN, fH, fW, 2
        )

        valid = (
            torch.isfinite(u)
            & torch.isfinite(v)
            & (u >= 0)
            & (u < Wraw)
            & (v >= 0)
            & (v < Hraw)
        ).reshape(BN, 1, fH, fW)

        grid = torch.nan_to_num(
            grid,
            nan=2.0,
            posinf=2.0,
            neginf=2.0,
        )

        aligned = F.grid_sample(
            rgb_tokens,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        )

        aligned = (
            aligned * valid.to(aligned.dtype)
        )

        return aligned, valid
    
    def forward(
        self,
        enc_out,
        depth_prob,
        view_transform,
        camera2lidar,
        camera_intrinsics,
        img_aug_matrix,
        lidar_aug_matrix,
        original_hw,
        camera_active_mask=None,
    ):
        """
        Returns:
            corr_bev: [B, 80, 180, 180]
            stats:    diagnostic dictionary
        """

        B, N = camera2lidar.shape[:2]
        BN = B * N

        D = int(view_transform.D)
        fH, fW = view_transform.feature_size

        # ----------------------------------
        # 1. Tensor shape verification
        # ----------------------------------

        if enc_out.ndim != 4:
            raise ValueError(
                f'enc_out shape: {enc_out.shape}'
            )

        if enc_out.shape[:2] != (
            BN, self.in_channels
        ):
            raise ValueError(
                f'enc_out mismatch: {enc_out.shape}'
            )

        if depth_prob.shape != (
            BN, D, fH, fW
        ):
            raise ValueError(
                f'depth mismatch: {depth_prob.shape}'
            )

        if img_aug_matrix.shape != (
            B, N, 4, 4
        ):
            raise ValueError(
                'img_aug_matrix shape mismatch'
            )

        if lidar_aug_matrix.shape != (
            B, 4, 4
        ):
            raise ValueError(
                'lidar_aug_matrix shape mismatch'
            )

        # ----------------------------------
        # 2. RGB SBS -> Image Feature Grid
        # ----------------------------------

        flat_aug = img_aug_matrix.reshape(
            BN, 4, 4
        ).float()

        aligned, visible = (
            self.align_rgb_tokens_to_image_plane(
                enc_out=enc_out,
                frustum=view_transform.frustum,
                img_aug_matrix=flat_aug,
                original_hw=original_hw,
            )
        )

        # [BN,312,fH,fW]

        # ----------------------------------
        # 3. Semantic Channel Adapter
        # ----------------------------------

        semantic = self.adapter(aligned)

        semantic = (
            semantic * visible.to(semantic.dtype)
        )

        # [BN,80,fH,fW]

        # ----------------------------------
        # 4. Active Camera Mask
        # ----------------------------------

        if camera_active_mask is not None:

            if camera_active_mask.shape != (B, N):
                raise ValueError(
                    'camera_active_mask mismatch'
                )

            mask = camera_active_mask.reshape(
                BN, 1, 1, 1
            ).to(semantic.dtype)

            semantic = semantic * mask

        # ----------------------------------
        # 5. Existing DepthLSS Probability
        # ----------------------------------

        depth = depth_prob.detach().float()

        lifted = (
            semantic.float().unsqueeze(2)
            * depth.unsqueeze(1)
        )

        # [BN,80,D,fH,fW]

        lifted = lifted.reshape(
            B,
            N,
            self.out_channels,
            D,
            fH,
            fW,
        )

        lifted = lifted.permute(
            0, 1, 3, 4, 5, 2
        ).contiguous()

        # [B,N,D,fH,fW,80]

        # ----------------------------------
        # 6. LGPC-Corrected Geometry
        # ----------------------------------

        c2l = camera2lidar.float()
        intr = camera_intrinsics.float()
        img_aug = img_aug_matrix.float()
        lidar_aug = lidar_aug_matrix.float()

        with torch.no_grad():

            geom = view_transform.get_geometry(
                c2l[..., :3, :3],
                c2l[..., :3, 3],
                intr[..., :3, :3],
                img_aug[..., :3, :3],
                img_aug[..., :3, 3],

                extra_rots=(
                    lidar_aug[..., :3, :3]
                ),

                extra_trans=(
                    lidar_aug[..., :3, 3]
                ),
            )

        # ----------------------------------
        # 7. BEV Pooling
        # ----------------------------------

        corr_bev = view_transform.bev_pool(
            geom,
            lifted,
        )

        corr_bev = view_transform.downsample(
            corr_bev
        )

        # ----------------------------------
        # 8. Diagnostics
        # ----------------------------------

        stats = {
            'rgb_token_coverage': float(
                visible.float().mean().detach()
            ),

            'aligned_feature_abs': float(
                semantic.abs().mean().detach()
            ),

            'bev_feature_abs': float(
                corr_bev.abs().mean().detach()
            ),
        }

        return corr_bev, stats