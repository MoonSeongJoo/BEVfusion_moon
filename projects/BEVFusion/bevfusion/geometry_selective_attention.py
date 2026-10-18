
import math

import torch
from torch import nn


class GeometrySelectiveAttention(nn.Module):

    def __init__(
        self,
        channels=128,
        num_heads=8,
        topk=16,
        radius_m=4.0,
        sigma_m=2.0,
        gate_init_bias=-2.2,
    ):
        super().__init__()

        assert channels % num_heads == 0
        assert topk > 0
        assert radius_m > 0
        assert sigma_m > 0

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.topk = topk
        self.radius_m = radius_m
        self.sigma_m = sigma_m

        # Feature projection
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)

        # Residual output
        self.out_proj = nn.Linear(channels, channels)

        # Query-specific reliability gate
        gate_hidden = max(32, channels // 2)

        self.gate_net = nn.Sequential(
            nn.Linear(2 * channels + 1, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 1),
        )

        # Important: exact identity initialization
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(
            self.gate_net[-1].bias,
            gate_init_bias,
        )

        self.last_stats = {}
        self._stats_recorded = False

    def forward(
        self,
        query_chw,
        query_grid_xy,
        camera_feat,
        corrected_cam_xyz_norm,
        pc_range,
        voxel_size,
        out_size_factor,
        proposal_valid_mask=None,
        active_cam_mask=None,
    ):

        B, C, Nq = query_chw.shape
        Np = camera_feat.shape[1]

        assert C == self.channels
        assert query_grid_xy.shape == (B, Nq, 2)
        assert camera_feat.shape == (B, Np, C)
        assert corrected_cam_xyz_norm.shape == (B, Np, 3)

        if Np == 0:
            return torch.zeros_like(query_chw)

        device = query_chw.device

        # --------------------------------------------
        # STEP A: Coordinate conversion
        # --------------------------------------------

        pcr = torch.as_tensor(
            pc_range,
            dtype=torch.float32,
            device=device,
        )

        vox = torch.as_tensor(
            voxel_size,
            dtype=torch.float32,
            device=device,
        )

        # BEV grid -> LiDAR meters
        query_metric_xy = (
            query_grid_xy.detach().float()
            * (
                vox[:2] * float(out_size_factor)
            ).view(1, 1, 2)
            + pcr[:2].view(1, 1, 2)
        )

        # Normalized Camera Proposal -> LiDAR meters
        xyz = corrected_cam_xyz_norm.detach().float()

        cam_metric_xy = (
            xyz[..., :2]
            * (pcr[3:5] - pcr[:2]).view(1, 1, 2)
            + pcr[:2].view(1, 1, 2)
        )

        # --------------------------------------------
        # STEP B: Query-to-Proposal distance
        # --------------------------------------------

        sqdist = (
            (
                query_metric_xy.unsqueeze(2)
                - cam_metric_xy.unsqueeze(1)
            ) ** 2
        ).sum(dim=-1)

        # [B, Nq, Np]

        # --------------------------------------------
        # STEP C: Proposal validity
        # --------------------------------------------

        if proposal_valid_mask is None:

            valid = torch.ones(
                (B, Np),
                dtype=torch.bool,
                device=device,
            )

        else:

            mask = proposal_valid_mask.bool().to(device)

            if mask.numel() != B * Np:
                raise ValueError(
                    "Invalid proposal mask shape"
                )

            if mask.ndim == 3:

                if active_cam_mask is not None:
                    active = active_cam_mask.bool().to(device)

                    assert active.shape == (
                        B, mask.shape[1]
                    )

                    mask = (
                        mask
                        & active.unsqueeze(-1)
                    )

            elif active_cam_mask is not None:
                raise ValueError(
                    "Expected 3D proposal mask"
                )

            valid = mask.reshape(B, Np)

        valid = valid & torch.isfinite(xyz).all(-1)

        valid = (
            valid
            & (xyz >= 0).all(-1)
            & (xyz <= 1).all(-1)
        )

        # --------------------------------------------
        # STEP D: Geometry-based Top-K selection
        # --------------------------------------------

        select_dist = sqdist.masked_fill(
            ~valid.unsqueeze(1),
            float("inf"),
        )

        K = min(self.topk, Np)

        best_d2, best_idx = torch.topk(
            select_dist,
            k=K,
            dim=-1,
            largest=False,
        )

        neighbors_valid = (
            torch.isfinite(best_d2)
            & (best_d2 <= self.radius_m ** 2)
        )

        has_neighbor = neighbors_valid.any(-1)

        # Replace invalid distances before softmax
        best_d2 = torch.where(
            neighbors_valid,
            best_d2,
            torch.zeros_like(best_d2),
        )

        batch_idx = torch.arange(
            B, device=device
        )[:, None, None]

        selected_feat = camera_feat[
            batch_idx,
            best_idx,
        ]

        # [B, Nq, K, C]

        # --------------------------------------------
        # STEP E: Multi-head selective attention
        # --------------------------------------------

        query = query_chw.transpose(1, 2).contiguous()

        H = self.num_heads
        D = self.head_dim

        q = self.q_proj(query).reshape(
            B, Nq, H, D
        )

        k = self.k_proj(selected_feat).reshape(
            B, Nq, K, H, D
        )

        v = self.v_proj(selected_feat).reshape(
            B, Nq, K, H, D
        )

        feature_score = (
            q.unsqueeze(2) * k
        ).sum(-1) / math.sqrt(D)

        geometry_score = (
            -best_d2.unsqueeze(-1)
            / (2 * self.sigma_m ** 2)
        )

        scores = (
            feature_score
            + geometry_score.to(feature_score.dtype)
        )

        scores = scores.masked_fill(
            ~neighbors_valid.unsqueeze(-1),
            -1e4,
        )

        weights = torch.softmax(
            scores.float(),
            dim=2,
        ).to(v.dtype)

        weights = (
            weights
            * neighbors_valid.unsqueeze(-1)
        )

        weights = (
            weights
            / weights.sum(
                dim=2,
                keepdim=True,
            ).clamp_min(1e-6)
        )

        context = (
            weights.unsqueeze(-1) * v
        ).sum(dim=2)

        context = context.reshape(B, Nq, C)

        # --------------------------------------------
        # STEP F: Reliability gate
        # --------------------------------------------

        valid_d2 = best_d2.masked_fill(
            ~neighbors_valid,
            self.radius_m ** 2,
        )

        min_d2 = valid_d2.min(-1).values

        reliability = torch.exp(
            -min_d2
            / (2 * self.sigma_m ** 2)
        )

        reliability = (
            reliability * has_neighbor.float()
        )

        gate_input = torch.cat(
            [
                query,
                context,
                reliability.to(query.dtype).unsqueeze(-1),
            ],
            dim=-1,
        )

        gate = torch.sigmoid(
            self.gate_net(gate_input)
        )

        gate = (
            gate
            * has_neighbor.unsqueeze(-1)
        )

        # --------------------------------------------
        # STEP G: Residual output
        # --------------------------------------------

        delta = (
            gate * self.out_proj(context)
        )

        # --------------------------------------------
        # Debug statistics
        # --------------------------------------------

        if not self._stats_recorded:

            self.last_stats = {
                "coverage": float(
                    has_neighbor.float().mean().item()
                ),
                "mean_gate": float(
                    gate.detach().mean().item()
                ),
                "mean_valid_neighbors": float(
                    neighbors_valid.float()
                    .sum(-1).mean().item()
                ),
            }

            self._stats_recorded = True

        return delta.transpose(1, 2).contiguous()
