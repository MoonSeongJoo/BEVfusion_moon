import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalCorrRefinementHead(nn.Module):
    """
    R2.1 Local Correspondence Refinement.

    Main differences from R2.0
    -------------------------
    1. No residual_mlp bypass.

    2. Final residual is determined directly from
       the local candidate probabilities.

    3. Candidate logits are exposed to BEVFusion so
       that GT candidate supervision can be applied.

    Inputs
    ------
    local_feature:
        [B, 512, H, 2W]

        Frozen ResNet layer2 SBS feature.

        Expected current shape:
            [B, 512, 24, 160]


    query_input:
        [B, Q, 2]

        LEFT image query in SBS-normalized coordinates.

        x:
            q_x = u / (2 * 1600)

        y:
            q_y = v / 900


    coarse_corr:
        [B, Q, 2]

        Frozen R1 COTR prediction on RIGHT SBS.

        x:
            0.5 + u' / (2 * 1600)

        y:
            v' / 900
    """

    def __init__(
        self,
        in_channels=512,
        proj_dim=64,
        hidden_dim=128,
        radius_px=64.0,
        grid_size=5,
        image_width=1600.0,
        image_height=900.0,
        temperature=1.0,
    ):
        super().__init__()

        # ============================================================
        # 0. Sanity checks
        # ============================================================

        if grid_size < 3 or grid_size % 2 == 0:

            raise ValueError(
                'grid_size must be odd and >= 3, '
                f'got {grid_size}'
            )


        if radius_px <= 0:

            raise ValueError(
                'radius_px must be positive, '
                f'got {radius_px}'
            )


        if temperature <= 0:

            raise ValueError(
                'temperature must be positive, '
                f'got {temperature}'
            )


        if proj_dim % 8 != 0:

            raise ValueError(
                'proj_dim must be divisible by 8 '
                'for GroupNorm.'
            )


        self.in_channels = int(
            in_channels
        )

        self.proj_dim = int(
            proj_dim
        )

        self.hidden_dim = int(
            hidden_dim
        )

        self.radius_px = float(
            radius_px
        )

        self.grid_size = int(
            grid_size
        )

        self.image_width = float(
            image_width
        )

        self.image_height = float(
            image_height
        )

        self.temperature = float(
            temperature
        )


        # ============================================================
        # 1. Trainable layer2 feature projection
        #
        # Frozen backbone feature:
        #
        # 512
        #  ↓
        # 64
        # ============================================================

        self.feature_proj = nn.Sequential(

            nn.Conv2d(
                self.in_channels,
                self.proj_dim,
                kernel_size=1,
                bias=False,
            ),

            nn.GroupNorm(
                num_groups=8,
                num_channels=self.proj_dim,
            ),

            nn.GELU(),
        )


        # ============================================================
        # 2. Candidate score MLP
        #
        # For each local candidate:
        #
        # source feature         D
        # target feature         D
        # abs(source-target)     D
        # source*target          D
        # normalized offset      2
        #
        # input = 4*D + 2
        # ============================================================

        score_input_dim = (
            4 * self.proj_dim
            + 2
        )


        self.score_mlp = nn.Sequential(

            nn.Linear(
                score_input_dim,
                self.hidden_dim,
            ),

            nn.GELU(),

            nn.Linear(
                self.hidden_dim,
                64,
            ),

            nn.GELU(),

            nn.Linear(
                64,
                1,
            ),
        )


        # ============================================================
        # CRITICAL R2.1:
        #
        # All candidate logits start at exactly zero.
        #
        # Therefore:
        #
        # probability = uniform over valid candidates.
        #
        # Together with baseline-offset subtraction below,
        #
        # delta_px = 0 exactly at initialization.
        # ============================================================

        nn.init.zeros_(
            self.score_mlp[-1].weight
        )

        nn.init.zeros_(
            self.score_mlp[-1].bias
        )


        # ============================================================
        # 3. Candidate offsets
        #
        # Default:
        #
        # radius = 64
        # grid   = 5
        #
        # axis =
        #
        # [-64, -32, 0, +32, +64]
        #
        # 25 candidates.
        # ============================================================

        axis = torch.linspace(

            -self.radius_px,

            self.radius_px,

            steps=self.grid_size,
        )


        try:

            yy, xx = torch.meshgrid(
                axis,
                axis,
                indexing='ij',
            )

        except TypeError:

            yy, xx = torch.meshgrid(
                axis,
                axis,
            )


        candidate_offsets_px = torch.stack(

            [
                xx.reshape(-1),
                yy.reshape(-1),
            ],

            dim=-1,
        )

        # [K,2]
        #
        # [:,0] = du
        # [:,1] = dv


        self.register_buffer(

            'candidate_offsets_px',

            candidate_offsets_px,

            persistent=False,
        )


    # ================================================================
    # Feature sampling
    # ================================================================

    @staticmethod
    def _sample_feature(
        feature,
        grid,
    ):
        """
        feature:
            [B,C,H,W]

        grid:
            [B,Q,K,2]
            grid_sample coordinates [-1,+1]

        return:
            [B,Q,K,C]
        """

        sampled = F.grid_sample(

            feature,

            grid,

            mode='bilinear',

            padding_mode='zeros',

            align_corners=True,
        )

        # [B,C,Q,K]


        sampled = sampled.permute(
            0,
            2,
            3,
            1,
        ).contiguous()

        # [B,Q,K,C]

        return sampled


    # ================================================================
    # Forward
    # ================================================================

    def forward(
        self,
        local_feature,
        query_input,
        coarse_corr,
    ):

        # ============================================================
        # 0. Shape validation
        # ============================================================

        if local_feature.ndim != 4:

            raise RuntimeError(
                'local_feature must be [B,C,H,2W], '
                f'actual={tuple(local_feature.shape)}'
            )


        if (
            query_input.ndim != 3
            or query_input.shape[-1] != 2
        ):

            raise RuntimeError(
                'query_input must be [B,Q,2], '
                f'actual={tuple(query_input.shape)}'
            )


        if coarse_corr.shape != query_input.shape:

            raise RuntimeError(
                'coarse_corr/query_input mismatch: '
                f'{tuple(coarse_corr.shape)} vs '
                f'{tuple(query_input.shape)}'
            )


        B, C, H, W_sbs = (
            local_feature.shape
        )

        _, Q, _ = (
            query_input.shape
        )


        if C != self.in_channels:

            raise RuntimeError(
                '[R2.1 LocalRefiner] '
                'feature channel mismatch: '
                f'expected={self.in_channels}, '
                f'actual={C}'
            )


        if W_sbs % 2 != 0:

            raise RuntimeError(
                '[R2.1 LocalRefiner] '
                f'SBS width must be even: {W_sbs}'
            )


        # ============================================================
        # 1. Project frozen layer2 feature
        # ============================================================

        projected = self.feature_proj(
            local_feature
        )

        # expected:
        #
        # [B,64,24,160]


        half_w = (
            projected.shape[-1]
            // 2
        )


        # ============================================================
        # 2. LEFT / RIGHT feature split
        # ============================================================

        left_feature = (
            projected[
                ...,
                :half_w
            ]
        )

        right_feature = (
            projected[
                ...,
                half_w:
            ]
        )

        # expected:
        #
        # [B,64,24,80]
        # [B,64,24,80]


        # ============================================================
        # 3. Sample source feature at LEFT query
        #
        # query x:
        #
        #   q_x = u / (2W)
        #
        # LEFT-half normalized x:
        #
        #   x_left = 2*q_x
        # ============================================================

        source_x01 = (
            query_input[..., 0]
            * 2.0
        )

        source_y01 = (
            query_input[..., 1]
        )


        source_grid = torch.stack(

            [
                source_x01 * 2.0 - 1.0,
                source_y01 * 2.0 - 1.0,
            ],

            dim=-1,
        ).unsqueeze(2)

        # [B,Q,1,2]


        source_feat = self._sample_feature(

            left_feature,

            source_grid,
        ).squeeze(2)

        # [B,Q,D]


        # ============================================================
        # 4. Coarse target position on RIGHT image
        #
        # SBS:
        #
        # x = 0.5 + u'/(2W)
        #
        # Right-half local coordinate:
        #
        # x_right = (x - 0.5)*2
        # ============================================================

        coarse_x01 = (

            (
                coarse_corr[..., 0]
                - 0.5
            )

            * 2.0
        )


        coarse_y01 = (
            coarse_corr[..., 1]
        )


        offsets_px = (
            self.candidate_offsets_px
            .to(
                device=coarse_corr.device,
                dtype=coarse_corr.dtype,
            )
        )


        K = offsets_px.shape[0]


        # ============================================================
        # 5. 5x5 local target coordinates
        #
        # offsets are ORIGINAL pixels.
        #
        # Since coarse_x01 is RIGHT-image normalized:
        #
        #   du_norm = du / 1600
        #
        # and:
        #
        #   dv_norm = dv / 900
        # ============================================================

        candidate_x01 = (

            coarse_x01.unsqueeze(-1)

            +

            offsets_px[
                :,
                0
            ].view(
                1,
                1,
                K,
            )

            / self.image_width
        )


        candidate_y01 = (

            coarse_y01.unsqueeze(-1)

            +

            offsets_px[
                :,
                1
            ].view(
                1,
                1,
                K,
            )

            / self.image_height
        )


        # ============================================================
        # 6. Candidate validity
        # ============================================================

        candidate_valid = (

            torch.isfinite(
                candidate_x01
            )

            & torch.isfinite(
                candidate_y01
            )

            & (
                candidate_x01 >= 0.0
            )

            & (
                candidate_x01 <= 1.0
            )

            & (
                candidate_y01 >= 0.0
            )

            & (
                candidate_y01 <= 1.0
            )
        )


        # ============================================================
        # 7. grid_sample coordinates
        # ============================================================

        candidate_grid = torch.stack(

            [
                candidate_x01 * 2.0 - 1.0,
                candidate_y01 * 2.0 - 1.0,
            ],

            dim=-1,
        )

        # [B,Q,K,2]


        target_feat = self._sample_feature(

            right_feature,

            candidate_grid,
        )

        # [B,Q,K,D]


        # ============================================================
        # 8. Candidate descriptor
        # ============================================================

        source_expand = (

            source_feat
            .unsqueeze(2)
            .expand(
                -1,
                -1,
                K,
                -1,
            )
        )


        abs_diff = torch.abs(

            source_expand
            - target_feat
        )


        product = (

            source_expand
            * target_feat
        )


        offset_norm = (

            offsets_px
            / self.radius_px
        )


        offset_norm = (

            offset_norm
            .view(
                1,
                1,
                K,
                2,
            )
            .expand(
                B,
                Q,
                -1,
                -1,
            )
        )


        candidate_descriptor = torch.cat(

            [
                source_expand,
                target_feat,
                abs_diff,
                product,
                offset_norm,
            ],

            dim=-1,
        )


        # ============================================================
        # 9. Candidate score logits
        # ============================================================

        score_logits_raw = (

            self.score_mlp(
                candidate_descriptor
            )
            .squeeze(-1)
        )

        # [B,Q,K]


        # ============================================================
        # Safety:
        # if every candidate is invalid, permit center only.
        # ============================================================

        all_invalid = (

            ~candidate_valid.any(
                dim=-1,
                keepdim=True,
            )
        )


        if all_invalid.any():

            center_idx = (
                K // 2
            )


            fallback_valid = (
                torch.zeros_like(
                    candidate_valid
                )
            )


            fallback_valid[
                ...,
                center_idx
            ] = True


            candidate_valid = torch.where(

                all_invalid,

                fallback_valid,

                candidate_valid,
            )


        # ============================================================
        # 10. Mask invalid logits
        # ============================================================

        score_logits = (
            score_logits_raw.masked_fill(
                ~candidate_valid,
                -1.0e4,
            )
        )


        # ============================================================
        # 11. Learned candidate probability
        # ============================================================

        weights = F.softmax(

            score_logits
            / self.temperature,

            dim=-1,
        )


        # ============================================================
        # 12. Uniform-valid baseline probability
        #
        # Important:
        #
        # At image boundaries the valid candidate set may be
        # asymmetric.
        #
        # If we simply use:
        #
        #     sum(weights * offsets)
        #
        # then even uniform weights can produce a non-zero offset
        # at an image boundary.
        #
        # Therefore subtract the uniform-valid baseline offset.
        #
        # At initialization:
        #
        # learned weights == uniform valid weights
        #
        # => delta_px == 0 exactly.
        # ============================================================

        baseline_weights = (
            candidate_valid.float()
        )


        baseline_weights = (

            baseline_weights

            /

            baseline_weights.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1.0)
        )


        # ============================================================
        # 13. Learned soft offset
        # ============================================================

        candidate_offsets = (
            offsets_px.view(
                1,
                1,
                K,
                2,
            )
        )


        soft_offset_px = (

            weights.unsqueeze(-1)

            * candidate_offsets

        ).sum(
            dim=2
        )


        baseline_offset_px = (

            baseline_weights.unsqueeze(-1)

            * candidate_offsets

        ).sum(
            dim=2
        )


        # ============================================================
        # CRITICAL R2.1:
        #
        # No residual MLP.
        #
        # Candidate probabilities themselves determine correction.
        # ============================================================

        delta_px = (

            soft_offset_px

            - baseline_offset_px
        )


        # ============================================================
        # 14. Refined correspondence
        # ============================================================

        refined_corr = (
            coarse_corr.clone()
        )


        # Horizontal SBS normalization:
        #
        # du / (2*1600)

        refined_corr[
            ...,
            0
        ] = (

            coarse_corr[
                ...,
                0
            ]

            +

            delta_px[
                ...,
                0
            ]

            / (
                2.0
                * self.image_width
            )
        )


        # Vertical:
        #
        # dv / 900

        refined_corr[
            ...,
            1
        ] = (

            coarse_corr[
                ...,
                1
            ]

            +

            delta_px[
                ...,
                1
            ]

            / self.image_height
        )


        # ============================================================
        # 15. Diagnostics
        # ============================================================

        max_weight = (
            weights.max(
                dim=-1
            ).values
        )


        valid_count = (

            candidate_valid
            .sum(
                dim=-1
            )
            .float()
        )


        entropy = -(

            weights

            * torch.log(
                weights.clamp_min(
                    1.0e-8
                )
            )

        ).sum(
            dim=-1
        )


        entropy_denominator = torch.log(

            valid_count.clamp_min(
                2.0
            )
        )


        entropy_norm = (

            entropy

            / entropy_denominator
        )


        diagnostics = {

            'delta_px':
                delta_px.detach(),

            'soft_offset_px':
                soft_offset_px.detach(),

            'baseline_offset_px':
                baseline_offset_px.detach(),

            'max_weight':
                max_weight.detach(),

            'candidate_valid_ratio':
                (
                    candidate_valid
                    .float()
                    .mean(
                        dim=-1
                    )
                    .detach()
                ),

            'candidate_entropy_norm':
                entropy_norm.detach(),
        }


        # ============================================================
        # 16. Training auxiliary tensors
        #
        # score_logits MUST NOT be detached.
        #
        # BEVFusion uses it for direct candidate supervision.
        # ============================================================

        train_aux = {

            'score_logits':
                score_logits,

            'candidate_valid':
                candidate_valid,

            'candidate_offsets_px':
                offsets_px,
        }


        return (
            refined_corr,
            diagnostics,
            train_aux,
        )