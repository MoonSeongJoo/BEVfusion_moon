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
        high_in_channels=312,

        proj_dim=64,
        hidden_dim=128,

        patch_embed_dim=32,
        patch_grid_size=3,

        patch_radius_l2_px=24.0,
        patch_radius_high_px=48.0,

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

        self.high_in_channels = int(
            high_in_channels
        )

        self.patch_embed_dim = int(
            patch_embed_dim
        )

        self.patch_grid_size = int(
            patch_grid_size
        )

        self.patch_radius_l2_px = float(
            patch_radius_l2_px
        )

        self.patch_radius_high_px = float(
            patch_radius_high_px
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
        # V1.2:
        # high-level / layer3-scale feature projection
        #
        # enc_out:
        #     [B,312,12,80]
        #
        # projected:
        #     [B,64,12,80]
        # ============================================================

        self.high_feature_proj = nn.Sequential(

            nn.Conv2d(
                self.high_in_channels,
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
        # V1.2 Multi-scale candidate descriptor
        #
        # Layer2:
        #   source        D
        #   target        D
        #   abs diff      D
        #   product       D
        #
        # High-scale:
        #   source        D
        #   target        D
        #   abs diff      D
        #   product       D
        #
        # Patch embeddings:
        #   layer2        patch_embed_dim
        #   high          patch_embed_dim
        #
        # Geometry:
        #   offset        2
        #
        # total =
        #   8*D
        #   + 2*patch_embed_dim
        #   + 2
        # ============================================================

        score_input_dim = (

            8
            * self.proj_dim

            +

            2
            * self.patch_embed_dim

            +

            2
        )


        self.score_input_dim = (

            score_input_dim
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

        # ============================================================
        # V1.2 Multi-scale patch offsets
        # ============================================================

        def _make_patch_offsets(
            radius_px,
            grid_size,
        ):

            axis_patch = torch.linspace(

                -radius_px,
                radius_px,

                steps=grid_size,
            )

            try:

                yy_patch, xx_patch = torch.meshgrid(
                    axis_patch,
                    axis_patch,
                    indexing='ij',
                )

            except TypeError:

                yy_patch, xx_patch = torch.meshgrid(
                    axis_patch,
                    axis_patch,
                )

            return torch.stack(
                [
                    xx_patch.reshape(-1),
                    yy_patch.reshape(-1),
                ],
                dim=-1,
            )


        patch_offsets_l2_px = _make_patch_offsets(

            self.patch_radius_l2_px,
            self.patch_grid_size,
        )


        patch_offsets_high_px = _make_patch_offsets(

            self.patch_radius_high_px,
            self.patch_grid_size,
        )


        self.register_buffer(

            'patch_offsets_l2_px',

            patch_offsets_l2_px,

            persistent=False,
        )


        self.register_buffer(

            'patch_offsets_high_px',

            patch_offsets_high_px,

            persistent=False,
        )


        self.patch_points = (

            self.patch_grid_size
            *
            self.patch_grid_size
        )

        # ============================================================
        # 5. V1.2 Patch relation encoders
        # ============================================================

        patch_relation_dim = (

            3
            *
            self.patch_points
        )


        self.patch_relation_dim = (

            patch_relation_dim
        )


        self.patch_encoder_l2 = nn.Sequential(

            nn.Linear(
                patch_relation_dim,
                self.patch_embed_dim,
            ),

            nn.GELU(),

            nn.Linear(
                self.patch_embed_dim,
                self.patch_embed_dim,
            ),

            nn.GELU(),
        )


        self.patch_encoder_high = nn.Sequential(

            nn.Linear(
                patch_relation_dim,
                self.patch_embed_dim,
            ),

            nn.GELU(),

            nn.Linear(
                self.patch_embed_dim,
                self.patch_embed_dim,
            ),

            nn.GELU(),
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
    # V1.2:
    # Source local-patch sampling
    # ================================================================

    def _sample_source_patch(
        self,
        feature,
        center_x01,
        center_y01,
        patch_offsets_px,
    ):
        """
        feature:
            [B,D,H,W]

        center_x01:
            [B,Q]

        center_y01:
            [B,Q]

        patch_offsets_px:
            [P,2]

        Returns
        -------
        sampled:
            [B,Q,P,D]

        valid:
            [B,Q,P]
        """

        offsets = patch_offsets_px.to(

            device=center_x01.device,
            dtype=center_x01.dtype,
        )


        P = offsets.shape[0]


        # ------------------------------------------------------------
        # Original-image pixel offset
        # -> normalized coordinate inside one LEFT/RIGHT image.
        # ------------------------------------------------------------

        patch_x01 = (

            center_x01.unsqueeze(-1)

            +

            offsets[
                :,
                0
            ].view(
                1,
                1,
                P,
            )

            / self.image_width
        )


        patch_y01 = (

            center_y01.unsqueeze(-1)

            +

            offsets[
                :,
                1
            ].view(
                1,
                1,
                P,
            )

            / self.image_height
        )


        # ------------------------------------------------------------
        # Patch-point validity
        # ------------------------------------------------------------

        valid = (

            torch.isfinite(
                patch_x01
            )

            &

            torch.isfinite(
                patch_y01
            )

            &

            (
                patch_x01 >= 0.0
            )

            &

            (
                patch_x01 <= 1.0
            )

            &

            (
                patch_y01 >= 0.0
            )

            &

            (
                patch_y01 <= 1.0
            )
        )


        # ------------------------------------------------------------
        # [0,1]
        # ->
        # grid_sample coordinate [-1,+1]
        # ------------------------------------------------------------

        grid = torch.stack(

            [
                patch_x01 * 2.0 - 1.0,
                patch_y01 * 2.0 - 1.0,
            ],

            dim=-1,
        )

        # [B,Q,P,2]


        sampled = self._sample_feature(

            feature,
            grid,
        )

        # [B,Q,P,D]


        return (
            sampled,
            valid,
        )


    # ================================================================
    # V1.2:
    # Candidate local-patch sampling
    # ================================================================

    def _sample_candidate_patch(
        self,
        feature,
        center_x01,
        center_y01,
        patch_offsets_px,
    ):
        """
        feature:
            [B,D,H,W]

        center_x01:
            [B,Q,K]

        center_y01:
            [B,Q,K]

        patch_offsets_px:
            [P,2]

        Returns
        -------
        sampled:
            [B,Q,K,P,D]

        valid:
            [B,Q,K,P]
        """

        offsets = patch_offsets_px.to(

            device=center_x01.device,
            dtype=center_x01.dtype,
        )


        B, Q, K = (
            center_x01.shape
        )


        P = offsets.shape[0]


        patch_x01 = (

            center_x01.unsqueeze(-1)

            +

            offsets[
                :,
                0
            ].view(
                1,
                1,
                1,
                P,
            )

            / self.image_width
        )


        patch_y01 = (

            center_y01.unsqueeze(-1)

            +

            offsets[
                :,
                1
            ].view(
                1,
                1,
                1,
                P,
            )

            / self.image_height
        )


        valid = (

            torch.isfinite(
                patch_x01
            )

            &

            torch.isfinite(
                patch_y01
            )

            &

            (
                patch_x01 >= 0.0
            )

            &

            (
                patch_x01 <= 1.0
            )

            &

            (
                patch_y01 >= 0.0
            )

            &

            (
                patch_y01 <= 1.0
            )
        )


        grid = torch.stack(

            [
                patch_x01 * 2.0 - 1.0,
                patch_y01 * 2.0 - 1.0,
            ],

            dim=-1,
        )

        # [B,Q,K,P,2]


        # ------------------------------------------------------------
        # Existing _sample_feature() expects:
        #
        #     [B,Q,N,2]
        #
        # so flatten K and P temporarily.
        # ------------------------------------------------------------

        grid_flat = grid.reshape(

            B,
            Q,
            K * P,
            2,
        )


        sampled_flat = self._sample_feature(

            feature,
            grid_flat,
        )

        # [B,Q,K*P,D]


        D = (
            sampled_flat.shape[-1]
        )


        sampled = sampled_flat.reshape(

            B,
            Q,
            K,
            P,
            D,
        )

        # [B,Q,K,P,D]


        return (
            sampled,
            valid,
        )


    # ================================================================
    # V1.2:
    # Build local patch relation descriptor
    # ================================================================

    def _build_patch_relation(
        self,
        source_patch,
        target_patch,
        source_valid,
        target_valid,
        encoder,
    ):
        """
        source_patch:
            [B,Q,P,D]

        target_patch:
            [B,Q,K,P,D]

        source_valid:
            [B,Q,P]

        target_valid:
            [B,Q,K,P]

        Returns
        -------
        patch_embedding:
            [B,Q,K,patch_embed_dim]
        """

        K = (
            target_patch.shape[2]
        )


        # ------------------------------------------------------------
        # Expand source patch across K candidates
        # ------------------------------------------------------------

        source_expand = (

            source_patch
            .unsqueeze(2)
            .expand(
                -1,
                -1,
                K,
                -1,
                -1,
            )
        )

        # [B,Q,K,P,D]


        # ------------------------------------------------------------
        # 1. Cosine similarity for each corresponding patch point
        # ------------------------------------------------------------

        source_norm = F.normalize(

            source_expand,

            p=2,
            dim=-1,

            eps=1.0e-6,
        )


        target_norm = F.normalize(

            target_patch,

            p=2,
            dim=-1,

            eps=1.0e-6,
        )


        patch_cos = (

            source_norm
            *
            target_norm

        ).sum(
            dim=-1
        )

        # [B,Q,K,P]


        # ------------------------------------------------------------
        # 2. Feature absolute difference at each patch point
        # ------------------------------------------------------------

        patch_absdiff = (

            torch.abs(

                source_expand
                -
                target_patch
            )

            .mean(
                dim=-1
            )
        )

        # [B,Q,K,P]


        # ------------------------------------------------------------
        # 3. Pair validity
        # ------------------------------------------------------------

        pair_valid = (

            source_valid
            .unsqueeze(2)

            &

            target_valid
        )

        # [B,Q,K,P]


        pair_valid_float = (

            pair_valid.to(
                dtype=patch_cos.dtype
            )
        )


        # Invalid patch samples should contribute no feature evidence.
        patch_cos = (

            patch_cos
            *
            pair_valid_float
        )


        patch_absdiff = (

            patch_absdiff
            *
            pair_valid_float
        )


        # ------------------------------------------------------------
        # Per patch point:
        #
        # cosine      P
        # absdiff     P
        # validity    P
        #
        # total = 3*P
        #
        # P=9 -> 27 dim
        # ------------------------------------------------------------

        patch_relation = torch.cat(

            [
                patch_cos,
                patch_absdiff,
                pair_valid_float,
            ],

            dim=-1,
        )

        # [B,Q,K,3*P]


        patch_embedding = encoder(

            patch_relation
        )

        # [B,Q,K,patch_embed_dim]


        return patch_embedding

    # ================================================================
    # Forward
    # ================================================================

    def forward(
        self,
        local_feature,
        high_feature,
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
        # V1.2:
        # high-level feature validation
        # ============================================================

        if high_feature.ndim != 4:

            raise RuntimeError(
                'high_feature must be [B,C,H,2W], '
                f'actual={tuple(high_feature.shape)}'
            )


        if (
            high_feature.shape[0]
            != local_feature.shape[0]
        ):

            raise RuntimeError(
                '[V1.2 LocalRefiner] '
                'local/high batch mismatch: '
                f'local={tuple(local_feature.shape)}, '
                f'high={tuple(high_feature.shape)}'
            )


        if (
            high_feature.shape[1]
            != self.high_in_channels
        ):

            raise RuntimeError(
                '[V1.2 LocalRefiner] '
                'high feature channel mismatch: '
                f'expected={self.high_in_channels}, '
                f'actual={high_feature.shape[1]}'
            )


        if (
            high_feature.shape[-1]
            % 2
            != 0
        ):

            raise RuntimeError(
                '[V1.2 LocalRefiner] '
                'high SBS width must be even: '
                f'{tuple(high_feature.shape)}'
            )

        # ============================================================
        # 1. V1.2 Multi-scale feature projection
        # ============================================================

        projected_l2 = self.feature_proj(

            local_feature
        )

        # expected:
        # [B,64,24,160]


        projected_high = self.high_feature_proj(

            high_feature
        )

        # expected:
        # [B,64,12,80]


        # ============================================================
        # 2. LEFT / RIGHT split - Layer2
        # ============================================================

        half_w_l2 = (

            projected_l2.shape[-1]
            // 2
        )


        left_l2 = (

            projected_l2[
                ...,
                :half_w_l2
            ]
        )


        right_l2 = (

            projected_l2[
                ...,
                half_w_l2:
            ]
        )


        # expected:
        #
        # left_l2:
        #     [B,64,24,80]
        #
        # right_l2:
        #     [B,64,24,80]


        # ============================================================
        # 3. LEFT / RIGHT split - high-level feature
        # ============================================================

        half_w_high = (

            projected_high.shape[-1]
            // 2
        )


        left_high = (

            projected_high[
                ...,
                :half_w_high
            ]
        )


        right_high = (

            projected_high[
                ...,
                half_w_high:
            ]
        )


        # expected:
        #
        # left_high:
        #     [B,64,12,40]
        #
        # right_high:
        #     [B,64,12,40]


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


        # ============================================================
        # V1.2 Source point feature - Layer2
        # ============================================================

        source_feat_l2 = self._sample_feature(

            left_l2,

            source_grid,

        ).squeeze(2)

        # [B,Q,64]


        # ============================================================
        # V1.2 Source point feature - high-level
        # ============================================================

        source_feat_high = self._sample_feature(

            left_high,

            source_grid,

        ).squeeze(2)

        # [B,Q,64]


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


        # ============================================================
        # V1.2 Candidate point feature - Layer2
        # ============================================================

        target_feat_l2 = self._sample_feature(

            right_l2,

            candidate_grid,
        )

        # [B,Q,K,64]


        # ============================================================
        # V1.2 Candidate point feature - high-level
        # ============================================================

        target_feat_high = self._sample_feature(

            right_high,

            candidate_grid,
        )

        # [B,Q,K,64]

        # ============================================================
        # 8. V1.2 Layer2 local patch matching
        # ============================================================

        (
            source_patch_l2,
            source_patch_valid_l2,
        ) = self._sample_source_patch(

            left_l2,

            source_x01,
            source_y01,

            self.patch_offsets_l2_px,
        )

        # source_patch_l2:
        #     [B,Q,9,64]


        (
            target_patch_l2,
            target_patch_valid_l2,
        ) = self._sample_candidate_patch(

            right_l2,

            candidate_x01,
            candidate_y01,

            self.patch_offsets_l2_px,
        )

        # target_patch_l2:
        #     [B,Q,K,9,64]


        patch_embedding_l2 = (

            self._build_patch_relation(

                source_patch_l2,
                target_patch_l2,

                source_patch_valid_l2,
                target_patch_valid_l2,

                self.patch_encoder_l2,
            )
        )

        # [B,Q,K,32]


        # ============================================================
        # 9. V1.2 High-level local patch matching
        # ============================================================

        (
            source_patch_high,
            source_patch_valid_high,
        ) = self._sample_source_patch(

            left_high,

            source_x01,
            source_y01,

            self.patch_offsets_high_px,
        )


        (
            target_patch_high,
            target_patch_valid_high,
        ) = self._sample_candidate_patch(

            right_high,

            candidate_x01,
            candidate_y01,

            self.patch_offsets_high_px,
        )


        patch_embedding_high = (

            self._build_patch_relation(

                source_patch_high,
                target_patch_high,

                source_patch_valid_high,
                target_patch_valid_high,

                self.patch_encoder_high,
            )
        )

        # [B,Q,K,32]

        # ============================================================
        # 10. V1.2 Multi-scale candidate descriptor
        # ============================================================


        # ------------------------------------------------------------
        # A. Layer2 point relation
        # ------------------------------------------------------------

        source_expand_l2 = (

            source_feat_l2
            .unsqueeze(2)
            .expand(
                -1,
                -1,
                K,
                -1,
            )
        )

        # [B,Q,K,64]


        abs_diff_l2 = torch.abs(

            source_expand_l2
            -
            target_feat_l2
        )


        product_l2 = (

            source_expand_l2
            *
            target_feat_l2
        )


        # ------------------------------------------------------------
        # B. High-level point relation
        # ------------------------------------------------------------

        source_expand_high = (

            source_feat_high
            .unsqueeze(2)
            .expand(
                -1,
                -1,
                K,
                -1,
            )
        )

        # [B,Q,K,64]


        abs_diff_high = torch.abs(

            source_expand_high
            -
            target_feat_high
        )


        product_high = (

            source_expand_high
            *
            target_feat_high
        )


        # ------------------------------------------------------------
        # C. Candidate geometric offset
        # ------------------------------------------------------------

        offset_norm = (

            offsets_px
            /
            self.radius_px
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


        # ------------------------------------------------------------
        # D. Final V1.2 descriptor
        #
        # Layer2 point:
        #     4*64 = 256
        #
        # High point:
        #     4*64 = 256
        #
        # Layer2 patch:
        #     32
        #
        # High patch:
        #     32
        #
        # Offset:
        #     2
        #
        # TOTAL:
        #     578
        # ------------------------------------------------------------

        candidate_descriptor = torch.cat(

            [
                # Layer2 fine-scale point evidence
                source_expand_l2,
                target_feat_l2,
                abs_diff_l2,
                product_l2,

                # High-level semantic point evidence
                source_expand_high,
                target_feat_high,
                abs_diff_high,
                product_high,

                # Multi-scale patch context
                patch_embedding_l2,
                patch_embedding_high,

                # Geometric candidate location
                offset_norm,
            ],

            dim=-1,
        )


        # ------------------------------------------------------------
        # Safety check
        # ------------------------------------------------------------

        if (
            candidate_descriptor.shape[-1]
            != self.score_input_dim
        ):

            raise RuntimeError(

                '[V1.2 LocalRefiner] '
                'candidate descriptor dim mismatch: '

                f'expected={self.score_input_dim}, '

                f'actual='
                f'{candidate_descriptor.shape[-1]}'
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