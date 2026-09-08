import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from mmengine.runner import load_checkpoint
from mmengine.logging import print_log

@MODELS.register_module()
class ZEstimator(BaseModule):
    """
    글로벌 컨텍스트, 로컬 특징, 위치 정보를 모두 융합하여 깊이를 추정하는 모듈.
    """
    def __init__(self, enc_channels=312, local_channels=312, uv_dim=2, hidden_dim=512, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        # 1. 글로벌 컨텍스트 추출기
        self.enc_adaptor = nn.Sequential(
            nn.Conv2d(enc_channels, 128, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # # 2. 위치 정보 임베딩
        self.uv_embed = nn.Linear(uv_dim, 64)

        # ============================================================
        # V2: Metric-depth embedding
        #
        # Input:
        #   normalized local LiDAR depth [0,1]
        #
        # Output:
        #   64-D depth representation
        #
        # Keeping 64-D allows us to fuse it with UV embedding
        # without changing the input dimension of self.fusion.
        # ============================================================

        self.depth_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.LeakyReLU(
                negative_slope=0.01
            ),
            nn.LayerNorm(64),
        )

        # ============================================================
        # V3: Lightweight Neighborhood Z Estimation
        #
        # Transformer-free:
        #
        #   5x5 neighborhood
        #       -> candidate feature / depth / offset
        #       -> tiny shared MLP scoring
        #       -> weighted depth anchor
        #       -> residual Z refinement
        # ============================================================

        self.neighborhood_grid_size = 5

        # Corr residual ~ 수십 pixel을 고려한 최초 실험 범위
        self.neighborhood_radius_px = 64.0

        # Candidate feature는 312 channel을 그대로 쓰지 않고
        # 32 channel로 먼저 projection하여 계산량을 줄임.
        self.candidate_feat_dim = 32


        # ============================================================
        # 1. Neighborhood offsets
        #
        # [-64, -32, 0, 32, 64] x
        # [-64, -32, 0, 32, 64]
        #
        # => K = 25 candidates
        # ============================================================

        offset_axis = torch.linspace(
            -self.neighborhood_radius_px,
            self.neighborhood_radius_px,
            steps=self.neighborhood_grid_size,
        )


        offset_y, offset_x = torch.meshgrid(
            offset_axis,
            offset_axis,
            indexing='ij',
        )


        neighborhood_offsets_px = torch.stack(
            [
                offset_x.reshape(-1),
                offset_y.reshape(-1),
            ],
            dim=-1,
        )
        # [25, 2]


        self.register_buffer(
            'neighborhood_offsets_px',
            neighborhood_offsets_px,
            persistent=False,
        )


        # ============================================================
        # 2. Lightweight candidate feature adaptor
        #
        # IMPORTANT:
        # 312-D candidate를 25번 직접 MLP에 넣지 않는다.
        #
        # 먼저 feature map:
        #
        #   312 -> 32
        #
        # 로 줄인 후 neighborhood sampling.
        #
        # 실시간성 측면에서 훨씬 유리.
        # ============================================================

        self.candidate_feat_adaptor = nn.Sequential(

            nn.Conv2d(
                local_channels,
                self.candidate_feat_dim,
                kernel_size=1,
            ),

            nn.LeakyReLU(
                negative_slope=0.01,
            ),
        )


        # ============================================================
        # 3. Candidate scoring
        #
        # Candidate descriptor:
        #
        #   feature      = 32
        #   depth        =  1
        #   offset       =  2
        #   validity     =  1
        # ----------------------
        #                  36
        #
        # Tiny MLP:
        #
        #   36 -> 32 -> 1
        # ============================================================

        candidate_descriptor_dim = (
            self.candidate_feat_dim
            + 1
            + 2
            + 1
        )


        self.candidate_score = nn.Sequential(

            nn.Linear(
                candidate_descriptor_dim,
                32,
            ),

            nn.LeakyReLU(
                negative_slope=0.01,
            ),

            nn.Linear(
                32,
                1,
            ),
        )


        # ============================================================
        # 4. Residual Z predictor
        #
        # z_final =
        #
        #     z_anchor + delta_z
        #
        # Normal local correction range:
        #
        #     +/- 15 m
        #
        # 시작 실험에서는 충분히 넓게 둔다.
        # ============================================================

        self.z_residual_range_m = 15.0


        self.residual_predictor = nn.Sequential(

            nn.Linear(
                hidden_dim,
                128,
            ),

            nn.LeakyReLU(
                negative_slope=0.01,
            ),

            nn.Linear(
                128,
                1,
            ),

            nn.Tanh(),
        )


        # ============================================================
        # Stable initialization
        #
        # Candidate scoring:
        #   최초에는 모든 valid candidate를 거의 동일하게 사용
        #
        # Residual:
        #   최초 delta_z = 0
        #
        # 즉 training 첫 순간:
        #
        #   z_pred ~= neighborhood weighted depth
        #
        # 로 시작하게 함.
        # ============================================================

        nn.init.zeros_(
            self.candidate_score[-1].weight
        )

        nn.init.zeros_(
            self.candidate_score[-1].bias
        )


        # residual_predictor[-2] =
        # 마지막 Linear(128, 1)
        nn.init.zeros_(
            self.residual_predictor[-2].weight
        )

        nn.init.zeros_(
            self.residual_predictor[-2].bias
        )
        
        # 3. 모든 특징을 융합하는 레이어
        # ✨ FIX: 로컬 특징(local_channels)이 추가되었으므로 입력 차원 확장
        fusion_in_channels = 128 + local_channels + 64  # Global + Local + Positional
        # fusion_in_channels = 128 + local_channels  # Global + Local : positional 빼고 해보기
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_channels, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(hidden_dim)
        )
        
        # 4. 깊이 예측기
        self.depth_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 1),
            # nn.ReLU()  # 깊이는 항상 0 이상이어야 하므로 최종 출력은 ReLU
            nn.Sigmoid(),
        )
        
        # # 5. LiDAR 포인트 신뢰도 예측기 (기존과 동일)
        # self.confidence_predictor = nn.Sequential(
        #     nn.Linear(1, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1),
        #     nn.Sigmoid()
        # )
        # ✨✨✨ START: 최종 레이어 편향(Bias) 초기화 ✨✨✨
        # self.depth_predictor의 마지막 Linear 레이어(-2 인덱스)를 타겟으로 합니다.
        # 편향(bias)을 작은 양수(예: 0.1)로 초기화합니다.
        # 이렇게 하면 초기 출력이 음수가 될 확률이 크게 줄어듭니다.
        torch.nn.init.constant_(self.depth_predictor[-2].bias.data, 0.1)
        # ✨✨✨ END: 편향 초기화 ✨✨✨

        # --- ✨ 2. 가중치 수동 로드 로직 (모든 레이어 생성 후) ---
        if self.init_cfg and self.init_cfg.get('type') == 'Pretrained':
            checkpoint_path = self.init_cfg.get('checkpoint')
            if checkpoint_path:
                print_log(f'Manually loading checkpoint for ZEstimator from: {checkpoint_path}', logger='current')
                
                # ⭐️ (중요) 'self' (ZEstimator 인스턴스)에 로드합니다.
                load_checkpoint(
                    self, 
                    checkpoint_path, 
                    map_location='cpu', 
                    strict=False,
                    
                    # ⭐️ (중요) 체크포인트의 접두사에 맞게 수정하세요.
                    # 예: 체크포인트 키가 'z_estimator.enc_adaptor...' 라면
                    revise_keys=[('^z_estimator\\.', '')]
                    # 예: 접두사가 없다면 이 'revise_keys' 라인을 삭제
                )
            else:
                print_log('No checkpoint path in init_cfg for ZEstimator.', logger='current', level='WARNING')

    def estimate_lidar_confidence(self, z_depth_real):
        return self.confidence_predictor(z_depth_real.unsqueeze(-1)).squeeze(-1)

    def forward(
        self,
        uv_sbs_normalized,
        uv_orig_pixels,
        depth_map,
        enc_out,
    ):
        """
        Lightweight Robust Neighborhood Z Estimator.

        Args:
            uv_sbs_normalized:
                [NV, Q, 2]

                CorrNet predicted correspondence
                in SBS normalized coordinates.

                Right depth side:
                    x ~= [0.5, 1.0]

            uv_orig_pixels:
                [NV, Q, 2]

                Corr predicted correspondence in
                original image coordinates.

                u: [0, 1600)
                v: [0, 900)

            depth_map:
                [NV, H, W]

                Broken dense LiDAR depth map.

            enc_out:
                [NV, C, H_feat, W_feat]

                CorrNet encoder feature map.

        Returns:
            dict
        """

        # ============================================================
        # 0. Shapes
        # ============================================================

        NV, Q, _ = (
            uv_sbs_normalized.shape
        )

        NV_depth, H, W = (
            depth_map.shape
        )

        C_local = (
            enc_out.shape[1]
        )

        device = (
            uv_sbs_normalized.device
        )

        dtype = (
            uv_orig_pixels.dtype
        )


        if NV_depth != NV:

            raise RuntimeError(
                '[ZEstimator V3] '
                'depth_map / UV batch mismatch: '
                f'UV NV={NV}, '
                f'depth NV={NV_depth}'
            )


        # ============================================================
        # 1. Global feature
        # ============================================================

        enc_reduced = (
            self.enc_adaptor(
                enc_out
            )
            .squeeze(-1)
            .squeeze(-1)
        )
        # [NV, 128]


        global_features = (
            enc_reduced
            .unsqueeze(1)
            .expand(
                NV,
                Q,
                128,
            )
            .reshape(
                NV * Q,
                128,
            )
        )
        # [NV*Q,128]


        # ============================================================
        # 2. Center local feature
        #
        # 기존 V2와 동일:
        # Corr predicted UV의 local feature
        # ============================================================

        center_grid = (
            uv_sbs_normalized
            * 2.0
            - 1.0
        )

        center_grid = (
            center_grid
            .unsqueeze(1)
        )
        # [NV,1,Q,2]


        center_feature_sampled = (
            F.grid_sample(
                enc_out,
                center_grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False,
            )
        )
        # [NV,C,1,Q]


        local_features_flat = (
            center_feature_sampled
            .squeeze(2)
            .permute(
                0,
                2,
                1,
            )
            .reshape(
                NV * Q,
                C_local,
            )
        )
        # [NV*Q,312]


        # ============================================================
        # 3. UV embedding
        # ============================================================

        uv_sbs_flat = (
            uv_sbs_normalized
            .reshape(
                NV * Q,
                2,
            )
        )


        uv_embedded = (
            self.uv_embed(
                uv_sbs_flat
            )
        )
        # [NV*Q,64]


        # ============================================================
        # 4. Build 5x5 neighborhood
        #
        # Center:
        #
        #   Corr predicted (u',v')
        #
        # Radius:
        #
        #   +/- 64 px
        #
        # K = 25
        # ============================================================

        offsets = (
            self.neighborhood_offsets_px
            .to(
                device=device,
                dtype=dtype,
            )
        )

        K = offsets.shape[0]
        # 25


        candidate_uv = (
            uv_orig_pixels
            .unsqueeze(2)
            +
            offsets
            .view(
                1,
                1,
                K,
                2,
            )
        )
        # [NV,Q,K,2]


        candidate_u = (
            candidate_uv[
                ...,
                0
            ]
        )

        candidate_v = (
            candidate_uv[
                ...,
                1
            ]
        )


        # ============================================================
        # 5. Candidate geometric validity
        # ============================================================

        candidate_in_bounds = (
            (candidate_u >= 0.0)
            & (candidate_u < float(W))
            & (candidate_v >= 0.0)
            & (candidate_v < float(H))
        )
        # [NV,Q,K]


        # ============================================================
        # 6. Candidate depth sampling
        #
        # Transformer 없음.
        #
        # 단순 vectorized integer depth lookup.
        # ============================================================

        depth_map_reshaped = (
            depth_map
            .reshape(
                NV,
                H,
                W,
            )
        )


        candidate_u_idx = (
            candidate_u
            .round()
            .long()
            .clamp(
                0,
                W - 1,
            )
        )


        candidate_v_idx = (
            candidate_v
            .round()
            .long()
            .clamp(
                0,
                H - 1,
            )
        )


        candidate_cam_idx = (
            torch.arange(
                NV,
                device=device,
            )
            .view(
                NV,
                1,
                1,
            )
            .expand(
                NV,
                Q,
                K,
            )
        )


        candidate_depth = (
            depth_map_reshaped[
                candidate_cam_idx,
                candidate_v_idx,
                candidate_u_idx,
            ]
        )
        # [NV,Q,K]


        candidate_valid = (
            candidate_in_bounds
            & torch.isfinite(
                candidate_depth
            )
            & (
                candidate_depth
                > 1e-3
            )
        )


        candidate_depth_safe = (
            torch.where(
                candidate_valid,
                candidate_depth,
                torch.zeros_like(
                    candidate_depth
                ),
            )
        )


        # ============================================================
        # 7. Candidate feature map projection
        #
        # BEFORE sampling:
        #
        #   [312,12,80]
        #       ↓ 1x1 Conv
        #   [32,12,80]
        #
        # 계산량 절감 핵심.
        # ============================================================

        candidate_feat_map = (
            self.candidate_feat_adaptor(
                enc_out
            )
        )
        # [NV,32,H_feat,W_feat]


        # ============================================================
        # 8. Original candidate pixel
        #       -> SBS normalized coordinate
        #
        # right half:
        #
        # x =
        #   0.5 + u / (2 * W)
        #
        # y =
        #   v / H
        # ============================================================

        candidate_sbs_x = (
            0.5
            +
            candidate_u
            / (
                2.0
                * float(W)
            )
        )


        candidate_sbs_y = (
            candidate_v
            / float(H)
        )


        # grid_sample [-1,1]

        candidate_grid_x = (
            candidate_sbs_x
            * 2.0
            - 1.0
        )

        candidate_grid_y = (
            candidate_sbs_y
            * 2.0
            - 1.0
        )


        candidate_grid = (
            torch.stack(
                [
                    candidate_grid_x,
                    candidate_grid_y,
                ],
                dim=-1,
            )
        )
        # [NV,Q,K,2]


        # ============================================================
        # 9. Sample candidate features
        # ============================================================

        candidate_features = (
            F.grid_sample(
                candidate_feat_map,
                candidate_grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False,
            )
        )
        # [NV,32,Q,K]


        candidate_features = (
            candidate_features
            .permute(
                0,
                2,
                3,
                1,
            )
            .contiguous()
        )
        # [NV,Q,K,32]


        # ============================================================
        # 10. Candidate depth feature
        #
        # 0~80m -> 0~1
        # ============================================================

        candidate_depth_normalized = (
            candidate_depth_safe
            .clamp(
                min=0.0,
                max=80.0,
            )
            / 80.0
        )

        candidate_depth_feature = (
            candidate_depth_normalized
            .unsqueeze(-1)
        )
        # [NV,Q,K,1]


        # ============================================================
        # 11. Candidate relative offset
        #
        # +/-64 px -> approximately [-1,1]
        # ============================================================

        candidate_offset_feature = (
            offsets
            /
            self.neighborhood_radius_px
        )

        candidate_offset_feature = (
            candidate_offset_feature
            .view(
                1,
                1,
                K,
                2,
            )
            .expand(
                NV,
                Q,
                K,
                2,
            )
        )


        # ============================================================
        # 12. Validity feature
        # ============================================================

        candidate_valid_feature = (
            candidate_valid
            .float()
            .unsqueeze(-1)
        )
        # [NV,Q,K,1]


        # ============================================================
        # 13. Candidate descriptor
        #
        # 32 feature
        #  1 depth
        #  2 offset
        #  1 valid
        # ----------
        # 36
        # ============================================================

        candidate_descriptor = (
            torch.cat(
                [
                    candidate_features,
                    candidate_depth_feature,
                    candidate_offset_feature,
                    candidate_valid_feature,
                ],
                dim=-1,
            )
        )
        # [NV,Q,K,36]


        # ============================================================
        # 14. Tiny shared candidate scoring MLP
        #
        # NO Transformer
        #
        # 36 -> 32 -> 1
        # ============================================================

        candidate_scores = (
            self.candidate_score(
                candidate_descriptor
            )
            .squeeze(-1)
        )
        # [NV,Q,K]


        # Invalid candidate cannot receive weight.

        candidate_scores = (
            candidate_scores
            .masked_fill(
                ~candidate_valid,
                -1e4,
            )
        )


        # ============================================================
        # 15. Soft candidate weighting
        # ============================================================

        candidate_weights = (
            torch.softmax(
                candidate_scores,
                dim=-1,
            )
        )


        # Completely remove invalid candidate weight.

        candidate_weights = (
            candidate_weights
            *
            candidate_valid.float()
        )


        weight_sum = (
            candidate_weights
            .sum(
                dim=-1,
                keepdim=True,
            )
        )


        candidate_weights = (
            candidate_weights
            /
            (
                weight_sum
                + 1e-6
            )
        )


        # ============================================================
        # 16. Robust neighborhood depth anchor
        #
        # z_anchor =
        #
        #   Σ wi * zi
        # ============================================================

        z_anchor = (
            candidate_weights
            *
            candidate_depth_safe
        ).sum(
            dim=-1
        )
        # [NV,Q]


        neighborhood_any_valid = (
            candidate_valid.any(
                dim=-1
            )
        )
        # [NV,Q]


        neighborhood_valid_ratio = (
            candidate_valid
            .float()
            .mean(
                dim=-1
            )
        )
        # [NV,Q]


        neighborhood_weight_max = (
            candidate_weights
            .max(
                dim=-1
            )
            .values
        )
        # [NV,Q]


        # ============================================================
        # 17. Anchor embedding
        #
        # z_anchor is no longer hard output.
        #
        # It becomes geometric evidence for residual refinement.
        # ============================================================

        z_anchor_normalized = (
            z_anchor
            .clamp(
                min=0.0,
                max=80.0,
            )
            / 80.0
        )


        z_anchor_embedding = (
            self.depth_embed(
                z_anchor_normalized
                .reshape(
                    NV * Q,
                    1,
                )
            )
        )
        # [NV*Q,64]


        # No neighborhood depth:
        # remove anchor embedding.

        z_anchor_embedding = (
            z_anchor_embedding
            *
            neighborhood_any_valid
            .reshape(
                NV * Q,
                1,
            )
            .float()
        )


        # ============================================================
        # 18. UV + geometric depth anchor
        #
        # fusion dimension remains 504.
        # ============================================================

        uv_anchor_embedded = (
            uv_embedded
            +
            z_anchor_embedding
        )


        # ============================================================
        # 19. Existing feature fusion
        # ============================================================

        combined = (
            torch.cat(
                [
                    global_features,
                    local_features_flat,
                    uv_anchor_embedded,
                ],
                dim=1,
            )
        )
        # [NV*Q,504]


        fused = (
            self.fusion(
                combined
            )
        )
        # [NV*Q,hidden_dim]


        # ============================================================
        # 20. Direct absolute depth fallback
        #
        # Neighborhood가 전혀 없을 때만 사용.
        #
        # 기존 depth predictor 유지.
        # ============================================================

        direct_raw_prediction = (
            self.depth_predictor(
                fused
            )
            .squeeze(-1)
        )


        z_direct_real = (
            direct_raw_prediction
            * 80.0
        )
        # [NV*Q]


        # ============================================================
        # 21. Residual depth correction
        #
        # delta_z in:
        #
        # [-15m, +15m]
        # ============================================================

        delta_z = (
            self.residual_predictor(
                fused
            )
            .squeeze(-1)
            *
            self.z_residual_range_m
        )
        # [NV*Q]


        z_anchor_flat = (
            z_anchor
            .reshape(
                NV * Q
            )
        )


        z_from_anchor = (
            z_anchor_flat
            +
            delta_z
        )


        # ============================================================
        # 22. Final Z
        #
        # Normal:
        #
        #   z_anchor + delta_z
        #
        # No valid candidate:
        #
        #   fallback direct Z
        # ============================================================

        z_estimated_real = (
            torch.where(
                neighborhood_any_valid
                .reshape(
                    NV * Q
                ),

                z_from_anchor,

                z_direct_real,
            )
        )


        z_estimated_real = (
            z_estimated_real
            .clamp(
                min=0.0,
                max=80.0,
            )
        )


        # ============================================================
        # 23. V1:
        #
        # learned Z == downstream Z
        # ============================================================

        z_final_real = (
            z_estimated_real
        )


        # ============================================================
        # 24. Center raw depth
        #
        # 기존 diagnostic을 유지하기 위해
        # Corr predicted CENTER pixel의 depth도 그대로 반환.
        # ============================================================

        uv_orig_flat = (
            uv_orig_pixels
            .reshape(
                NV * Q,
                2,
            )
        )


        center_u = (
            uv_orig_flat[
                :,
                0
            ]
        )

        center_v = (
            uv_orig_flat[
                :,
                1
            ]
        )


        center_in_bounds = (
            (center_u >= 0.0)
            & (center_u < float(W))
            & (center_v >= 0.0)
            & (center_v < float(H))
        )


        center_u_idx = (
            center_u
            .round()
            .long()
            .clamp(
                0,
                W - 1,
            )
        )


        center_v_idx = (
            center_v
            .round()
            .long()
            .clamp(
                0,
                H - 1,
            )
        )


        center_cam_idx = (
            torch.arange(
                NV,
                device=device,
            )
            .unsqueeze(1)
            .expand(
                NV,
                Q,
            )
            .reshape(-1)
        )


        z_depth_real = (
            depth_map_reshaped[
                center_cam_idx,
                center_v_idx,
                center_u_idx,
            ]
        )


        center_depth_valid = (
            center_in_bounds
            & torch.isfinite(
                z_depth_real
            )
            & (
                z_depth_real
                > 1e-3
            )
        )


        # ============================================================
        # 25. Outputs
        # ============================================================

        return {

            # Final learned Z used by CalibHead
            'depth':
                z_final_real
                .reshape(
                    NV,
                    Q,
                    1,
                ),


            # Existing raw center depth diagnostic
            'z_lidar_real':
                z_depth_real
                .reshape(
                    NV,
                    Q,
                    1,
                ),


            # Existing compatibility field
            'confidence':
                center_depth_valid
                .float()
                .reshape(
                    NV,
                    Q,
                    1,
                ),


            # Final learned prediction
            'z_estimated_real':
                z_estimated_real
                .reshape(
                    NV,
                    Q,
                    1,
                ),


            # ========================================================
            # V3 diagnostics
            # ========================================================

            'z_anchor_real':
                z_anchor
                .reshape(
                    NV,
                    Q,
                    1,
                ),


            'neighbor_valid_ratio':
                neighborhood_valid_ratio
                .reshape(
                    NV,
                    Q,
                    1,
                ),


            'neighbor_weight_max':
                neighborhood_weight_max
                .reshape(
                    NV,
                    Q,
                    1,
                ),


            'neighbor_any_valid':
                neighborhood_any_valid
                .float()
                .reshape(
                    NV,
                    Q,
                    1,
                ),
        }