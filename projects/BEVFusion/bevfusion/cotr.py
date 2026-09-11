import easydict
import torch
import torch.nn as nn
from torch.nn import functional as F
from .COTR.COTR_models.cotr_model_moon_Ver12_0 import build
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule 
from mmengine.runner import load_checkpoint 
from mmengine import print_log             

@MODELS.register_module()
class COTR(BaseModule):
    # __init__ 시그니처를 cfg 파일로부터 파라미터를 받도록 수정합니다.
    def __init__(self,
                 num_kp=200,
                 max_corrs=1000,
                 dim_feedforward=1024,
                 backbone='resnet50',
                 hidden_dim=312,
                 dilation=False,
                 dropout=0.1,
                 nheads=8,
                 layer='layer3',
                 enc_layers=6,
                 dec_layers=6,
                 position_embedding='lin_sine',
                 load_weights_freeze=False, # cfg에서 받을 수 있도록 추가
                 enable_cycle=False,
                #  frozen=False,  # <--- ✨ 1. frozen 인자를 추가합니다 (기본값 False).
                 init_cfg=None): # mmdet3d의 표준 가중치 초기화를 위해 init_cfg를 받습니다.
        # super() 호출 시 init_cfg를 전달해야 Pretrained 가중치 로딩이 동작합니다.
        super(COTR, self).__init__(init_cfg)
        self.num_kp = num_kp

        self.enable_cycle = (
            enable_cycle
        )

        # __init__ 함수 내에서 build 함수에 전달할 설정 딕셔너리를 동적으로 생성합니다.
        cotr_config = {
            "max_corrs": max_corrs,
            "dim_feedforward": dim_feedforward,
            "backbone": backbone,
            "hidden_dim": hidden_dim,
            "dilation": dilation,
            "dropout": dropout,
            "nheads": nheads,
            "layer": layer,
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "position_embedding": position_embedding,
            "load_weights_freeze": load_weights_freeze,
            # init_cfg를 통해 가중치 경로를 전달받습니다.
            "load_weights_path": self.init_cfg.get('checkpoint') if self.init_cfg else None,
            # 아래 파라미터들은 모델 빌드에 직접 필요하지 않을 수 있으나 호환성을 위해 유지
            "out_dir": None,
        }

        ##### CORR network #######
        self.corr = build(easydict.EasyDict(cotr_config))

        # --- ✨ 2. 가중치 수동 로드 로직 추가 ---
        if self.init_cfg and self.init_cfg['type'] == 'Pretrained':
            checkpoint_path = self.init_cfg.get('checkpoint')
            if checkpoint_path:
                # self.corr 모듈에 직접 체크포인트를 로드합니다.
                load_checkpoint(
                    self.corr, 
                    checkpoint_path, 
                    map_location='cpu', 
                    strict=False,
                    # 키 이름의 맨 앞에 있는 'corr.' 문자열을 제거하는 정규식
                    revise_keys=[('^corr\\.', '')] # <-- 이 라인을 추가!
                )
                print_log(f'Manually loading checkpoint for self.corr from: {checkpoint_path}', logger='current')
            else:
                print_log('No checkpoint path in init_cfg for COTR.', logger='current', level='WARNING')
        
    def forward(self, sbs_img, query_input):

        corrs_pred, enc_out = self.corr(
            sbs_img,
            query_input
        )
        # ============================================================
        # Direct correspondence only
        #
        # When cycle is disabled, avoid the SECOND Transformer forward.
        # ============================================================

        if not self.enable_cycle:

            cycle = torch.zeros_like(
                query_input
            )

            mask = torch.zeros(
                query_input.shape[:-1],
                dtype=torch.bool,
                device=query_input.device,
            )

            return (
                corrs_pred,
                cycle,
                mask,
                enc_out,
            )

        img_reverse_input = torch.cat(
            [
                sbs_img[..., 640:],
                sbs_img[..., :640]
            ],
            dim=-1
        )

        query_reverse = corrs_pred.clone()
        query_reverse[..., 0] -= 0.5

        cycle, _ = self.corr(
            img_reverse_input,
            query_reverse
        )

        # Do not modify network output in-place.
        cycle_aligned = cycle.clone()
        cycle_aligned[..., 0] -= 0.5

        mask = (
            torch.norm(
                cycle_aligned - query_input,
                dim=-1
            )
            < 30 / 640
        )

        return (
            corrs_pred,
            cycle_aligned,
            mask,
            enc_out
        )

@MODELS.register_module()
class CorrelationCycleLoss(nn.Module):

    def __init__(
        self,
        corr_weight=1.0,
        cycle_weight=0.1,

        # NEW
        image_width=1600.0,
        image_height=900.0,
        huber_beta_px=32.0,
    ):
        super().__init__()

        self.corr_weight = corr_weight
        self.cycle_weight = cycle_weight

        # =========================================================
        # NEW:
        # Original image geometry
        #
        # SBS x:
        #   x = 0.5 + u / (2 * W)
        #
        # y:
        #   y = v / H
        #
        # Therefore:
        #
        #   dx_norm = du / (2W)
        #   dy_norm = dv / H
        #
        # To make the loss isotropic in PIXEL space:
        #
        #   dx_balanced = dx_norm * (2W/H)
        #               = du / H
        #
        #   dy_balanced = dy_norm
        #               = dv / H
        # =========================================================

        self.image_width = float(
            image_width
        )

        self.image_height = float(
            image_height
        )

        self.huber_beta_px = float(
            huber_beta_px
        )

        self.x_balance_scale = (
            2.0
            * self.image_width
            / self.image_height
        )

        self.huber_beta_norm = (
            self.huber_beta_px
            / self.image_height
        )


    def forward(
        self,
        corr_pred,
        corr_target,
        cycle,
        queries,
        cycle_mask,
        corr_valid_mask=None,
    ):

        # =========================================================
        # 1. Valid correspondence mask
        # =========================================================

        if corr_valid_mask is None:

            corr_valid_mask = torch.ones(
                corr_pred.shape[:-1],
                dtype=torch.bool,
                device=corr_pred.device,
            )


        # =========================================================
        # 2. Direct correspondence loss
        #
        # OLD:
        #
        #   smooth_l1(corr_pred, corr_target)
        #
        # Problem:
        #
        #   x uses /3200
        #   y uses /900
        #
        # so the same pixel error receives a different loss scale.
        #
        # NEW:
        #
        #   pixel-balanced Huber loss
        # =========================================================

        if corr_valid_mask.any():

            corr_error = (
                corr_pred
                - corr_target
            )
            # [B,Q,2]


            # -----------------------------------------------------
            # x error:
            #
            # normalized dx = du / 3200
            #
            # multiply by 3200 / 900
            #
            # -> du / 900
            #
            # y error already:
            #
            # -> dv / 900
            # -----------------------------------------------------

            corr_error_balanced = torch.stack(
                [
                    corr_error[..., 0]
                    * self.x_balance_scale,

                    corr_error[..., 1],
                ],
                dim=-1,
            )


            valid_error = (
                corr_error_balanced[
                    corr_valid_mask
                ]
            )


            corr_match_loss = F.smooth_l1_loss(

                valid_error,

                torch.zeros_like(
                    valid_error
                ),

                beta=
                    self.huber_beta_norm,

                reduction='mean',
            )

        else:

            # Keep valid autograd graph
            corr_match_loss = (
                corr_pred.sum()
                * 0.0
            )


        # =========================================================
        # 3. Cycle consistency loss
        #
        # Current experiment:
        # enable_cycle=False
        #
        # Therefore this normally remains zero.
        #
        # Keep existing behavior unchanged.
        # =========================================================

        valid_cycle_mask = (
            cycle_mask
            & corr_valid_mask
        )


        if valid_cycle_mask.any():

            cycle_loss = F.smooth_l1_loss(
                cycle[
                    valid_cycle_mask
                ],
                queries[
                    valid_cycle_mask
                ],
                reduction='mean',
            )

        else:

            cycle_loss = (
                cycle.sum()
                * 0.0
            )


        # =========================================================
        # 4. Total Corr loss
        # =========================================================

        total_loss = (
            self.corr_weight
            * corr_match_loss
            +
            self.cycle_weight
            * cycle_loss
        )


        return (
            total_loss,
            corr_match_loss.detach(),
            cycle_loss.detach(),
        )

class PointDistanceLoss(nn.Module):
    def __init__(self, distance_weight=1.0):
        super().__init__()
        self.point_distance_weight = distance_weight
    
    def forward(self, points_pred, points_gt):
        return self.point_distance_loss(points_pred, points_gt) * self.point_distance_weight
    
    def chamfer_loss(self, points_a, points_b):
        """
        Chamfer Distance Loss 계산 메서드
        Args:
            points_a: (N, 3) 형태의 텐서 [detection_xyz_normal[...,2:]]
            points_b: (M, 3) 형태의 텐서 [pts_lidar_mis_normalized[mask_valid_mis]]
        """
        # 입력 차원 검증
        if points_a.size(0) == 0 or points_b.size(0) == 0:
            print ("chmfer loss points_a or points_b is empty") 
        assert points_a.dim() == 2 and points_b.dim() == 2, "Input must be 2D tensors"
        points_b = points_b.float()
        # 유효 포인트 필터링
        valid_a = torch.isfinite(points_a).all(dim=1)
        valid_b = torch.isfinite(points_b).all(dim=1)
        points_a = points_a[valid_a]
        points_b = points_b[valid_b]

        # 거리 행렬 계산
        dist_matrix = torch.cdist(points_a, points_b, p=2)
        
        # 양방향 최소 거리 계산
        min_a_to_b = torch.min(dist_matrix, dim=1)[0]
        min_b_to_a = torch.min(dist_matrix, dim=0)[0]
        
        # 평균 손실 계산
        return (min_a_to_b.mean() + min_b_to_a.mean()) / 2.0
    
    def point_distance_loss(self, points_pred, points_gt):
        """
        1:1 대응 포인트 거리 손실 계산
        - points_pred: [N,3], points_gt: [N,3]
        """
        if points_pred.size(0) == 0 or points_gt.size(0) == 0:
            return torch.tensor(0.0, device=points_pred.device)
        
        assert points_pred.size() == points_gt.size(), "포인트 개수 불일치"
        
        # 각 포인트별 L2 거리 계산 → [N]
        error = torch.norm(points_pred - points_gt, p=2, dim=1)
        
        # 값 클램핑 (100 이하로 제한)
        error = error.clamp(max=100.0)
        
        # 평균 손실 계산
        return error.mean()