from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from mmengine.utils import is_list_of
from torch import Tensor
from torch.nn import functional as F
from torchvision.transforms import functional as tvtf
from mmengine.structures import InstanceData
from mmdet.structures.bbox import bbox_overlaps ,bbox2roi
from mmengine.runner import loops

from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList
from .ops import Voxelization
import cv2
from .imageprocessing_unit import (dense_map_from_depth_batch_v2, 
                                   batch_colormap,two_images_side_by_side_gpu,
                                   display_depth_maps,
                                   save_batch_predictions_to_file,
                                   visualize_bev_proposals,
                                   visualize_ours_fusion_result,
                                   )
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import math
from .calib_head import axis_angle_to_matrix,geodesic_distance_loss,quaternion_to_matrix ,identity_matrix_loss

# class CalibrationCorrectionHead(nn.Module):
#     """
#     특징 맵을 입력받아 6-DoF 보정 파라미터를 예측하는 헤드.

#     Args:
#         in_channels (int): 입력 특징 맵의 채널 수.
#         hidden_dim (int): MLP의 중간층 차원.
#         out_dim (int): 출력 차원. 기본값은 6 (rot 3 + trans 3).
#     """
#     def __init__(self, in_channels: int, hidden_dim: int = 256, out_dim: int = 6):
#         super().__init__()
        
#         # 1. 공간 차원(H, W)을 없애고 채널 정보만 남기기 위한 풀링 레이어
#         self.pool = nn.AdaptiveAvgPool2d(1)
        
#         # 2. 풀링된 특징 벡터를 최종 6-DoF 값으로 매핑하는 MLP
#         self.mlp = nn.Sequential(
#             nn.Linear(in_channels, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, out_dim)
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x (torch.Tensor): 입력 특징 맵 (B*N, C, H, W)
        
#         Returns:
#             torch.Tensor: 예측된 6-DoF 파라미터 (B*N, 6)
#         """
#         # (B*N, C, H, W) -> (B*N, C, 1, 1)
#         x = self.pool(x)
        
#         # (B*N, C, 1, 1) -> (B*N, C)
#         x = torch.flatten(x, 1)
        
#         # (B*N, C) -> (B*N, 6)
#         pred_delta_6dof = self.mlp(x)
        
#         return pred_delta_6dof

def chamfer_distance(pred_points, gt_points):
    """
    두 포인트 클라우드 간의 Chamfer Distance를 계산합니다.
    Args:
        pred_points (Tensor): [B, N, 3] 예측 포인트 클라우드.
        gt_points (Tensor): [B, M, 3] Ground Truth 포인트 클라우드.
    Returns:
        Tensor: 배치별 Chamfer Distance 값 [B].
    """
    pred_points = pred_points.float()
    gt_points = gt_points.float()

    # pred -> gt 거리 계산
    diff_pred_gt = pred_points.unsqueeze(2) - gt_points.unsqueeze(1) # [B, N, M, 3]
    dist_pred_gt = torch.sum(diff_pred_gt**2, dim=3) # [B, N, M]
    min_dist_pred_gt, _ = torch.min(dist_pred_gt, dim=2) # [B, N]

    # gt -> pred 거리 계산
    diff_gt_pred = gt_points.unsqueeze(2) - pred_points.unsqueeze(1) # [B, M, N, 3]
    dist_gt_pred = torch.sum(diff_gt_pred**2, dim=3) # [B, M, N]
    min_dist_gt_pred, _ = torch.min(dist_gt_pred, dim=2) # [B, M]

    # 두 거리의 평균 합
    loss = torch.mean(min_dist_pred_gt, dim=1) + torch.mean(min_dist_gt_pred, dim=1)
    return loss / 2.0 # 평균 반환

@MODELS.register_module()
class BEVFusion(Base3DDetector):

    def __init__(
        self,
        enable_selective_freezing,
        class_names: List[str],
        data_preprocessor: OptConfigType = None,
        calibration_mode: str = 'pcc_full',
        pts_voxel_encoder: Optional[dict] = None,
        pts_middle_encoder: Optional[dict] = None,
        fusion_layer: Optional[dict] = None,
        img_backbone: Optional[dict] = None,
        pts_backbone: Optional[dict] = None,
        view_transform: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        pts_neck: Optional[dict] = None,
        bbox_head: Optional[dict] = None,
        img_bbox_head: Optional[dict] = None,
        lgpc_train_stage: str = 'joint',
        corr: Optional[dict] = None,
        corr_loss: Optional[dict] = None,
        z_estimator: Optional[dict] = None,
        calib_head: Optional[dict] = None,
        init_cfg: OptMultiConfig = None,
        seg_head: Optional[dict] = None,
        train_cfg=None,  
        test_cfg=None,  
        **kwargs,
    ) -> None:
        voxelize_cfg = data_preprocessor.pop('voxelize_cfg')
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.voxelize_reduce = voxelize_cfg.pop('voxelize_reduce')
        self.pts_voxel_layer = Voxelization(**voxelize_cfg)

        self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)

        self.img_backbone = MODELS.build(
            img_backbone) if img_backbone is not None else None
        self.img_neck = MODELS.build(
            img_neck) if img_neck is not None else None
        self.view_transform = MODELS.build(
            view_transform) if view_transform is not None else None
        self.pts_middle_encoder = MODELS.build(pts_middle_encoder)

        self.fusion_layer = MODELS.build(
            fusion_layer) if fusion_layer is not None else None

        self.pts_backbone = MODELS.build(pts_backbone)
        self.pts_neck = MODELS.build(pts_neck)

        self.init_weights()

        # modified by sjmoon
        self.calibration_mode = calibration_mode

        valid_calibration_modes = {
            'clean',
            'broken',
            'oracle',
            'pcc_calib_only',
            'pcc_full',
            'lccnet',
            # NEW:
            # Broken geometry + RRRF feature refinement
            # Stage-1 SE(3) correction is NOT applied.
            'pcc_broken_refine',
            'pcc_clean_refine',
            'geo_oracle_gtrot',
        }

        if self.calibration_mode not in valid_calibration_modes:
            raise ValueError(
                f'Unsupported calibration_mode: '
                f'{self.calibration_mode}'
            )
        
        print(
            f'[BEVFusion] calibration_mode = '
            f'{self.calibration_mode}'
        )
        
        self.bbox_head = MODELS.build(bbox_head)
        self.img_bbox_head = MODELS.build(img_bbox_head)
        self.corr = MODELS.build(corr)
        self.corr_loss = (
            MODELS.build(corr_loss)
            if corr_loss is not None
            else None
        )
        self.z_estimator = MODELS.build(z_estimator)
        self.calib_head = MODELS.build(calib_head)

        self._lgpc_base_trainability = {

            'corr': {
                name: p.requires_grad
                for name, p
                in self.corr.named_parameters()
            },

            'z_estimator': {
                name: p.requires_grad
                for name, p
                in self.z_estimator.named_parameters()
            },

            'calib_head': {
                name: p.requires_grad
                for name, p
                in self.calib_head.named_parameters()
            },
        }

        self.is_lgpc_stage1 = (
            getattr(
                self.bbox_head,
                'rrrf_mode',
                None,
            )
            == 'lgpc_only'
        )

        # ============================================================
        # LGPC internal training stage
        #
        # corr  : CorrNet only
        # z     : ZEstimator only
        # calib : CalibHead only
        # z_calib : ZEstimator + CalibHead, Corr frozen
        # joint : Corr + Z + Calib end-to-end
        # ============================================================

        self.lgpc_train_stage = lgpc_train_stage

        valid_lgpc_train_stages = {
            'corr',
            'z',
            'calib',
            'z_calib',
            'joint',
        }

        if self.lgpc_train_stage not in valid_lgpc_train_stages:

            raise ValueError(
                f'Unknown lgpc_train_stage='
                f'{self.lgpc_train_stage}'
            )


        print(
            '[BEVFusion] LGPC train stage = '
            f'{self.lgpc_train_stage}'
        )

        print(
            '[BEVFusion] LGPC Stage-1 standalone = '
            f'{self.is_lgpc_stage1}'
        )
        
        self.class_names = class_names
        self.name_to_idx = {name: i for i, name in enumerate(self.class_names)}

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        feat_dim_original = 312 # 입력 차원은 det_feat의 원래 특징 차원입니다 (12 * 64 = 768).
        hidden_channel = bbox_head['hidden_channel'] # (128) 출력 차원은 TransFusionHead의 hidden_channel과 반드시 일치해야 합니다.
        self.feat_projector = nn.Linear(feat_dim_original, hidden_channel)

        # ============================================================
        # Configure LGPC Stage-1 after ALL modules are constructed.
        # ============================================================

        if self.is_lgpc_stage1:

            if enable_selective_freezing:

                raise RuntimeError(
                    'LGPC Stage-1 must use '
                    'enable_selective_freezing=False. '
                    'Legacy _freeze_modules() freezes CorrNet.'
                )

            self._configure_lgpc_substage()

        # =====================================================================
        # ✨ START: Code added for selective module freezing
        # =====================================================================
        # Set this flag to True to freeze parts of the network during training.
        # The specific modules to be frozen are defined in the _freeze_modules() method.
        self.enable_selective_freezing = enable_selective_freezing
        if (
            self.enable_selective_freezing
            and not self.is_lgpc_stage1
        ):

            self._freeze_modules()
        # =====================================================================
        # ✨ END: Code added for selective module freezing
        # =====================================================================
        corr_total_params = sum(
            p.numel()
            for p in self.corr.parameters()
        )

        corr_trainable_params = sum(
            p.numel()
            for p in self.corr.parameters()
            if p.requires_grad
        )

        print(
            '[PHASE-A] CorrNet trainable params = '
            f'{corr_trainable_params} / '
            f'{corr_total_params}'
        )

        self.vis_step_counter = 0
        self.training_step = 0
        # self.pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.pc_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]

    def _build_calib_z_reliability(
        self,
        z_output,
    ):
        """
        Build inference-available Z reliability descriptor
        for CalibHead.

        NO GT INFORMATION IS USED.

        Returns:
            [M,Q,4]

            0: neighborhood valid ratio
            1: neighborhood max candidate weight
            2: center depth valid indicator
            3: anchor-center disagreement / 80m
        """

        neighbor_valid_ratio = (
            z_output[
                'neighbor_valid_ratio'
            ]
            .float()
        )


        neighbor_weight_max = (
            z_output[
                'neighbor_weight_max'
            ]
            .float()
        )


        center_valid = (
            z_output[
                'confidence'
            ]
            .float()
        )


        z_anchor = torch.nan_to_num(

            z_output[
                'z_anchor_real'
            ],

            nan=0.0,

            posinf=80.0,

            neginf=0.0,
        )


        z_center = torch.nan_to_num(

            z_output[
                'z_lidar_real'
            ],

            nan=0.0,

            posinf=80.0,

            neginf=0.0,
        )


        # ------------------------------------------------------------
        # How much does the center raw depth disagree
        # with the neighborhood anchor?
        #
        # This can be a strong cue that a single Corr pixel
        # has landed on the wrong surface.
        # ------------------------------------------------------------

        anchor_center_gap = (
            torch.abs(
                z_anchor
                - z_center
            )
            .clamp(
                min=0.0,
                max=80.0,
            )
            / 80.0
        )


        # center depth does not exist:
        # disagreement itself is meaningless.
        anchor_center_gap = (
            anchor_center_gap
            * center_valid
        )


        reliability = torch.cat(

            [
                neighbor_valid_ratio,
                neighbor_weight_max,
                center_valid,
                anchor_center_gap,
            ],

            dim=-1,
        )


        reliability = torch.nan_to_num(

            reliability,

            nan=0.0,

            posinf=1.0,

            neginf=0.0,
        )


        return (
            reliability
            .clamp(
                0.0,
                1.0,
            )
        )
    
    def _configure_lgpc_substage(self):

        stage = self.lgpc_train_stage
        # ============================================================
        # 1. Always freeze fixed query generator
        # ============================================================

        self._freeze_stage1_2d_detector()


        # ============================================================
        # 2. Always freeze BEVFusion / RRRF
        # ============================================================

        self._freeze_stage1_unused_modules()


        # ============================================================
        # 3. LGPC sub-stage
        # ============================================================

        train_corr = (
            stage in {
                'corr',
                'joint',
            }
        )

        train_z = (
            stage in {
                'z',
                'z_calib',
                'joint',
            }
        )

        train_calib = (
            stage in {
                'calib',
                'z_calib',
                'joint',
            }
        )


        self._set_lgpc_module_trainable(
            'corr',
            train_corr,
        )

        self._set_lgpc_module_trainable(
            'z_estimator',
            train_z,
        )

        self._set_lgpc_module_trainable(
            'calib_head',
            train_calib,
        )


        print(
            '\n'
            '=========================================\n'
            '[LGPC SUB-STAGE CONFIG]\n'
            '========================================='
        )

        print(
            f'stage       = {stage}'
        )

        print(
            f'CorrNet     = '
            f'{"TRAIN" if train_corr else "FROZEN"}'
        )

        print(
            f'ZEstimator  = '
            f'{"TRAIN" if train_z else "FROZEN"}'
        )

        print(
            f'CalibHead   = '
            f'{"TRAIN" if train_calib else "FROZEN"}'
        )

        print(
            '=========================================\n'
        )
    
    def _set_lgpc_module_trainable(
        self,
        module_name,
        enabled,
    ):

        module = getattr(
            self,
            module_name
        )

        original_mask = (
            self._lgpc_base_trainability[
                module_name
            ]
        )


        for name, p in module.named_parameters():

            if enabled:

                # Restore original intended trainability.
                #
                # Important for CorrNet:
                # ImageNet ResNet remains frozen.
                p.requires_grad = (
                    original_mask[name]
                )

            else:

                p.requires_grad = False


        if enabled:

            module.train()

        else:

            module.eval()
    
    def train(self, mode: bool = True):
        """
        Override nn.Module.train() so that LGPC staged training
        preserves the intended train/eval state of each module.

        Important:
        MMEngine calls model.train() after __init__().
        Without this override, modules frozen with module.eval()
        during __init__ are recursively switched back to train mode.
        """

        # First let PyTorch/MMEngine set the normal global state.
        super().train(mode)

        # ------------------------------------------------------------
        # Evaluation mode:
        # super().train(False) already puts everything in eval mode.
        # Nothing else is needed.
        # ------------------------------------------------------------
        if not mode:
            return self

        # ------------------------------------------------------------
        # Non-LGPC Stage-1:
        # keep normal full-model training behavior.
        # ------------------------------------------------------------
        if not getattr(
            self,
            'is_lgpc_stage1',
            False,
        ):
            return self


        stage = self.lgpc_train_stage


        # ============================================================
        # 1. Fixed pretrained 2D detector
        #
        # Always frozen/eval during LGPC Stage-1.
        # ============================================================

        for module in [
            self.img_backbone,
            self.img_neck,
            self.img_bbox_head,
        ]:

            if module is not None:
                module.eval()


        # ============================================================
        # 2. Unused BEVFusion / RRRF modules
        #
        # Always frozen/eval during LGPC Stage-1.
        # ============================================================

        for module in [
            self.pts_voxel_encoder,
            self.pts_middle_encoder,
            self.pts_backbone,
            self.pts_neck,
            self.view_transform,
            self.fusion_layer,
            self.bbox_head,
            self.feat_projector,
        ]:

            if module is not None:
                module.eval()


        # ============================================================
        # 3. LGPC sub-stage modes
        # ============================================================

        if stage == 'corr':

            # CorrNet is being optimized.
            self.corr.train()

            # Not used / frozen.
            self.z_estimator.eval()
            self.calib_head.eval()


        elif stage == 'z':

            # CorrNet must be deterministic feature provider.
            self.corr.eval()

            # Only ZEstimator is optimized.
            self.z_estimator.train()

            # Not used / frozen.
            self.calib_head.eval()


        elif stage == 'calib':

            # Frozen deterministic feature providers.
            self.corr.eval()
            self.z_estimator.eval()

            # Only CalibHead is optimized.
            self.calib_head.train()
        
        elif stage == 'z_calib':

            # CorrNet is the fixed epoch-10 correspondence provider.
            self.corr.eval()

            # Train both downstream LGPC modules.
            self.z_estimator.train()
            self.calib_head.train()

        elif stage == 'joint':

            # Full LGPC end-to-end fine tuning.
            self.corr.train()
            self.z_estimator.train()
            self.calib_head.train()


        else:

            raise RuntimeError(
                f'Unknown LGPC train stage: {stage}'
            )


        return self
    
    def _freeze_stage1_2d_detector(self):
        """
        Stage-1 LGPC training:

        pretrained 2D detector is used only as a fixed
        object-center query generator.
        """

        modules = {
            'img_backbone': self.img_backbone,
            'img_neck': self.img_neck,
            'img_bbox_head': self.img_bbox_head,
        }

        print(
            '\n'
            '=========================================\n'
            '[LGPC STAGE1] Freeze 2D detector\n'
            '========================================='
        )

        for name, module in modules.items():

            if module is None:
                continue

            for param in module.parameters():
                param.requires_grad = False

            module.eval()

            total = sum(
                p.numel()
                for p in module.parameters()
            )

            trainable = sum(
                p.numel()
                for p in module.parameters()
                if p.requires_grad
            )

            print(
                f'{name}: '
                f'total={total:,}, '
                f'trainable={trainable:,}'
            )

        print(
            '=========================================\n'
        )
    
    def _freeze_stage1_unused_modules(self):
        """
        Modules that are not used by LGPC Stage-1.

        They remain part of the full PCC model,
        but they must not participate in optimizer/DDP gradient work.
        """

        modules = {
            # BEVFusion LiDAR path
            'pts_voxel_encoder':
                self.pts_voxel_encoder,

            'pts_middle_encoder':
                self.pts_middle_encoder,

            'pts_backbone':
                self.pts_backbone,

            'pts_neck':
                self.pts_neck,

            # Camera -> BEV path
            'view_transform':
                self.view_transform,

            'fusion_layer':
                self.fusion_layer,

            # Stage-2 / detection
            'bbox_head':
                self.bbox_head,

            # Camera proposal feature projection
            'feat_projector':
                self.feat_projector,
        }


        print(
            '\n'
            '=========================================\n'
            '[LGPC STAGE1] Freeze unused modules\n'
            '========================================='
        )


        for name, module in modules.items():

            if module is None:
                continue

            for p in module.parameters():

                p.requires_grad = False

            module.eval()


            total = sum(
                p.numel()
                for p in module.parameters()
            )

            print(
                f'{name:<22}'
                f'total={total:>12,d} '
                f'trainable=0'
            )


        print(
            '=========================================\n'
        )
    
    def _freeze_modules(self):
            """
            Selectively freezes parts of the network for targeted training.
            This configuration trains ONLY the image 2D detection pipeline.
            """
            # --- STRATEGY: Freeze everything EXCEPT the Image 2D Detection pipeline. ---
            print("Freezing all modules EXCEPT the Image 2D Detection pipeline.")
            
            # # 동결할 모듈 목록 (2D 탐지 관련 모듈 제외)
            modules_to_freeze = {
                # LiDAR Path
                # 'pts_voxel_layer': self.pts_voxel_layer,
                # 'pts_voxel_encoder': self.pts_voxel_encoder,
                # 'pts_middle_encoder': self.pts_middle_encoder,
                # 'pts_backbone': self.pts_backbone,
                # 'pts_neck': self.pts_neck,
                
                # # 3D Detection Head
                # 'bbox_head': self.bbox_head,
                
                # # Fusion & View Transform
                # 'view_transform': self.view_transform,
                # 'fusion_layer': self.fusion_layer,
                
                # Custom Modules
                'corr': self.corr,
                # 'z_estimator': self.z_estimator,
            }
            # modules_to_freeze = {
            #     # # LiDAR Path
            #     # 'pts_voxel_layer': self.pts_voxel_layer,
            #     # 'pts_voxel_encoder': self.pts_voxel_encoder,
            #     # 'pts_middle_encoder': self.pts_middle_encoder,
            #     # 'pts_backbone': self.pts_backbone,
            #     # 'pts_neck': self.pts_neck,
                
            #     # # 3D Detection Head
            #     # 'bbox_head': self.bbox_head,
                
            #     # # Fusion & View Transform
            #     # 'view_transform': self.view_transform,
            #     # 'fusion_layer': self.fusion_layer,
                
            #     # # Custom Modules
            #     'corr': self.corr,
            #     # 'z_estimator': self.z_estimator,
            # }

            # 선택된 모듈들의 파라미터 업데이트를 중지
            for name, module in modules_to_freeze.items():
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad = False
                    print(f" - ❄️ Module '{name}' has been frozen.")
                else:
                    print(f" - Module '{name}' is None, skipping.")
            
            print("\n - 🔥 The following modules will be trained: 'img_backbone', 'img_neck', 'img_bbox_head'.")
            print("---------------------------------")

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass

    def parse_losses(
        self,
        losses: Dict[str, torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        """
        Parse optimization losses and logging metrics.

        IMPORTANT:
        Empty optimization-loss handling must be done inside loss(),
        i.e. inside the DDP forward graph.

        parse_losses() must NOT directly attach model parameters
        to a newly-created zero loss.
        """

        log_vars = []


        # ============================================================
        # 1. Convert values to scalar tensors
        # ============================================================

        for loss_name, loss_value in losses.items():

            if isinstance(
                loss_value,
                torch.Tensor,
            ):

                log_vars.append(
                    [
                        loss_name,
                        loss_value.mean(),
                    ]
                )


            elif is_list_of(
                loss_value,
                torch.Tensor,
            ):

                if len(loss_value) == 0:

                    raise RuntimeError(
                        '[parse_losses] '
                        f'Empty tensor list: {loss_name}'
                    )


                value = sum(
                    (
                        item.mean()
                        for item in loss_value
                    ),
                    loss_value[0].new_zeros(()),
                )


                log_vars.append(
                    [
                        loss_name,
                        value,
                    ]
                )


            else:

                raise TypeError(
                    f'{loss_name} is not a Tensor '
                    f'or list of Tensors. '
                    f'Got {type(loss_value)}'
                )


        # ============================================================
        # 2. Optimization loss only
        # ============================================================

        loss_terms = [
            value
            for key, value in log_vars
            if 'loss' in key
        ]


        if len(loss_terms) == 0:

            raise RuntimeError(
                '[parse_losses] '
                'loss() returned no optimization loss. '
                f'keys={list(losses.keys())}'
            )


        loss = sum(
            loss_terms[1:],
            loss_terms[0],
        )


        # ============================================================
        # 3. Aggregate loss logging
        # ============================================================

        log_vars.insert(
            0,
            [
                'loss',
                loss,
            ],
        )


        log_vars = OrderedDict(
            log_vars
        )


        # ============================================================
        # 4. Distributed logging reduction
        # ============================================================

        for loss_name, loss_value in log_vars.items():

            log_value = (
                loss_value
                .detach()
                .clone()
            )


            if (
                dist.is_available()
                and dist.is_initialized()
            ):

                dist.all_reduce(
                    log_value
                )

                log_value = (
                    log_value
                    / dist.get_world_size()
                )


            log_vars[
                loss_name
            ] = log_value.item()


        return (
            loss,
            log_vars,
        )
    
    def init_weights(self) -> None:
        if self.img_backbone is not None:
            self.img_backbone.init_weights()

    @property
    def with_bbox_head(self):
        """bool: Whether the detector has a box head."""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_seg_head(self):
        """bool: Whether the detector has a segmentation head.
        """
        return hasattr(self, 'seg_head') and self.seg_head is not None
    
    def extract_img_feat(
        self,
        # ✨ MODIFIED: 입력이 이제 Neck의 출력이므로 이름을 x_neck으로 변경
        x_neck: tuple,
        points: List[torch.Tensor],
        lidar2image: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        camera2lidar: torch.Tensor,
        img_aug_matrix: torch.Tensor,
        lidar_aug_matrix: torch.Tensor,
        img_metas: List[Dict],
    ) -> Tuple[torch.Tensor, tuple]:
        """
        이미지 넥(Neck) 특징과 Calibration 정보를 받아 View Transform을 거쳐 
        BEV 특징과 5D 이미지 특징을 생성합니다.
        """
        # --- ❗️ REMOVED: Neck 중복 계산 로직 삭제 ---
        # x_neck_tuple_4d = self.img_neck(x_backbone)
        # 이제 x_neck이 바로 입력으로 들어옵니다.
        x_neck_tuple_4d = x_neck

        # --- 2. 2D Head용 특징 재구성 (기존과 동일) ---
        B, N = len(img_metas), len(img_metas[0]['cam2img'])
        img_feature_tuple_5d = []
        for feat_4d in x_neck_tuple_4d:
            _BN, C_feat, H_feat, W_feat = feat_4d.size()
            feat_5d = feat_4d.view(B, N, C_feat, H_feat, W_feat)
            img_feature_tuple_5d.append(feat_5d)
        img_feature_tuple_5d = tuple(img_feature_tuple_5d)

        # --- 3. View Transform용 특징 선택 (기존과 동일) ---
        x_for_bev = x_neck_tuple_4d[0]
        BN, C_bev, H_bev, W_bev = x_for_bev.size()
        x_for_bev_5d = x_for_bev.view(B, N, C_bev, H_bev, W_bev)

        # --- 4. View Transform 수행 (기존과 동일) ---
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            bev_feature = self.view_transform(
                x_for_bev_5d,
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                img_metas,
            )
        
        # --- 5. ✨ MODIFIED: 반환값 수정 ---
        # Docstring에 명시된 대로 BEV 특징과 5D 이미지 특징 튜플을 모두 반환합니다.
        return bev_feature
   
    # def extract_img_feat(
    #     self,
    #     x,
    #     points,
    #     lidar2image,
    #     camera_intrinsics,
    #     camera2lidar,
    #     img_aug_matrix,
    #     lidar_aug_matrix,
    #     img_metas,
    # ) -> tuple[torch.Tensor, tuple]: # 반환 타입 힌트 수정
    #     B, N, C, H, W = x.size()
    #     x_reshaped_4d = x.view(B * N, C, H, W).contiguous()

    #     x_backbone = self.img_backbone(x_reshaped_4d)
        
    #     # 1. img_neck의 출력(튜플)을 별도의 변수에 저장합니다.
    #     x_neck_tuple_4d = self.img_neck(x_backbone)

    #     # --- ✨ 2D 헤드용 img_feature를 생성하는 새로운 로직 시작 ---
    #     # 이 로직은 기존 x의 흐름에 영향을 주지 않습니다.
    #     img_feature_tuple_5d = []
    #     for feat_4d in x_neck_tuple_4d:
    #         # 각 4D 피처 (B*N, C, H, W)를 5D (B, N, C, H, W)로 변환
    #         _BN, C_feat, H_feat, W_feat = feat_4d.size()
    #         feat_5d = feat_4d.view(B, N, C_feat, H_feat, W_feat)
    #         img_feature_tuple_5d.append(feat_5d)
    #     img_feature_tuple_5d = tuple(img_feature_tuple_5d)
    #     # --- 새로운 로직 끝 ---

    #     # --- 아래는 view_transform의 입력을 만들기 위한 기존 로직 (그대로 유지) ---
    #     x_for_bev = x_neck_tuple_4d
    #     if not isinstance(x_for_bev, torch.Tensor):
    #         x_for_bev = x_for_bev[0]

    #     BN, C_bev, H_bev, W_bev = x_for_bev.size()
    #     x_for_bev_5d = x_for_bev.view(B, int(BN / B), C_bev, H_bev, W_bev)

    #     with torch.autocast(device_type='cuda', dtype=torch.float32):
    #         bev_feature = self.view_transform(
    #             x_for_bev_5d, # 기존과 동일한 단일 5D 텐서 전달
    #             points,
    #             lidar2image,
    #             camera_intrinsics,
    #             camera2lidar,
    #             img_aug_matrix,
    #             lidar_aug_matrix,
    #             img_metas,
    #         )
    #     # --- 기존 로직 끝 ---
        
    #     # 최종적으로 BEV 피처와, 2D 헤드용으로 새롭게 가공된 이미지 피처 튜플을 반환
    #     return bev_feature, img_feature_tuple_5d
    
    def extract_pts_feat(self, batch_inputs_dict) -> torch.Tensor:
        points = batch_inputs_dict['points']
        with torch.autocast('cuda', enabled=False):
            points = [point.float() for point in points]
            feats, coords, sizes = self.voxelize(points)
            batch_size = coords[-1, 0] + 1
        x = self.pts_middle_encoder(feats, coords, batch_size)
        return x

    @torch.no_grad()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.pts_voxel_layer(res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode='constant', value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(
                    dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()

        return feats, coords, sizes

    # def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
    #             batch_data_samples: List[Det3DDataSample],
    #             **kwargs) -> List[Det3DDataSample]:
    #     """Forward of testing.

    #     Args:
    #         batch_inputs_dict (dict): The model input dict which include
    #             'points' keys.

    #             - points (list[torch.Tensor]): Point cloud of each sample.
    #         batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
    #             Samples. It usually includes information such as
    #             `gt_instance_3d`.

    #     Returns:
    #         list[:obj:`Det3DDataSample`]: Detection results of the
    #         input sample. Each Det3DDataSample usually contain
    #         'pred_instances_3d'. And the ``pred_instances_3d`` usually
    #         contains following keys.

    #         - scores_3d (Tensor): Classification scores, has a shape
    #             (num_instances, )
    #         - labels_3d (Tensor): Labels of bboxes, has a shape
    #             (num_instances, ).
    #         - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
    #             contains a tensor with shape (num_instances, 7).
    #     """
    #     batch_input_metas = [item.metainfo for item in batch_data_samples]
    #     feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

    #     if self.with_bbox_head:
    #         outputs = self.bbox_head.predict(feats, batch_input_metas)

    #     res = self.add_pred_to_datasample(batch_data_samples, outputs)

    #     return res
    
    def extract_feat(
        self,
        batch_inputs_dict: Dict[str, torch.Tensor],
        batch_input_metas: List[Dict],
        # ✨ 보정된 Calibration 딕셔너리를 선택적으로 받음
        corrected_calib: Optional[Dict[str, torch.Tensor]] = None,
        # ✨ 미리 계산된 이미지 백본 특징을 선택적으로 받음
        precomputed_img_feats: Optional[tuple] = None,
        return_lidar_query_feat=False,
    ) -> tuple:
        """
        이미지와 포인트 클라우드 특징을 추출하고 융합합니다.

        Args:
            batch_inputs_dict (Dict): 'imgs', 'points' 등을 포함하는 입력 데이터 딕셔너리.
            batch_input_metas (List[Dict]): 데이터 샘플의 메타 정보 리스트.
            corrected_calib (Optional[Dict]): 온라인으로 보정된 Calibration 파라미터.
                제공되면 이 값을 우선적으로 사용하여 융합을 수행합니다.
            precomputed_img_feats (Optional[tuple]): 미리 계산된 이미지 백본 특징.
                제공되면 이미지 백본 계산을 건너뛰어 효율성을 높입니다.

        Returns:
            tuple: 최종 3D 특징, 원본 이미지 특징, 그리고 융합에 사용된 
                lidar2image, camera_intrinsics, camera2lidar 행렬들을 반환합니다.
        """
        imgs = batch_inputs_dict.get('imgs')
        points = batch_inputs_dict.get('points')

        # --- 1. 융합에 사용할 Calibration 파라미터 결정 ---
        if corrected_calib is not None:
            # 'corrected_calib'가 제공되면 (학습 시), 보정된 값을 사용합니다.
            lidar2image = corrected_calib['lidar2img']
            camera_intrinsics = corrected_calib.get('cam2img')
            camera2lidar = corrected_calib['cam2lidar']
        else:
            # 제공되지 않으면 (추론 시), 기존 방식대로 metas에서 값을 로드합니다.
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            for meta in batch_input_metas:
                lidar2image.append(meta['lidar2img'])
                camera_intrinsics.append(meta['cam2img'])
                camera2lidar.append(meta['cam2lidar'])
            
            lidar2image = imgs.new_tensor(np.asarray(lidar2image))
            camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
            camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))

        # Augmentation 행렬은 Calibration과 별개로 항상 metas에서 로드합니다.
        img_aug_matrix, lidar_aug_matrix = [], []
        for meta in batch_input_metas:
            img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
            lidar_aug_matrix.append(meta.get('lidar_aug_matrix', np.eye(4)))
        img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
        lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))

        # --- 2. 각 센서의 특징 추출 ---
        # 이미지 특징 추출: 미리 계산된 값이 있으면 사용하고, 없으면 새로 계산합니다.
        if precomputed_img_feats is None:
            # loss 함수에서 미리 계산하지 않은 경우 (예: 독립적인 추론)
            img_feats_from_backbone = self.img_backbone(batch_inputs_dict)
        else:
            # loss 함수에서 전달받은 값을 사용 (중복 계산 방지)
            img_feats_from_backbone = precomputed_img_feats

        # ✨ '올바른' Calibration과 미리 계산된 이미지 특징을 extract_img_feat에 전달
        img_bev_feature = self.extract_img_feat(
            img_feats_from_backbone,  # 원본 이미지 대신 백본 특징 전달
            deepcopy(points),
            lidar2image, 
            camera_intrinsics,
            camera2lidar, 
            img_aug_matrix,
            lidar_aug_matrix,
            batch_input_metas
        )
        
        # 포인트 클라우드 특징 추출 (카메라와 무관)
        pts_feature = self.extract_pts_feat(batch_inputs_dict)

        # ============================================================
        # NEW: LiDAR-only feature for query feasibility test
        #
        # IMPORTANT:
        # - no camera feature
        # - no ConvFuser
        # - reuse existing SECOND + SECONDFPN
        # - test/inference feasibility only
        # ============================================================

        lidar_query_feat = None

        if return_lidar_query_feat:

            lidar_query_feat = self.pts_backbone(
                pts_feature
            )

            lidar_query_feat = self.pts_neck(
                lidar_query_feat
            )
        
        # --- 3. 특징 융합 및 3D 후처리 ---
        features = [img_bev_feature, pts_feature]
        
        if self.fusion_layer is not None:
            x = self.fusion_layer(features)
        else:
            # 기본 융합: BEV 특징을 기본으로 사용
            x = features[0] 

        x = self.pts_backbone(x)
        x = self.pts_neck(x)

        # --- 4. 결과 반환 ---

        if return_lidar_query_feat:
            return x, lidar_query_feat
        
        return x
    
    def extract_sbs_img(
        self,
        batch_inputs_dict,
        batch_input_metas,
        visualize=False,
        depth_mode='broken',
        **kwargs,
    ):
        imgs = batch_inputs_dict.get('img_original', None)
        points = batch_inputs_dict.get('perturbed_points', None)
        
        if imgs is not None:
            imgs = imgs.contiguous()
            lidar_depth_mis, lidar_depth_gt = [], []
            for i, meta in enumerate(batch_input_metas):
                lidar_depth_mis.append(meta['lidar_depth_mis'])
                lidar_depth_gt.append(meta['lidar_depth_gt'])

            lidar_depth_mis = imgs.new_tensor(np.asarray(lidar_depth_mis))
            lidar_depth_gt = imgs.new_tensor(np.asarray(lidar_depth_gt))

            # dense_depth_map_mis = dense_map_from_depth_batch_v2(lidar_depth_mis,grid=3,iterations=3)
            # dense_depth_img_mis = dense_depth_map_mis.to(dtype=torch.uint8)
            # dense_depth_img_color_mis = batch_colormap(dense_depth_img_mis)
            # dense_depth_map = dense_map_from_depth_batch_v2(lidar_depth_gt,grid=3,iterations=3)
            # # dense_depth_img = dense_depth_map.to(dtype=torch.uint8)
            # # dense_depth_img_color = batch_colormap(dense_depth_img)

            # ============================================================
            # Build both BROKEN and CLEAN depth maps
            # ============================================================
            dense_depth_map_mis = dense_map_from_depth_batch_v2(
                lidar_depth_mis,
                grid=3,
                iterations=3
            )

            dense_depth_map_gt = dense_map_from_depth_batch_v2(
                lidar_depth_gt,
                grid=3,
                iterations=3
            )
            # ============================================================
            # Select depth actually given to CorrNet / Z-estimator
            # ============================================================

            if depth_mode == 'clean':

                dense_depth_map_selected = (
                    dense_depth_map_gt
                )

            elif depth_mode == 'broken':

                dense_depth_map_selected = (
                    dense_depth_map_mis
                )

            else:

                raise ValueError(
                    f'Unknown depth_mode: {depth_mode}'
                )


            dense_depth_img_selected = (
                dense_depth_map_selected.to(
                    dtype=torch.uint8
                )
            )

            dense_depth_img_color_selected = (
                batch_colormap(
                    dense_depth_img_selected
                )
            )



            # 픽셀 값을 0.0 ~ 1.0 범위로 정규화하여 imshow가 올바르게 표시하도록 함
            img_min, img_max = imgs.min(), imgs.max()
            imgs = (imgs - img_min) / (img_max - img_min + 1e-8) # 0으로 나누는 것 방지

            N, V, C, H, W = imgs.shape
            imgs_reshaped = imgs.view(N * V, C, H, W)
            depth_reshaped_mis = dense_depth_img_color_selected.view(N * V, C, H, W)

            img_resized = F.interpolate(imgs_reshaped, size=[192, 640], mode="bilinear")
            lidar_depth_mis_resized = F.interpolate(depth_reshaped_mis, size=[192, 640], mode="bilinear")

            sbs_img = two_images_side_by_side_gpu(img_resized, lidar_depth_mis_resized)
            sbs_img = sbs_img.permute(0,3,1,2)
            sbs_img = tvtf.normalize(sbs_img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            sbs_img = sbs_img.view(N, V, C, 192, 640*2)

            # ############## input display ##########################
            if visualize and sbs_img is not None:
                display_depth_maps(imgs,dense_depth_img_color_selected,sbs_img)
                print("input dispaly end")
        
        return sbs_img, points ,dense_depth_map_selected,dense_depth_map_gt # CorrNet/Z가 실제 사용할 depth # GT clean depth는 diagnostic용
    
    def box_iou(self, bboxes1, bboxes2):
        """
        두 바운딩 박스 그룹 간의 IoU(Intersection over Union)를 계산합니다.
        
        Args:
            bboxes1 (torch.Tensor): [N, 6] 형태의 텐서 (x1, y1, x2, y2, score, label)
            bboxes2 (torch.Tensor): [M, 6] 형태의 텐서 (x1, y1, x2, y2, score, label)
            
        Returns:
            torch.Tensor: [N, M] 형태의 IoU 텐서
        """
        # IoU 계산에는 좌표값만 필요하므로, score와 label을 제외한 앞의 4개 값만 사용합니다.
        bboxes1_coords = bboxes1[:, :4]
        bboxes2_coords = bboxes2[:, :4]
        
        return bbox_overlaps(bboxes1_coords, bboxes2_coords)

    # def _prepare_2d_head_inputs(
    #     self,
    #     img_feats: tuple,
    #     batch_data_samples: List[Det3DDataSample]
    # ) -> Tuple[tuple, List[Det3DDataSample]]:
    #     """
    #     Multi-view 이미지 피처와 DataSample을 2D 탐지 헤드에 맞게 변환합니다.
    #     (입력 img_feats가 4D 텐서일 경우를 처리하도록 수정됨)
    #     """
    #     # --- ✨ MODIFIED: 4D 텐서를 그대로 사용 ---
    #     # 입력 img_feats는 이미 (B*N, C, H, W) 형태이므로, 
    #     # 추가적인 reshape 없이 그대로 사용합니다.
    #     reshaped_img_feats = img_feats
        
    #     # --- (이후 DataSample 처리 로직은 기존과 동일) ---
    #     camera_types = [
    #         'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
    #         'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    #     ]
    #     reshaped_data_samples = []
    #     # img_feats[0]이 None일 가능성을 대비하여 device를 안전하게 가져옴
    #     device = img_feats[0].device if img_feats and img_feats[0] is not None else 'cpu' 

    #     for sample in batch_data_samples:
    #         # metainfo 키 존재 여부 확인
    #         if 'ann_info_aug_2d_per_cam' not in sample.metainfo:
    #             continue
    #         multi_cam_2d_anns = sample.metainfo['ann_info_aug_2d_per_cam']

    #         for cam_name in camera_types:
    #             if cam_name not in multi_cam_2d_anns:
    #                 continue
                    
    #             new_sample = Det3DDataSample()
    #             new_sample.set_metainfo(sample.metainfo)
    #             if 'gt_instances_3d' in sample:
    #                 new_sample.gt_instances_3d = sample.gt_instances_3d

    #             cam_gt = multi_cam_2d_anns[cam_name]
    #             gt_instances_2d = InstanceData()
                
    #             # Bbox 데이터 처리
    #             gt_bboxes = cam_gt.get('gt_bboxes', []) # 키가 없을 경우 빈 리스트 반환
    #             bboxes_tensor = torch.as_tensor(
    #                 gt_bboxes, dtype=torch.float32, device=device)
    #             gt_instances_2d.bboxes = bboxes_tensor.reshape(-1, 4)
                
    #             # 라벨 데이터 처리
    #             string_labels = cam_gt.get('gt_labels', []) # 키가 없을 경우 빈 리스트 반환
    #             numeric_labels = [self.name_to_idx.get(name, -1) for name in string_labels]
    #             labels_tensor = torch.as_tensor(
    #                 numeric_labels, dtype=torch.long, device=device)
    #             gt_instances_2d.labels = labels_tensor.reshape(-1)
                
    #             new_sample.gt_instances = gt_instances_2d
    #             reshaped_data_samples.append(new_sample)
        
    #     return reshaped_img_feats, reshaped_data_samples
    
    def _prepare_2d_head_inputs(
        self,
        img_feats: tuple,
        batch_data_samples: List[Det3DDataSample]
    ) -> Tuple[tuple, List[Det3DDataSample]]:
        """
        Multi-view 이미지 피처와 DataSample을 2D 탐지 헤드에 맞게 변환합니다.
        학습 모드와 추론 모드를 구분하여 처리합니다.
        """
        # img_feats는 이미 (B*N, C, H, W) 형태이므로 그대로 사용합니다.
        reshaped_img_feats = img_feats
        reshaped_data_samples = []

        if self.training:
            # --- 학습(Training) 경로 ---
            # 기존과 동일하게 GT 어노테이션을 처리하여 loss 계산에 사용합니다.
            camera_types = [
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ]
            device = img_feats[0].device if img_feats and img_feats[0] is not None else 'cpu'

            for sample in batch_data_samples:
                if 'ann_info_aug_2d_per_cam' not in sample.metainfo:
                    continue
                multi_cam_2d_anns = sample.metainfo['ann_info_aug_2d_per_cam']

                for cam_name in camera_types:
                    if cam_name not in multi_cam_2d_anns:
                        continue
                    
                    # metainfo를 생성자에 전달하여 새 샘플 생성
                    new_sample = type(sample)(metainfo=sample.metainfo)
                    if 'gt_instances_3d' in sample:
                        new_sample.gt_instances_3d = sample.gt_instances_3d

                    cam_gt = multi_cam_2d_anns[cam_name]
                    gt_instances_2d = InstanceData()
                    
                    gt_bboxes = cam_gt.get('gt_bboxes', [])
                    bboxes_tensor = torch.as_tensor(
                        gt_bboxes, dtype=torch.float32, device=device)
                    gt_instances_2d.bboxes = bboxes_tensor.reshape(-1, 4)
                    
                    string_labels = cam_gt.get('gt_labels', [])
                    numeric_labels = [self.name_to_idx.get(name, -1) for name in string_labels]
                    labels_tensor = torch.as_tensor(
                        numeric_labels, dtype=torch.long, device=device)
                    gt_instances_2d.labels = labels_tensor.reshape(-1)
                    
                    new_sample.gt_instances = gt_instances_2d
                    reshaped_data_samples.append(new_sample)
        else:
            # --- 추론(Inference) 경로 ---
            # predict 함수가 메타정보를 필요로 하므로, GT 없이 메타정보만 담은
            # 더미 DataSample 리스트를 생성합니다.
            for sample in batch_data_samples:
                # metainfo에서 카메라 개수를 가져옵니다. (e.g., lidar2img shape)
                # 하드코딩보다 안정적인 방법입니다.
                num_cameras = sample.metainfo['lidar2img'].shape[0]
                for _ in range(num_cameras):
                    dummy_sample = type(sample)(metainfo=sample.metainfo)
                    reshaped_data_samples.append(dummy_sample)

        return reshaped_img_feats, reshaped_data_samples

    def process_2d_detections(self, det_results_list, device):
        """
        Processes a list of InstanceData objects from the modern .predict() API.

        Args:
            det_results_list (List[InstanceData]): List of detection results
                from `self.img_bbox_head.predict()`. Length is N*V.
            device (torch.device): The target device.

        Returns:
            list[torch.Tensor]: A list of detection tensors. Each tensor has
                a shape of [num_boxes, 6] (x1, y1, x2, y2, score, label).
        """
        detections = []
        for instance_data in det_results_list:
            # InstanceData에서 bbox, score, label을 추출합니다.
            bboxes = instance_data.bboxes
            scores = instance_data.scores.unsqueeze(1)
            labels = instance_data.labels.unsqueeze(1).to(bboxes.dtype)
            
            # (x1, y1, x2, y2, score, label) 형태의 [num_boxes, 6] 텐서로 결합합니다.
            detection = torch.cat([bboxes, scores, labels], dim=1)
            detections.append(detection)
        return detections
    
    def process_2d_gt(self, gt_bboxes, gt_labels, device):
        """
        :param gt_bboxes:
            gt_bboxes: list[boxes] of size BATCH_SIZE
            boxes: [num_boxes, 4->(x1, y1, x2, y2)]
        :param gt_labels:
        :return:
        """
        return [torch.cat(
            [bboxes.to(device), torch.ones([len(labels), 1], dtype=bboxes.dtype, device=device),
             labels.unsqueeze(-1).to(bboxes.dtype)], dim=-1).to(device)
                for bboxes, labels in zip(gt_bboxes, gt_labels)]
    
    def complement_2d_gt(self, detections, gts, thr=0.35):
        # detections: [n, 6], gts: [m, 6]
        if len(gts) == 0:
            return detections
        if len(detections) == 0:
            return gts
        iou = self.box_iou(gts, detections)
        max_iou = iou.max(-1)[0]
        complement_ids = max_iou < thr
        min_bbox_size = self.train_cfg['detection_proposal'].get('min_bbox_size', 0)
        wh = gts[:, 2:4] - gts[:, 0:2]
        valid_ids = (wh >= min_bbox_size).all(dim=1)
        complement_gts = gts[complement_ids & valid_ids]
        return torch.cat([detections, complement_gts], dim=0)
    
    def display_2d_results(self,
                           images,
                           predictions,
                           ground_truths,
                           score_thr=0.3,
                           save_dir='vis_2d_results'):
        """
        2D 탐지 결과와 Ground Truth를 이미지 위에 그려서 저장합니다.

        Args:
            images (torch.Tensor): (N*V, C, H, W) 형태의 원본 이미지 텐서.
            predictions (list[torch.Tensor]): (N*V) 길이의 리스트. 각 텐서는
                [num_boxes, 6] (x1, y1, x2, y2, score, label) 형태.
            ground_truths (list[Det3DDataSample]): (N*V) 길이의 데이터 샘플 리스트.
            score_thr (float): 시각화할 최소 신뢰도 점수.
            save_dir (str): 이미지를 저장할 디렉토리.
        """
        # 저장할 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)
        
        # 배치 내의 각 이미지를 순회 (총 N*V개)
        for i, (img_tensor, preds, gt_sample) in enumerate(zip(images, predictions, ground_truths)):
            # 1. 텐서를 시각화 가능한 NumPy 배열로 변환
            # (C, H, W) -> (H, W, C)
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            
            # 2. Figure 생성
            fig, ax = plt.subplots(1, figsize=(16, 9))
            ax.imshow(img_np)
            ax.axis('off')

            # 3. 예측(Prediction) 바운딩 박스 그리기 (빨간색)
            preds_np = preds.cpu().numpy()
            # 신뢰도 점수가 임계값 이상인 예측만 선택
            high_conf_preds = preds_np[preds_np[:, 4] > score_thr]
            for box in high_conf_preds:
                x1, y1, x2, y2, score, label_idx = box
                w, h = x2 - x1, y2 - y1
                rect = patches.Rectangle(
                    (x1, y1), w, h, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(
                    x1, y1 - 10,
                    f'{self.class_names[int(label_idx)]} {score:.2f}',
                    bbox=dict(facecolor='r', alpha=0.5),
                    fontsize=10, color='white')

            # # 4. 정답(Ground Truth) 바운딩 박스 그리기 (녹색)
            # if 'bboxes' in gt_sample.gt_instances:
            #     gt_bboxes = gt_sample.gt_instances.bboxes.cpu().numpy()
            #     gt_labels = gt_sample.gt_instances.labels.cpu().numpy()
            #     for box, label_idx in zip(gt_bboxes, gt_labels):
            #         x1, y1, x2, y2 = box
            #         w, h = x2 - x1, y2 - y1
            #         rect = patches.Rectangle(
            #             (x1, y1), w, h, linewidth=2, edgecolor='g', facecolor='none')
            #         ax.add_patch(rect)
            #         ax.text(
            #             x1, y1 + h + 20,
            #             f'{self.class_names[int(label_idx)]}',
            #             bbox=dict(facecolor='g', alpha=0.5),
            #             fontsize=10, color='white')
            
            # 5. 이미지 저장 및 종료
            plt.savefig(f'{save_dir}/result_sample_{i}.png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            
            # (디버깅 속도를 위해 우선 첫 5개 이미지만 시각화)
            if i >= 4:
                break
        
        print(f"Visualization saved to '{save_dir}' directory.")

    def _generate_and_process_2d_dets(self,
                                      reshaped_img_feats: tuple,
                                      reshaped_data_samples: List[Det3DDataSample],
                                      batch_inputs_dict: Dict,
                                      visualize: bool = False
                                     ) -> List[torch.Tensor]:
        """
        2D 헤드를 사용해 탐지 결과를 생성하고, GT로 보강 및 시각화합니다.

        Args:
            reshaped_img_feats (tuple): 2D 헤드 입력용으로 변환된 피처맵 튜플.
            reshaped_data_samples (List[Det3DDataSample]):
                각 뷰에 맞는 2D GT를 포함하는 데이터 샘플 리스트.
            batch_inputs_dict (Dict): 원본 'img_original'을 포함하는 입력 딕셔너리.
            visualize (bool): 시각화 실행 여부.

        Returns:
            List[torch.Tensor]: 최종적으로 처리된 2D 탐지 결과 리스트.
        """
        detections_2d = None
        
        # 1. 2D 탐지 결과(detection) 생성 (추론 모드)
        if self.with_bbox_head:
            with torch.no_grad():
                det_results_list = self.img_bbox_head.predict(
                    reshaped_img_feats, reshaped_data_samples)
            
            # 2. 생성된 2D 탐지 결과를 후처리
            device = reshaped_img_feats[0].device
            detections_2d = self.process_2d_detections(det_results_list, device)

            # 3. 설정에 따라 Ground Truth로 2D 탐지 결과를 보강
            # if self.train_cfg.get('complement_2d_gt', -1) > 0:
            # self.training 조건을 추가하여 학습 모드일 때만 이 블록이 실행되도록 합니다.
            if self.training and not self.is_lgpc_stage1 and self.train_cfg.get('complement_2d_gt', -1) > 0:
                gt_bboxes_list = [sample.gt_instances.bboxes for sample in reshaped_data_samples]
                gt_labels_list = [sample.gt_instances.labels for sample in reshaped_data_samples]
                
                detections_gt = self.process_2d_gt(gt_bboxes_list, gt_labels_list, device)
                
                complemented_detections = []
                for det, det_gt in zip(detections_2d, detections_gt):
                    complemented_det = self.complement_2d_gt(
                        det,
                        det_gt,
                        thr=self.train_cfg.get('complement_2d_gt')
                    )
                    complemented_detections.append(complemented_det)
                detections_2d = complemented_detections

        # 4. 시각화 옵션이 켜져 있으면 결과 그리기
        if visualize and detections_2d is not None:
            # orig_images = batch_inputs_dict['img_original']
            aug_images = batch_inputs_dict['imgs']
            # N, V, C, H, W = orig_images.shape
            N, V, C, H, W = aug_images.shape
            # orig_images_reshaped = orig_images.view(N * V, C, H, W)
            aug_images_reshaped = aug_images.view(N * V, C, H, W)
            
            self.display_2d_results(
                images=aug_images_reshaped,
                predictions=detections_2d,
                ground_truths=reshaped_data_samples
            )
            
        return detections_2d
    
    def _generate_rois_from_detections(self, detections_2d: List[torch.Tensor]) -> tuple:
        """
        탐지 결과 리스트를 RoI 텐서와 proposal 리스트로 변환합니다.
        - RoI 텐서에 object_index를 추가합니다.
        - 빈 탐지 예외 처리를 포함합니다.
        """
        proposal_list = detections_2d
        
        # 빈 탐지 예외 처리
        if sum([len(p) for p in proposal_list]) == 0:
            dummy_det = torch.tensor(
                [[0.0, 0.0, 100.0, 100.0, 1.0, 0.0]],
                dtype=proposal_list[0].dtype,
                device=proposal_list[0].device
            )
            proposal_list[0] = dummy_det

        # --- ✨ Object Index 추가 로직 시작 ✨ ---
        
        proposals_with_obj_idx = []
        for p in proposal_list:
            # 각 텐서(이미지)의 박스 개수만큼 [0, 1, 2, ...] 인덱스 생성
            # shape: [num_boxes, 1]
            obj_indices = torch.arange(
                len(p), dtype=p.dtype, device=p.device).unsqueeze(1)
            
            # 기존 proposal 텐서(num_boxes, 6) 앞에 인덱스 텐서(num_boxes, 1)를 합침
            # 결과 shape: [num_boxes, 7] -> (obj_idx, x1, y1, x2, y2, score, label)
            proposals_with_obj_idx.append(torch.cat([obj_indices, p], dim=1))
        
        # --- Object Index 추가 로직 끝 ---

        # object_index가 추가된 새로운 리스트를 bbox2roi에 전달
        rois = bbox2roi(proposals_with_obj_idx)

        # rois와 (더미가 추가될 수 있는) 원본 형식의 proposal_list를 반환
        return rois, proposal_list
    
    def get_center_points(self, rois_with_indices):
        """
        Args:
            rois_with_indices: Tensor of shape [num_obj, 6] (cam_id, obj_id, x_min, y_min, x_max, y_max)
        Returns:
            center_points: Tensor of shape [num_obj, 4] (cam_id, obj_id, center_x, center_y)
        """
        cam_ids = rois_with_indices[:, 0]
        obj_ids = rois_with_indices[:, 1]
        x_min = rois_with_indices[:, 2]
        y_min = rois_with_indices[:, 3]
        x_max = rois_with_indices[:, 4]
        y_max = rois_with_indices[:, 5]
        score = rois_with_indices[:, 6]
        cls_lable = rois_with_indices[:, 7]

        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0

        center_points = torch.stack([cam_ids, obj_ids, center_x, center_y], dim=1)
        return center_points
    
    
    def batch_rois_center_by_cam_id(self, rois_center, batch_size=100):
        """
        rois_center 텐서에서 카메라 ID를 읽어, 항상 6개의 카메라에 대한
        고정된 크기의 배치(batch) 텐서를 생성합니다.
        존재하지 않는 카메라 ID의 슬롯은 0으로 채워집니다.
        """
        device = rois_center.device
        
        # ✨ 1. 카메라 수를 6으로 고정합니다.
        NUM_CAMS = 6 
        
        # ✨ 2. 출력 텐서를 고정된 [6, batch_size, 4] 크기로 생성합니다.
        batched_centers = torch.zeros((NUM_CAMS, batch_size, 4), device=device)

        # ============================================================
        # NEW:
        # Which query slots correspond to REAL detections?
        #
        # CorrNet still receives exactly batch_size=200 queries.
        #
        # True:
        #     original detection
        #
        # False:
        #     repeated filler used only to satisfy fixed-Q CorrNet
        # ============================================================

        query_unique_mask = torch.zeros(
            (
                NUM_CAMS,
                batch_size,
            ),
            dtype=torch.bool,
            device=device,
        )

        # 입력 텐서가 비어있는 경우, 위에서 생성한 제로 텐서를 그대로 반환
        if rois_center.shape[0] == 0:
            return (
                batched_centers,
                query_unique_mask,
            )

        # 1. rois_center의 0열에서 모든 카메라 인덱스를 추출합니다.
        cam_indices_tensor = rois_center[:, 0]
        
        # 2. 존재하는 고유한 카메라 ID 목록을 찾습니다.
        unique_cam_ids = torch.unique(cam_indices_tensor).long().cpu().tolist()
        
        # ✨ 3. 입력 데이터에 기반해 크기를 정하던 로직은 삭제되었습니다.
        
        # 4. 실제 존재하는 카메라 ID들을 순회하며 batched_centers의 해당 위치를 채웁니다.
        for cam_id in unique_cam_ids:
            # cam_id가 6 이상인 예외적인 데이터가 들어올 경우를 대비
            if cam_id >= NUM_CAMS:
                continue
                
            cam_mask = (rois_center[:, 0] == cam_id)
            cam_centers = rois_center[cam_mask]
            n = cam_centers.size(0)
            
            if n == 0:
                continue

            # ============================================================
            # Number of actual detector queries BEFORE repetition.
            # ============================================================

            num_real = min(
                n,
                batch_size,
            )

            query_unique_mask[
                cam_id,
                :num_real,
            ] = True
                
            # --- (내부 샘플링 로직은 기존과 동일) ---
            obj_ids = cam_centers[:, 1].cpu().numpy()
            unique_obj_ids = np.unique(obj_ids)
            num_unique_objs = len(unique_obj_ids)
            
            if num_unique_objs <= batch_size:
                if n < batch_size:
                    repeat_factor = (batch_size + n - 1) // n
                    cam_centers = cam_centers.repeat(repeat_factor, 1)[:batch_size]
            else:
                selected_indices = []
                for obj_id in unique_obj_ids:
                    obj_indices = np.where(obj_ids == obj_id)[0]
                    selected_idx = np.random.choice(obj_indices)
                    selected_indices.append(selected_idx)
                    
                if len(selected_indices) < batch_size:
                    remaining = batch_size - len(selected_indices)
                    all_indices = np.arange(n)
                    pool = np.setdiff1d(all_indices, selected_indices)
                    replace = len(pool) < remaining
                    extra_indices = np.random.choice(
                        pool,
                        size=remaining,
                        replace=replace
                    )
                    selected_indices.extend(extra_indices)
                    
                selected_indices = torch.tensor(selected_indices, device=device, dtype=torch.long)
                cam_centers = cam_centers[selected_indices]
            
            batched_centers[cam_id] = cam_centers[:batch_size]

        return (
            batched_centers,
            query_unique_mask,
        )
    
    def remove_duplicate_objs(self,corrs_pred_with_obj):
        """
        객체 ID 기준 중복 제거 및 결과 포맷 변환
        - PyTorch 버전 호환성 해결
        - 객체 ID 연속성 체크 추가
        - 텐서 크기 불일치 해결
        
        Args:
            corrs_pred_with_obj: [num_cams, batch_size, 3] 텐서 
                (obj_id, center_pred_x, center_pred_y)
                
        Returns:
            [number_of_unique_obj, 4] 텐서 
            (cam_id, obj_id, center_pred_x, center_pred_y)
        """
        num_cams, batch_size, _ = corrs_pred_with_obj.shape
        
        # 1. 카메라 ID 텐서 생성
        cam_ids = torch.arange(num_cams, device=corrs_pred_with_obj.device)
        cam_ids = cam_ids.view(-1, 1, 1).expand(-1, batch_size, 1)
        
        # 2. 모든 정보 결합 [cam_id, obj_id, pred_x, pred_y]
        combined = torch.cat([cam_ids.float(), corrs_pred_with_obj], dim=-1)
        
        # 3. 배치 차원 병합 [num_cams * batch_size, 4]
        flat_combined = combined.view(-1, 4)
        
        # 4. 객체 ID 추출 및 연속성 체크
        obj_ids = flat_combined[:, 1]
        unique_ids, counts = torch.unique(obj_ids, return_counts=True)
        
        # 5. 객체 ID 연속성 검증
        if not torch.all(torch.diff(unique_ids) == 1):
            print("경고: 객체 ID가 연속적이지 않음. 누락된 객체 존재 가능")
        
        # 6. 중복 제거 (첫 번째 발생만 유지)
        _, unique_indices = torch.unique(obj_ids, return_inverse=True)
        first_occurrence = torch.zeros_like(obj_ids, dtype=torch.bool)
        
        for obj_id in unique_ids:
            indices = (obj_ids == obj_id).nonzero(as_tuple=True)[0]
            if indices.numel() > 0:
                first_occurrence[indices[0]] = True
        
        # 7. 고유 객체 선택
        unique_objs = flat_combined[first_occurrence]
        
        # 8. 크기 검증
        if unique_objs.size(0) != unique_ids.size(0):
            print(f"크기 불일치: 고유 객체 {unique_ids.size(0)}개, 결과 {unique_objs.size(0)}개")
        
        return unique_objs
    
    def uvz_to_lidar_xyz(self, estimated_uvz: torch.Tensor, lidar2img: torch.Tensor) -> torch.Tensor:
        """
        이미지 좌표계의 (u, v, depth) 포인트를 LiDAR 좌표계의 (x, y, z)로 변환합니다.

        Args:
            estimated_uvz (torch.Tensor): (N*V, Num_Points, 3) 형태의 텐서.
                                        각 포인트는 (u, v, z) 정보를 가집니다.
                                        z는 카메라 좌표계에서의 깊이(depth)입니다.
            lidar2img (torch.Tensor): (N, V, 4, 4) 형태의 변환 행렬.

        Returns:
            torch.Tensor: (N*V, Num_Points, 3) 형태의 LiDAR 좌표계 (x, y, z) 텐서.
        """
        # 1. 데이터 형태(Shape) 준비
        N, V, _, _ = lidar2img.shape
        # (N, V, 4, 4) -> (N*V, 4, 4)
        lidar2img_reshaped = lidar2img.view(N * V, 4, 4)
        # (N*V, Num_Points, 3)
        num_points = estimated_uvz.shape[1]

        # 2. 역행렬 계산 (image -> lidar 변환)
        img2lidar = torch.inverse(lidar2img_reshaped)

        # 3. (u, v, z)를 4D 동차 좌표(Homogeneous Coordinate)로 변환
        # (u, v, z) -> (u*z, v*z, z, 1)
        uv = estimated_uvz[..., 0:2]
        depth = estimated_uvz[..., 2:3] # 차원을 유지하기 위해 [..., 2:3] 사용

        # (u, v) * depth -> (u*z, v*z)
        points_2d_multiplied_by_depth = uv * depth
        
        # (u*z, v*z, z, 1) 형태의 동차 좌표 생성
        points_img_homogeneous = torch.cat(
            [points_2d_multiplied_by_depth, depth, torch.ones_like(depth)], 
            dim=-1
        ) # shape: (N*V, Num_Points, 4)

        # 4. 좌표 변환 (행렬 곱셈)
        # img2lidar: (N*V, 4, 4)
        # points_img_homogeneous: (N*V, Num_Points, 4)
        # einsum을 사용하여 배치별 행렬 곱셈 수행
        # 결과 shape: (N*V, Num_Points, 4)
        points_lidar_homogeneous = torch.einsum(
            'bmn,bqn->bqm', 
            img2lidar, 
            points_img_homogeneous
        )

        # 5. 3D 좌표로 변환
        # 동차 좌표 (x, y, z, w)를 w로 나누어 (x, y, z)를 얻음
        points_lidar_xyz = points_lidar_homogeneous[..., :3] / (points_lidar_homogeneous[..., 3:] + 1e-8)

        return points_lidar_xyz
    
    def _prepare_camera_proposals(self, det_xyz, det_feat_sampled, B, N_cam):
        """
        카메라 기반 제안(proposal)들을 배치(batch) 우선 형태로 전처리합니다.

        (B * N_cam, ...) 모양의 텐서를 (B, N_cam * ..., ...) 모양으로 재구성합니다.

        Args:
            det_xyz (torch.Tensor): [B * N_cam, N_proposals, 3] 모양의 좌표 텐서
            det_feat_sampled (torch.Tensor): [B * N_cam, N_proposals, C] 모양의 샘플링된 특징 텐서
            B (int): 배치 크기
            N_cam (int): 카메라 수

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 전처리된 (det_xyz, det_feat) 텐서.
                                            Shape: ([B, N_total, 3], [B, N_total, C])
        """
        # 1. det_xyz 전처리: (B * N_cam, ...) -> (B, ...)
        N_proposals = det_xyz.shape[1]
        # [B * N_cam, N, 3] -> [B, N_cam * N, 3]
        det_xyz_preprocessed = det_xyz.reshape(B, N_cam * N_proposals, 3)

        # 2. det_feat_sampled 전처리: (B * N_cam, ...) -> (B, ...)
        # 이제 det_xyz와 동일한 방식으로 모양만 재구성해주면 됩니다.
        det_feat_preprocessed = det_feat_sampled.reshape(B, N_cam * N_proposals, -1)
        det_feat_preprocessed = self.feat_projector(det_feat_preprocessed)

        return det_xyz_preprocessed, det_feat_preprocessed
    

    def _sample_features_from_grid(self, feature_map, coords):
        """
        주어진 2D 좌표를 이용해 2D 특징 맵에서 특징 벡터를 샘플링합니다.

        F.grid_sample을 사용하여, 각 좌표에 해당하는 특징을
        bilinear interpolation을 통해 정확하게 추출합니다.

        Args:
            feature_map (torch.Tensor): 샘플링할 원본 2D 특징 맵.
                Shape: [B_N, C, H, W]
            coords (torch.Tensor): 샘플링할 위치의 (u, v) 좌표.
                [0, 1] 또는 [-1, 1] 범위로 가정하며, 내부에서 [-1, 1]로 변환합니다.
                Shape: [B_N, N_proposals, 2]

        Returns:
            torch.Tensor: 각 좌표에서 샘플링된 특징 벡터.
                Shape: [B_N, N_proposals, C]
        """
        # 입력 텐서들로부터 필요한 차원 정보를 가져옵니다.
        B_N, C, H, W = feature_map.shape
        _, N_proposals, _ = coords.shape

        # <<< [수정] 좌표 범위를 [0, 1] -> [-1, 1]로 변환 >>>
        # F.grid_sample은 -1 ~ 1 범위의 좌표를 기대합니다.
        coords_normalized = coords * 2.0 - 1.0

        # F.grid_sample의 입력 형식에 맞게 좌표 텐서의 모양을 변경합니다.
        # [B_N, N_proposals, 2] -> [B_N, 1, N_proposals, 2]
        sampling_grid = coords_normalized.view(B_N, 1, N_proposals, 2)

        # F.grid_sample을 사용하여 N_proposals개 좌표 위치의 특징을 정확히 샘플링합니다.
        # 결과 sampled_feat의 모양: [B_N, C, 1, N_proposals]
        sampled_feat = F.grid_sample(feature_map, sampling_grid, mode='bilinear', align_corners=True)

        # 최종적으로 원하는 모양인 [B_N, N_proposals, C] 형태로 정리합니다.
        # [B_N, C, 1, N_proposals] -> [B_N, C, N_proposals] -> [B_N, N_proposals, C]
        sampled_feat_final = sampled_feat.squeeze(2).permute(0, 2, 1)

        return sampled_feat_final
    
    # def convert_boxes_to_original_scale(
    #     self,
    #     pred_results_list: List[torch.Tensor],
    #     data_samples_list: List
    # ) -> List[torch.Tensor]:
    #     """
    #     증강된 이미지 좌표계의 BBox를 원본으로 역변환합니다. (최종 버전)
    #     너비/높이 변수 할당 오류를 수정하여 모든 변환을 최종적으로 해결합니다.
    #     """
    #     num_cameras = 6 
    #     converted_results = []

    #     for i, (pred_tensor, data_sample) in enumerate(zip(pred_results_list, data_samples_list)):
            
    #         if pred_tensor.shape[0] == 0:
    #             converted_results.append(pred_tensor)
    #             continue
                
    #         aug_bboxes = pred_tensor[:, :4].clone()

    #         # # --- ✨ 1. 초기 증강 BBox 코너 좌표 (동차) ✨ ---
    #         # x1, y1, x2, y2 = aug_bboxes.T
    #         # corners = torch.stack([x1, y1, x2, y1, x2, y2, x1, y2], dim=-1).view(-1, 4, 2)
    #         # corners_hom = torch.cat([corners, torch.ones(corners.shape[0], 4, 1, device=corners.device)], dim=-1)

    #         # if i == 0: # 첫 번째 카메라 이미지에 대해서만 로그 출력
    #         #     print(f"\n--- Camera Index: {i} ---")
    #         #     print(f"[LOG] Initial Augmented BBox (first box): {aug_bboxes[0].detach().cpu().numpy()}")
    #         #     print(f"[LOG] Initial Corner Hom (first box, first corner): {corners_hom[0, 0].detach().cpu().numpy()}")

    #         if not hasattr(data_sample, 'img_aug_params'):
    #             converted_results.append(pred_tensor)
    #             continue

    #         params_list = data_sample.img_aug_params
    #         cam_index = i % num_cameras
    #         params_dict = params_list[cam_index]

    #         resize, crop, flip, rotate = params_dict['resize'], params_dict['crop'], params_dict['flip'], params_dict['rotate']
    #         fH, fW = params_dict['final_dim'] # [Height, Width] 순서로 할당

    #         # ✨✨✨ 2. 사용된 증강 파라미터 확인 ✨✨✨
    #         if i == 0:
    #             print(f"[DEBUG] Aug Params: resize={resize:.4f}, crop={crop}, flip={flip}, rotate={rotate:.4f}, fH={fH}, fW={fW}")
            
    #         # --- 역변환 행렬 구성 ---
    #         M_resize_inv = torch.eye(3, device=pred_tensor.device, dtype=torch.float32)
    #         M_resize_inv[0, 0] = 1 / resize
    #         M_resize_inv[1, 1] = 1 / resize

    #         M_crop_inv = torch.eye(3, device=pred_tensor.device, dtype=torch.float32)
    #         M_crop_inv[0, 2] = crop[0]
    #         M_crop_inv[1, 2] = crop[1]

    #         M_flip_inv = torch.eye(3, device=pred_tensor.device, dtype=torch.float32)
    #         # if flip:
    #         #     M_flip_inv[0, 0] = -1
    #         #     M_flip_inv[0, 2] = fW - 1
    #         if flip:
    #             # ✨ 시도: img_transform과 유사하게 crop 후 너비(fW)를 기준 축으로 사용
    #             flip_axis_x = crop[2] - crop[0] # = fW
    #             M_flip_inv[0, 0] = -1
    #             M_flip_inv[0, 2] = flip_axis_x - 1 # fW - 1 대신 사용 시도

    #         M_rotate_inv = torch.eye(3, device=pred_tensor.device, dtype=torch.float32)
    #         if rotate != 0:
    #             angle = math.radians(rotate)
    #             cos, sin = math.cos(angle), math.sin(angle)
    #             cx, cy = (fW - 1) / 2, (fH - 1) / 2
    #             T1 = torch.tensor([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], device=pred_tensor.device, dtype=torch.float32)
    #             # 올바른 방향인 시계 방향(Clockwise) 역회전 행렬
    #             R_inv = torch.tensor([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]], device=pred_tensor.device, dtype=torch.float32)
    #             T2 = torch.tensor([[1, 0, cx], [0, 1, cy], [0, 0, 1]], device=pred_tensor.device, dtype=torch.float32)
    #             M_rotate_inv = T2 @ R_inv @ T1

    #             # # ✨ 시도: img_transform과 유사하게 crop 후 크기(fW, fH)를 중심 계산에 사용
    #             # center_x = (crop[2] - crop[0]) / 2.0 # = fW / 2
    #             # center_y = (crop[3] - crop[1]) / 2.0 # = fH / 2

    #             # T1 = torch.tensor([[1, 0, -center_x], [0, 1, -center_y], [0, 0, 1]], device=pred_tensor.device, dtype=torch.float32)
    #             # R_inv = torch.tensor([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]], device=pred_tensor.device, dtype=torch.float32)
    #             # T2 = torch.tensor([[1, 0, center_x], [0, 1, center_y], [0, 0, 1]], device=pred_tensor.device, dtype=torch.float32)
    #             # M_rotate_inv = T2 @ R_inv @ T1

    #         # --- ✨ 2. 단계별 역변환 적용 및 로그 출력 ✨ ---
    #         corners_current = corners_hom.clone() # 원본 복사

    #         # 단계 1: InvRotate
    #         corners_after_rotate = (M_rotate_inv.to(corners_current.dtype) @ corners_current.transpose(1, 2)).transpose(1, 2)
    #         if i == 0: print(f"[LOG] Corner After InvRotate: {corners_after_rotate[0, 0].detach().cpu().numpy()}")
    #         corners_current = corners_after_rotate

    #         # 단계 2: InvFlip
    #         corners_after_flip = (M_flip_inv.to(corners_current.dtype) @ corners_current.transpose(1, 2)).transpose(1, 2)
    #         if i == 0: print(f"[LOG] Corner After InvFlip:   {corners_after_flip[0, 0].detach().cpu().numpy()}")
    #         corners_current = corners_after_flip

    #         # 단계 3: InvCrop
    #         corners_after_crop = (M_crop_inv.to(corners_current.dtype) @ corners_current.transpose(1, 2)).transpose(1, 2)
    #         if i == 0: print(f"[LOG] Corner After InvCrop:   {corners_after_crop[0, 0].detach().cpu().numpy()}")
    #         corners_current = corners_after_crop

    #         # 단계 4: InvResize
    #         corners_after_resize = (M_resize_inv.to(corners_current.dtype) @ corners_current.transpose(1, 2)).transpose(1, 2)
    #         if i == 0: print(f"[LOG] Corner After InvResize: {corners_after_resize[0, 0].detach().cpu().numpy()}")
    #         transformed_corners_hom = corners_after_resize # 최종 결과
                
    #         # 최종 역변환 행렬 계산
    #         M_total_inv = M_rotate_inv @ M_flip_inv @ M_crop_inv @ M_resize_inv

    #         # Bounding Box 변환 적용
    #         x1, y1, x2, y2 = aug_bboxes.T
    #         corners = torch.stack([x1, y1, x2, y1, x2, y2, x1, y2], dim=-1).view(-1, 4, 2)
    #         corners_hom = torch.cat([corners, torch.ones(corners.shape[0], 4, 1, device=corners.device)], dim=-1)
            
    #         M_total_inv = M_total_inv.to(corners_hom.dtype)
    #         transformed_corners_hom = (M_total_inv @ corners_hom.transpose(1, 2)).transpose(1, 2)
            
    #         transformed_corners = transformed_corners_hom[..., :2] / transformed_corners_hom[..., 2, None]
            
    #         min_coords = torch.min(transformed_corners, dim=1).values
    #         max_coords = torch.max(transformed_corners, dim=1).values
    #         original_bboxes = torch.cat([min_coords, max_coords], dim=1)

    #         if i == 0:
    #             print(f"[LOG] Final Transformed Corner 2D: {transformed_corners[0, 0].detach().cpu().numpy()}")
    #             print(f"[LOG] Final Original BBox (first box): {original_bboxes[0].detach().cpu().numpy()}")
            
    #         new_pred_tensor = pred_tensor.clone()
    #         new_pred_tensor[:, :4] = original_bboxes
    #         converted_results.append(new_pred_tensor)

    #     return converted_results
    
    def convert_boxes_to_original_scale(
        self,
        pred_results_list: List[torch.Tensor],
        data_samples_list: List
    ) -> List[torch.Tensor]:
        """
        증강된 이미지 좌표계의 BBox를 원본으로 역변환합니다. (최종 버전)
        너비/높이 변수 할당 오류를 수정하여 모든 변환을 최종적으로 해결합니다.
        """
        num_cameras = 6 
        converted_results = []

        for i, (pred_tensor, data_sample) in enumerate(zip(pred_results_list, data_samples_list)):
            
            if pred_tensor.shape[0] == 0:
                converted_results.append(pred_tensor)
                continue
                
            aug_bboxes = pred_tensor[:, :4].clone()
        
            if not hasattr(data_sample, 'img_aug_params'):
                        converted_results.append(pred_tensor)
                        continue

            params_list = data_sample.img_aug_params
            cam_index = i % num_cameras
            params_dict = params_list[cam_index]

            resize, crop, flip, rotate = params_dict['resize'], params_dict['crop'], params_dict['flip'], params_dict['rotate']
            fH, fW = params_dict['final_dim'] # 증강 후 최종 크기 (H, W)
            
            # --- ✨ OpenCV를 사용한 아핀 변환 행렬 계산 ✨ ---
            # 1. Resize 변환 행렬 (순방향)
            M_resize = np.float32([[resize, 0, 0], [0, resize, 0]])

            # 2. Crop 변환 행렬 (순방향) - 이동(Translation)
            #    Crop은 (x_offset, y_offset, x_offset+width, y_offset+height)
            M_crop = np.float32([[1, 0, -crop[0]], [0, 1, -crop[1]]]) # 오프셋만큼 빼기

            # 3. Flip 변환 행렬 (순방향)
            M_flip = np.float32([[1, 0, 0], [0, 1, 0]])
            if flip:
                M_flip = np.float32([[-1, 0, fW - 1], [0, 1, 0]]) # crop 후 너비(fW) 기준

            # 4. Rotate 변환 행렬 (순방향)
            center_x_aug, center_y_aug = fW / 2.0, fH / 2.0 # 증강 후 이미지 중심
            M_rotate = cv2.getRotationMatrix2D((center_x_aug, center_y_aug), rotate, 1.0) # OpenCV는 반시계 방향이 +

            # 5. 모든 순방향 변환 행렬 결합 (OpenCV 스타일, 2x3 행렬)
            #    순서: Resize -> Crop -> Flip -> Rotate
            #    OpenCV는 3x3 동차 행렬 곱과 약간 다르게 결합해야 함
            
            # 3x3 행렬로 변환하여 곱셈 (더 직관적)
            def to_3x3(M):
                return np.vstack([M, [0, 0, 1]])

            M_resize_3x3 = to_3x3(M_resize)
            M_crop_3x3 = to_3x3(M_crop)
            M_flip_3x3 = to_3x3(M_flip)
            M_rotate_3x3 = to_3x3(M_rotate)
            
            # 순방향 전체 변환 행렬 (오른쪽부터 적용됨)
            M_forward_total_3x3 = M_rotate_3x3 @ M_flip_3x3 @ M_crop_3x3 @ M_resize_3x3

            # 6. 최종 역변환 행렬 계산
            M_total_inv_np = np.linalg.inv(M_forward_total_3x3)
            M_total_inv = torch.from_numpy(M_total_inv_np).to(dtype=torch.float32, device=pred_tensor.device)

            # if i == 0:
            #     print(f"[LOG] M_total_inv (OpenCV based):\n{M_total_inv.detach().cpu().numpy()}")

            # --- Bounding Box 변환 적용 (이하 로직 동일) ---
            x1, y1, x2, y2 = aug_bboxes.T
            corners = torch.stack([x1, y1, x2, y1, x2, y2, x1, y2], dim=-1).view(-1, 4, 2)
            corners_hom = torch.cat([corners, torch.ones(corners.shape[0], 4, 1, device=corners.device)], dim=-1)
            
            M_total_inv = M_total_inv.to(corners_hom.dtype)
            transformed_corners_hom = (M_total_inv @ corners_hom.transpose(1, 2)).transpose(1, 2)
            
            transformed_corners = transformed_corners_hom[..., :2] / transformed_corners_hom[..., 2, None]
            
            min_coords = torch.min(transformed_corners, dim=1).values
            max_coords = torch.max(transformed_corners, dim=1).values
            original_bboxes = torch.cat([min_coords, max_coords], dim=1)

            # if i == 0:
            #     # ... (기존 로그 출력) ...
            #     print(f"[LOG] Final Original BBox (first box, OpenCV): {original_bboxes[0].detach().cpu().numpy()}")

            new_pred_tensor = pred_tensor.clone()
            new_pred_tensor[:, :4] = original_bboxes
            converted_results.append(new_pred_tensor)

        return converted_results

    def extract_multiscale_img_feats(self, batch_inputs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        입력 딕셔너리에서 이미지를 추출하고, 
        이미지 백본을 통과시켜 특징(features)을 반환합니다.
        """
        imgs = batch_inputs_dict['imgs']
        B, N, C, H, W = imgs.size()
        
        # 이미지를 (B*N, C, H, W) 형태로 재구성하여 백본에 전달
        imgs_reshaped = imgs.view(B * N, C, H, W).contiguous()
        
        # 이미지 백본을 통과시켜 특징 추출
        x_backbone  = self.img_backbone(imgs_reshaped)

        # 2. ✨ 핵심 수정: 백본 출력을 넥(FPN 등)에 통과시킵니다.
        x_neck = self.img_neck(x_backbone)
        
        return x_neck
    
    def _build_calib_dict_from_cam2lidar(
        self,
        camera2lidar,
        camera_intrinsics,
    ):

        # Camera -> LiDAR
        #       inverse
        # LiDAR -> Camera
        lidar2camera = torch.linalg.inv(
            camera2lidar
        )

        K = camera_intrinsics[
            ..., :3, :3
        ]

        lidar2img_3x4 = (
            K
            @ lidar2camera[..., :3, :]
        )

        B, N = camera2lidar.shape[:2]

        bottom = torch.zeros(
            B,
            N,
            1,
            4,
            device=camera2lidar.device,
            dtype=camera2lidar.dtype,
        )

        bottom[..., 0, 3] = 1.0

        lidar2img = torch.cat(
            [
                lidar2img_3x4,
                bottom,
            ],
            dim=-2
        )

        return {
            'lidar2img': lidar2img,
            'cam2img': camera_intrinsics,
            'cam2lidar': camera2lidar,
        }
    
    @torch.no_grad()
    def _solve_translation_given_rotation(
        self,
        query_input,
        source_uv_pixels,
        source_z,
        point_valid_mask,
        broken_camera2lidar,
        camera_intrinsics,
        delta_rot,
    ):
        """
        Solve Delta translation t when Delta rotation R is known.

        Convention:
            broken_C2L = Delta @ clean_C2L

            Delta = [R, t]

        Therefore:

            corrected_C2L
                = inv(Delta) @ broken_C2L

        Inputs
        ------
        query_input:
            [A,Q,2]
            Left-SBS target query.

        source_uv_pixels:
            [A,Q,2]
            Corr/source pixel on broken projection.

        source_z:
            [A,Q,1]
            Optical-axis depth [m].

        point_valid_mask:
            [A,Q]

        broken_camera2lidar:
            [A,4,4]

        camera_intrinsics:
            [A,4,4]

        delta_rot:
            [A,3]
            axis-angle rotation of Delta.

        Returns
        -------
        pred_trans:
            [A,3]

        diagnostics:
            dict
        """

        device = query_input.device
        out_dtype = query_input.dtype

        A_cam, Q, _ = query_input.shape

        # ============================================================
        # 1. Original-image target pixels
        #
        # query x occupies LEFT SBS half:
        #
        # q_x = (u / 1600) / 2
        # q_y = v / 900
        # ============================================================

        target_u = (
            query_input[..., 0]
            * 2.0
            * 1600.0
        )

        target_v = (
            query_input[..., 1]
            * 900.0
        )


        # ============================================================
        # 2. Camera intrinsics
        # ============================================================

        K = (
            camera_intrinsics[
                ...,
                :3,
                :3
            ]
        )

        K_inv = torch.linalg.inv(K)


        # ============================================================
        # 3. Target pixel -> camera ray
        #
        # r = K^-1 [u,v,1]
        # ============================================================

        target_pix_h = torch.stack(
            [
                target_u,
                target_v,
                torch.ones_like(target_u),
            ],
            dim=-1,
        )
        # [A,Q,3]


        target_ray = torch.einsum(
            'aij,aqj->aqi',
            K_inv,
            target_pix_h,
        )


        # Unit normalization improves numerical conditioning.
        target_ray = F.normalize(
            target_ray,
            p=2,
            dim=-1,
            eps=1e-8,
        )


        # ============================================================
        # 4. Broken source pixel -> camera XYZ
        #
        # Z is optical-axis depth:
        #
        # X = (u-cx)/fx * Z
        # Y = (v-cy)/fy * Z
        # ============================================================

        source_u = source_uv_pixels[..., 0]
        source_v = source_uv_pixels[..., 1]

        source_pix_h = torch.stack(
            [
                source_u,
                source_v,
                torch.ones_like(source_u),
            ],
            dim=-1,
        )


        source_ray = torch.einsum(
            'aij,aqj->aqi',
            K_inv,
            source_pix_h,
        )


        source_xyz_cam = (
            source_ray
            * source_z
        )
        # [A,Q,3]


        # ============================================================
        # 5. Broken camera XYZ -> LiDAR XYZ
        #
        # broken C2L:
        #
        # X_L = R_B X_C + t_B
        # ============================================================

        R_broken = (
            broken_camera2lidar[
                ...,
                :3,
                :3
            ]
        )

        t_broken = (
            broken_camera2lidar[
                ...,
                :3,
                3
            ]
        )


        source_xyz_lidar = (
            torch.einsum(
                'aij,aqj->aqi',
                R_broken,
                source_xyz_cam,
            )
            +
            t_broken.unsqueeze(1)
        )


        # ============================================================
        # 6. Known Delta rotation
        # ============================================================

        R_delta = axis_angle_to_matrix(
            delta_rot
        )
        # [A,3,3]


        # ============================================================
        # 7. Translation linear system
        #
        # corrected L2C:
        #
        #     inv(corrected_C2L)
        #   = inv(inv(Delta) @ Broken)
        #   = inv(Broken) @ Delta
        #
        # Therefore target camera point:
        #
        # X_C =
        # R_B^T ( R_delta X_L + t_delta - t_B )
        #
        # It must lie on target ray r:
        #
        # [r]_x X_C = 0
        #
        # giving:
        #
        # [r]_x R_B^T t_delta
        #
        #   =
        #
        # - [r]_x R_B^T
        #       (R_delta X_L - t_B)
        #
        # This is:
        #
        # A t = b
        # ============================================================

        pred_trans = torch.zeros(
            (
                A_cam,
                3,
            ),
            device=device,
            dtype=torch.float64,
        )


        valid_counts = []
        residuals = []
        cond_numbers = []


        for cam_idx in range(A_cam):

            valid = (
                point_valid_mask[
                    cam_idx
                ]
                &
                torch.isfinite(
                    source_xyz_lidar[
                        cam_idx
                    ]
                ).all(dim=-1)
                &
                torch.isfinite(
                    target_ray[
                        cam_idx
                    ]
                ).all(dim=-1)
                &
                torch.isfinite(
                    source_z[
                        cam_idx,
                        :,
                        0
                    ]
                )
                &
                (
                    source_z[
                        cam_idx,
                        :,
                        0
                    ]
                    > 1e-3
                )
            )


            num_valid = int(
                valid.sum().item()
            )

            valid_counts.append(
                num_valid
            )


            if num_valid < 6:

                raise RuntimeError(
                    '[GEO SOLVER] '
                    f'cam={cam_idx}: '
                    f'only {num_valid} valid points.'
                )


            r = (
                target_ray[
                    cam_idx,
                    valid
                ]
                .double()
            )
            # [Nv,3]


            X = (
                source_xyz_lidar[
                    cam_idx,
                    valid
                ]
                .double()
            )


            Rb = (
                R_broken[
                    cam_idx
                ]
                .double()
            )


            tb = (
                t_broken[
                    cam_idx
                ]
                .double()
            )


            Rd = (
                R_delta[
                    cam_idx
                ]
                .double()
            )


            # --------------------------------------------------------
            # skew(r)
            #
            # [ 0  -rz  ry ]
            # [ rz  0  -rx ]
            # [-ry  rx  0  ]
            # --------------------------------------------------------

            rx = r[:, 0]
            ry = r[:, 1]
            rz = r[:, 2]

            zero = torch.zeros_like(rx)


            skew_r = torch.stack(
                [
                    zero, -rz,   ry,
                    rz,   zero, -rx,
                    -ry,  rx,    zero,
                ],
                dim=-1,
            ).reshape(
                -1,
                3,
                3,
            )


            Rb_T = Rb.transpose(0, 1)


            # A_i = [r_i]_x R_B^T
            A_block = (
                skew_r
                @ Rb_T
            )
            # [Nv,3,3]


            rotated_X = (
                torch.einsum(
                    'ij,qj->qi',
                    Rd,
                    X,
                )
            )


            rhs_point = (
                rotated_X
                - tb.unsqueeze(0)
            )


            # b_i =
            # - [r_i]_x R_B^T (R_delta X_i - t_B)

            tmp = torch.einsum(
                'ij,qj->qi',
                Rb_T,
                rhs_point,
            )


            b_block = -torch.einsum(
                'qij,qj->qi',
                skew_r,
                tmp,
            )


            A_matrix = A_block.reshape(
                -1,
                3,
            )

            b_vector = b_block.reshape(
                -1,
                1,
            )


            # ========================================================
            # 8. Least-squares solve
            # ========================================================

            solution = torch.linalg.lstsq(
                A_matrix,
                b_vector,
            ).solution[
                :3,
                0
            ]


            pred_trans[
                cam_idx
            ] = solution


            # ========================================================
            # Diagnostics
            # ========================================================

            residual = (
                A_matrix
                @ solution.unsqueeze(-1)
                - b_vector
            )


            residuals.append(
                torch.sqrt(
                    torch.mean(
                        residual ** 2
                    )
                )
            )


            normal_matrix = (
                A_matrix.transpose(0, 1)
                @ A_matrix
            )


            cond_numbers.append(
                torch.linalg.cond(
                    normal_matrix
                )
            )


        diagnostics = {
            'valid_count':
                torch.tensor(
                    valid_counts,
                    device=device,
                    dtype=torch.float32,
                ),

            'residual_rmse':
                torch.stack(
                    residuals
                ).float(),

            'condition_number':
                torch.stack(
                    cond_numbers
                ).float(),
        }


        return (
            pred_trans.to(out_dtype),
            diagnostics,
        )
    
    def _make_delta_matrix(
        self,
        delta_rot,
        delta_trans,
    ):

        R = axis_angle_to_matrix(
            delta_rot
        )

        B, N = delta_rot.shape[:2]

        T = torch.eye(
            4,
            device=delta_rot.device,
            dtype=delta_rot.dtype,
        )

        T = T.view(
            1, 1, 4, 4
        ).repeat(
            B, N, 1, 1
        )

        T[..., :3, :3] = R
        T[..., :3, 3] = delta_trans

        return T
        
    def _stack_sample_tensor(
        self,
        batch_data_samples,
        key,
        device,
    ):
        values = []

        for sample in batch_data_samples:

            value = getattr(
                sample,
                key,
                None
            )

            if value is None:

                value = sample.metainfo.get(
                    key,
                    None
                )

            if value is None:
                raise KeyError(
                    f"Missing '{key}' "
                    f"in Det3DDataSample."
                )

            if torch.is_tensor(value):

                value = value.to(
                    device=device,
                    dtype=torch.float32
                )

            else:

                value = torch.tensor(
                    value,
                    device=device,
                    dtype=torch.float32
                )

            values.append(value)

        return torch.stack(
            values,
            dim=0
        )

    def _predict_bevfusion_with_calib(
        self,
        batch_inputs_dict,
        batch_data_samples,
        calib_dict,
        img_feats=None,
    ):

        batch_input_metas = [
            item.metainfo
            for item in batch_data_samples
        ]

        if img_feats is None:

            img_feats = (
                self.extract_multiscale_img_feats(
                    batch_inputs_dict
                )
            )

        feats = self.extract_feat(
            batch_inputs_dict=
                batch_inputs_dict,

            batch_input_metas=
                batch_input_metas,

            corrected_calib=
                calib_dict,

            precomputed_img_feats=
                img_feats,
        )

        # --------------------------------------------------
        # 매우 중요:
        #
        # PCC camera proposal은 사용하지 않는다.
        # 순수 BEVFusion detection
        # --------------------------------------------------

        results_list_3d = (
            self.bbox_head.predict(
                feats,
                None,
                None,
                batch_input_metas,
            )
        )

        return self.add_pred_to_datasample(
            batch_data_samples,
            results_list_3d,
        )

    def _predict_calibration_baseline(
        self,
        batch_inputs_dict,
        batch_data_samples,
        mode,
    ):

        device = batch_inputs_dict[
            'imgs'
        ].device

        clean_c2l = (
            self._stack_sample_tensor(
                batch_data_samples,
                'camera2lidar',
                device,
            )
        )

        broken_c2l = (
            self._stack_sample_tensor(
                batch_data_samples,
                'broken_camera2lidar',
                device,
            )
        )

        intrinsics = (
            self._stack_sample_tensor(
                batch_data_samples,
                'broken_camera_intrinsics',
                device,
            )
        )

        # ==============================================
        # CLEAN
        # ==============================================

        if mode == 'clean':

            selected_c2l = clean_c2l

        # ==============================================
        # BROKEN
        # ==============================================

        elif mode == 'broken':

            selected_c2l = broken_c2l


        # ==============================================
        # ORACLE
        # ==============================================

        elif mode == 'oracle':

            gt_delta_rot = (
                self._stack_sample_tensor(
                    batch_data_samples,
                    'gt_delta_rot',
                    device,
                )
            )

            gt_delta_trans = (
                self._stack_sample_tensor(
                    batch_data_samples,
                    'gt_delta_trans',
                    device,
                )
            )

            delta_T_gt = (
                self._make_delta_matrix(
                    gt_delta_rot,
                    gt_delta_trans,
                )
            )

            # ------------------------------------------
            # Gate A:
            #
            # broken = Delta_GT @ clean
            # ------------------------------------------

            broken_reconstructed = (
                delta_T_gt
                @ clean_c2l
            )

            delta_definition_error = (
                broken_reconstructed
                - broken_c2l
            ).abs().max()

            if delta_definition_error > 1e-4:

                raise RuntimeError(
                    "Delta convention mismatch: "
                    f"{delta_definition_error.item():.8e}"
                )

            # ------------------------------------------
            # Oracle
            #
            # clean =
            # inv(Delta_GT) @ broken
            # ------------------------------------------

            selected_c2l = (
                torch.linalg.inv(
                    delta_T_gt
                )
                @ broken_c2l
            )

            restore_error = (
                selected_c2l
                - clean_c2l
            ).abs().max()

            if restore_error > 1e-4:

                raise RuntimeError(
                    "Oracle restoration failed: "
                    f"{restore_error.item():.8e}"
                )
            
            # print(
            #     "[ORACLE CHECK] "
            #     f"broken reconstruction = "
            #     f"{delta_definition_error.item():.8e}, "
            #     f"GT restore = "
            #     f"{restore_error.item():.8e}"
            # )

        else:

            raise ValueError(
                f'Unsupported mode: {mode}'
            )


        calib_dict = (
            self._build_calib_dict_from_cam2lidar(
                selected_c2l,
                intrinsics,
            )
        )

        clean_calib = (
            self._build_calib_dict_from_cam2lidar(
                clean_c2l,
                intrinsics,
            )
        )

        oracle_calib = (
            self._build_calib_dict_from_cam2lidar(
                selected_c2l,
                intrinsics,
            )
        )

        # projection_error = (
        #     clean_calib['lidar2img']
        #     - oracle_calib['lidar2img']
        # ).abs().max()

        # print(
        #     "[ORACLE PROJECTION CHECK]",
        #     projection_error.item()
        # )

        # # ======================================================
        # # CLEAN vs ORIGINAL BEVFusion METADATA CHECK
        # # ======================================================

        # meta_lidar2img = torch.stack([
        #     torch.as_tensor(
        #         s.metainfo['lidar2img'],
        #         device=device,
        #         dtype=torch.float32
        #     )
        #     for s in batch_data_samples
        # ])

        # meta_cam2img = torch.stack([
        #     torch.as_tensor(
        #         s.metainfo['cam2img'],
        #         device=device,
        #         dtype=torch.float32
        #     )
        #     for s in batch_data_samples
        # ])

        # meta_cam2lidar = torch.stack([
        #     torch.as_tensor(
        #         s.metainfo['cam2lidar'],
        #         device=device,
        #         dtype=torch.float32
        #     )
        #     for s in batch_data_samples
        # ])


        # err_c2l = (
        #     clean_c2l
        #     - meta_cam2lidar
        # ).abs().max()

        # err_K = (
        #     intrinsics
        #     - meta_cam2img
        # ).abs().max()

        # err_l2i = (
        #     calib_dict['lidar2img']
        #     - meta_lidar2img
        # ).abs().max()


        # print(
        #     "\n[CLEAN vs ORIGINAL META]"
        # )

        # print(
        #     "cam2lidar error =",
        #     err_c2l.item()
        # )

        # print(
        #     "cam2img error   =",
        #     err_K.item()
        # )

        # print(
        #     "lidar2img error =",
        #     err_l2i.item()
        # )

        return self._predict_bevfusion_with_calib(
            batch_inputs_dict,
            batch_data_samples,
            calib_dict,
        )    
        
    # ########## old code (이력관리- 나중에 까먹지 않기 !!) ############
    # def _get_corrected_calib_from_prediction(
    #     self,
    #     pred_delta_rot: torch.Tensor, # ✨ (B, N, 3)
    #     pred_delta_trans: torch.Tensor, # (B, N, 3)
    #     broken_camera2lidar: torch.Tensor,
    #     broken_camera_intrinsics: torch.Tensor
    # ) -> Dict[str, torch.Tensor]:
    #     """
    #     예측된 delta 값과 broken calibration을 사용하여 
    #     보정된 calibration 파라미터 딕셔너리를 생성합니다.
    #     """
    #     # --- 1. 예측된 오차를 사용하여 corrected_camera2lidar 생성 ---
    #     pred_delta_rot_mat = axis_angle_to_matrix(pred_delta_rot)
    #     broken_rots = broken_camera2lidar[..., :3, :3]
    #     broken_trans = broken_camera2lidar[..., :3, 3]
        
    #     # 🛑 [수정] 1. 회전 행렬 순서 수정 (T_broken @ T_pred에 맞춤)
    #     # T_{Cam->L} = T_{Cam->MisL} @ T_{MisL->L}
    #     corrected_camera2lidar_rots =  broken_rots @ pred_delta_rot_mat
    #     # 🛑 [수정] 2. 이동 벡터 계산 수정 (회전 반영)
    #     # t_{new} = R_{broken} @ t_{pred} + t_{broken}
    #     # (R_broken @ pred_delta_trans.unsqueeze(-1))는 R_{broken} * t_{pred} 입니다.
    #     corrected_camera2lidar_trans = (
    #         torch.matmul(broken_rots, pred_delta_trans.unsqueeze(-1)).squeeze(-1) + broken_trans
    #     )
        
    #     # 1. 상단 3x4 부분 [R_corr | t_corr] 생성
    #     top_3x4 = torch.cat(
    #         [corrected_camera2lidar_rots, corrected_camera2lidar_trans.unsqueeze(-1)], 
    #         dim=-1
    #     ) # shape: (B, N, 3, 4)

    #     # 2. 하단 1x4 부분 [0, 0, 0, 1] 생성
    #     B, N = broken_camera2lidar.shape[:2]
    #     bottom_row = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], 
    #                             device=broken_camera2lidar.device, 
    #                             dtype=broken_camera2lidar.dtype)
    #     bottom_row = bottom_row.expand(B, N, -1, -1) # shape: (B, N, 1, 4)

    #     # 3. 상단과 하단을 합쳐 최종 4x4 행렬 생성
    #     corrected_camera2lidar = torch.cat([top_3x4, bottom_row], dim=-2)
        
    #     # --- (이후 lidar2img 계산 로직은 기존과 동일) ---
    #     corrected_lidar2camera_rots = corrected_camera2lidar_rots.transpose(-1, -2)
    #     corrected_lidar2camera_trans = -torch.matmul(
    #         corrected_lidar2camera_rots,
    #         corrected_camera2lidar_trans.unsqueeze(-1)
    #     ).squeeze(-1)
    #     corrected_lidar2camera_3x4 = torch.cat(
    #         [corrected_lidar2camera_rots, corrected_lidar2camera_trans.unsqueeze(-1)], dim=-1
    #     )

    #     # 최종 투영 행렬 계산 (3x4)
    #     intrinsics_3x3 = broken_camera_intrinsics[..., :3, :3]
    #     corrected_lidar2imag_3x4 = intrinsics_3x3 @ corrected_lidar2camera_3x4
        
    #     # 4x4 동차 좌표계 행렬로 변환
    #     B, N, _, _ = corrected_lidar2imag_3x4.shape
    #     bottom_row = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], 
    #                             device=corrected_lidar2imag_3x4.device, 
    #                             dtype=corrected_lidar2imag_3x4.dtype)
    #     bottom_row = bottom_row.expand(B, N, -1, -1)
    #     corrected_lidar2imag_4x4 = torch.cat([corrected_lidar2imag_3x4, bottom_row], dim=-2)

    #     # --- 3. 최종 결과 딕셔너리 반환 ---
    #     corrected_calib_dict = {
    #         'lidar2img': corrected_lidar2imag_4x4,
    #         'cam2img': broken_camera_intrinsics,
    #         'cam2lidar': corrected_camera2lidar
    #     }
        
    #     return corrected_calib_dict

    def _build_corr_target_from_clean_depth(
        self,
        query_input_active,
        active_cam_indices,
        dense_depth_map_gt,
        clean_camera2lidar,
        broken_camera2lidar,
        camera_intrinsics,
    ):
        """
        Build GT CorrNet target for object-centric image queries.

        query_input_active:
            [A, Q, 2]
            SBS-normalized coordinates.
            RGB side occupies x=[0, 0.5].

        Returns:
            corr_target:
                [A, Q, 2]
                Target coordinates on BROKEN depth side.
                SBS-normalized.

            valid_mask:
                [A, Q]

            z_broken_gt:
                [A, Q]
                Depth of the SAME physical 3D point
                under the broken camera geometry.
        """

        B, N = clean_camera2lidar.shape[:2]

        # Current PCC training path is effectively B=1.
        if B != 1:
            raise RuntimeError(
                "Corr target builder currently expects B=1. "
                f"Got B={B}."
            )


        H_img = 900
        W_img = 1600

        A, Q, _ = query_input_active.shape


        # ============================================================
        # 1. SBS-normalized RGB query -> original image pixel
        #
        # q_x = (u / 1600) / 2
        # q_y = v / 900
        # ============================================================

        u_clean = (
            query_input_active[..., 0]
            * 2.0
            * W_img
        )

        v_clean = (
            query_input_active[..., 1]
            * H_img
        )


        # ============================================================
        # 2. Sample CLEAN depth at object-center query
        # ============================================================

        clean_depth_all = (
            dense_depth_map_gt
            .view(
                B * N,
                H_img,
                W_img,
            )
        )

        clean_depth_active = (
            clean_depth_all[
                active_cam_indices
            ]
            .unsqueeze(1)
        )
        # [A,1,H,W]


        grid_x = (
            2.0
            * u_clean
            / (W_img - 1)
            - 1.0
        )

        grid_y = (
            2.0
            * v_clean
            / (H_img - 1)
            - 1.0
        )

        sample_grid = torch.stack(
            [
                grid_x,
                grid_y,
            ],
            dim=-1,
        ).unsqueeze(2)
        # [A,Q,1,2]


        z_clean = F.grid_sample(
            clean_depth_active,
            sample_grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )

        z_clean = (
            z_clean
            .squeeze(1)
            .squeeze(-1)
        )
        # [A,Q]


        # ============================================================
        # 3. Build CLEAN and BROKEN projection matrices
        # ============================================================

        clean_calib = (
            self._build_calib_dict_from_cam2lidar(
                clean_camera2lidar,
                camera_intrinsics,
            )
        )

        broken_calib = (
            self._build_calib_dict_from_cam2lidar(
                broken_camera2lidar,
                camera_intrinsics,
            )
        )


        clean_l2i_active = (
            clean_calib[
                'lidar2img'
            ][
                0,
                active_cam_indices,
            ]
        )

        broken_l2i_active = (
            broken_calib[
                'lidar2img'
            ][
                0,
                active_cam_indices,
            ]
        )
        # [A,4,4]


        # ============================================================
        # 4. Clean (u,v,z) -> physical LiDAR XYZ
        # ============================================================

        clean_uvz = torch.stack(
            [
                u_clean,
                v_clean,
                z_clean,
            ],
            dim=-1,
        )
        # [A,Q,3]


        xyz_lidar = self.uvz_to_lidar_xyz(
            clean_uvz,
            clean_l2i_active.unsqueeze(0),
        )
        # [A,Q,3]


        # ============================================================
        # 5. Same physical XYZ -> BROKEN image projection
        # ============================================================

        xyz_h = torch.cat(
            [
                xyz_lidar,
                torch.ones_like(
                    xyz_lidar[..., :1]
                ),
            ],
            dim=-1,
        )


        proj_broken = torch.einsum(
            'aij,aqj->aqi',
            broken_l2i_active,
            xyz_h,
        )


        z_broken = (
            proj_broken[..., 2]
        )


        z_safe = torch.clamp(
            z_broken,
            min=1e-6,
        )


        u_broken = (
            proj_broken[..., 0]
            / z_safe
        )

        v_broken = (
            proj_broken[..., 1]
            / z_safe
        )


        # ============================================================
        # 6. Valid correspondence
        # ============================================================

        valid_mask = (
            (z_clean > 1e-3)
            & (z_broken > 1e-3)

            & torch.isfinite(
                u_broken
            )

            & torch.isfinite(
                v_broken
            )

            & (u_broken >= 0)
            & (u_broken < W_img)

            & (v_broken >= 0)
            & (v_broken < H_img)
        )


        # ============================================================
        # 7. BROKEN pixel -> SBS normalized target
        #
        # Depth image is the RIGHT half:
        #
        # x target = 0.5 + u / (2*1600)
        # ============================================================

        target_x = (
            0.5
            +
            u_broken
            / (2.0 * W_img)
        )

        target_y = (
            v_broken
            / H_img
        )


        corr_target = torch.stack(
            [
                target_x,
                target_y,
            ],
            dim=-1,
        )


        return (
            corr_target.detach(),
            valid_mask.detach(),
            z_broken.detach(),
        )

    def _get_corrected_calib_from_prediction(
        self,
        pred_delta_rot: torch.Tensor, # (B, N, 3) - 예측된 오차 각도
        pred_delta_trans: torch.Tensor, # (B, N, 3) - 예측된 오차 이동량
        broken_camera2lidar: torch.Tensor, # (B, N, 4, 4) - T_Cam->MisL
        broken_camera_intrinsics: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        예측된 델타의 역행렬을 계산하여 broken calibration에 적용함으로써 보정된 
        T_Cam->Lidar 행렬을 생성합니다. (T_corrected = T_broken @ T_correction_inverse)
        """
        
        B, N = broken_camera2lidar.shape[:2]
        device = broken_camera2lidar.device
        dtype = broken_camera2lidar.dtype
        
        # --- 1. 예측된 델타 (오차) 행렬 T_pred_delta (4x4) 생성 ---
        # T_pred_delta는 네트워크가 예측한 오차 그 자체 (T_Lidar -> MisLidar)
        pred_delta_rot_mat = axis_angle_to_matrix(pred_delta_rot)

        # 1.1 torch.eye(4)를 생성하고 expand 합니다.
        delta_matrix_base = torch.eye(4, dtype=dtype, device=device)
        delta_matrix_expanded = delta_matrix_base.expand(B, N, -1, -1) 
        # 1.2 expand된 뷰를 clone()하여 새로운 메모리 공간을 확보합니다.
        delta_matrix = delta_matrix_expanded.clone()
        # 1.3 이제 안전하게 할당
        delta_matrix[..., :3, :3] = pred_delta_rot_mat
        delta_matrix[..., :3, 3] = pred_delta_trans

        # --- 2. 예측된 델타의 역행렬 (T_correction) 계산 및 적용 (핵심 로직) ---
        # T_Correction = T_pred_delta_inverse
        try:
            # 2.1 예측된 오차 행렬의 역행렬을 구함 (이것이 보정 행렬 T_MisL->Lidar 임)
            correction_matrix = torch.linalg.inv(delta_matrix)
        except torch.linalg.LinAlgError:
            # 역행렬 계산 실패 시, 보정 없는 단위 행렬로 대체 (학습 초기 발산 방지)
            correction_matrix = torch.eye(4, dtype=dtype, device=device).expand(B, N, -1, -1)
            
        # 2.2 Broken T_c2l 에 correction_matrix 를 곱하여 최종 보정
        """
        Predicted error Delta_pred is inverted and left-multiplied:

            T_corrected
                = inv(Delta_pred) @ T_broken
        """
        corrected_camera2lidar = torch.matmul(correction_matrix,broken_camera2lidar)
        
        # --- 3. 행렬 추출 및 역변환 ---
        # T_Cam->Lidar (corrected_camera2lidar) 에서 R, t 추출
        corrected_camera2lidar_rots = corrected_camera2lidar[..., :3, :3]
        corrected_camera2lidar_trans = corrected_camera2lidar[..., :3, 3] # (B, N, 3)
        
        # T_Lidar->Camera 역행렬 계산 (3x4)
        corrected_lidar2camera_rots = corrected_camera2lidar_rots.transpose(-1, -2)
        corrected_lidar2camera_trans = -torch.matmul(
            corrected_lidar2camera_rots,
            corrected_camera2lidar_trans.unsqueeze(-1)
        ).squeeze(-1)
        corrected_lidar2camera_3x4 = torch.cat(
            [corrected_lidar2camera_rots, corrected_lidar2camera_trans.unsqueeze(-1)], dim=-1
        )

        # --- 4. 최종 투영 행렬 (Lidar->Image) 계산 ---
        intrinsics_3x3 = broken_camera_intrinsics[..., :3, :3]
        corrected_lidar2imag_3x4 = torch.matmul(intrinsics_3x3, corrected_lidar2camera_3x4)
        
        # 4x4 동차 좌표계 행렬로 변환 (LSS/BEV 파이프라인용)
        bottom_row_proj = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        bottom_row_proj = bottom_row_proj.expand(B, N, -1, -1)
        corrected_lidar2imag_4x4 = torch.cat([corrected_lidar2imag_3x4, bottom_row_proj], dim=-2)

        # --- 5. 최종 결과 딕셔너리 반환 ---
        corrected_calib_dict = {
            'lidar2img': corrected_lidar2imag_4x4,
            'cam2img': broken_camera_intrinsics,
            'cam2lidar': corrected_camera2lidar # 최종 보정된 T_c2l (4x4)
        }
        
        return corrected_calib_dict

    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        
            target_device = batch_inputs_dict['imgs'].device
            batch_input_metas = [item.metainfo for item in batch_data_samples]
            # batch_data_samples에서 직접 Calibration 관련 텐서를 가져옵니다.
            broken_camera2lidar = torch.stack([s.broken_camera2lidar for s in batch_data_samples]).to(target_device)
            broken_camera_intrinsics = torch.stack([s.broken_camera_intrinsics for s in batch_data_samples]).to(target_device)
            original_camera2lidar = torch.stack([s.camera2lidar for s in batch_data_samples]).to(target_device)
            gt_delta_rot = torch.stack([s.gt_delta_rot for s in batch_data_samples]).to(target_device)
            gt_delta_trans = torch.stack([s.gt_delta_trans for s in batch_data_samples]).to(target_device)

            # img_feats = self.extract_multiscale_img_feats(batch_inputs_dict)

            # ============================================================
            # Pretrained 2D query generator
            #
            # Stage-1:
            #   no gradient through 2D detector
            # ============================================================

            if self.is_lgpc_stage1:

                with torch.no_grad():

                    img_feats = (
                        self.extract_multiscale_img_feats(
                            batch_inputs_dict
                        )
                    )

            else:

                img_feats = (
                    self.extract_multiscale_img_feats(
                        batch_inputs_dict
                    )
                )

            reshaped_img_feats, reshaped_data_samples = self._prepare_2d_head_inputs(
                img_feats, batch_data_samples)
            
            # losses = dict()

            # # ============================================================
            # # LGPC Stage-1 standalone training
            # # ============================================================
            
            # if self.with_bbox_head:
            #     losses_2d = self.img_bbox_head.loss(reshaped_img_feats, reshaped_data_samples)
            # else:
            #     losses_2d = dict()
            
            # # 4. 계산된 2D 로스를 최종 로스 딕셔너리에 'img_' 접두사와 함께 추가
            # total_losses = dict()
            # for k, v in losses_2d.items():
            #     total_losses[f'img_{k}'] = v # 예: 'loss_cls' -> 'img_loss_cls'
            
            # # losses 딕셔너리를 total_losses로 초기화하여 2D loss를 먼저 담습니다.
            # losses = total_losses

            losses = dict()

            # ============================================================
            # 2D detector
            #
            # Stage-1 LGPC:
            #   pretrained 2D detector is used ONLY to generate
            #   object-center queries.
            #
            #   Do NOT optimize it.
            # ============================================================

            if not self.is_lgpc_stage1:

                if self.img_bbox_head is not None:

                    losses_2d = self.img_bbox_head.loss(
                        reshaped_img_feats,
                        reshaped_data_samples
                    )

                    for k, v in losses_2d.items():

                        losses[
                            f'img_{k}'
                        ] = v

            detections_2d = self._generate_and_process_2d_dets(
                reshaped_img_feats, 
                reshaped_data_samples, 
                batch_inputs_dict, 
                visualize=False
            )

            detections_2d_orig_coords = self.convert_boxes_to_original_scale(
                pred_results_list=detections_2d,
                data_samples_list=reshaped_data_samples
            )

            rois , proposla_list = self._generate_rois_from_detections(detections_2d_orig_coords)
            rois_center = self.get_center_points(rois)
            
            if rois_center.numel() > 0:
                active_cam_indices = torch.unique(rois_center[:, 0]).long()
            else:
                active_cam_indices = torch.tensor([], dtype=torch.long, device=rois_center.device)
            

            # ============================================================
            # Calib-only DDP empty-query safety
            #
            # Some DDP ranks may occasionally receive a sample for which
            # no usable 2D object query is generated.
            #
            # In calib-only training:
            #
            #   CorrNet      = frozen
            #   ZEstimator   = frozen
            #   CalibHead    = trainable
            #
            # If active_cam_indices is empty, the normal CalibHead path
            # cannot produce loss_calib_rot / loss_calib_trans.
            #
            # Therefore create a graph-connected ZERO loss HERE,
            # inside model.loss() / DDP forward.
            #
            # IMPORTANT:
            # Do NOT create this zero loss inside parse_losses().
            # ============================================================

            if (
                self.is_lgpc_stage1
                and self.lgpc_train_stage == 'calib'
                and len(active_cam_indices) == 0
            ):

                zero_calib_loss = None


                for param in self.calib_head.parameters():

                    if not param.requires_grad:
                        continue


                    zero_term = (
                        param.sum()
                        * 0.0
                    )


                    if zero_calib_loss is None:

                        zero_calib_loss = zero_term

                    else:

                        zero_calib_loss = (
                            zero_calib_loss
                            + zero_term
                        )


                if zero_calib_loss is None:

                    raise RuntimeError(
                        '[CALIB EMPTY BATCH] '
                        'CalibHead has no trainable parameter.'
                    )


                # Name contains "loss", therefore MMEngine optimizer
                # will receive this graph-connected zero.
                losses[
                    'loss_calib_empty_batch'
                ] = zero_calib_loss


                # Logging only
                losses[
                    'calib_empty_query_batch'
                ] = torch.ones(
                    (),
                    device=target_device,
                    dtype=torch.float32,
                )


                rank = (
                    dist.get_rank()
                    if (
                        dist.is_available()
                        and dist.is_initialized()
                    )
                    else 0
                )


                print(
                    '\n'
                    '=========================================\n'
                    '[CALIB EMPTY QUERY BATCH]\n'
                    f'rank = {rank}\n'
                    f'rois_center shape = '
                    f'{tuple(rois_center.shape)}\n'
                    f'active_cam_indices = []\n'
                    'This batch is skipped with a '
                    'DDP-safe zero CalibHead loss.\n'
                    '=========================================\n'
                )

                return losses
            
            (
                trimed_center_pts,
                query_unique_mask,
            ) = self.batch_rois_center_by_cam_id(

                rois_center,

                batch_size=200,
            )

            query_coords = trimed_center_pts[..., 2:]
            q_x = (query_coords[..., 0] / 1600) / 2
            q_y = query_coords[..., 1] / 900
            if query_coords.shape[-1] > 2:
                q_z = query_coords[..., 2]
                query_input = torch.stack([q_x, q_y, q_z], dim=-1)
            else:
                query_input = torch.stack([q_x, q_y], dim=-1)

            # sbs_img, pertubed_points,dense_depth_map,dense_depth_map_gt = self.extract_sbs_img(batch_inputs_dict, batch_input_metas,visualize=False)
           
            # ============================================================
            # Select CorrNet / Z-estimator input condition
            # ============================================================
            if self.calibration_mode == 'pcc_clean_refine':
                depth_mode = 'clean'
            else:
                depth_mode = 'broken'


            sbs_img, pertubed_points, dense_depth_map, dense_depth_map_gt = \
                self.extract_sbs_img(
                    batch_inputs_dict,
                    batch_input_metas,
                    visualize=False,
                    # NEW
                    depth_mode=depth_mode,
                )
           
            B,N,C,H,W = sbs_img.shape
            d_model = 312
            # feat_h, feat_w = 12, 64
            feat_h, feat_w = 12, 80

            # --- ✨ FIX 1: 모든 텐서를 미리 0으로 초기화 (else 블록과 공유) ---
            raw_corrs_shape = (B * N, query_input.shape[1], query_input.shape[2])
            enc_out_shape = (B * N, feat_h * feat_w, d_model)
            esitmated_uvz_shape = (B * N, query_input.shape[1], 3) # (u,v,z)

            raw_corrs = torch.zeros(raw_corrs_shape, device=query_input.device)
            enc_out = torch.zeros(enc_out_shape, device=query_input.device)
            esitmated_uvz = torch.zeros(esitmated_uvz_shape, device=query_input.device)
            pred_delta_6dof = torch.zeros(B, N, 6, device=target_device)
            
            # --- ✨ FIX 2: "안전장치" if 블록 유지 ---
            if len(active_cam_indices) > 0:
                sbs_img_filtered = sbs_img[:, active_cam_indices]
                query_input_filtered = query_input[active_cam_indices]
                query_unique_mask_filtered = (
                    query_unique_mask[
                        active_cam_indices
                    ]
                )
                
                num_active_cams = sbs_img_filtered.shape[1]
                sbs_view = sbs_img_filtered.view(B * num_active_cams, C, H, W)

                if B != 1:

                    raise RuntimeError(
                        '[LGPC CalibHead V2] '
                        f'Current active-camera path expects B=1, got B={B}.'
                    )

                camera_intrinsics_active = (
                    broken_camera_intrinsics[
                        0,
                        active_cam_indices,
                    ]
                )
                
                # ============================================================
                # LGPC internal training stage
                #
                # corr  : CorrNet only
                # z     : ZEstimator only
                # calib : CalibHead only
                # joint : Corr + Z + Calib
                #
                # Non-Stage1 PCC always behaves as joint.
                # ============================================================

                if self.is_lgpc_stage1:
                    stage = self.lgpc_train_stage
                else:
                    stage = 'joint'


                # ============================================================
                # 1. CorrNet Forward
                #
                # corr / joint:
                #     CorrNet needs gradient.
                #
                # z / calib:
                #     CorrNet is only a frozen feature provider.
                # ============================================================

                if stage in {
                    'corr',
                    'joint',
                }:

                    (
                        raw_corrs_active,
                        cycle,
                        corr_mask,
                        enc_out_active_4d,
                    ) = self.corr(
                        sbs_view,
                        query_input_filtered,
                    )

                else:

                    with torch.no_grad():

                        (
                            raw_corrs_active,
                            cycle,
                            corr_mask,
                            enc_out_active_4d,
                        ) = self.corr(
                            sbs_view,
                            query_input_filtered,
                        )


                # ============================================================
                # Runtime shape check
                # ============================================================

                b_act, c_f, h_f, w_f = (
                    enc_out_active_4d.shape
                )

                if (
                    c_f != 312
                    or h_f != 12
                    or w_f != 80
                ):

                    raise RuntimeError(
                        '[CorrNet shape mismatch] '
                        f'expected=(312,12,80), '
                        f'actual=({c_f},{h_f},{w_f})'
                    )


                enc_out_active_3d = (
                    enc_out_active_4d
                    .flatten(2)
                    .permute(0, 2, 1)
                )


                # ============================================================
                # 2. Direct Correspondence Supervision
                #
                # IMPORTANT:
                #
                # Only Corr and Joint stages calculate L_corr.
                #
                # z/calib stages do NOT need corr_target generation.
                # ============================================================

                corr_target_active = None
                corr_valid_mask = None
                z_broken_geom_gt = None

                if stage in {
                    'corr',
                    'z',
                    'z_calib',   # <<< NEW
                    'joint',
                }:

                    corr_target_active, corr_valid_mask,z_broken_geom_gt, = (
                        self._build_corr_target_from_clean_depth(
                            query_input_active=
                                query_input_filtered,

                            active_cam_indices=
                                active_cam_indices,

                            dense_depth_map_gt=
                                dense_depth_map_gt,

                            clean_camera2lidar=
                                original_camera2lidar,

                            broken_camera2lidar=
                                broken_camera2lidar,

                            camera_intrinsics=
                                broken_camera_intrinsics,
                        )
                    )

                    corr_supervision_mask = (
                        corr_valid_mask
                        &
                        query_unique_mask_filtered
                    )

                    if stage in {
                        'corr',
                        'joint',
                    }:
                        if self.corr_loss is None:

                            raise RuntimeError(
                                'Corr/Joint LGPC stage requires '
                                'corr_loss, but self.corr_loss is None.'
                            )


                        (
                            loss_corr,
                            corr_match_raw,
                            corr_cycle_raw,
                        ) = self.corr_loss(
                            corr_pred=
                                raw_corrs_active,

                            corr_target=
                                corr_target_active,

                            cycle=
                                cycle,

                            queries=
                                query_input_filtered,

                            cycle_mask=
                                corr_mask,

                            corr_valid_mask=
                                corr_supervision_mask,
                        )


                        losses[
                            'loss_corr'
                        ] = loss_corr


                        # --------------------------------------------------------
                        # Diagnostics only
                        # --------------------------------------------------------

                        losses[
                            'corr_match_raw'
                        ] = corr_match_raw

                        losses[
                            'corr_cycle_raw'
                        ] = corr_cycle_raw

                        losses[
                            'corr_valid_ratio'
                        ] = (
                            corr_supervision_mask
                            .float()
                            .mean()
                            .detach()
                        )


                        # ========================================================
                        # Pixel-domain correspondence diagnostics
                        #
                        # 1) corr_epe_px
                        #    CorrNet prediction vs geometric GT
                        #
                        # 2) identity_epe_px
                        #    No-correction baseline vs geometric GT
                        #
                        # 3) corr_recovery
                        #    How much of the identity error is recovered
                        # ========================================================

                        with torch.no_grad():

                            W_IMG = 1600.0
                            H_IMG = 900.0


                            # ====================================================
                            # A. CorrNet prediction -> original image pixels
                            #
                            # CorrNet output is on RIGHT SBS image:
                            #
                            # x_norm = 0.5 + u / (2*1600)
                            # y_norm = v / 900
                            # ====================================================

                            pred_u = (
                                (
                                    raw_corrs_active[
                                        ...,
                                        0
                                    ]
                                    - 0.5
                                )
                                * 2.0
                                * W_IMG
                            )

                            pred_v = (
                                raw_corrs_active[
                                    ...,
                                    1
                                ]
                                * H_IMG
                            )


                            # ====================================================
                            # B. Geometric GT correspondence -> pixels
                            # ====================================================

                            gt_u = (
                                (
                                    corr_target_active[
                                        ...,
                                        0
                                    ]
                                    - 0.5
                                )
                                * 2.0
                                * W_IMG
                            )

                            gt_v = (
                                corr_target_active[
                                    ...,
                                    1
                                ]
                                * H_IMG
                            )


                            # ====================================================
                            # C. CorrNet EPE
                            #
                            # distance:
                            #
                            #   prediction (u'_pred, v'_pred)
                            #             vs
                            #   GT         (u'_GT,   v'_GT)
                            # ====================================================

                            corr_epe = torch.sqrt(
                                (
                                    pred_u
                                    - gt_u
                                ) ** 2
                                +
                                (
                                    pred_v
                                    - gt_v
                                ) ** 2
                            )


                            # ====================================================
                            # D. Identity / No-Correction baseline
                            #
                            # query_input_filtered is LEFT SBS coordinate:
                            #
                            # q_x = (u / 1600) / 2
                            # q_y = v / 900
                            #
                            # If CorrNet performs NO correction,
                            # the corresponding point on the right image is
                            # simply the SAME original pixel (u,v).
                            #
                            # Therefore:
                            #
                            # identity_u = q_x * 2 * 1600
                            # identity_v = q_y * 900
                            # ====================================================

                            identity_u = (
                                query_input_filtered[
                                    ...,
                                    0
                                ]
                                * 2.0
                                * W_IMG
                            )

                            identity_v = (
                                query_input_filtered[
                                    ...,
                                    1
                                ]
                                * H_IMG
                            )


                            # ====================================================
                            # E. Identity EPE
                            #
                            # distance:
                            #
                            #   no-correction (u,v)
                            #             vs
                            #   GT            (u'_GT,v'_GT)
                            #
                            # This tells us how large the correspondence error
                            # caused by calibration corruption originally was.
                            # ====================================================

                            identity_epe = torch.sqrt(
                                (
                                    identity_u
                                    - gt_u
                                ) ** 2
                                +
                                (
                                    identity_v
                                    - gt_v
                                ) ** 2
                            )


                            # ====================================================
                            # F. IMPORTANT:
                            #
                            # Use EXACTLY the same valid correspondence mask
                            # for CorrNet and Identity baseline.
                            #
                            # Otherwise the comparison is not fair.
                            # ====================================================

                            if corr_valid_mask.any():

                                corr_epe_px = (
                                    corr_epe[
                                        corr_supervision_mask
                                    ]
                                    .mean()
                                )


                                identity_epe_px = (
                                    identity_epe[
                                        corr_supervision_mask
                                    ]
                                    .mean()
                                )

                            else:

                                corr_epe_px = (
                                    corr_epe
                                    .new_tensor(
                                        0.0
                                    )
                                )

                                identity_epe_px = (
                                    identity_epe
                                    .new_tensor(
                                        0.0
                                    )
                                )

                        # ============================================================
                        # Corr deterministic evaluation cache
                        #
                        # Diagnostic only.
                        # Does NOT participate in training loss.
                        # Only populated when model.eval() is active.
                        # ============================================================

                        if not self.training:

                            if corr_supervision_mask.any():

                                eval_mask = (
                                    corr_supervision_mask
                                )

                                self._corr_eval_cache = {

                                    'epe_px':
                                        corr_epe[
                                            eval_mask
                                        ].detach().cpu(),

                                    'identity_epe_px':
                                        identity_epe[
                                            eval_mask
                                        ].detach().cpu(),

                                    'u_abs_px':
                                        torch.abs(
                                            pred_u
                                            - gt_u
                                        )[
                                            eval_mask
                                        ].detach().cpu(),

                                    'v_abs_px':
                                        torch.abs(
                                            pred_v
                                            - gt_v
                                        )[
                                            eval_mask
                                        ].detach().cpu(),

                                }

                            else:

                                empty = torch.empty(
                                    0,
                                    dtype=torch.float32,
                                )

                                self._corr_eval_cache = {
                                    'epe_px': empty,
                                    'identity_epe_px': empty,
                                    'u_abs_px': empty,
                                    'v_abs_px': empty,
                                }
                        # ====================================================
                        # G. Correspondence recovery ratio
                        #
                        # recovery =
                        #
                        #      1 - CorrNet_EPE / Identity_EPE
                        #
                        # Example:
                        #
                        # Identity = 100 px
                        # CorrNet  =  30 px
                        #
                        # recovery = 0.70 = 70 %
                        #
                        # negative:
                        # CorrNet made correspondence worse
                        # ====================================================

                        corr_recovery = torch.where(
                            identity_epe_px > 1e-6,

                            1.0
                            - (
                                corr_epe_px
                                / identity_epe_px
                            ),

                            torch.zeros_like(
                                identity_epe_px
                            ),
                        )


                        # ====================================================
                        # H. Logging diagnostics
                        #
                        # IMPORTANT:
                        #
                        # These names do NOT contain "loss",
                        # so parse_losses() will NOT add them
                        # to optimizer loss.
                        # ====================================================

                        losses[
                            'corr_epe_px'
                        ] = corr_epe_px


                        losses[
                            'identity_epe_px'
                        ] = identity_epe_px


                        losses[
                            'corr_recovery'
                        ] = corr_recovery

                        
                        losses[
                            'corr_valid_count'
                        ] = (
                            corr_supervision_mask
                            .float()
                            .sum()
                            .detach()
                        )

                        losses[
                            'corr_total_count'
                        ] = torch.tensor(
                            float(
                                corr_supervision_mask.numel()
                            ),
                            device=corr_valid_mask.device,
                        )


                # ============================================================
                # 3. CORR-ONLY STAGE ENDS HERE
                #
                # Do NOT execute:
                #   ZEstimator
                #   CalibHead
                #
                # This is the key memory saving for Corr pretraining.
                # ============================================================

                if (
                    self.is_lgpc_stage1
                    and stage == 'corr'
                ):

                    return losses


                # ============================================================
                # 4. Corr output -> original broken-image pixel
                # ============================================================

                r_x = (
                    (
                        raw_corrs_active[
                            ...,
                            0
                        ]
                        - 0.5
                    )
                    * 2.0
                    * 1600.0
                )

                r_y = (
                    raw_corrs_active[
                        ...,
                        1
                    ]
                    * 900.0
                )


                uv_pixels_from_corr = (
                    torch.stack(
                        [
                            r_x,
                            r_y,
                        ],
                        dim=-1,
                    )
                )

                # # ============================================================
                # # 4. TEMP ORACLE CORR UV TEST
                # #
                # # ZEstimator에 CorrNet prediction 대신
                # # geometric GT correspondence를 입력
                # # ============================================================

                # oracle_corrs_active = (
                #     corr_target_active
                #     .detach()
                # )

                # r_x = (
                #     (
                #         oracle_corrs_active[
                #             ...,
                #             0
                #         ]
                #         - 0.5
                #     )
                #     * 2.0
                #     * 1600.0
                # )

                # r_y = (
                #     oracle_corrs_active[
                #         ...,
                #         1
                #     ]
                #     * 900.0
                # )

                # uv_pixels_from_corr = (
                #     torch.stack(
                #         [
                #             r_x,
                #             r_y,
                #         ],
                #         dim=-1,
                #     )
                # )


                # ============================================================
                # 5. BROKEN depth for ZEstimator
                # ============================================================

                depth_map_reshaped_BROKEN = (
                    dense_depth_map.view(
                        B * N,
                        900,
                        1600,
                    )
                )

                depth_map_active_BROKEN = (
                    depth_map_reshaped_BROKEN[
                        active_cam_indices
                    ]
                )


                # ============================================================
                # 6. ZEstimator Forward
                #
                # z / joint:
                #     gradient ON
                #
                # calib:
                #     ZEstimator is frozen/no_grad
                # ============================================================

                if stage in {
                    'z',
                    'z_calib',    # <<< NEW
                    'joint',
                }:

                    esitmated_z_active = (
                        self.z_estimator(
                            uv_sbs_normalized=
                                raw_corrs_active,

                            uv_orig_pixels=
                                uv_pixels_from_corr,

                            depth_map=
                                depth_map_active_BROKEN,

                            enc_out=
                                enc_out_active_4d,
                        )
                    )

                else:

                    # --------------------------------------------------------
                    # calib stage:
                    # CorrNet and ZEstimator only provide fixed inputs.
                    # --------------------------------------------------------

                    with torch.no_grad():

                        esitmated_z_active = (
                            self.z_estimator(
                                uv_sbs_normalized=
                                    raw_corrs_active,

                                uv_orig_pixels=
                                    uv_pixels_from_corr,

                                depth_map=
                                    depth_map_active_BROKEN,

                                enc_out=
                                    enc_out_active_4d,
                            )
                        )

                # ============================================================
                # 7. Z outputs
                # ============================================================

                z_estimated_active = (
                    esitmated_z_active[
                        'z_estimated_real'
                    ]
                )

                z_estimated_hybrid = (
                    esitmated_z_active[
                        'depth'
                    ]
                )

                # ============================================================
                # Z inputs for CalibHead
                #
                # z_pred_for_calib:
                #     final ZEstimator V3 prediction
                #
                # z_raw_for_calib:
                #     raw center LiDAR depth
                #
                # calib / z_calib:
                #     calibration loss must not update ZEstimator
                #
                # joint:
                #     allow end-to-end gradient
                # ============================================================

                if stage == 'joint':

                    z_pred_for_calib = (
                        z_estimated_hybrid
                    )

                    z_raw_for_calib = (
                        esitmated_z_active[
                            'z_lidar_real'
                        ]
                    )

                else:

                    z_pred_for_calib = (
                        z_estimated_hybrid
                        .detach()
                    )

                    z_raw_for_calib = (
                        esitmated_z_active[
                            'z_lidar_real'
                        ]
                        .detach()
                    )


                # ============================================================
                # Z supervision
                #
                # z:
                #     ZEstimator training
                #
                # z_calib:
                #     ZEstimator + CalibHead concurrent training
                #
                # joint:
                #     full LGPC training
                #
                # calib:
                #     ZEstimator frozen, therefore no Z GT is required
                # ============================================================

                if stage in {
                    'z',
                    'z_calib',
                    'joint',
                }:

                    z_target_broken = (
                        z_broken_geom_gt
                        .detach()
                    )

                    z_prediction = (
                        z_estimated_active
                        .squeeze(-1)
                    )

                    pred_u = (
                        uv_pixels_from_corr[
                            ...,
                            0
                        ]
                    )

                    pred_v = (
                        uv_pixels_from_corr[
                            ...,
                            1
                        ]
                    )

                    pred_uv_valid = (
                        (pred_u >= 0.0)
                        & (pred_u < 1600.0)
                        & (pred_v >= 0.0)
                        & (pred_v < 900.0)
                    )

                    valid_z_mask = (
                        corr_valid_mask
                        & pred_uv_valid
                        & torch.isfinite(
                            z_target_broken
                        )
                        & torch.isfinite(
                            z_prediction
                        )
                        & (
                            z_target_broken
                            > 1e-3
                        )
                    )

                    # ============================================================
                    # Diagnostic:
                    #
                    # OLD Z target vs NEW geometry-consistent Z target
                    #
                    # OLD:
                    #   z_lidar_real
                    #   = broken depth associated with Corr predicted UV
                    #
                    # NEW:
                    #   z_broken_geom_gt
                    #   = depth of the SAME physical 3D point used for
                    #     geometric CorrNet GT.
                    #
                    # IMPORTANT:
                    # - diagnostic only
                    # - detached
                    # - NOT included in optimizer loss
                    # ============================================================

                    if stage in {
                        'z',
                        'z_calib',
                        'joint',
                    }:

                        z_old_target_broken = (
                            esitmated_z_active[
                                'z_lidar_real'
                            ]
                            .detach()
                            .squeeze(-1)
                        )


                        # --------------------------------------------------------
                        # Fair comparison mask
                        #
                        # Both old/new targets must be:
                        # - based on valid geometric correspondence
                        # - Corr prediction inside the image
                        # - finite
                        # - positive depth
                        # --------------------------------------------------------

                        z_target_compare_mask = (
                            corr_valid_mask
                            & pred_uv_valid

                            & torch.isfinite(
                                z_old_target_broken
                            )

                            & torch.isfinite(
                                z_target_broken
                            )

                            & (
                                z_old_target_broken
                                > 1e-3
                            )

                            & (
                                z_target_broken
                                > 1e-3
                            )
                        )


                        with torch.no_grad():

                            if z_target_compare_mask.any():

                                # ========================================================
                                # 1. RAW local LiDAR depth error
                                #
                                # z_old_target_broken:
                                #   Corr predicted UV에서 실제로 읽은 broken depth
                                #
                                # z_target_broken:
                                #   same-physical-point geometry GT
                                # ========================================================

                                z_raw_error_same_mask = torch.abs(
                                    z_old_target_broken[
                                        z_target_compare_mask
                                    ]
                                    -
                                    z_target_broken[
                                        z_target_compare_mask
                                    ]
                                )


                                z_raw_mae_same_mask_m = (
                                    z_raw_error_same_mask
                                    .mean()
                                )


                                # ========================================================
                                # 2. ZEstimator prediction error
                                #
                                # IMPORTANT:
                                # RAW depth와 EXACTLY SAME mask를 사용한다.
                                #
                                # 따라서 아래 두 값은 직접 비교 가능:
                                #
                                #   z_raw_mae_same_mask_m
                                #   z_pred_mae_same_mask_m
                                # ========================================================

                                z_pred_error_same_mask = torch.abs(
                                    z_prediction[
                                        z_target_compare_mask
                                    ]
                                    -
                                    z_target_broken[
                                        z_target_compare_mask
                                    ]
                                )


                                z_pred_mae_same_mask_m = (
                                    z_pred_error_same_mask
                                    .mean()
                                )


                                # ========================================================
                                # 3. Depth recovery ratio
                                #
                                # recovery =
                                #
                                #   1 - Pred_Error / Raw_Error
                                #
                                # Example:
                                #
                                # raw  = 5.0 m
                                # pred = 3.0 m
                                #
                                # recovery = 1 - 3/5 = 0.40
                                #
                                # => raw depth error의 40%를 복구
                                #
                                # > 0 : ZEstimator better
                                # = 0 : same
                                # < 0 : ZEstimator worse
                                # ========================================================

                                z_depth_recovery = torch.where(
                                    z_raw_mae_same_mask_m > 1e-6,

                                    1.0
                                    -
                                    (
                                        z_pred_mae_same_mask_m
                                        /
                                        z_raw_mae_same_mask_m
                                    ),

                                    torch.zeros_like(
                                        z_raw_mae_same_mask_m
                                    ),
                                )


                                # ========================================================
                                # Existing target disagreement diagnostics
                                #
                                # 이것은 사실상 RAW depth error의 MAE/RMSE이다.
                                # 기존 로그 호환성을 위해 유지.
                                # ========================================================

                                z_target_disagreement = (
                                    z_raw_error_same_mask
                                )


                                z_target_disagree_mae_m = (
                                    z_target_disagreement
                                    .mean()
                                )


                                z_target_disagree_rmse_m = (
                                    torch.sqrt(
                                        torch.mean(
                                            z_target_disagreement ** 2
                                        )
                                    )
                                )


                                z_target_compare_ratio = (
                                    z_target_compare_mask
                                    .float()
                                    .mean()
                                )

                                # ========================================================
                                # V3: Neighborhood anchor quality
                                # ========================================================

                                z_anchor_real = (
                                    esitmated_z_active[
                                        'z_anchor_real'
                                    ]
                                    .detach()
                                    .squeeze(-1)
                                )


                                z_anchor_error_same_mask = torch.abs(
                                    z_anchor_real[
                                        z_target_compare_mask
                                    ]
                                    -
                                    z_target_broken[
                                        z_target_compare_mask
                                    ]
                                )


                                z_anchor_mae_same_mask_m = (
                                    z_anchor_error_same_mask
                                    .mean()
                                )


                                z_anchor_recovery = torch.where(
                                    z_raw_mae_same_mask_m > 1e-6,

                                    1.0
                                    -
                                    (
                                        z_anchor_mae_same_mask_m
                                        /
                                        z_raw_mae_same_mask_m
                                    ),

                                    torch.zeros_like(
                                        z_raw_mae_same_mask_m
                                    ),
                                )


                            else:

                                z_target_disagree_mae_m = (
                                    z_prediction
                                    .new_tensor(0.0)
                                )

                                z_target_disagree_rmse_m = (
                                    z_prediction
                                    .new_tensor(0.0)
                                )

                                z_target_compare_ratio = (
                                    z_prediction
                                    .new_tensor(0.0)
                                )


                                # NEW
                                z_raw_mae_same_mask_m = (
                                    z_prediction
                                    .new_tensor(0.0)
                                )

                                z_pred_mae_same_mask_m = (
                                    z_prediction
                                    .new_tensor(0.0)
                                )

                                z_depth_recovery = (
                                    z_prediction
                                    .new_tensor(0.0)
                                )

                                z_anchor_mae_same_mask_m = (
                                    z_prediction
                                    .new_tensor(0.0)
                                )
                                z_anchor_recovery = (
                                    z_prediction
                                    .new_tensor(0.0)
                                )
                            

                        # --------------------------------------------------------
                        # Diagnostic names intentionally DO NOT contain "loss"
                        # --------------------------------------------------------

                        losses[
                            'z_target_disagree_mae_m'
                        ] = (
                            z_target_disagree_mae_m
                        )

                        losses[
                            'z_target_disagree_rmse_m'
                        ] = (
                            z_target_disagree_rmse_m
                        )


                        losses[
                            'z_target_compare_ratio'
                        ] = (
                            z_target_compare_ratio
                        )

                        losses[
                            'z_raw_mae_same_mask_m'
                        ] = (
                            z_raw_mae_same_mask_m
                        )


                        losses[
                            'z_pred_mae_same_mask_m'
                        ] = (
                            z_pred_mae_same_mask_m
                        )


                        losses[
                            'z_depth_recovery'
                        ] = (
                            z_depth_recovery
                        )

                        losses[
                            'z_anchor_mae_same_mask_m'
                        ] = (
                            z_anchor_mae_same_mask_m
                        )


                        losses[
                            'z_anchor_recovery'
                        ] = (
                            z_anchor_recovery
                        )


                        losses[
                            'z_neighbor_valid_ratio'
                        ] = (
                            esitmated_z_active[
                                'neighbor_valid_ratio'
                            ]
                            .detach()
                            .mean()
                        )


                        losses[
                            'z_neighbor_weight_max'
                        ] = (
                            esitmated_z_active[
                                'neighbor_weight_max'
                            ]
                            .detach()
                            .mean()
                        )


                    if (
                        self.is_lgpc_stage1
                        and stage == 'z'
                        and not hasattr(
                            self,
                            '_z_mode_debug_done'
                        )
                    ):

                        corr_trainable = sum(
                            p.numel()
                            for p in self.corr.parameters()
                            if p.requires_grad
                        )

                        z_trainable = sum(
                            p.numel()
                            for p in self.z_estimator.parameters()
                            if p.requires_grad
                        )

                        calib_trainable = sum(
                            p.numel()
                            for p in self.calib_head.parameters()
                            if p.requires_grad
                        )

                        img_trainable = sum(
                            p.numel()
                            for p in self.img_backbone.parameters()
                            if p.requires_grad
                        )

                        print(
                            '\n'
                            '========================================='
                        )

                        print('[Z STAGE MODE DEBUG]')

                        print(
                            'BEVFusion.training =',
                            self.training
                        )

                        print(
                            'Corr.training =',
                            self.corr.training,
                            '/ trainable params =',
                            corr_trainable
                        )

                        print(
                            'ZEstimator.training =',
                            self.z_estimator.training,
                            '/ trainable params =',
                            z_trainable
                        )

                        print(
                            'CalibHead.training =',
                            self.calib_head.training,
                            '/ trainable params =',
                            calib_trainable
                        )

                        print(
                            'img_backbone.training =',
                            self.img_backbone.training,
                            '/ trainable params =',
                            img_trainable
                        )

                        print(
                            '=========================================\n'
                        )

                        self._z_mode_debug_done = True
                    # ============================================================
                    # 8. Z loss
                    #
                    # IMPORTANT:
                    #
                    # Only z and joint stages calculate L_z.
                    # ============================================================

                    if stage in {
                        'z',
                        'z_calib',     # <<< NEW
                        'joint',
                    }:

                        if valid_z_mask.any():

                            loss_z_estimation = (
                                F.smooth_l1_loss(
                                    z_prediction[
                                        valid_z_mask
                                    ],

                                    z_target_broken[
                                        valid_z_mask
                                    ],

                                    reduction='mean',
                                    beta=1.0,
                                )
                            )

                            losses[
                                'loss_z_estimation'
                            ] = loss_z_estimation

                        else:

                            # Keep valid autograd graph
                            losses[
                                'loss_z_estimation'
                            ] = (
                                z_prediction.sum()
                                * 0.0
                            )

                    # ============================================================
                    # ZEstimator physical diagnostics
                    #
                    # z_prediction / z_target_broken are real depth values.
                    # ============================================================
                    if stage in {
                        'z',
                        'z_calib',
                        'joint',
                    }:
                        with torch.no_grad():

                            if valid_z_mask.any():

                                z_error = (
                                    z_prediction[
                                        valid_z_mask
                                    ]
                                    -
                                    z_target_broken[
                                        valid_z_mask
                                    ]
                                )

                                # Mean Absolute Error [m]
                                z_mae_m = (
                                    torch.abs(
                                        z_error
                                    )
                                    .mean()
                                )

                                # Root Mean Squared Error [m]
                                z_rmse_m = torch.sqrt(
                                    torch.mean(
                                        z_error ** 2
                                    )
                                )

                                z_valid_ratio = (
                                    valid_z_mask
                                    .float()
                                    .mean()
                                )

                            else:

                                z_mae_m = (
                                    z_prediction
                                    .new_tensor(0.0)
                                )

                                z_rmse_m = (
                                    z_prediction
                                    .new_tensor(0.0)
                                )

                                z_valid_ratio = (
                                    z_prediction
                                    .new_tensor(0.0)
                                )


                            losses[
                                'z_mae_m'
                            ] = z_mae_m

                            losses[
                                'z_rmse_m'
                            ] = z_rmse_m

                            losses[
                                'z_valid_ratio'
                            ] = z_valid_ratio    


                    # ============================================================
                    # 9. Z-ONLY STAGE ENDS HERE
                    #
                    # Do NOT execute CalibHead.
                    # ============================================================

                    if (
                        self.is_lgpc_stage1
                        and stage == 'z'
                    ):

                        return losses

                # ============================================================
                # 11. Build Corr + Z correspondence
                #
                # raw_corrs_active:
                #     [NumActive, Q, 2]
                #
                # z_for_calib:
                #     [NumActive, Q, 1]
                #
                # result:
                #     [NumActive, Q, 3]
                #     = [u', v', z']
                # ============================================================

                corrs_3d_active = torch.cat(

                    [
                        raw_corrs_active,
                        z_pred_for_calib,
                    ],

                    dim=-1,
                )

                # ============================================================
                # CalibHead point-valid mask
                #
                # IMPORTANT:
                # CorrNet itself still processed all 200 slots.
                #
                # CalibHead will use only original detector queries.
                # ============================================================

                corr_finite = (
                    torch.isfinite(
                        raw_corrs_active
                    )
                    .all(
                        dim=-1
                    )
                )


                corr_in_range = (

                    (
                        raw_corrs_active[
                            ...,
                            0
                        ]
                        >= 0.5
                    )

                    & (
                        raw_corrs_active[
                            ...,
                            0
                        ]
                        <= 1.0
                    )

                    & (
                        raw_corrs_active[
                            ...,
                            1
                        ]
                        >= 0.0
                    )

                    & (
                        raw_corrs_active[
                            ...,
                            1
                        ]
                        <= 1.0
                    )
                )


                calib_point_valid_mask = (

                    query_unique_mask_filtered

                    & corr_finite

                    & corr_in_range
                )


                # ============================================================
                # 12. Build runtime-available Z reliability
                #
                # IMPORTANT:
                #
                # This uses only inference-available ZEstimator V3 outputs.
                # No GT information is included.
                #
                # Expected:
                #     [NumActive, Q, 4]
                # ============================================================

                z_reliability_active = (
                    self._build_calib_z_reliability(
                        esitmated_z_active
                    )
                )


                # ============================================================
                # z_calib V1:
                #
                # Calib loss must NOT modify:
                #
                #   - neighborhood candidate scoring
                #   - Z anchor construction
                #   - ZEstimator reliability path
                #
                # ZEstimator is trained only by loss_z_estimation.
                # ============================================================

                if stage == 'z_calib':

                    z_reliability_active = (
                        z_reliability_active
                        .detach()
                    )


                # ============================================================
                # 13. CalibHead V1
                # ============================================================
                (
                    pred_delta_6dof_active,
                    calib_diag,
                ) = self.calib_head(

                    enc_out=
                        enc_out_active_4d,

                    query_input=
                        query_input_filtered,

                    corrs_pred_3d=
                        corrs_3d_active,

                    z_reliability=
                        z_reliability_active,

                    # NEW: Adaptive Z Gate
                    z_raw=
                        z_raw_for_calib,

                    # NEW: K-aware geometry
                    camera_intrinsics=
                        camera_intrinsics_active,

                    # Duplicate / invalid Corr exclusion
                    point_valid_mask=
                        calib_point_valid_mask,
                )

                # ============================================================
                # 14. CalibHead diagnostics
                #
                # IMPORTANT:
                #
                # These are logging metrics only.
                # Their names do NOT contain "loss", so MMEngine will
                # not include them in the optimization loss.
                # ============================================================

                for key, value in calib_diag.items():

                    losses[
                        key
                    ] = value.detach()

                # ============================================================
                # 12. Store output only when downstream PCC is needed
                #
                # LGPC Stage1 will return before BEVFusion/RRRF.
                # ============================================================

                if (
                    not self.is_lgpc_stage1
                    and B == 1
                ):

                    raw_corrs[
                        active_cam_indices
                    ] = raw_corrs_active

                    enc_out[
                        active_cam_indices
                    ] = enc_out_active_3d

                    pred_delta_6dof[
                        0,
                        active_cam_indices
                    ] = pred_delta_6dof_active


                    esitmated_uvz_active = (
                        torch.cat(
                            [
                                uv_pixels_from_corr,
                                esitmated_z_active[
                                    'depth'
                                ],
                            ],
                            dim=-1,
                        )
                    )

                    esitmated_uvz[
                        active_cam_indices
                    ] = esitmated_uvz_active


                # ============================================================
                # 13. Calibration supervision
                # ============================================================

                pred_rot_filtered = (
                    pred_delta_6dof_active[
                        ...,
                        :3
                    ]
                )

                pred_trans_filtered = (
                    pred_delta_6dof_active[
                        ...,
                        3:
                    ]
                )


                gt_rot_filtered = (
                    gt_delta_rot[
                        :,
                        active_cam_indices
                    ]
                    .squeeze(0)
                )

                gt_trans_filtered = (
                    gt_delta_trans[
                        :,
                        active_cam_indices
                    ]
                    .squeeze(0)
                )


                R_pred_calib = (
                    axis_angle_to_matrix(
                        pred_rot_filtered
                    )
                )

                R_gt_calib = (
                    axis_angle_to_matrix(
                        gt_rot_filtered
                    )
                )


                # ============================================================
                # 14. Calib losses
                #
                # calib / joint only reach this point.
                # ============================================================

                losses[
                    'loss_calib_rot'
                ] = (
                    identity_matrix_loss(
                        R_pred_calib,
                        R_gt_calib,
                    )
                    * 100.0
                )


                losses[
                    'loss_calib_trans'
                ] = (
                    F.smooth_l1_loss(
                        pred_trans_filtered,
                        gt_trans_filtered,
                        reduction='mean',
                    )
                    * 50.0
                )

                # ============================================================
                # Physical Calib diagnostics
                # ============================================================

                with torch.no_grad():

                    rot_error_rad = (
                        geodesic_distance_loss(
                            R_pred_calib,
                            R_gt_calib,
                        )
                    )

                    calib_rot_err_deg = (
                        rot_error_rad
                        .mean()
                        * (
                            180.0
                            / math.pi
                        )
                    )


                    trans_error = (
                        pred_trans_filtered
                        - gt_trans_filtered
                    )


                    calib_trans_mae_m = (
                        torch.abs(
                            trans_error
                        )
                        .mean()
                    )


                    calib_trans_l2_m = (
                        torch.linalg.norm(
                            trans_error,
                            dim=-1,
                        )
                        .mean()
                    )


                losses[
                    'calib_rot_err_deg'
                ] = (
                    calib_rot_err_deg
                )


                losses[
                    'calib_trans_mae_m'
                ] = (
                    calib_trans_mae_m
                )


                losses[
                    'calib_trans_l2_m'
                ] = (
                    calib_trans_l2_m
                )
            
            # ============================================================
            # Stage-1 LGPC standalone training ends here.
            #
            # Objective:
            #   loss_corr
            #   loss_z_estimation
            #   loss_calib_rot
            #   loss_calib_trans
            #
            # Do NOT train RRRF / BEVFusion detector here.
            # ============================================================


            # ============================================================
            # LGPC Stage-1 safety/debug:
            # check whether a REAL optimization loss was produced.
            # ============================================================

            if self.is_lgpc_stage1:

                has_optimization_loss = any(
                    'loss' in key
                    for key in losses.keys()
                )


                if not has_optimization_loss:

                    rank = (
                        dist.get_rank()
                        if (
                            dist.is_available()
                            and dist.is_initialized()
                        )
                        else 0
                    )


                    print(
                        '\n'
                        '=========================================\n'
                        '[LGPC NO-LOSS BATCH DEBUG]\n'
                        f'rank = {rank}\n'
                        f'stage = {stage}\n'
                        f'active_cam_indices = '
                        f'{active_cam_indices.detach().cpu().tolist()}\n'
                        f'num_rois = {rois_center.shape[0]}\n'
                        f'loss keys = {list(losses.keys())}\n'
                        '=========================================\n'
                    )


                return losses

            enc_out = enc_out.permute(0, 2, 1).reshape(-1, d_model, feat_h, feat_w)
            
            pred_delta_rot = pred_delta_6dof[..., :3]
            pred_delta_trans = pred_delta_6dof[..., 3:]



            # # ##### 검증용 display ######
            # from .imageprocessing_unit import draw_correspondences
            # # gt_corrs = torch.cat([query_input,corr_target],dim=-1)
            # pred_corrs = torch.cat([query_input,raw_corrs],dim=-1)
            # # vis_step_counter는 __init__에서 0으로 초기화 되어야 합니다.
            # self.vis_step_counter += 1
            # for cid in range(6):
            #     # idx = id_to_idx[cid.item()]
            #     # draw_correspondences(
            #     #     trimed_corrs = gt_corrs[cid][:10,...],  # 첫 번째 배치 선택
            #     #     sbs_img=sbs_img[cid],
            #     #     save_path='correspondence_visualization_gt.jpg'
            #     # )
            #     bboxes_for_this_view = detections_2d_orig_coords[cid]
            #     draw_correspondences(
            #         trimed_corrs = pred_corrs[cid][:20,...],  # 첫 번째 배치 선택
            #         sbs_img=sbs_img.view(B*N,C,H,W)[cid],
            #         save_path='correspondence_visualization_pred.jpg',
            #         bboxes_to_draw = bboxes_for_this_view, # 원본 좌표계 BBox 전달
            #         score_thr = 0.4
            #     )
            #     # # --- 2. 원본 vs 증강 BBox 비교 시각화 저장 (요청하신 부분) ---
            #     # save_batch_predictions_to_file(
            #     #         batch_inputs_dict=batch_inputs_dict,
            #     #         reshaped_data_samples=reshaped_data_samples,
            #     #         augmented_preds_list=detections_2d,
            #     #         original_preds_list=detections_2d_orig_coords,
            #     #         current_step=self.vis_step_counter,
            #     #         save_dir='work_dirs/my_exp/vis_results',
            #     #         view_index=cid, # 루프 변수 cid를 view_index로 사용
            #     #         score_thr=0.4
            #     #     )
            #     print ("end")

            corrected_calib_dict = self._get_corrected_calib_from_prediction(
                            pred_delta_rot,
                            pred_delta_trans,
                            broken_camera2lidar,
                            broken_camera_intrinsics
                        )

            # ✨ '보정된' lidar2imag를 사용하여 3D 좌표 변환 수행
            # ✨ (esitmated_uvz는 이제 if/else 로직에 의해 올바르게 채워졌습니다)
            det_xyz = self.uvz_to_lidar_xyz(esitmated_uvz, corrected_calib_dict['lidar2img'])
        
            # ... (이하 Chamfer Loss) ...
            # gt_lidar_points = batch_inputs_dict['points'][0][:, :3]
            # gt_lidar_points = gt_lidar_points.unsqueeze(0).to(target_device)
            
            # num_queries_per_cam = det_xyz.shape[1]
            # det_xyz_batch = det_xyz.reshape(B, N * num_queries_per_cam, 3)

            # if gt_lidar_points.shape[1] > 2048:
            #     indices = torch.randperm(gt_lidar_points.shape[1], device=target_device)[:2048]
            #     gt_lidar_points_sampled = gt_lidar_points[:, indices, :]
            # else:
            #     gt_lidar_points_sampled = gt_lidar_points
                
            # if det_xyz_batch.shape[1] > 2048:
            #     indices = torch.randperm(det_xyz_batch.shape[1], device=target_device)[:2048]
            #     det_xyz_batch_sampled = det_xyz_batch[:, indices, :]
            # else:
            #     det_xyz_batch_sampled = det_xyz_batch

            # loss_chamfer = chamfer_distance(det_xyz_batch_sampled, gt_lidar_points_sampled).mean()

            # losses['loss_chamfer_xyz'] = loss_chamfer * 0.005

            det_xyz_ref = det_xyz.clone()
            det_xyz_ref[..., 0:1] = (det_xyz_ref[..., 0:1] - self.pc_range[0]) / (
                    self.pc_range[3] - self.pc_range[0])
            det_xyz_ref[..., 1:2] = (det_xyz_ref[..., 1:2] - self.pc_range[1]) / (
                    self.pc_range[4] - self.pc_range[1])
            det_xyz_ref[..., 2:3] = (det_xyz_ref[..., 2:3] - self.pc_range[2]) / (
                    self.pc_range[5] - self.pc_range[2])
            det_xyz_ref_clamped = det_xyz_ref.clamp(min=0, max=1)
            
            det_feat_sampled = self._sample_features_from_grid(feature_map=enc_out, coords=query_input)
            det_xyz_proc, det_feat_proc = self._prepare_camera_proposals(det_xyz_ref_clamped,det_feat_sampled,B=B,N_cam=N)

            feats = self.extract_feat(batch_inputs_dict=batch_inputs_dict,
                                    batch_input_metas=batch_input_metas,
                                    corrected_calib=corrected_calib_dict,
                                    precomputed_img_feats=img_feats)

            if self.with_bbox_head:
                bbox_loss = self.bbox_head.loss(
                                feats, 
                                det_xyz_proc, 
                                det_feat_proc, 
                                batch_data_samples,
                                pred_delta_rot=pred_delta_rot,
                                pred_delta_trans=pred_delta_trans,
                                gt_delta_rot=gt_delta_rot,
                                gt_delta_trans=gt_delta_trans
                            )
            
            # --- ✨ 2. 손실과 예측값 분리 ---
            # 시각화를 위해 예측값을 별도 변수로 빼내고, 딕셔너리에서 제거
            pred_delta_rot_batch = bbox_loss.pop('pred_delta_rot')
            pred_delta_trans_batch = bbox_loss.pop('pred_delta_trans')

            losses.update(bbox_loss)

            # # --- 4. ✨ VERIFICATION 2: 2nd Stage 시각적 검증 ---
            # if hasattr(self, 'training_step') and self.training_step % 50 == 0:
            #     with torch.no_grad():
                    
            #         # --- 헬퍼 함수: 포인트 투영 ---
            #         def project_points(points_tensor, P_matrix, height, width):
            #             points_h = torch.cat([points_tensor[:, :3], torch.ones_like(points_tensor[:, :1])], dim=-1).to(P_matrix.device)
            #             points_proj_raw = (P_matrix @ points_h.T).T
            #             uv = points_proj_raw[:, :2] / (points_proj_raw[:, 2:3] + 1e-8)
            #             z = points_proj_raw[:, 2:3]
            #             points_proj = torch.cat([uv, z], dim=-1)
            #             mask = (points_proj[:, 0] >= 0) & (points_proj[:, 0] < width) & \
            #                    (points_proj[:, 1] >= 0) & (points_proj[:, 1] < height) & (points_proj[:, 2] > 0)
            #             return points_proj[mask].cpu().numpy()

            #         # --- 1. 시각화에 필요한 데이터 준비 ---
            #         cam_idx = 0
            #         batch_idx = 0 # (첫 번째 배치 샘플 사용)
                    
            #         img_tensor_chw = batch_inputs_dict['img_original'][batch_idx][cam_idx].cpu().numpy()
            #         img_for_vis = img_tensor_chw.transpose(1, 2, 0)
            #         if img_for_vis.dtype in [np.float32, np.float64] and img_for_vis.max() > 1.0:
            #             img_for_vis = img_for_vis / 255.0
                        
            #         points_for_vis_raw = batch_inputs_dict['points_original'][batch_idx].tensor
            #         h, w = img_for_vis.shape[:2]
                    
            #         K = broken_camera_intrinsics[batch_idx, cam_idx, :3, :3]
                    
            #         # --- 2. 네 가지 상태의 Extrinsics (T_c2l, 4x4) 준비 ---
                    
            #         # a) Ground Truth (정답)
            #         T_c2l_orig = original_camera2lidar[batch_idx, cam_idx]
                    
            #         # b) Broken (문제)
            #         T_c2l_broken = broken_camera2lidar[batch_idx, cam_idx]

            #         # --- ⬇️⬇️⬇️ [수정된 부분] ⬇️⬇️⬇️ ---
            #         # c) 1st-Stage Correction (from calib_head)
            #         # (이 딕셔너리는 bbox_head 호출 전에 1st-stage가 계산한 결과)
            #         T_c2l_corr_1st = corrected_calib_dict['cam2lidar'][batch_idx, cam_idx]
                    
            #         # c-1) Head에서 예측한 Delta 값 (배치에서 0번째 샘플 추출)
            #         pred_delta_rot = pred_delta_rot_batch[batch_idx] # [3]
            #         pred_delta_trans = pred_delta_trans_batch[batch_idx] # [3]
                    
            #         # c-2) 2nd-Stage Delta (Rotation, Translation) -> 4x4 행렬 (Delta T_2nd)
            #         delta_R = axis_angle_to_matrix(pred_delta_rot)
            #         delta_T_2nd_stage = torch.eye(4, device=K.device)
            #         delta_T_2nd_stage[:3, :3] = delta_R
            #         delta_T_2nd_stage[:3, 3] = pred_delta_trans

            #         try:
            #             # T_Correction = T_Error_inverse
            #             T_2nd_Correction = torch.linalg.inv(delta_T_2nd_stage)
            #         except torch.linalg.LinAlgError:
            #             # 역행렬 계산이 불가능한 경우 단위 행렬로 대체 (보정 없음)
            #             T_2nd_Correction = torch.eye(4, dtype=delta_T_2nd_stage.dtype, device=delta_T_2nd_stage.device).expand_as(delta_T_2nd_stage)
                    
            #         # c-3) ✨[핵심 수정] 1st Stage Correction에 2nd Stage Delta를 적용
            #         # 1단계 보정 행렬 T_c2l_corr_1st에 2단계 보정 행렬 delta_T_2nd_stage를 곱합니다.
            #         T_c2l_corr_2nd = torch.matmul(T_2nd_Correction, T_c2l_corr_1st)

            #         # --- 3. 세 가지 상태의 투영 행렬(P = K @ T_l2c, 3x4) 계산 ---
                    
            #         T_l2c_orig = torch.inverse(T_c2l_orig)[:3, :]
            #         P_orig = K @ T_l2c_orig
                    
            #         T_l2c_broken = torch.inverse(T_c2l_broken)[:3, :]
            #         P_broken = K @ T_l2c_broken

            #         # [NEW] 1st Stage Projection
            #         T_l2c_corr_1st = torch.inverse(T_c2l_corr_1st)[:3, :]
            #         P_corr_1st = K @ T_l2c_corr_1st
                    
            #         # ✨[핵심] 위에서 계산한 2nd stage T_c2l_corr를 사용
            #         T_l2c_corr_2nd = torch.inverse(T_c2l_corr_2nd)[:3, :]
            #         P_corr_2nd = K @ T_l2c_corr_2nd

            #         # --- 3.5. 포인트 필터링 및 샘플링 ---
            #         z_threshold = 0.3
            #         height_mask = points_for_vis_raw[:, 2] > z_threshold
            #         filtered_points = points_for_vis_raw[height_mask]
            #         max_points = 2000
            #         if len(filtered_points) > max_points:
            #             indices = np.random.choice(len(filtered_points), max_points, replace=False)
            #             filtered_points = filtered_points[indices]
            #         if len(filtered_points) < 100: 
            #             if len(points_for_vis_raw) > max_points:
            #                 indices = np.random.choice(len(points_for_vis_raw), max_points, replace=False)
            #                 points_to_plot = points_for_vis_raw[indices]
            #             else:
            #                 points_to_plot = points_for_vis_raw
            #         else:
            #             points_to_plot = filtered_points

            #         # --- 4. 각 상태에 대해 포인트 투영 실행 ---
            #         pts_gt = project_points(points_to_plot, P_orig, h, w)
            #         pts_broken = project_points(points_to_plot, P_broken, h, w)
            #         pts_corr_1st = project_points(points_to_plot, P_corr_1st, h, w) # [NEW]
            #         pts_corr_2nd = project_points(points_to_plot, P_corr_2nd, h, w) # [RENAME]
                    
            #         # --- 5. 최종 시각화 ---
            #         save_dir = "work_dirs/calib_verification" 
            #         os.makedirs(save_dir, exist_ok=True)
                    
            #         # --- Figure 1: Broken / GT / 1st Stage ---
            #         plt.figure(figsize=(16, 9))
            #         plt.imshow(img_for_vis)
            #         plt.scatter(pts_broken[:, 0], pts_broken[:, 1], color='red', s=2, alpha=0.6, label='Broken (Problem)')
            #         plt.scatter(pts_gt[:, 0], pts_gt[:, 1], color='lime', s=8, alpha=0.6, label='Ground Truth (Answer)')
            #         plt.scatter(pts_corr_1st[:, 0], pts_corr_1st[:, 1], c='gold', s=2, alpha=0.6, label='Corrected by 1st Stage')
            #         plt.title(f"Calibration Verification (1st Stage) @ Step {self.training_step}")
            #         plt.legend()
            #         plt.axis('off')
            #         save_path_1st = f"{save_dir}/calib_verification_1st_stage_step_{self.training_step}.jpg"
            #         plt.savefig(save_path_1st, bbox_inches='tight', pad_inches=0)
            #         plt.close()
            #         print(f"✅ 1st Stage calib verification image saved to {save_path_1st}")

            #         # --- Figure 2: Broken / GT / 2nd Stage ---
            #         plt.figure(figsize=(16, 9))
            #         plt.imshow(img_for_vis)
            #         plt.scatter(pts_broken[:, 0], pts_broken[:, 1], color='red', s=2, alpha=0.6, label='Broken (Problem)')
            #         plt.scatter(pts_gt[:, 0], pts_gt[:, 1], color='lime', s=8, alpha=0.6, label='Ground Truth (Answer)')
            #         plt.scatter(pts_corr_2nd[:, 0], pts_corr_2nd[:, 1], c='cyan', s=2, alpha=0.6, label='Corrected by 2nd Stage (Final)')
            #         plt.title(f"Calibration Verification (2nd Stage) @ Step {self.training_step}")
            #         plt.legend()
            #         plt.axis('off')
            #         save_path_2nd = f"{save_dir}/calib_verification_2nd_stage_step_{self.training_step}.jpg"
            #         plt.savefig(save_path_2nd, bbox_inches='tight', pad_inches=0)
            #         plt.close()
            #         print(f"✅ 2nd Stage calib verification image saved to {save_path_2nd}")

            #         plt.figure(figsize=(16, 9))
            #         plt.imshow(img_for_vis)
            #         plt.scatter(pts_broken[:, 0], pts_broken[:, 1], color='red', s=2, alpha=0.6, label='Broken (Problem)')
            #         plt.scatter(pts_gt[:, 0], pts_gt[:, 1], color='lime', s=8, alpha=0.6, label='Ground Truth (Answer)')
            #         # [NEW] 1st Stage 플롯 (자홍색)
            #         plt.scatter(pts_corr_1st[:, 0], pts_corr_1st[:, 1], c='gold', s=2, alpha=0.6, label='Corrected by 1st Stage')
            #         # [UPDATED] 2nd Stage 플롯 (하늘색)
            #         plt.scatter(pts_corr_2nd[:, 0], pts_corr_2nd[:, 1], c='cyan', s=2, alpha=0.6, label='Corrected by 2nd Stage (Final)')
            #         plt.title(f"Calibration Verification @ Step {self.training_step}")
            #         plt.legend()
            #         plt.axis('off')
            #         save_path_intg = f"{save_dir}/calib_verification_1and2_stage_step_{self.training_step}.jpg"
            #         plt.savefig(save_path_intg, bbox_inches='tight', pad_inches=0)
            #         plt.close()
            #         print(f"✅ 1nd2nd Stage calib verification image saved to {save_path_intg}")
            
            # self.training_step += 1

            return losses

    def predict(self, batch_inputs_dict: Dict[str, Tensor],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        
        # ======================================================
        # Calibration baseline routing
        # ======================================================

        if self.calibration_mode in {
            'clean',
            'broken',
            'oracle',
        }:

            return self._predict_calibration_baseline(
                batch_inputs_dict,
                batch_data_samples,
                self.calibration_mode,
            )


        if self.calibration_mode not in {
            'pcc_calib_only',
            'pcc_full',
            'lccnet',
            # NEW
            'pcc_broken_refine',
            'pcc_clean_refine',
            'geo_oracle_gtrot',

        }:

            raise ValueError(
                f'Unknown calibration_mode: '
                f'{self.calibration_mode}'
            )
        
        """
        (Function description remains the same)
        """
        # """
        # 전체 네트워크 추론 시간을 측정합니다.
        # """
        # # 1. 측정을 위한 CUDA Event 생성
        # start_event = torch.cuda.Event(enable_timing=True)
        # end_event = torch.cuda.Event(enable_timing=True)

        # # 2. 이전 GPU 연산이 끝날 때까지 대기 후 기록 시작
        # torch.cuda.synchronize()
        # start_event.record()

        # --- 1. & 2. Data Prep and 2D Detections (Same as before) ---
        target_device = batch_inputs_dict['imgs'].device
        batch_input_metas = [item.metainfo for item in batch_data_samples]

        broken_camera2lidar = torch.stack([s.broken_camera2lidar for s in batch_data_samples]).to(target_device)
        broken_camera_intrinsics = torch.stack([s.broken_camera_intrinsics for s in batch_data_samples]).to(target_device)
        # ============================================================
        # NEW: clean physical calibration
        # ============================================================

        clean_camera2lidar = torch.stack([
            s.camera2lidar
            for s in batch_data_samples
        ]).to(target_device)

        # ============================================================
        # Oracle geometric-solver diagnostics
        # ============================================================

        gt_delta_rot = torch.stack(
            [
                s.gt_delta_rot
                for s in batch_data_samples
            ]
        ).to(
            target_device
        )


        gt_delta_trans = torch.stack(
            [
                s.gt_delta_trans
                for s in batch_data_samples
            ]
        ).to(
            target_device
        )
        
        img_feats = self.extract_multiscale_img_feats(batch_inputs_dict)
        reshaped_img_feats, reshaped_data_samples = self._prepare_2d_head_inputs(
            img_feats, batch_data_samples)
        
        detections_2d = self._generate_and_process_2d_dets(
            reshaped_img_feats, reshaped_data_samples, batch_inputs_dict, visualize=False)

        detections_2d_orig_coords = self.convert_boxes_to_original_scale(
            pred_results_list=detections_2d, data_samples_list=reshaped_data_samples)

        # --- 3. Query Point Generation (Same as before) ---
        rois, _ = self._generate_rois_from_detections(detections_2d_orig_coords)
        rois_center = self.get_center_points(rois)
        
        if rois_center.numel() > 0:
            active_cam_indices = torch.unique(rois_center[:, 0]).long()
        else:
            active_cam_indices = torch.tensor([], dtype=torch.long, device=rois_center.device)
        
        (
            trimed_center_pts,
            query_unique_mask,
        ) = self.batch_rois_center_by_cam_id(

            rois_center,

            batch_size=200,
        )

        query_coords = trimed_center_pts[..., 2:]
        q_x = (query_coords[..., 0] / 1600) / 2
        q_y = query_coords[..., 1] / 900
        if query_coords.shape[-1] > 2:
            q_z = query_coords[..., 2]
            query_input = torch.stack([q_x, q_y, q_z], dim=-1)
        else:
            query_input = torch.stack([q_x, q_y], dim=-1)


        # sbs_img, _, dense_depth_map = self.extract_sbs_img(
        #     batch_inputs_dict, batch_input_metas, visualize=False)
        if self.calibration_mode == 'pcc_clean_refine':

            depth_mode = 'clean'

        else:

            depth_mode = 'broken'


        (
            sbs_img,
            pertubed_points,
            dense_depth_map,
            dense_depth_map_gt,
        ) = self.extract_sbs_img(

            batch_inputs_dict,
            batch_input_metas,

            visualize=False,

            depth_mode=depth_mode,
        )
        
        B, N, C, H, W = sbs_img.shape
        d_model = 312
        # feat_h, feat_w = 12, 64 # 우리가 확인한 실제 피처맵 크기
        feat_h, feat_w = 12, 80 # 우리가 확인한 실제 피처맵 크기

        raw_corrs_shape = (B * N, query_input.shape[1], query_input.shape[2])
        enc_out_shape = (B * N, feat_h * feat_w, d_model) # [B*N, 768, 312]
        esitmated_uvz_shape = (B * N, query_input.shape[1], 3) # (u,v,z)

        raw_corrs = torch.zeros(raw_corrs_shape, device=query_input.device)
        enc_out = torch.zeros(enc_out_shape, device=query_input.device)
        esitmated_uvz = torch.zeros(esitmated_uvz_shape, device=query_input.device)
        pred_delta_6dof = torch.zeros(B, N, 6, device=target_device)

        # Stores A/B/C/D geometric-oracle diagnostics for this sample.
        geo_oracle_summary = None

        # Handle the two cases: with or without active cameras
        if len(active_cam_indices) > 0:
            # Filter inputs for active cameras
            sbs_img_filtered = sbs_img[:, active_cam_indices]
            query_input_filtered = query_input[active_cam_indices]
            query_unique_mask_filtered = (
                query_unique_mask[
                    active_cam_indices
                ]
            )
            
            num_active_cams = sbs_img_filtered.shape[1]
            sbs_view = sbs_img_filtered.view(B * num_active_cams, C, H, W)

            # ============================================================
            # CalibHead V2 intrinsic for inference
            # ============================================================

            if B != 1:

                raise RuntimeError(
                    '[LGPC CalibHead V2 Predict] '
                    f'Current active-camera path expects B=1, got B={B}.'
                )


            camera_intrinsics_filtered = (
                broken_camera_intrinsics[
                    0,
                    active_cam_indices,
                ]
            )
                        
            # # Call the correlation network with filtered data
            # # --- ⏱️ Stage-1 (LGPC) 순수 오버헤드 측정 시작 ---
            # s1_start = torch.cuda.Event(enable_timing=True)
            # s1_end = torch.cuda.Event(enable_timing=True)

            # torch.cuda.synchronize() # 이전 연산(2D Detection 등) 완료 보장
            # s1_start.record()

            raw_corrs_filtered, _, _, enc_out_filtered_4d  = self.corr(sbs_view, query_input_filtered)

            # s1_end.record()
            # torch.cuda.synchronize() # LGPC 연산 완료 대기
            
            # t1_ms = s1_start.elapsed_time(s1_end)
            # # --- ⏱️ Stage-1 (LGPC) 측정 종료 ---

            # print(f"✅ [Stage-1: LGPC Latency]: {t1_ms:.2f} ms")

            # ============================================================
            # NEW:
            # Runtime-valid point mask for CalibHead
            #
            # CorrNet still receives fixed Q=200.
            #
            # query_unique_mask_filtered:
            #   True  = original detector query
            #   False = repeated filler query
            #
            # raw_corrs_filtered:
            #   CorrNet prediction on RIGHT SBS image
            # ============================================================

            corr_finite = (
                torch.isfinite(
                    raw_corrs_filtered
                )
                .all(
                    dim=-1
                )
            )


            # CorrNet correspondence output:
            #
            # x:
            #   right SBS region = [0.5, 1.0]
            #
            # y:
            #   normalized image range = [0.0, 1.0]
            corr_in_range = (

                (
                    raw_corrs_filtered[
                        ...,
                        0
                    ]
                    >= 0.5
                )

                & (
                    raw_corrs_filtered[
                        ...,
                        0
                    ]
                    <= 1.0
                )

                & (
                    raw_corrs_filtered[
                        ...,
                        1
                    ]
                    >= 0.0
                )

                & (
                    raw_corrs_filtered[
                        ...,
                        1
                    ]
                    <= 1.0
                )
            )


            # ============================================================
            # Final mask used ONLY by CalibHead.
            #
            # No GT information is used.
            # Therefore safe for inference.
            # ============================================================

            calib_point_valid_mask = (

                query_unique_mask_filtered

                & corr_finite

                & corr_in_range
            )
                
            # 1. Convert normalized corrs to pixel coordinates
            r_x = (raw_corrs_filtered[..., 0] - 0.5) * 2 * 1600 
            r_y = raw_corrs_filtered[..., 1] * 900
            uv_pixels_from_corr_filtered = torch.stack([r_x, r_y], dim=-1) # (NumActive, Q, 2)

            # 2. Get the "BROKEN" depth map for Z-Estimation
            depth_map_reshaped_BROKEN = dense_depth_map.view(B * N, 900, 1600)
            depth_map_active_BROKEN = depth_map_reshaped_BROKEN[active_cam_indices]

            # 3. Call ZEstimator
            esitmated_z_filtered = self.z_estimator(
                uv_sbs_normalized=raw_corrs_filtered,       # (특징 좌표)
                uv_orig_pixels=uv_pixels_from_corr_filtered,    # (GT 샘플링용 픽셀 좌표)
                depth_map=depth_map_active_BROKEN,
                enc_out=enc_out_filtered_4d # 4D enc_out
            )

            # 4. Get the predicted Z value
            # z_estimated_filtered = esitmated_z_filtered['z_estimated_real'] # [NumActive, Q, 1]
            z_estimated_hybrid = esitmated_z_filtered['depth']
            z_raw_for_calib = (
                esitmated_z_filtered[
                    'z_lidar_real'
                ]
            )

            corrs_3d_filtered = torch.cat(

                [
                    raw_corrs_filtered,
                    z_estimated_hybrid,
                ],

                dim=-1,
            )


            z_reliability_filtered = (
                self._build_calib_z_reliability(
                    esitmated_z_filtered
                )
            )

            # ============================================================
            # GEOMETRIC SOLVER ORACLE: 2x2 BOTTLENECK MATRIX
            #
            # A = GT Corr   + GT Z              + GT Rotation
            # B = Pred Corr + GT Z              + GT Rotation
            # C = GT Corr   + Pred Z @ GT Corr  + GT Rotation
            # D = Pred Corr + Pred Z @ Pred Corr+ GT Rotation
            #
            # IMPORTANT:
            #   Case C MUST re-run ZEstimator at GT Corr. Reusing
            #   z_estimated_hybrid would leak Pred-Corr localization error
            #   into the Z-only test.
            # ============================================================

            if self.calibration_mode == 'geo_oracle_gtrot':

                if B != 1:
                    raise RuntimeError(
                        '[GEO ORACLE] '
                        f'Current diagnostic expects B=1, got B={B}.'
                    )

                gt_rot_active = gt_delta_rot[
                    0,
                    active_cam_indices,
                ]

                gt_trans_active = gt_delta_trans[
                    0,
                    active_cam_indices,
                ]

                broken_c2l_active = broken_camera2lidar[
                    0,
                    active_cam_indices,
                ]

                # ========================================================
                # Build perfect GT source correspondence and GT depth.
                # ========================================================
                (
                    corr_target_gt,
                    corr_valid_gt,
                    z_broken_gt,
                ) = self._build_corr_target_from_clean_depth(
                    query_input_active=query_input_filtered,
                    active_cam_indices=active_cam_indices,
                    dense_depth_map_gt=dense_depth_map_gt,
                    clean_camera2lidar=clean_camera2lidar,
                    broken_camera2lidar=broken_camera2lidar,
                    camera_intrinsics=broken_camera_intrinsics,
                )

                # GT Corr is represented on the RIGHT SBS image.
                # Convert it back to original-image pixel coordinates.
                gt_source_u = (
                    (corr_target_gt[..., 0] - 0.5)
                    * 2.0
                    * 1600.0
                )
                gt_source_v = corr_target_gt[..., 1] * 900.0

                gt_source_uv = torch.stack(
                    [gt_source_u, gt_source_v],
                    dim=-1,
                )

                gt_solver_mask = (
                    query_unique_mask_filtered
                    & corr_valid_gt
                    & torch.isfinite(z_broken_gt)
                    & (z_broken_gt > 1e-3)
                )

                # ========================================================
                # A. SOLVER SANITY
                # GT Corr + GT Z + GT Rotation
                # ========================================================
                (
                    trans_a,
                    diag_a,
                ) = self._solve_translation_given_rotation(
                    query_input=query_input_filtered,
                    source_uv_pixels=gt_source_uv,
                    source_z=z_broken_gt.unsqueeze(-1),
                    point_valid_mask=gt_solver_mask,
                    broken_camera2lidar=broken_c2l_active,
                    camera_intrinsics=camera_intrinsics_filtered,
                    delta_rot=gt_rot_active,
                )

                # ========================================================
                # D-common:
                # Pred Corr + Pred Z, but evaluated on exactly the same
                # GT-valid subset as A/B/C.
                #
                # Diagnostic only. GT mask is NOT for inference.
                # ========================================================

                d_common_mask = (
                    query_unique_mask_filtered
                    &
                    calib_point_valid_mask
                    &
                    corr_valid_gt
                    &
                    torch.isfinite(z_broken_gt)
                    &
                    (z_broken_gt > 1e-3)
                )


                (
                    trans_d_common,
                    diag_d_common,
                ) = self._solve_translation_given_rotation(

                    query_input=
                        query_input_filtered,

                    source_uv_pixels=
                        uv_pixels_from_corr_filtered,

                    source_z=
                        z_estimated_hybrid,

                    point_valid_mask=
                        d_common_mask,

                    broken_camera2lidar=
                        broken_c2l_active,

                    camera_intrinsics=
                        camera_intrinsics_filtered,

                    delta_rot=
                        gt_rot_active,
                )


                d_common_mae = (
                    torch.abs(
                        trans_d_common
                        - gt_trans_active
                    )
                    .mean()
                )


                d_common_l2 = (
                    torch.linalg.norm(
                        trans_d_common
                        - gt_trans_active,
                        dim=-1,
                    )
                    .mean()
                )

                a_mae = torch.abs(
                    trans_a - gt_trans_active
                ).mean()

                a_l2 = torch.linalg.norm(
                    trans_a - gt_trans_active,
                    dim=-1,
                ).mean()

                # ========================================================
                # B. CORRNET-ONLY ERROR ISOLATION
                # Pred Corr + GT Z + GT Rotation
                #
                # GT depth belongs to the correct physical point;
                # only the UV source location is replaced by CorrNet output.
                # ========================================================
                b_valid_mask = (
                    query_unique_mask_filtered
                    & corr_finite
                    & corr_in_range
                    & corr_valid_gt
                    & torch.isfinite(z_broken_gt)
                    & (z_broken_gt > 1e-3)
                )

                (
                    trans_b,
                    diag_b,
                ) = self._solve_translation_given_rotation(
                    query_input=query_input_filtered,
                    source_uv_pixels=uv_pixels_from_corr_filtered,
                    source_z=z_broken_gt.unsqueeze(-1),
                    point_valid_mask=b_valid_mask,
                    broken_camera2lidar=broken_c2l_active,
                    camera_intrinsics=camera_intrinsics_filtered,
                    delta_rot=gt_rot_active,
                )

                b_mae = torch.abs(
                    trans_b - gt_trans_active
                ).mean()

                b_l2 = torch.linalg.norm(
                    trans_b - gt_trans_active,
                    dim=-1,
                ).mean()

                # ========================================================
                # C. ZESTIMATOR-ONLY ERROR ISOLATION
                # GT Corr + Pred Z @ GT Corr + GT Rotation
                #
                # Re-run ZEstimator at GT Corr. DO NOT reuse
                # z_estimated_hybrid, because that was estimated at
                # Pred Corr and would contaminate this test with Corr error.
                # ========================================================
                with torch.no_grad():
                    estimated_z_at_gtcorr = self.z_estimator(
                        uv_sbs_normalized=corr_target_gt,
                        uv_orig_pixels=gt_source_uv,
                        depth_map=depth_map_active_BROKEN,
                        enc_out=enc_out_filtered_4d,
                    )

                z_pred_at_gtcorr = estimated_z_at_gtcorr['depth']

                c_valid_mask = (
                    query_unique_mask_filtered
                    & corr_valid_gt
                    & torch.isfinite(z_pred_at_gtcorr[..., 0])
                    & (z_pred_at_gtcorr[..., 0] > 1e-3)
                )

                (
                    trans_c,
                    diag_c,
                ) = self._solve_translation_given_rotation(
                    query_input=query_input_filtered,
                    source_uv_pixels=gt_source_uv,
                    source_z=z_pred_at_gtcorr,
                    point_valid_mask=c_valid_mask,
                    broken_camera2lidar=broken_c2l_active,
                    camera_intrinsics=camera_intrinsics_filtered,
                    delta_rot=gt_rot_active,
                )

                c_mae = torch.abs(
                    trans_c - gt_trans_active
                ).mean()

                c_l2 = torch.linalg.norm(
                    trans_c - gt_trans_active,
                    dim=-1,
                ).mean()

                # ========================================================
                # D. REAL PROVIDER TEST
                # Pred Corr + Pred Z @ Pred Corr + GT Rotation
                # ========================================================
                d_valid_mask = (
                    calib_point_valid_mask
                    & torch.isfinite(z_estimated_hybrid[..., 0])
                    & (z_estimated_hybrid[..., 0] > 1e-3)
                )

                (
                    trans_d,
                    diag_d,
                ) = self._solve_translation_given_rotation(
                    query_input=query_input_filtered,
                    source_uv_pixels=uv_pixels_from_corr_filtered,
                    source_z=z_estimated_hybrid,
                    point_valid_mask=d_valid_mask,
                    broken_camera2lidar=broken_c2l_active,
                    camera_intrinsics=camera_intrinsics_filtered,
                    delta_rot=gt_rot_active,
                )

                d_mae = torch.abs(
                    trans_d - gt_trans_active
                ).mean()

                d_l2 = torch.linalg.norm(
                    trans_d - gt_trans_active,
                    dim=-1,
                ).mean()

                # Keep scalar diagnostics so they can be attached to
                # DataSample.metainfo as well as printed.
                geo_oracle_summary = {
                    'geo_A_mae_m': float(a_mae.detach().cpu()),
                    'geo_A_l2_m': float(a_l2.detach().cpu()),
                    'geo_A_valid_count': float(
                        diag_a['valid_count'].float().mean().detach().cpu()
                    ),
                    'geo_A_residual_rmse': float(
                        diag_a['residual_rmse'].mean().detach().cpu()
                    ),
                    'geo_B_mae_m': float(b_mae.detach().cpu()),
                    'geo_B_l2_m': float(b_l2.detach().cpu()),
                    'geo_B_valid_count': float(
                        diag_b['valid_count'].float().mean().detach().cpu()
                    ),
                    'geo_B_residual_rmse': float(
                        diag_b['residual_rmse'].mean().detach().cpu()
                    ),
                    'geo_C_mae_m': float(c_mae.detach().cpu()),
                    'geo_C_l2_m': float(c_l2.detach().cpu()),
                    'geo_C_valid_count': float(
                        diag_c['valid_count'].float().mean().detach().cpu()
                    ),
                    'geo_C_residual_rmse': float(
                        diag_c['residual_rmse'].mean().detach().cpu()
                    ),
                    'geo_D_mae_m': float(d_mae.detach().cpu()),
                    'geo_D_l2_m': float(d_l2.detach().cpu()),
                    'geo_D_valid_count': float(
                        diag_d['valid_count'].float().mean().detach().cpu()
                    ),
                    'geo_D_residual_rmse': float(
                        diag_d['residual_rmse'].mean().detach().cpu()
                    ),
                    'geo_D_common_mae_m': float(
                        d_common_mae.detach().cpu()
                    ),

                    'geo_D_common_l2_m': float(
                        d_common_l2.detach().cpu()
                    ),

                    'geo_D_common_valid_count': float(
                        diag_d_common[
                            'valid_count'
                        ].float().mean().detach().cpu()
                    ),

                    'geo_D_common_residual_rmse': float(
                        diag_d_common[
                            'residual_rmse'
                        ].mean().detach().cpu()
                    ),
                }

                # Print up to 200 samples so a deterministic 200-sample
                # run can be parsed directly from the log.
                if not hasattr(self, '_geo_oracle_debug_count'):
                    self._geo_oracle_debug_count = 0

                if self._geo_oracle_debug_count < 200:
                    print(
                        '\n'
                        '=====================================================\n'
                        '[GEO ORACLE 2x2 BOTTLENECK MATRIX]\n'
                        '=====================================================\n'
                        '\n'
                        '[A] GT Corr + GT Z + GT Rotation\n'
                        f'MAE      = {a_mae.item():.6f} m\n'
                        f'L2       = {a_l2.item():.6f} m\n'
                        f'valid    = {diag_a["valid_count"].float().mean().item():.2f}\n'
                        f'residual = {diag_a["residual_rmse"].mean().item():.6f}\n'
                        '\n'
                        '[B] Pred Corr + GT Z + GT Rotation\n'
                        f'MAE      = {b_mae.item():.6f} m\n'
                        f'L2       = {b_l2.item():.6f} m\n'
                        f'valid    = {diag_b["valid_count"].float().mean().item():.2f}\n'
                        f'residual = {diag_b["residual_rmse"].mean().item():.6f}\n'
                        '\n'
                        '[C] GT Corr + Pred Z@GT-Corr + GT Rotation\n'
                        f'MAE      = {c_mae.item():.6f} m\n'
                        f'L2       = {c_l2.item():.6f} m\n'
                        f'valid    = {diag_c["valid_count"].float().mean().item():.2f}\n'
                        f'residual = {diag_c["residual_rmse"].mean().item():.6f}\n'
                        '\n'
                        '[D] Pred Corr + Pred Z@Pred-Corr + GT Rotation\n'
                        f'MAE      = {d_mae.item():.6f} m\n'
                        f'L2       = {d_l2.item():.6f} m\n'
                        f'valid    = {diag_d["valid_count"].float().mean().item():.2f}\n'
                        f'residual = {diag_d["residual_rmse"].mean().item():.6f}\n'
                        '\n'
                        '[D-common] Pred Corr + Pred Z + GT-R '
                        'on A/B/C common mask\n'
                        f'MAE      = {d_common_mae.item():.6f} m\n'
                        f'L2       = {d_common_l2.item():.6f} m\n'
                        f'valid    = '
                        f'{diag_d_common["valid_count"].float().mean().item():.2f}\n'
                        f'residual = '
                        f'{diag_d_common["residual_rmse"].mean().item():.6f}\n'
                        '=====================================================\n'
                    )

                    self._geo_oracle_debug_count += 1

                # For the existing stage-1 metadata path, use D:
                # GT rotation + translation solved from real Pred Corr/Z.
                pred_delta_6dof_filtered = torch.cat(
                    [gt_rot_active, trans_d],
                    dim=-1,
                )

            else:
                # ========================================================
                # Existing learned CalibHead
                # ========================================================
                (
                    pred_delta_6dof_filtered,
                    _
                ) = self.calib_head(
                    enc_out=enc_out_filtered_4d,
                    query_input=query_input_filtered,
                    corrs_pred_3d=corrs_3d_filtered,
                    z_reliability=z_reliability_filtered,
                    z_raw=z_raw_for_calib,
                    camera_intrinsics=camera_intrinsics_filtered,
                    point_valid_mask=calib_point_valid_mask,
                )

            # Convert 4D enc_out to 3D for storage
            b_act, c_f, h_f, w_f = enc_out_filtered_4d.shape
            enc_out_filtered_3d = enc_out_filtered_4d.flatten(2).permute(0, 2, 1)

            if B == 1:
                raw_corrs[active_cam_indices] = raw_corrs_filtered
                enc_out[active_cam_indices] = enc_out_filtered_3d # Store 3D version
                pred_delta_6dof[0, active_cam_indices] = pred_delta_6dof_filtered # Store 6DoF
                
                # ✨ Store (u, v, z) in pixel coordinates + depth
                esitmated_uvz_filtered = torch.cat(
                    [uv_pixels_from_corr_filtered, esitmated_z_filtered['depth']], dim=-1
                )
                esitmated_uvz[active_cam_indices] = esitmated_uvz_filtered

        # --- ✨ FIX: Reshape enc_out to 4D *after* the if-block (like in loss) ---
        enc_out = enc_out.permute(0, 2, 1).reshape(-1, d_model, feat_h, feat_w)
            
        # --- ✨ FIX: Split 6DoF (3 rot + 3 trans) ---
        pred_delta_rot = pred_delta_6dof[..., :3]
        pred_delta_trans = pred_delta_6dof[..., 3:]

        # Current active-camera prediction path is B==1 only.
        if B != 1:
            raise RuntimeError(
                "B>1이면 active_cam_indices를 배치별로 따로 만들어 저장해야 합니다."
            )

        for b in range(B):
            # stage1 per-cam
            batch_input_metas[b]['stage1_pred_delta_rot'] = pred_delta_rot[b].detach()
            batch_input_metas[b]['stage1_pred_delta_trans'] = pred_delta_trans[b].detach()
            batch_input_metas[b]['stage1_active_cam_indices'] = active_cam_indices.detach()

            # ✅ metric이 찾는 alias도 같이
            batch_input_metas[b]['pred_delta_rot_1st'] = batch_input_metas[b]['stage1_pred_delta_rot']
            batch_input_metas[b]['pred_delta_trans_1st'] = batch_input_metas[b]['stage1_pred_delta_trans']

            # =========================================================
            # NEW
            # 실제 어떤 calibration mode를 적용하는 실험인지 기록
            # =========================================================
            batch_input_metas[b]['calibration_mode'] = (
                self.calibration_mode
            )

            # ✅ data_sample.metainfo "덮어쓰기" 금지: 기존 metainfo 보존 + update
            mi = dict(batch_data_samples[b].metainfo)  # 기존 메타 복사 (gt_delta_* 유지됨)

            mi.update({
                'stage1_pred_delta_rot': batch_input_metas[b]['stage1_pred_delta_rot'],
                'stage1_pred_delta_trans': batch_input_metas[b]['stage1_pred_delta_trans'],
                'stage1_active_cam_indices': batch_input_metas[b]['stage1_active_cam_indices'],
                'pred_delta_rot_1st': batch_input_metas[b]['pred_delta_rot_1st'],
                'pred_delta_trans_1st': batch_input_metas[b]['pred_delta_trans_1st'],
                'calibration_mode': self.calibration_mode,
            })

            if geo_oracle_summary is not None:
                mi.update(geo_oracle_summary)

            batch_data_samples[b].set_metainfo(mi)

        # Oracle diagnostic stops before BEVFusion detection.
        # Metadata for the whole batch has already been attached above.
        if self.calibration_mode == 'geo_oracle_gtrot':
            if geo_oracle_summary is None:
                raise RuntimeError(
                    '[GEO ORACLE] No active camera / no oracle summary was produced.'
                )
            return batch_data_samples

        # ===================== END: LOGIC ALIGNMENT WITH LOSS FUNCTION =====================

        # --- 5. Correct Calibration Matrices (Now safe to run) ---
        corrected_calib_dict = self._get_corrected_calib_from_prediction(
            pred_delta_rot,   # (B, N, 3)
            pred_delta_trans, # (B, N, 3)
            broken_camera2lidar,
            broken_camera_intrinsics
        )

        # ============================================================
        # Build BROKEN calibration dictionary
        # ============================================================

        broken_calib_dict = (
            self._build_calib_dict_from_cam2lidar(
                broken_camera2lidar,
                broken_camera_intrinsics,
            )
        )

        # ============================================================
        # NEW: CLEAN calibration dictionary
        # ============================================================

        clean_calib_dict = (
            self._build_calib_dict_from_cam2lidar(
                clean_camera2lidar,
                broken_camera_intrinsics,
            )
        )

        # ============================================================
        # Calibration actually used by BEVFusion and RRRF
        # ============================================================

        if self.calibration_mode == 'pcc_broken_refine':

            # Broken geometry + FeatureRefine
            active_calib_dict = (
                broken_calib_dict
            )


        elif self.calibration_mode == 'pcc_clean_refine':

            # Clean geometry + FeatureRefine
            # No LGPC physical correction needed.
            active_calib_dict = (
                clean_calib_dict
            )

        else:

            # pcc_full etc.
            active_calib_dict = (
                corrected_calib_dict
            )

        # ============================================================
        # DEBUG: verify which physical calibration is actually active
        # ============================================================
        if (
            self.calibration_mode == 'pcc_clean_refine'
            and not hasattr(self, '_clean_refine_debug_done')
        ):

            active_c2l = active_calib_dict['cam2lidar']

            err_to_clean = (
                active_c2l
                - clean_camera2lidar
            ).abs().max().item()

            err_to_broken = (
                active_c2l
                - broken_camera2lidar
            ).abs().max().item()

            print(
                '\n'
                '=========================================\n'
                '[CLEAN REFINE SANITY]\n'
                f'mode={self.calibration_mode}\n'
                f'active_vs_clean_maxerr  = {err_to_clean:.8e}\n'
                f'active_vs_broken_maxerr = {err_to_broken:.8e}\n'
                '=========================================\n'
            )

            self._clean_refine_debug_done = True

        if self.calibration_mode == 'pcc_calib_only':

            results = (
                self._predict_bevfusion_with_calib(
                    batch_inputs_dict,
                    batch_data_samples,
                    corrected_calib_dict,
                    img_feats=img_feats,
                )
            )

            return self._attach_calib_prediction_meta(
                results,
                pred_delta_rot,
                pred_delta_trans,
                active_cam_indices,
            )

        det_xyz = self.uvz_to_lidar_xyz(esitmated_uvz, active_calib_dict['lidar2img'])

        # --- ✨ START: Logic copied from loss function ---
        # loss 함수와 동일하게 좌표를 pc_range로 정규화 및 클램핑합니다.
        det_xyz_ref = det_xyz.clone()
        det_xyz_ref[..., 0:1] = (det_xyz_ref[..., 0:1] - self.pc_range[0]) / (
                self.pc_range[3] - self.pc_range[0])
        det_xyz_ref[..., 1:2] = (det_xyz_ref[..., 1:2] - self.pc_range[1]) / (
                self.pc_range[4] - self.pc_range[1])
        det_xyz_ref[..., 2:3] = (det_xyz_ref[..., 2:3] - self.pc_range[2]) / (
                self.pc_range[5] - self.pc_range[2])
        det_xyz_ref_clamped = det_xyz_ref.clamp(min=0, max=1)
        # --- ✨ END: Logic copied from loss function ---
        
        det_feat_sampled = self._sample_features_from_grid(feature_map=enc_out, coords=query_input)
        det_xyz_proc, det_feat_proc = self._prepare_camera_proposals(
            det_xyz_ref_clamped, det_feat_sampled, B=B, N_cam=N)

        # --- 7. & 8. Final 3D Detection and Formatting (Same as before) ---
        # feats = self.extract_feat(
        #     batch_inputs_dict=batch_inputs_dict,
        #     batch_input_metas=batch_input_metas,
        #     corrected_calib=active_calib_dict,
        #     precomputed_img_feats=img_feats)

        use_lidar_query = (
            getattr(
                self.bbox_head,
                'query_source',
                'fused'
            )
            == 'lidar'
        )


        if use_lidar_query:

            (
                feats,
                lidar_query_feats
            ) = self.extract_feat(
                batch_inputs_dict=
                    batch_inputs_dict,

                batch_input_metas=
                    batch_input_metas,

                corrected_calib=
                    active_calib_dict,

                precomputed_img_feats=
                    img_feats,

                return_lidar_query_feat=
                    True,
            )

        else:

            feats = self.extract_feat(
                batch_inputs_dict=
                    batch_inputs_dict,

                batch_input_metas=
                    batch_input_metas,

                corrected_calib=
                    active_calib_dict,

                precomputed_img_feats=
                    img_feats,
            )

            lidar_query_feats = None
        
        # # --- ⏱️ Stage-2 (RRRF) 순수 오버헤드 측정 시작 ---
        # s2_start = torch.cuda.Event(enable_timing=True)
        # s2_end = torch.cuda.Event(enable_timing=True)

        # torch.cuda.synchronize() # 이전 연산 완료 보장
        # s2_start.record()
        
        # results_list_3d = self.bbox_head.predict(
        #     feats, det_xyz_proc, det_feat_proc, batch_input_metas)
        
        results_list_3d = self.bbox_head.predict(
            feats,
            det_xyz_proc,
            det_feat_proc,
            batch_input_metas,

            lidar_query_feats=
                lidar_query_feats,
        )
        
        results = self.add_pred_to_datasample(batch_data_samples,
                                            results_list_3d)
        
        # s2_end.record()
        # torch.cuda.synchronize() # Stage-2 연산 완료 대기
        
        # t2_ms = s2_start.elapsed_time(s2_end)
        # # --- ⏱️ Stage-2 (RRRF) 측정 종료 ---

        # print(f"✅ [Stage-2: RRRF Latency]: {t2_ms:.2f} ms")
        
        # # 3. 측정 종료 및 동기화
        # end_event.record()
        # torch.cuda.synchronize()

        # # 4. 시간 계산 (단위: ms)
        # elapsed_time_ms = start_event.elapsed_time(end_event)
        
        # # 결과 출력 (터미널에서 바로 확인 가능)
        # print(f"\n🚀 [Ours Network Inference Time]: {elapsed_time_ms:.2f} ms | FPS: {1000.0 / elapsed_time_ms:.1f}")
        
        # ==========================================================
        # [CRITICAL] metric이 보는 "최종 results(DataSample)"에 stage1 pred delta를 주입
        #  - Tensor 그대로 넣으면 metric의 `or` 체인에서 bool(tensor) 이슈가 날 수 있으니
        #    안전하게 list 로 저장 (또는 numpy)
        # ==========================================================
        stage1_rot_cpu = pred_delta_rot.detach().to('cpu')        # (B, N_cam, 3)
        stage1_trans_cpu = pred_delta_trans.detach().to('cpu')    # (B, N_cam, 3)

        if torch.is_tensor(active_cam_indices):
            active_cam_indices_cpu = active_cam_indices.detach().to('cpu')
        else:
            active_cam_indices_cpu = active_cam_indices

        for b, ds in enumerate(results):
            mi = dict(ds.metainfo)  # 기존 gt_delta_* 등 보존

            mi.update({
                # stage1 canonical
                'stage1_pred_delta_rot': stage1_rot_cpu[b].tolist(),
                'stage1_pred_delta_trans': stage1_trans_cpu[b].tolist(),
                'stage1_active_cam_indices': active_cam_indices_cpu.tolist()
                    if torch.is_tensor(active_cam_indices_cpu) else active_cam_indices_cpu,

                # metric alias (stage1)
                'pred_delta_rot_1st': stage1_rot_cpu[b].tolist(),
                'pred_delta_trans_1st': stage1_trans_cpu[b].tolist(),

                # =====================================================
                # NEW
                # evaluator에게 실제 calibration application mode 전달
                # =====================================================
                'calibration_mode':
                    self.calibration_mode,
            })

            ds.set_metainfo(mi)
        
        if self.test_cfg is not None and self.test_cfg.get('visualize_ours', False):
            visualize_ours_fusion_result(
                batch_inputs_dict=batch_inputs_dict,
                results=results,
                corrected_calib=active_calib_dict,  # mode에 따라 실제 적용된 calibration
                save_path=f'work_dirs/vis/ours_step_{self.training_step}.png'
            )
        
        return results

    def _attach_calib_prediction_meta(
        self,
        results,
        pred_delta_rot,
        pred_delta_trans,
        active_cam_indices,
    ):

        rot_cpu = (
            pred_delta_rot
            .detach()
            .cpu()
        )

        trans_cpu = (
            pred_delta_trans
            .detach()
            .cpu()
        )

        if torch.is_tensor(
            active_cam_indices
        ):

            active_cpu = (
                active_cam_indices
                .detach()
                .cpu()
                .tolist()
            )

        else:

            active_cpu = (
                active_cam_indices
            )

        for b, ds in enumerate(results):

            mi = dict(
                ds.metainfo
            )

            mi.update({
                'stage1_pred_delta_rot':
                    rot_cpu[b].tolist(),

                'stage1_pred_delta_trans':
                    trans_cpu[b].tolist(),

                'stage1_active_cam_indices':
                    active_cpu,

                'pred_delta_rot_1st':
                    rot_cpu[b].tolist(),

                'pred_delta_trans_1st':
                    trans_cpu[b].tolist(),

                # NEW
                'calibration_mode':
                    self.calibration_mode,
            })

            ds.set_metainfo(mi)

        return results