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
                                   visualize_bev_proposals,batched_trim_corrs,
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
        corr: Optional[dict] = None,
        corr_loss :Optional[dict] = None,
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
        # self.view_transform = MODELS.build(
        #     view_transform) if view_transform is not None else None
        self.pts_middle_encoder = MODELS.build(pts_middle_encoder)

        # self.fusion_layer = MODELS.build(
        #     fusion_layer) if fusion_layer is not None else None

        self.pts_backbone = MODELS.build(pts_backbone)
        self.pts_neck = MODELS.build(pts_neck)

        self.init_weights()

        # modified by sjmoon
        self.bbox_head = MODELS.build(bbox_head)
        self.img_bbox_head = MODELS.build(img_bbox_head)
        self.corr = MODELS.build(corr)
        self.corr_loss = MODELS.build(corr_loss)
        self.z_estimator = MODELS.build(z_estimator)
        self.calib_head = MODELS.build(calib_head)
        
        self.class_names = class_names
        self.name_to_idx = {name: i for i, name in enumerate(self.class_names)}

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        feat_dim_original = 312 # 입력 차원은 det_feat의 원래 특징 차원입니다 (12 * 64 = 768).
        hidden_channel = bbox_head['hidden_channel'] # (128) 출력 차원은 TransFusionHead의 hidden_channel과 반드시 일치해야 합니다.
        # self.feat_projector = nn.Linear(feat_dim_original, hidden_channel)

        # =====================================================================
        # ✨ START: Code added for selective module freezing
        # =====================================================================
        # Set this flag to True to freeze parts of the network during training.
        # The specific modules to be frozen are defined in the _freeze_modules() method.
        self.enable_selective_freezing = enable_selective_freezing
        if self.enable_selective_freezing:
            print("\n!!! WARNING: Selectively freezing parts of the network. !!!\n")
            self._freeze_modules()
        # =====================================================================
        # ✨ END: Code added for selective module freezing
        # =====================================================================

        self.vis_step_counter = 0
        self.training_step = 0
        self.pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    
    # def _freeze_modules(self):
    #         """
    #         Selectively freezes parts of the network for targeted training.
    #         This configuration trains ONLY the image 2D detection pipeline.
    #         """
    #         # --- STRATEGY: Freeze everything EXCEPT the Image 2D Detection pipeline. ---
    #         print("Freezing all modules EXCEPT the Image 2D Detection pipeline.")
            
    #         # # 동결할 모듈 목록 (2D 탐지 관련 모듈 제외)
    #         modules_to_freeze = {
    #             # LiDAR Path
    #             # 'pts_voxel_layer': self.pts_voxel_layer,
    #             # 'pts_voxel_encoder': self.pts_voxel_encoder,
    #             # 'pts_middle_encoder': self.pts_middle_encoder,
    #             # 'pts_backbone': self.pts_backbone,
    #             # 'pts_neck': self.pts_neck,
                
    #             # # 3D Detection Head
    #             # 'bbox_head': self.bbox_head,
                
    #             # # Fusion & View Transform
    #             # 'view_transform': self.view_transform,
    #             # 'fusion_layer': self.fusion_layer,
                
    #             # Custom Modules
    #             'corr': self.corr,
    #             # 'z_estimator': self.z_estimator,
    #         }
    #         # modules_to_freeze = {
    #         #     # # LiDAR Path
    #         #     # 'pts_voxel_layer': self.pts_voxel_layer,
    #         #     # 'pts_voxel_encoder': self.pts_voxel_encoder,
    #         #     # 'pts_middle_encoder': self.pts_middle_encoder,
    #         #     # 'pts_backbone': self.pts_backbone,
    #         #     # 'pts_neck': self.pts_neck,
                
    #         #     # # 3D Detection Head
    #         #     # 'bbox_head': self.bbox_head,
                
    #         #     # # Fusion & View Transform
    #         #     # 'view_transform': self.view_transform,
    #         #     # 'fusion_layer': self.fusion_layer,
                
    #         #     # # Custom Modules
    #         #     'corr': self.corr,
    #         #     # 'z_estimator': self.z_estimator,
    #         # }

    #         # 선택된 모듈들의 파라미터 업데이트를 중지
    #         for name, module in modules_to_freeze.items():
    #             if module is not None:
    #                 for param in module.parameters():
    #                     param.requires_grad = False
    #                 print(f" - ❄️ Module '{name}' has been frozen.")
    #             else:
    #                 print(f" - Module '{name}' is None, skipping.")
            
    #         print("\n - 🔥 The following modules will be trained: 'img_backbone', 'img_neck', 'img_bbox_head'.")
    #         print("---------------------------------")
    
    def _freeze_modules(self):
            """
            Selectively freezes parts of the network for targeted training.
            [Stage 1] Trains ONLY the Z-Estimator and the full calibration
            fusion pipeline (1st stage fusion, 2nd stage predictor, 2nd stage fusion).
            """
            # --- [Stage 1] STRATEGY: Freeze backbones, heads, and 'corr' module.
            # Train z_estimator and all modules needed for loss_calib_rot_pred. ---
            
            print("--- [Stage 1] Freezing modules for Z-Estimator + 2nd Stage Calib training ---")
            
            # [Stage 1]의 목표:
            # - 🔥 Train: z_estimator
            # - 🔥 Train: calibration_predictor (loss_calib_rot_pred 계산)
            # - 🔥 Train: 1단계/2단계 퓨전 모듈들 (calibration_predictor의 입력 생성)
            # - ❄️ Freeze: LiDAR/Image 백본, 3D/2D 헤드, 1단계 'corr' 모듈, 디코더
            
            # 학습할 모듈 목록 (이 모듈들을 제외하고 모두 동결):
            modules_to_train = [
                'z_estimator',                  # 1. Z-Estimator
                'calib_head',                   # 2. 1단계 오차 예측 헤드
                # 'fusion_cross_attention',       # 3. 1단계 퓨전 (Predictor의 입력)
                # 'fusion_norm1',
                # 'fusion_ffn',
                # 'refined_attention',            # 4. 2단계 퓨전
                # 'refined_norm1',
                # 'refined_ffn',
                # 'bev_query_pos_embedding',      # 5. 퓨전용 임베딩
                # 'camera_proposal_pos_embedding'
            ]
            
            # modules_to_freeze 딕셔너리 (학습할 모듈 제외)
            modules_to_freeze = {
                # LiDAR Path
                'pts_voxel_layer': self.pts_voxel_layer,
                'pts_voxel_encoder': self.pts_voxel_encoder,
                'pts_middle_encoder': self.pts_middle_encoder,
                'pts_backbone': self.pts_backbone,
                'pts_neck': self.pts_neck,

                # Image Path (2D Detection Head)
                # 'img_backbone': self.img_backbone,
                # 'img_neck': self.img_neck,
                # 'img_bbox_head': self.img_bbox_head,

                # 1st stage calibration Head
                # 'calib_head': self.calib_head,
                
                # 3D Detection Head
                'bbox_head': self.bbox_head,
                
                # # Fusion Backbone
                # 'shared_conv': self.bbox_head.shared_conv,

                # # Query Generation
                # 'heatmap_head': self.bbox_head.heatmap_head,
                # 'class_encoding': self.bbox_head.class_encoding,

                # # Decoders & Final Prediction Heads
                # 'decoder': self.bbox_head.decoder,
                # 'prediction_heads': self.bbox_head.prediction_heads,
                
                # Custom Modules
                # 'corr': self.corr, # 1st stage calib (Freeze)
            }

            # 선택된 모듈들의 파라미터 업데이트를 중지
            print("--- Freezing Modules (❄️) ---")
            for name, module in modules_to_freeze.items():
                if module is not None:
                    # 모듈이 ModuleList인 경우 (예: decoder, prediction_heads)
                    if isinstance(module, (torch.nn.ModuleList, list)):
                        for sub_module in module:
                            for param in sub_module.parameters():
                                param.requires_grad = False
                    else: # 단일 모듈인 경우
                        for param in module.parameters():
                            param.requires_grad = False
                    print(f" - ❄️ Module '{name}' has been frozen.")
                else:
                    print(f" - Module '{name}' is None, skipping.")

            # (확인 사살) 학습 대상 모듈의 파라미터가 확실히 학습되도록 설정
            print("\n--- Training Modules (🔥) ---")
            for name in modules_to_train:
                if hasattr(self, name):
                    module = getattr(self, name)
                    if module is not None:
                        # 모듈이 ModuleList인 경우
                        if isinstance(module, (torch.nn.ModuleList, list)):
                            for sub_module in module:
                                for param in sub_module.parameters():
                                    param.requires_grad = True
                        else: # 단일 모듈인 경우
                            for param in module.parameters():
                                param.requires_grad = True
                        print(f" - 🔥 Module '{name}' will be trained.")
                    else:
                        print(f" - WARNING: Trainable module '{name}' is None or not found!")
                else:
                    print(f" - WARNING: Trainable module '{name}' attribute does not exist!")
            
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
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parses the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: There are two elements. The first is the
            loss tensor passed to optim_wrapper which may be a weighted sum
            of all losses, and the second is log_vars which will be sent to
            the logger.
        """
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append(
                    [loss_name,
                     sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(value for key, value in log_vars if 'loss' in key)
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars  # type: ignore

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
        precomputed_img_feats: Optional[tuple] = None
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
        return x

    # def extract_feat(
    #     self,
    #     batch_inputs_dict,
    #     batch_input_metas,
    #     **kwargs,
    # ):
    #     imgs = batch_inputs_dict.get('imgs', None)
    #     points = batch_inputs_dict.get('points', None)
    #     features = []
    #     if imgs is not None:
    #         imgs = imgs.contiguous()
    #         lidar2image, camera_intrinsics, camera2lidar = [], [], []
    #         img_aug_matrix, lidar_aug_matrix = [], []
    #         for i, meta in enumerate(batch_input_metas):
    #             lidar2image.append(meta['lidar2img'])
    #             camera_intrinsics.append(meta['cam2img'])
    #             camera2lidar.append(meta['cam2lidar'])
    #             img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
    #             lidar_aug_matrix.append(
    #                 meta.get('lidar_aug_matrix', np.eye(4)))

    #         lidar2image = imgs.new_tensor(np.asarray(lidar2image))
    #         camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
    #         camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
    #         img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
    #         lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))
    #         img_feature ,raw_img_feature = self.extract_img_feat(imgs, deepcopy(points),
    #                                             lidar2image, camera_intrinsics,
    #                                             camera2lidar, img_aug_matrix,
    #                                             lidar_aug_matrix,
    #                                             batch_input_metas)
    #         features.append(img_feature)
    #     pts_feature = self.extract_pts_feat(batch_inputs_dict)
    #     features.append(pts_feature)

    #     if self.fusion_layer is not None:
    #         x = self.fusion_layer(features)
    #     else:
    #         assert len(features) == 1, features
    #         x = features[0]

    #     x = self.pts_backbone(x)
    #     x = self.pts_neck(x)

    #     return x, raw_img_feature,lidar2image, camera_intrinsics, camera2lidar
    
    def extract_sbs_img(
        self,
        batch_inputs_dict,
        batch_input_metas,
        visualize=False,
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

            dense_depth_map_mis = dense_map_from_depth_batch_v2(lidar_depth_mis,grid=3,iterations=3)
            dense_depth_img_mis = dense_depth_map_mis.to(dtype=torch.uint8)
            dense_depth_img_color_mis = batch_colormap(dense_depth_img_mis)

            dense_depth_map = dense_map_from_depth_batch_v2(lidar_depth_gt,grid=3,iterations=3)
            # dense_depth_img = dense_depth_map.to(dtype=torch.uint8)
            # dense_depth_img_color = batch_colormap(dense_depth_img)

            # 픽셀 값을 0.0 ~ 1.0 범위로 정규화하여 imshow가 올바르게 표시하도록 함
            img_min, img_max = imgs.min(), imgs.max()
            imgs = (imgs - img_min) / (img_max - img_min + 1e-8) # 0으로 나누는 것 방지

            N, V, C, H, W = imgs.shape
            imgs_reshaped = imgs.view(N * V, C, H, W)
            depth_reshaped_mis = dense_depth_img_color_mis.view(N * V, C, H, W)

            img_resized = F.interpolate(imgs_reshaped, size=[192, 640], mode="bilinear")
            lidar_depth_mis_resized = F.interpolate(depth_reshaped_mis, size=[192, 640], mode="bilinear")

            sbs_img = two_images_side_by_side_gpu(img_resized, lidar_depth_mis_resized)
            sbs_img = sbs_img.permute(0,3,1,2)
            sbs_img = tvtf.normalize(sbs_img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            sbs_img = sbs_img.view(N, V, C, 192, 640*2)

            # ############## input display ##########################
            if visualize and sbs_img is not None:
                display_depth_maps(imgs,dense_depth_img_color_mis,sbs_img)
                print("input dispaly end")
        
        return sbs_img, points ,dense_depth_map_mis,dense_depth_map
    
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
            if self.training and self.train_cfg.get('complement_2d_gt', -1) > 0:
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
    
    ##### old code ######
    # def batch_rois_center_by_cam_id(self, rois_center, batch_size=100):
    #     """
    #     rois_center 텐서에서 카메라 ID를 읽어, 항상 6개의 카메라에 대한
    #     고정된 크기의 배치(batch) 텐서를 생성합니다.
    #     존재하지 않는 카메라 ID의 슬롯은 0으로 채워집니다.
    #     """
    #     device = rois_center.device
        
    #     # 입력 텐서가 비어있는 경우, 빈 텐서를 반환
    #     if rois_center.shape[0] == 0:
    #         # num_cams를 알 수 없으므로 기본값 6으로 설정하거나, 호출하는 쪽에서 처리
    #         return torch.zeros((6, batch_size, 4), device=device)

    #     # 1. rois_center의 0열에서 모든 카메라 인덱스를 추출합니다.
    #     cam_indices_tensor = rois_center[:, 0]
        
    #     # 2. 존재하는 고유한 카메라 ID 목록을 찾습니다.
    #     unique_cam_ids = torch.unique(cam_indices_tensor).long().cpu().tolist()
        
    #     # 3. 최대 카메라 ID를 기반으로 출력 텐서의 크기를 결정합니다.
    #     #    예: [0, 1, 5]가 있다면, 크기가 6인 텐서 (0~5)를 생성합니다.
    #     max_cam_id = int(torch.max(cam_indices_tensor).item())
    #     num_total_cams = max_cam_id + 1
        
    #     batched_centers = torch.zeros((num_total_cams, batch_size, 4), device=device)
        
    #     # 원본 객체 ID 저장 (검증 로직은 그대로 유지)
    #     original_obj_ids = rois_center[:, 1].cpu().numpy()
        
    #     # 4. 하드코딩된 range(num_cams) 대신, 실제 존재하는 카메라 ID들을 순회합니다.
    #     for cam_id in unique_cam_ids:
    #         cam_mask = (rois_center[:, 0] == cam_id)
    #         cam_centers = rois_center[cam_mask]
    #         n = cam_centers.size(0)
            
    #         if n == 0:
    #             continue
                
    #         # --- (내부 샘플링 로직은 기존과 동일) ---
    #         obj_ids = cam_centers[:, 1].cpu().numpy()
    #         unique_obj_ids = np.unique(obj_ids)
    #         num_unique_objs = len(unique_obj_ids)
            
    #         if num_unique_objs <= batch_size:
    #             if n < batch_size:
    #                 repeat_factor = (batch_size + n - 1) // n
    #                 cam_centers = cam_centers.repeat(repeat_factor, 1)[:batch_size]
    #         else:
    #             selected_indices = []
    #             for obj_id in unique_obj_ids:
    #                 obj_indices = np.where(obj_ids == obj_id)[0]
    #                 selected_idx = np.random.choice(obj_indices)
    #                 selected_indices.append(selected_idx)
                    
    #             if len(selected_indices) < batch_size:
    #                 remaining = batch_size - len(selected_indices)
    #                 all_indices = np.arange(n)
    #                 # 이미 선택된 인덱스를 제외하고 남은 풀에서 추가 선택
    #                 pool = np.setdiff1d(all_indices, selected_indices)
    #                 # 만약 풀이 부족하면 복원 추출 허용
    #                 replace = len(pool) < remaining
    #                 extra_indices = np.random.choice(
    #                     pool,
    #                     size=remaining,
    #                     replace=replace
    #                 )
    #                 selected_indices.extend(extra_indices)
                    
    #             selected_indices = torch.tensor(selected_indices, device=device, dtype=torch.long)
    #             cam_centers = cam_centers[selected_indices]
            
    #         batched_centers[cam_id, :cam_centers.size(0)] = cam_centers[:batch_size]
        
    #     return batched_centers
    
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

        # 입력 텐서가 비어있는 경우, 위에서 생성한 제로 텐서를 그대로 반환
        if rois_center.shape[0] == 0:
            return batched_centers

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

        return batched_centers
    
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
    
    # def _get_corrected_calib_from_prediction(
    #     self,
    #     pred_delta_rot: torch.Tensor, # ✨ 입력이 (B, N, 3) 
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

    #     corrected_camera2lidar_trans = broken_trans + pred_delta_trans
    #     corrected_camera2lidar_rots = pred_delta_rot_mat @ broken_rots
        
    #     # --- ✨ FIX: In-place 할당 대신 torch.cat으로 새로운 4x4 행렬 조립 ---
        
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
    
    def _get_corrected_calib_from_prediction(
        self,
        pred_delta_rot: torch.Tensor, # (B, N, 3)
        pred_delta_trans: torch.Tensor, # (B, N, 3)
        broken_camera2lidar: torch.Tensor, # (B, N, 4, 4)
        broken_camera_intrinsics: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        예측된 delta 값으로 4x4 보정 행렬을 만들고, 
        broken calibration에 행렬 곱셈을 수행하여 보정합니다.
        """
        B, N = broken_camera2lidar.shape[:2]
        device = broken_camera2lidar.device
        dtype = broken_camera2lidar.dtype

        # --- 1. 예측값으로 4x4 보정 행렬 (T_correction) 생성 ---
        # T_correction: Misaligned LiDAR -> Corrected LiDAR (lidar2lidar)
        
        # 1-1. 회전 행렬 (3x3)
        pred_rot_mat = axis_angle_to_matrix(pred_delta_rot) # (B, N, 3, 3)
        
        # 1-2. 이동 벡터 (3x1)
        pred_trans_vec = pred_delta_trans.unsqueeze(-1) # (B, N, 3, 1)
        
        # 1-3. 4x4 행렬 조립
        # [ R_pred  t_pred ]
        # [   0       1    ]
        top_row = torch.cat([pred_rot_mat, pred_trans_vec], dim=-1) # (B, N, 3, 4)
        
        bottom_row = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        bottom_row = bottom_row.expand(B, N, -1, -1) # (B, N, 1, 4)
        
        T_correction = torch.cat([top_row, bottom_row], dim=-2) # (B, N, 4, 4)
        
        # --- 2. 행렬 곱셈으로 보정 적용 (핵심 수정) ---
        # Corrected = Correction @ Broken
        # T_{Cam->Lidar_Corrected} = T_{Lidar_Mis->Lidar_Corr} @ T_{Cam->Lidar_Mis}
        corrected_camera2lidar = torch.matmul(T_correction, broken_camera2lidar)
        
        # --- 3. lidar2img 계산 (기존 로직 활용) ---
        # corrected_camera2lidar의 역행렬 계산 (lidar2camera)
        # (일반적인 inverse보다 R.T를 이용한 방식이 수치적으로 더 안정적일 수 있음)
        R_corr = corrected_camera2lidar[..., :3, :3]
        t_corr = corrected_camera2lidar[..., :3, 3:4]
        
        corrected_lidar2camera_rots = R_corr.transpose(-1, -2)
        corrected_lidar2camera_trans = -torch.matmul(corrected_lidar2camera_rots, t_corr)
        
        corrected_lidar2camera_3x4 = torch.cat(
            [corrected_lidar2camera_rots, corrected_lidar2camera_trans], dim=-1
        ) # (B, N, 3, 4)

        # 최종 투영 행렬 계산 (3x4)
        intrinsics_3x3 = broken_camera_intrinsics[..., :3, :3]
        corrected_lidar2imag_3x4 = torch.matmul(intrinsics_3x3, corrected_lidar2camera_3x4)
        
        # 4x4 변환
        bottom_row_proj = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        bottom_row_proj = bottom_row_proj.expand(B, N, -1, -1)
        corrected_lidar2imag_4x4 = torch.cat([corrected_lidar2imag_3x4, bottom_row_proj], dim=-2)

        # --- 4. 결과 반환 ---
        corrected_calib_dict = {
            'lidar2img': corrected_lidar2imag_4x4,
            'cam2img': broken_camera_intrinsics,
            'cam2lidar': corrected_camera2lidar
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

            img_feats = self.extract_multiscale_img_feats(batch_inputs_dict)

            reshaped_img_feats, reshaped_data_samples = self._prepare_2d_head_inputs(
                img_feats, batch_data_samples)
            
            losses = dict()
            if self.with_bbox_head:
                losses_2d = self.img_bbox_head.loss(reshaped_img_feats, reshaped_data_samples)
            else:
                losses_2d = dict()
            
            # 4. 계산된 2D 로스를 최종 로스 딕셔너리에 'img_' 접두사와 함께 추가
            total_losses = dict()
            for k, v in losses_2d.items():
                total_losses[f'img_{k}'] = v # 예: 'loss_cls' -> 'img_loss_cls'
            
            # losses 딕셔너리를 total_losses로 초기화하여 2D loss를 먼저 담습니다.
            losses = total_losses

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
            
            trimed_center_pts =self.batch_rois_center_by_cam_id(rois_center,batch_size=200)

            query_coords = trimed_center_pts[..., 2:]
            q_x = (query_coords[..., 0] / 1600) / 2
            q_y = query_coords[..., 1] / 900
            if query_coords.shape[-1] > 2:
                q_z = query_coords[..., 2]
                query_input_bbox_centers = torch.stack([q_x, q_y, q_z], dim=-1)
            else:
                query_input_bbox_centers = torch.stack([q_x, q_y], dim=-1)

            # --- 3. SBS 이미지 및 깊이 맵 생성 (공통) ---
            sbs_img, pertubed_points,dense_depth_map,dense_depth_map_gt = self.extract_sbs_img(batch_inputs_dict, batch_input_metas,visualize=False)
            B,N,C,H,W = sbs_img.shape
            d_model = 312
            feat_h, feat_w = 12, 64

            if batch_data_samples[0].matched_uvset is not None and self.training: # 학습 시에만 실행
            # --- ✨ [NEW] Task A: Corr 네트워크 학습 (mv2d reference code 삽입) ---
                uv_set = batch_data_samples[0].matched_uvset.to(target_device)
                    # (mv2d ref) 랜덤 대응점 GT 준비
                trimed_uvset = batched_trim_corrs(uv_set).to(dtype=torch.float32, device=target_device)
                
                # (mv2d ref) Task A 쿼리 (랜덤 포인트)
                query_input_random = trimed_uvset[..., :2]
                query_input_random[..., 0] /= batch_data_samples[0].img_shape[1]  # img.shape[3] = 1600
                query_input_random[..., 1] /= batch_data_samples[0].img_shape[0]  # sbs_img.shape[2] =900
                query_input_random[:,:,0] = query_input_random[:,:,0]/2
                query_input_random[:,:,1] = query_input_random[:,:,1]

                # (mv2d ref) Task A 타겟 (랜덤 포인트)
                corr_target_random = trimed_uvset[...,2:]
                corr_target_random[...,0] = corr_target_random[...,0] / batch_data_samples[0].img_shape[1]
                corr_target_random[...,1] = corr_target_random[...,1] / batch_data_samples[0].img_shape[0]
                corr_target_random[:,:,0] = corr_target_random[:,:,0]/2 + 0.5
                corr_target_random[:,:,1] = corr_target_random[:,:,1]
                
                # (mv2d ref) Task A 실행
                # (주의: sbs_img, query_input_random의 B, N 차원이 맞아야 함)
                sbs_img_flat = sbs_img.view(B*N, C, H, W)
                raw_corrs_rand, cycle_rand, corr_mask_rand, enc_out_rand = self.corr(sbs_img_flat, query_input_random)

                # (mv2d ref) Task A 로스 계산
                loss_corr = self.corr_loss(raw_corrs_rand, corr_target_random, cycle_rand, query_input_random, corr_mask_rand)
                
                # ✨ [중요] 로스 가중치를 적용하여 서열 정리
                losses['loss_corr'] = loss_corr * 1000.0 # (예시: 대응점 학습에 높은 가중치 부여)
            
                # (u', v') 정규화 좌표 -> (u', v') 픽셀 좌표 변환
                r_x_rand = (raw_corrs_rand[..., 0] - 0.5) * 2 * 1600 
                r_y_rand = raw_corrs_rand[..., 1] * 900
                uv_pixels_rand = torch.stack([r_x_rand, r_y_rand], dim=-1) # [B*N, Q, 2]
                
                # "문제지" (BROKEN) - Z-Est는 내부적으로 사용 안 함 (학습용)
                depth_map_reshaped_BROKEN = dense_depth_map.view(B * N, 900, 1600)
                
                # ZEstimator 호출 (Task A의 결과물 사용)
                esitmated_z_rand = self.z_estimator(
                    uv_sbs_normalized=raw_corrs_rand,
                    uv_orig_pixels=uv_pixels_rand,
                    depth_map=depth_map_reshaped_BROKEN, # "문제지" 전달
                    enc_out=enc_out_rand
                )
                z_estimated_rand = esitmated_z_rand['z_estimated_real'] # [B*N, Q, 1]

                # "정답지" (TRUE) 준비
                depth_map_reshaped_TRUE = dense_depth_map_gt.view(B * N, 900, 1600)
                
                # GT 샘플링
                num_active_rand, Q_rand, _ = uv_pixels_rand.shape
                H_gt, W_gt = 900, 1600
                uv_orig_flat_rand = uv_pixels_rand.view(-1, 2)
                cam_ids_flat_rand = torch.arange(num_active_rand, device=target_device).unsqueeze(1).expand(num_active_rand, Q_rand).reshape(-1)
                u_coords_flat_rand = uv_orig_flat_rand[:, 0].round().long().clamp(0, W_gt - 1)
                v_coords_flat_rand = uv_orig_flat_rand[:, 1].round().long().clamp(0, H_gt - 1)

                z_lidar_sparse_gt_TRUE_flat_rand = depth_map_reshaped_TRUE[cam_ids_flat_rand, v_coords_flat_rand, u_coords_flat_rand]
                z_lidar_sparse_gt_TRUE_rand = z_lidar_sparse_gt_TRUE_flat_rand.view(num_active_rand, Q_rand)

                # Z-Estimator 손실 계산 (랜덤 포인트 기준)
                valid_mask_rand = (z_lidar_sparse_gt_TRUE_rand > 0)
                if valid_mask_rand.any():
                    loss_z_estimation = F.smooth_l1_loss(
                                        z_estimated_rand.squeeze(-1)[valid_mask_rand],
                                        z_lidar_sparse_gt_TRUE_rand[valid_mask_rand],
                                        reduction='mean',
                                        beta=1.0
                                    )
                    # ✨ 가중치 적용 (예: 1.0 또는 0.1)
                    losses['loss_z_estimation'] = loss_z_estimation * 1.0 
                else:
                    losses['loss_z_estimation'] = torch.tensor(0.0, device=target_device)
                
                z_hybrid = esitmated_z_rand['depth'] # Teacher-Forcing용 'depth' 사용
                corrs_3d_hybrid = torch.cat([raw_corrs_rand, z_hybrid], dim=-1)             
                pred_delta_6dof_random = self.calib_head(
                    enc_out_rand, 
                    query_input_random, 
                    corrs_3d_hybrid,
                    # ✨[제안] 여기에도 FPN 특징을 추가로 전달하면
                    # calib_head의 정체 현상을 더 확실히 풀 수 있습니다.
                    # fpn_feats=img_feats[active_cam_indices]
                )
                # (기존) Calib-Head Loss 계산
                pred_rot_random = pred_delta_6dof_random[..., :3]
                pred_trans_random = pred_delta_6dof_random[..., 3:]
                R_pred_calib = axis_angle_to_matrix(pred_rot_random)
                R_gt_calib = axis_angle_to_matrix(gt_delta_rot.squeeze(0))
                
                # ✨ 가중치 조절
                losses['loss_calib_rot'] = identity_matrix_loss(R_pred_calib, R_gt_calib) * 2.0
                losses['loss_calib_trans'] = F.smooth_l1_loss(pred_trans_random, gt_delta_trans.squeeze(0), reduction='mean') * 1.0

            else:
                losses['loss_corr'] = torch.tensor(0.0, device=target_device)
                losses['loss_z_estimation'] = torch.tensor(0.0, device=target_device) # Z-Est 로스도 0으로 초기화
                losses['loss_calib_rot'] = torch.tensor(0.0, device=target_device, requires_grad=True)
                losses['loss_calib_trans'] = torch.tensor(0.0, device=target_device, requires_grad=True)

            # --- 5. ✨ [Task B] Downstream 태스크 (Z-Est, Calib) (기존 로직) ---
        
            # (기존) 텐서 초기화
            raw_corrs_shape = (B * N, query_input_bbox_centers.shape[1], query_input_bbox_centers.shape[2])
            enc_out_shape = (B * N, feat_h * feat_w, d_model)
            esitmated_uvz_shape = (B * N, query_input_bbox_centers.shape[1], 3)
            
            raw_corrs = torch.zeros(raw_corrs_shape, device=target_device)
            enc_out = torch.zeros(enc_out_shape, device=target_device)
            esitmated_uvz = torch.zeros(esitmated_uvz_shape, device=target_device)

            # (기존) 2D Bbox가 감지된 카메라에 대해서만 Task B 실행
            if len(active_cam_indices) > 0:
                sbs_img_filtered = sbs_img[:, active_cam_indices]
                query_input_filtered = query_input_bbox_centers[active_cam_indices] # Bbox 중심점 쿼리 사용
                
                num_active_cams = sbs_img_filtered.shape[1]
                sbs_view = sbs_img_filtered.view(B * num_active_cams, C, H, W)
                
                # (기존) Task B 실행
                # ✨[MTL] self.corr는 이제 Task A의 loss_corr에서도 그래디언트를 받음
                raw_corrs_active, cycle, corr_mask, enc_out_active_4d = self.corr(sbs_view, query_input_filtered)

                b_act, c_f, h_f, w_f = enc_out_active_4d.shape
                enc_out_active_3d = enc_out_active_4d.flatten(2).permute(0, 2, 1)
                
                # (기존) Z-Estimator 로직
                r_x = (raw_corrs_active[..., 0] - 0.5) * 2 * 1600 
                r_y = raw_corrs_active[..., 1] * 900
                uv_pixels_from_corr = torch.stack([r_x, r_y], dim=-1)
                
                depth_map_reshaped_BROKEN = dense_depth_map.view(B * N, 900, 1600)
                depth_map_active_BROKEN = depth_map_reshaped_BROKEN[active_cam_indices]

                with torch.no_grad(): # ✨ Calib-Head 학습에 Z-Est가 영향 주지 않도록 no_grad
                    esitmated_z_active = self.z_estimator(
                        uv_sbs_normalized=raw_corrs_active,
                        uv_orig_pixels=uv_pixels_from_corr,
                        depth_map=depth_map_active_BROKEN,
                        enc_out=enc_out_active_4d
                    )
                
                # (기존) 텐서 인덱싱
                if B == 1:
                    raw_corrs[active_cam_indices] = raw_corrs_active
                    enc_out[active_cam_indices] = enc_out_active_3d
                    esitmated_uvz_active = torch.cat(
                        [uv_pixels_from_corr, esitmated_z_active['depth']], dim=-1
                    )
                    esitmated_uvz[active_cam_indices] = esitmated_uvz_active
            
            enc_out = enc_out.permute(0, 2, 1).reshape(-1, d_model, feat_h, feat_w)
            pred_delta_rot = pred_rot_random
            pred_delta_trans = pred_trans_random

            # # ##### 검증용 display ######
            # from .imageprocessing_unit import draw_correspondences
            # # gt_corrs = torch.cat([query_input,corr_target],dim=-1)
            # pred_corrs = torch.cat([query_input_bbox_centers,raw_corrs_active],dim=-1)
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
            #         trimed_corrs = pred_corrs[cid][:3,...],  # 첫 번째 배치 선택
            #         sbs_img=sbs_img.view(B*N,C,H,W)[cid],
            #         save_path='correspondence_visualization_pred.jpg',
            #         bboxes_to_draw = bboxes_for_this_view, # 원본 좌표계 BBox 전달
            #         score_thr = 0.4
            #     )
            #     # --- 2. 원본 vs 증강 BBox 비교 시각화 저장 (요청하신 부분) ---
            #     save_batch_predictions_to_file(
            #             batch_inputs_dict=batch_inputs_dict,
            #             reshaped_data_samples=reshaped_data_samples,
            #             augmented_preds_list=detections_2d,
            #             original_preds_list=detections_2d_orig_coords,
            #             current_step=self.vis_step_counter,
            #             save_dir='work_dirs/my_exp/vis_results',
            #             view_index=cid, # 루프 변수 cid를 view_index로 사용
            #             score_thr=0.4
            #         )
            #     print ("end")
   

            #### 1st stage end - 2nd stage start ################
            # corrected_calib_dict = self._get_corrected_calib_from_prediction(
            #                 pred_delta_rot,
            #                 pred_delta_trans,
            #                 broken_camera2lidar,
            #                 broken_camera_intrinsics
            #             )

            # # ✨ '보정된' lidar2imag를 사용하여 3D 좌표 변환 수행
            # # ✨ (esitmated_uvz는 이제 if/else 로직에 의해 올바르게 채워졌습니다)
            # det_xyz = self.uvz_to_lidar_xyz(esitmated_uvz, corrected_calib_dict['lidar2img'])
        
            # # ... (이하 Chamfer Loss, BBox Head 등 나머지 코드는 동일) ...

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

            # ##### loss 3d point cloud loss 
            # # losses['loss_chamfer_xyz'] = loss_chamfer * 0.005

            # det_xyz_ref = det_xyz.clone()
            # det_xyz_ref[..., 0:1] = (det_xyz_ref[..., 0:1] - self.pc_range[0]) / (
            #         self.pc_range[3] - self.pc_range[0])
            # det_xyz_ref[..., 1:2] = (det_xyz_ref[..., 1:2] - self.pc_range[1]) / (
            #         self.pc_range[4] - self.pc_range[1])
            # det_xyz_ref[..., 2:3] = (det_xyz_ref[..., 2:3] - self.pc_range[2]) / (
            #         self.pc_range[5] - self.pc_range[2])
            # det_xyz_ref_clamped = det_xyz_ref.clamp(min=0, max=1)
            
            # det_feat_sampled = self._sample_features_from_grid(feature_map=enc_out, coords=query_input_bbox_centers)
            # det_xyz_proc, det_feat_proc = self._prepare_camera_proposals(det_xyz_ref_clamped,det_feat_sampled,B=B,N_cam=N)

            # feats = self.extract_feat(batch_inputs_dict=batch_inputs_dict,
            #                         batch_input_metas=batch_input_metas,
            #                         corrected_calib=corrected_calib_dict,
            #                         precomputed_img_feats=img_feats)

            # if self.with_bbox_head:
            #     bbox_loss = self.bbox_head.loss(
            #                     feats, 
            #                     det_xyz_proc, 
            #                     det_feat_proc, 
            #                     batch_data_samples,
            #                     pred_delta_rot=pred_delta_rot,
            #                     pred_delta_trans=pred_delta_trans,
            #                     gt_delta_rot=gt_delta_rot,
            #                     gt_delta_trans=gt_delta_trans
            #                 )
            
            # # --- ✨ 2. 손실과 예측값 분리 ---
            # # 시각화를 위해 예측값을 별도 변수로 빼내고, 딕셔너리에서 제거
            # pred_delta_rot_batch = bbox_loss.pop('pred_delta_rot')
            # pred_delta_trans_batch = bbox_loss.pop('pred_delta_trans')

            # losses.update(bbox_loss)

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
            #         pred_delta_rot = pred_delta_rot_batch[batch_idx] # [4]
            #         pred_delta_trans = pred_delta_trans_batch[batch_idx] # [3]
                    
            #         # c-2) Delta (Rotation, Translation) -> 4x4 행렬 (Delta T)
            #         delta_R = axis_angle_to_matrix(pred_delta_rot)
            #         delta_T_2nd_stage = torch.eye(4, device=K.device)
            #         delta_T_2nd_stage[:3, :3] = delta_R
            #         delta_T_2nd_stage[:3, 3] = pred_delta_trans
                    
            #         # c-3) ✨[핵심] Broken Extrinsic에 2nd-stage Delta를 적용
            #         # T_c2l_corr_2nd = T_c2l_broken @ delta_T_2nd_stage
            #         T_c2l_corr_2nd = T_c2l_corr_1st @ delta_T_2nd_stage
            #         # --- ⬆️⬆️⬆️ [수정된 부분] ⬆️⬆️⬆️ ---

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
        """
        (Function description remains the same)
        """
        # --- 1. & 2. Data Prep and 2D Detections (Same as before) ---
        target_device = batch_inputs_dict['imgs'].device
        batch_input_metas = [item.metainfo for item in batch_data_samples]

        broken_camera2lidar = torch.stack([s.broken_camera2lidar for s in batch_data_samples]).to(target_device)
        broken_camera_intrinsics = torch.stack([s.broken_camera_intrinsics for s in batch_data_samples]).to(target_device)
        
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
        
        trimed_center_pts = self.batch_rois_center_by_cam_id(rois_center, batch_size=200)

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
        sbs_img, pertubed_points,dense_depth_map,dense_depth_map_gt = self.extract_sbs_img(
            batch_inputs_dict, batch_input_metas,visualize=False)
        B, N, C, H, W = sbs_img.shape
        d_model = 312
        feat_h, feat_w = 12, 64 # 우리가 확인한 실제 피처맵 크기

        raw_corrs_shape = (B * N, query_input.shape[1], query_input.shape[2])
        enc_out_shape = (B * N, feat_h * feat_w, d_model) # [B*N, 768, 312]
        esitmated_uvz_shape = (B * N, query_input.shape[1], 3) # (u,v,z)

        raw_corrs = torch.zeros(raw_corrs_shape, device=query_input.device)
        enc_out = torch.zeros(enc_out_shape, device=query_input.device)
        esitmated_uvz = torch.zeros(esitmated_uvz_shape, device=query_input.device)
        pred_delta_6dof = torch.zeros(B, N, 6, device=target_device)

        # Handle the two cases: with or without active cameras
        if len(active_cam_indices) > 0:
            # Filter inputs for active cameras
            sbs_img_filtered = sbs_img[:, active_cam_indices]
            query_input_filtered = query_input[active_cam_indices]
            
            num_active_cams = sbs_img_filtered.shape[1]
            sbs_view = sbs_img_filtered.view(B * num_active_cams, C, H, W)
            
            # Call the correlation network with filtered data
            raw_corrs_filtered, _, _, enc_out_filtered_4d  = self.corr(sbs_view, query_input_filtered)
            
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
            z_estimated_filtered = esitmated_z_filtered['z_estimated_real'] # [NumActive, Q, 1]

            # 5. Create 3D correspondences (u', v', z')
            corrs_3d_filtered = torch.cat([raw_corrs_filtered, z_estimated_filtered], dim=-1) # (NumActive, Q, 3)

            # 6. Call calib_head with 3D correspondences
            pred_delta_6dof_filtered = self.calib_head(
                enc_out_filtered_4d,    # 4D tensor
                query_input_filtered,   # (NumActive, Q, 2 or 3)
                corrs_3d_filtered       # (NumActive, Q, 3)
            ) # 출력 shape: (NumActive, 6)

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

        # ===================== END: LOGIC ALIGNMENT WITH LOSS FUNCTION =====================

        # --- 5. Correct Calibration Matrices (Now safe to run) ---
        corrected_calib_dict = self._get_corrected_calib_from_prediction(
            pred_delta_rot,   # (B, N, 3)
            pred_delta_trans, # (B, N, 3)
            broken_camera2lidar,
            broken_camera_intrinsics
        )

        det_xyz = self.uvz_to_lidar_xyz(esitmated_uvz, corrected_calib_dict['lidar2img'])

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
        feats = self.extract_feat(
            batch_inputs_dict=batch_inputs_dict,
            batch_input_metas=batch_input_metas,
            corrected_calib=corrected_calib_dict,
            precomputed_img_feats=img_feats)
        
        results_list_3d = self.bbox_head.predict(
            feats, det_xyz_proc, det_feat_proc, batch_input_metas)
        
        results = self.add_pred_to_datasample(batch_data_samples,
                                            results_list_3d)
        return results