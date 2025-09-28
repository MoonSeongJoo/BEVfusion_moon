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

from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList
from .ops import Voxelization
from .imageprocessing_unit import (dense_map_from_depth_batch_v2, 
                                   batch_colormap,two_images_side_by_side_gpu,
                                   display_depth_maps,
                                   save_batch_predictions_to_file
                                   )
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import os
import math


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
        z_estimator: Optional[dict] = None,
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
        self.bbox_head = MODELS.build(bbox_head)
        self.img_bbox_head = MODELS.build(img_bbox_head)
        self.corr = MODELS.build(corr)
        self.z_estimator = MODELS.build(z_estimator)
        
        self.class_names = class_names
        self.name_to_idx = {name: i for i, name in enumerate(self.class_names)}

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        feat_dim_original = 312 # 입력 차원은 det_feat의 원래 특징 차원입니다 (12 * 64 = 768).
        hidden_channel = bbox_head['hidden_channel'] # (128) 출력 차원은 TransFusionHead의 hidden_channel과 반드시 일치해야 합니다.
        self.feat_projector = nn.Linear(feat_dim_original, hidden_channel)

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
    
    def _freeze_modules(self):
            """
            Selectively freezes parts of the network for targeted training.
            This configuration trains ONLY the image 2D detection pipeline.
            """
            # --- STRATEGY: Freeze everything EXCEPT the Image 2D Detection pipeline. ---
            print("Freezing all modules EXCEPT the Image 2D Detection pipeline.")
            
            # # 동결할 모듈 목록 (2D 탐지 관련 모듈 제외)
            # modules_to_freeze = {
            #     # LiDAR Path
            #     'pts_voxel_layer': self.pts_voxel_layer,
            #     'pts_voxel_encoder': self.pts_voxel_encoder,
            #     'pts_middle_encoder': self.pts_middle_encoder,
            #     'pts_backbone': self.pts_backbone,
            #     'pts_neck': self.pts_neck,
                
            #     # 3D Detection Head
            #     'bbox_head': self.bbox_head,
                
            #     # Fusion & View Transform
            #     'view_transform': self.view_transform,
            #     'fusion_layer': self.fusion_layer,
                
            #     # Custom Modules
            #     'corr': self.corr,
            #     'z_estimator': self.z_estimator,
            # }
            modules_to_freeze = {
                # # LiDAR Path
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
                
                # # Custom Modules
                'corr': self.corr,
                # 'z_estimator': self.z_estimator,
            }

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
        x,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> tuple[torch.Tensor, tuple]: # 반환 타입 힌트 수정
        B, N, C, H, W = x.size()
        x_reshaped_4d = x.view(B * N, C, H, W).contiguous()

        x_backbone = self.img_backbone(x_reshaped_4d)
        
        # 1. img_neck의 출력(튜플)을 별도의 변수에 저장합니다.
        x_neck_tuple_4d = self.img_neck(x_backbone)

        # --- ✨ 2D 헤드용 img_feature를 생성하는 새로운 로직 시작 ---
        # 이 로직은 기존 x의 흐름에 영향을 주지 않습니다.
        img_feature_tuple_5d = []
        for feat_4d in x_neck_tuple_4d:
            # 각 4D 피처 (B*N, C, H, W)를 5D (B, N, C, H, W)로 변환
            _BN, C_feat, H_feat, W_feat = feat_4d.size()
            feat_5d = feat_4d.view(B, N, C_feat, H_feat, W_feat)
            img_feature_tuple_5d.append(feat_5d)
        img_feature_tuple_5d = tuple(img_feature_tuple_5d)
        # --- 새로운 로직 끝 ---

        # --- 아래는 view_transform의 입력을 만들기 위한 기존 로직 (그대로 유지) ---
        x_for_bev = x_neck_tuple_4d
        if not isinstance(x_for_bev, torch.Tensor):
            x_for_bev = x_for_bev[0]

        BN, C_bev, H_bev, W_bev = x_for_bev.size()
        x_for_bev_5d = x_for_bev.view(B, int(BN / B), C_bev, H_bev, W_bev)

        with torch.autocast(device_type='cuda', dtype=torch.float32):
            bev_feature = self.view_transform(
                x_for_bev_5d, # 기존과 동일한 단일 5D 텐서 전달
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                img_metas,
            )
        # --- 기존 로직 끝 ---
        
        # 최종적으로 BEV 피처와, 2D 헤드용으로 새롭게 가공된 이미지 피처 튜플을 반환
        return bev_feature, img_feature_tuple_5d
    
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

    def predict(self, batch_inputs_dict: Dict[str, Tensor],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """
        Args:
            batch_inputs_dict (dict): The model input dict which contains
                `points`, `img` keys.
            batch_data_samples (List[Det3DDataSample]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            list[Det3DDataSample]: Detection results of the
            input images. Each Det3DDataSample usually contains
            'pred_instances_3d'.
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        
        # 1. loss 메서드와 동일하게 모든 특징과 proposal을 생성합니다.
        feats, img_feats, lidar2imag, _, _ = self.extract_feat(
            batch_inputs_dict, batch_input_metas)
        sbs_img, _, dense_depth_map = self.extract_sbs_img(
            batch_inputs_dict, batch_input_metas, visualize=False)
        reshaped_img_feats, reshaped_data_samples = self._prepare_2d_head_inputs(
            img_feats, batch_data_samples)
        detections_2d = self._generate_and_process_2d_dets(
            reshaped_img_feats, reshaped_data_samples, batch_inputs_dict, visualize=False)
        rois, _ = self._generate_rois_from_detections(detections_2d)
        rois_center = self.get_center_points(rois)
        trimed_center_pts = self.batch_rois_center_by_cam_id(rois_center, batch_size=200)
        query_input = trimed_center_pts[..., 2:].clone()
        # ... (query_input 정규화 로직) ...
        query_input[..., 0] /= 1600
        query_input[..., 1] /= 900
        query_input[:,:,0] = query_input[:,:,0]/2
        
        B, N, C, H, W = sbs_img.shape
        raw_corrs, _, _, enc_out = self.corr(sbs_img.view(B*N, C, H, W), query_input)
        # ... (det_xyz, det_feat_sampled, projected_feat 생성 로직) ...
        raw_pred_center_pts = raw_corrs.clone()
        raw_pred_center_pts[..., 0] = (raw_pred_center_pts[..., 0] - 0.5) * 2
        raw_pred_center_pts[..., 0] *= 1600
        raw_pred_center_pts[..., 1] *= 900
        esitmated_z = self.z_estimator(raw_pred_center_pts, dense_depth_map, enc_out)
        esitmated_uvz = torch.cat([raw_pred_center_pts, esitmated_z['depth']], dim=-1)
        det_xyz = self.uvz_to_lidar_xyz(esitmated_uvz, lidar2imag)
        det_feat_sampled = self._sample_features_from_grid(feature_map=enc_out, coords=query_input)
        det_xyz_proc, det_feat_proc = self._prepare_camera_proposals(
            det_xyz, det_feat_sampled, B=B, N_cam=N)

        # 2. 이제 헤드의 predict 메서드에 모든 인자를 전달합니다.
        results_list_3d = self.bbox_head.predict(
            feats, det_xyz_proc, det_feat_proc, batch_input_metas)

        # 예측 결과(results_list_3d)를 원본 데이터 샘플(batch_data_samples)에 합쳐줍니다.
        results = self.add_pred_to_datasample(batch_data_samples,
                                              results_list_3d)
        return results

    def extract_feat(
        self,
        batch_inputs_dict,
        batch_input_metas,
        **kwargs,
    ):
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        features = []
        if imgs is not None:
            imgs = imgs.contiguous()
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for i, meta in enumerate(batch_input_metas):
                lidar2image.append(meta['lidar2img'])
                camera_intrinsics.append(meta['cam2img'])
                camera2lidar.append(meta['cam2lidar'])
                img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
                lidar_aug_matrix.append(
                    meta.get('lidar_aug_matrix', np.eye(4)))

            lidar2image = imgs.new_tensor(np.asarray(lidar2image))
            camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
            camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
            img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))
            img_feature ,raw_img_feature = self.extract_img_feat(imgs, deepcopy(points),
                                                lidar2image, camera_intrinsics,
                                                camera2lidar, img_aug_matrix,
                                                lidar_aug_matrix,
                                                batch_input_metas)
            features.append(img_feature)
        pts_feature = self.extract_pts_feat(batch_inputs_dict)
        features.append(pts_feature)

        if self.fusion_layer is not None:
            x = self.fusion_layer(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        x = self.pts_backbone(x)
        x = self.pts_neck(x)

        return x, raw_img_feature,lidar2image, camera_intrinsics, camera2lidar
    
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

            dense_depth_map = dense_map_from_depth_batch_v2(lidar_depth_mis,grid=3,iterations=3)
            dense_depth_img_mis = dense_depth_map.to(dtype=torch.uint8)
            dense_depth_img_color_mis = batch_colormap(dense_depth_img_mis)

            # 픽셀 값을 0.0 ~ 1.0 범위로 정규화하여 imshow가 올바르게 표시하도록 함
            img_min, img_max = imgs.min(), imgs.max()
            imgs = (imgs - img_min) / (img_max - img_min + 1e-8) # 0으로 나누는 것 방지

            N, V, C, H, W = imgs.shape
            imgs_reshaped = imgs.view(N * V, C, H, W)
            depth_reshaped = dense_depth_img_color_mis.view(N * V, C, H, W)

            img_resized = F.interpolate(imgs_reshaped, size=[192, 640], mode="bilinear")
            lidar_depth_mis_resized = F.interpolate(depth_reshaped, size=[192, 640], mode="bilinear")

            sbs_img = two_images_side_by_side_gpu(img_resized, lidar_depth_mis_resized)
            sbs_img = sbs_img.permute(0,3,1,2)
            sbs_img = tvtf.normalize(sbs_img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            sbs_img = sbs_img.view(N, V, C, 192, 640*2)

            # ############## input display ##########################
            if visualize and sbs_img is not None:
                display_depth_maps(imgs,dense_depth_img_color_mis,sbs_img)
                print("input dispaly end")
        
        return sbs_img, points ,dense_depth_map
    
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

    def _prepare_2d_head_inputs(
            self,
            img_feats: tuple,
            batch_data_samples: List[Det3DDataSample]
    ) -> tuple[tuple, List[Det3DDataSample]]:
        """
        Multi-view 이미지 피처와 DataSample을 2D 탐지 헤드에 맞게 변환합니다.
        (GT가 없는 샘플도 안전하게 2D 텐서로 처리하도록 수정됨)
        """
        N, V = img_feats[0].shape[:2]
        reshaped_img_feats = []
        for feat in img_feats:
            _N, _V, C, H, W = feat.shape
            reshaped_img_feats.append(feat.view(_N * _V, C, H, W))
        reshaped_img_feats = tuple(reshaped_img_feats)

        camera_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
            'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]
        reshaped_data_samples = []
        device = img_feats[0].device

        for sample in batch_data_samples:
            multi_cam_2d_anns = sample.metainfo['ann_info_aug_2d_per_cam']
            for cam_name in camera_types:
                new_sample = Det3DDataSample()
                new_sample.set_metainfo(sample.metainfo)
                if 'gt_instances_3d' in sample:
                    new_sample.gt_instances_3d = sample.gt_instances_3d

                cam_gt = multi_cam_2d_anns[cam_name]
                gt_instances_2d = InstanceData()
                
                # --- ✨ 핵심 수정 로직 ---
                # 1. bbox 데이터를 가져옵니다. (비어있을 수 있음)
                gt_bboxes = cam_gt['gt_bboxes']
                bboxes_tensor = torch.as_tensor(
                    gt_bboxes, dtype=torch.float32, device=device)
                
                # 2. 텐서가 비어있더라도 항상 (N, 4) 형태의 2D가 되도록 보장합니다.
                #    비어있을 경우 shape=(0, 4)가 됩니다.
                gt_instances_2d.bboxes = bboxes_tensor.reshape(-1, 4)
                
                # 3. 라벨도 동일하게 처리합니다.
                string_labels = cam_gt['gt_labels']
                numeric_labels = [self.name_to_idx.get(name, -1) for name in string_labels]
                labels_tensor = torch.as_tensor(
                    numeric_labels, dtype=torch.long, device=device)
                gt_instances_2d.labels = labels_tensor.reshape(-1)
                # -------------------------
                
                new_sample.gt_instances = gt_instances_2d
                reshaped_data_samples.append(new_sample)
        
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
            if self.train_cfg.get('complement_2d_gt', -1) > 0:
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
        rois_center 텐서에서 직접 카메라 ID를 읽어, 존재하는 카메라에 대해서만
        데이터를 배치(batch) 형태로 변환합니다.
        """
        device = rois_center.device
        
        # 입력 텐서가 비어있는 경우, 빈 텐서를 반환
        if rois_center.shape[0] == 0:
            # num_cams를 알 수 없으므로 기본값 6으로 설정하거나, 호출하는 쪽에서 처리
            return torch.zeros((6, batch_size, 4), device=device)

        # 1. rois_center의 0열에서 모든 카메라 인덱스를 추출합니다.
        cam_indices_tensor = rois_center[:, 0]
        
        # 2. 존재하는 고유한 카메라 ID 목록을 찾습니다.
        unique_cam_ids = torch.unique(cam_indices_tensor).long().cpu().tolist()
        
        # 3. 최대 카메라 ID를 기반으로 출력 텐서의 크기를 결정합니다.
        #    예: [0, 1, 5]가 있다면, 크기가 6인 텐서 (0~5)를 생성합니다.
        max_cam_id = int(torch.max(cam_indices_tensor).item())
        num_total_cams = max_cam_id + 1
        
        batched_centers = torch.zeros((num_total_cams, batch_size, 4), device=device)
        
        # 원본 객체 ID 저장 (검증 로직은 그대로 유지)
        original_obj_ids = rois_center[:, 1].cpu().numpy()
        
        # 4. 하드코딩된 range(num_cams) 대신, 실제 존재하는 카메라 ID들을 순회합니다.
        for cam_id in unique_cam_ids:
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
                    # 이미 선택된 인덱스를 제외하고 남은 풀에서 추가 선택
                    pool = np.setdiff1d(all_indices, selected_indices)
                    # 만약 풀이 부족하면 복원 추출 허용
                    replace = len(pool) < remaining
                    extra_indices = np.random.choice(
                        pool,
                        size=remaining,
                        replace=replace
                    )
                    selected_indices.extend(extra_indices)
                    
                selected_indices = torch.tensor(selected_indices, device=device, dtype=torch.long)
                cam_centers = cam_centers[selected_indices]
            
            batched_centers[cam_id, :cam_centers.size(0)] = cam_centers[:batch_size]
        
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
    
    def convert_boxes_to_original_scale_FINAL_DEBUG(
        self,
        pred_results_list: List[torch.Tensor],
        data_samples_list: List
    ) -> List[torch.Tensor]:
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
            fW, fH = params_dict['final_dim']
            
            # ==========================================================
            # ===== 버그를 수정한 올바른 corners 생성 코드입니다. =====
            # ==========================================================
            x1, y1, x2, y2 = aug_bboxes.T
            corners = torch.stack([x1, y1, x2, y1, x2, y2, x1, y2], dim=-1).view(-1, 4, 2)

            # ==================== DEBUG PRINT CODE ====================
            if flip and pred_tensor.shape[0] > 0:
                print("\n--- STARTING FINAL DEBUG FOR FLIPPED IMAGE ---")
                print(f"Params: {params_dict}")
                print(f"Initial Corners (Corrected):\n{corners[0].cpu().numpy().round(2)}")
            # ==========================================================

            # Sequential Inverse Transformation
            if rotate != 0:
                angle = math.radians(rotate)
                cos, sin = math.cos(angle), math.sin(angle)
                cx, cy = (fW - 1) / 2, (fH - 1) / 2
                R_inv = torch.tensor([[cos, sin], [-sin, cos]], device=corners.device, dtype=torch.float32)
                corners = (corners - torch.tensor([cx, cy], device=corners.device)) @ R_inv.T + torch.tensor([cx, cy], device=corners.device)
                if flip and pred_tensor.shape[0] > 0: print(f"After Inverse Rotate:\n{corners[0].cpu().numpy().round(2)}")

            if flip:
                corners[..., 0] = (fW - 1) - corners[..., 0]
                if pred_tensor.shape[0] > 0: print(f"After Inverse Flip:\n{corners[0].cpu().numpy().round(2)}")

            corners[..., 0] += crop[0]
            corners[..., 1] += crop[1]
            if flip and pred_tensor.shape[0] > 0: print(f"After Inverse Crop:\n{corners[0].cpu().numpy().round(2)}")

            corners /= resize
            if flip and pred_tensor.shape[0] > 0:
                print(f"After Inverse Resize (Final Coords):\n{corners[0].cpu().numpy().round(2)}")
                print("--- ENDING FINAL DEBUG ---")

            min_coords = torch.min(corners, dim=1).values
            max_coords = torch.max(corners, dim=1).values
            original_bboxes = torch.cat([min_coords, max_coords], dim=1)
            
            new_pred_tensor = pred_tensor.clone()
            new_pred_tensor[:, :4] = original_bboxes
            converted_results.append(new_pred_tensor)

        return converted_results
    
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
            
            # ====================================================================
            # ===== 여기가 모든 문제의 원인이었던 변수 할당 오류 수정 부분입니다 =====
            # ====================================================================
            fH, fW = params_dict['final_dim'] # [Height, Width] 순서로 할당
            
            # --- 역변환 행렬 구성 ---
            M_resize_inv = torch.eye(3, device=pred_tensor.device, dtype=torch.float32)
            M_resize_inv[0, 0] = 1 / resize
            M_resize_inv[1, 1] = 1 / resize

            M_crop_inv = torch.eye(3, device=pred_tensor.device, dtype=torch.float32)
            M_crop_inv[0, 2] = crop[0]
            M_crop_inv[1, 2] = crop[1]

            M_flip_inv = torch.eye(3, device=pred_tensor.device, dtype=torch.float32)
            if flip:
                M_flip_inv[0, 0] = -1
                M_flip_inv[0, 2] = fW - 1

            M_rotate_inv = torch.eye(3, device=pred_tensor.device, dtype=torch.float32)
            if rotate != 0:
                angle = math.radians(rotate)
                cos, sin = math.cos(angle), math.sin(angle)
                cx, cy = (fW - 1) / 2, (fH - 1) / 2
                
                T1 = torch.tensor([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], device=pred_tensor.device, dtype=torch.float32)
                # 올바른 방향인 시계 방향(Clockwise) 역회전 행렬
                R_inv = torch.tensor([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]], device=pred_tensor.device, dtype=torch.float32)
                T2 = torch.tensor([[1, 0, cx], [0, 1, cy], [0, 0, 1]], device=pred_tensor.device, dtype=torch.float32)
                M_rotate_inv = T2 @ R_inv @ T1

            # 최종 역변환 행렬 계산
            M_total_inv = M_resize_inv @ M_crop_inv @ M_flip_inv @ M_rotate_inv

            # Bounding Box 변환 적용
            x1, y1, x2, y2 = aug_bboxes.T
            corners = torch.stack([x1, y1, x2, y1, x2, y2, x1, y2], dim=-1).view(-1, 4, 2)
            corners_hom = torch.cat([corners, torch.ones(corners.shape[0], 4, 1, device=corners.device)], dim=-1)
            
            M_total_inv = M_total_inv.to(corners_hom.dtype)
            transformed_corners_hom = (M_total_inv @ corners_hom.transpose(1, 2)).transpose(1, 2)
            
            transformed_corners = transformed_corners_hom[..., :2] / transformed_corners_hom[..., 2, None]
            
            min_coords = torch.min(transformed_corners, dim=1).values
            max_coords = torch.max(transformed_corners, dim=1).values
            original_bboxes = torch.cat([min_coords, max_coords], dim=1)
            
            new_pred_tensor = pred_tensor.clone()
            new_pred_tensor[:, :4] = original_bboxes
            converted_results.append(new_pred_tensor)

        return converted_results

    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats,img_feats,lidar2imag,camera_intrinsics,camera2lidar = self.extract_feat(
                                                                    batch_inputs_dict, batch_input_metas)
        sbs_img, pertubed_points,dense_depth_map = self.extract_sbs_img(batch_inputs_dict, batch_input_metas,visualize=False)

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

        # ✨ 2. 헬퍼 함수를 호출하여 2D 탐지 결과 생성, 처리, 시각화를 한 번에 수행
        #    시각화가 필요할 때 visualize=True로 설정
        detections_2d = self._generate_and_process_2d_dets(
            reshaped_img_feats, 
            reshaped_data_samples, 
            batch_inputs_dict, 
            visualize=False # 디버깅 시 True, 평소에는 False
        )

        detections_2d_orig_coords = self.convert_boxes_to_original_scale(
            pred_results_list=detections_2d,
            data_samples_list=reshaped_data_samples
        )

        rois , proposla_list = self._generate_rois_from_detections(detections_2d_orig_coords)
        rois_center = self.get_center_points(rois)
        trimed_center_pts =self.batch_rois_center_by_cam_id(rois_center,batch_size=200)
        cam_ids = trimed_center_pts[..., 0]
        object_ids = trimed_center_pts[..., 1]

        query_input = trimed_center_pts[..., 2:].clone()
        query_input[..., 0] /= 1600
        query_input[..., 1] /= 900
        query_input[:,:,0] = query_input[:,:,0]/2    # recaling points for sbs image resizing
        query_input[:,:,1] = query_input[:,:,1]

        B,N,C,H,W = sbs_img.shape
        raw_corrs, cycle, corr_mask, enc_out = self.corr(sbs_img.view(B*N,C,H,W), query_input)

        # # ##### 검증용 display ######
        # from .imageprocessing_unit import draw_correspondences
        # # gt_corrs = torch.cat([query_input,corr_target],dim=-1)
        # pred_corrs = torch.cat([query_input,raw_corrs],dim=-1)
        # # vis_step_counter는 __init__에서 0으로 초기화 되어야 합니다.
        # self.vis_step_counter += 1
        # for cid in range(12):
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

        raw_pred_center_pts = raw_corrs.clone()
        raw_pred_center_pts[..., 0] = (raw_pred_center_pts[..., 0] - 0.5) * 2
        raw_pred_center_pts[..., 0] *= 1600
        raw_pred_center_pts[..., 1] *= 900

        esitmated_z = self.z_estimator(raw_pred_center_pts, dense_depth_map,enc_out)
        esitmated_uvz =torch.cat([raw_pred_center_pts, esitmated_z['depth']],dim=-1)

        det_xyz = self.uvz_to_lidar_xyz(esitmated_uvz, lidar2imag)
        #    이때 query_input은 -1~1 범위로 정규화된 상태여야 합니다.
        det_feat_sampled = self._sample_features_from_grid(feature_map=enc_out, coords=query_input)
        det_xyz_proc, det_feat_proc = self._prepare_camera_proposals(det_xyz,det_feat_sampled,B=B,N_cam=N)

        if self.with_bbox_head:
            bbox_loss = self.bbox_head.loss(feats, det_xyz_proc, det_feat_proc, batch_data_samples)

        losses.update(bbox_loss)

        return losses

