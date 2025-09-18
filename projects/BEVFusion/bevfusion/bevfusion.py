from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from mmengine.utils import is_list_of
from torch import Tensor
from torch.nn import functional as F
from torchvision.transforms import functional as tvtf
from mmengine.structures import InstanceData
from mmdet.structures.bbox import bbox_overlaps 

from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList
from .ops import Voxelization
from .imageprocessing_unit import (dense_map_from_depth_batch_v2, 
                                   batch_colormap,two_images_side_by_side_gpu,
                                   display_depth_maps
                                   )
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import os


@MODELS.register_module()
class BEVFusion(Base3DDetector):

    def __init__(
        self,
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
        cotr: Optional[dict] = None,
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

        self.bbox_head = MODELS.build(bbox_head)
        self.img_bbox_head = MODELS.build(img_bbox_head)
        self.cotr = MODELS.build(cotr)

        self.init_weights()

        # 이 부분이 이미 있다면 그대로 두고, 없다면 추가하세요.
        self.class_names = class_names
        self.name_to_idx = {name: i for i, name in enumerate(self.class_names)}

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

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
    # ) -> torch.Tensor:
    #     B, N, C, H, W = x.size()
    #     x = x.view(B * N, C, H, W).contiguous()

    #     x = self.img_backbone(x)
    #     x = self.img_neck(x)

    #     if not isinstance(x, torch.Tensor):
    #         x = x[0]

    #     BN, C, H, W = x.size()
    #     x = x.view(B, int(BN / B), C, H, W)
    #     img_feature = x.clone()

    #     with torch.autocast(device_type='cuda', dtype=torch.float32):
    #         x = self.view_transform(
    #             x,
    #             points,
    #             lidar2image,
    #             camera_intrinsics,
    #             camera2lidar,
    #             img_aug_matrix,
    #             lidar_aug_matrix,
    #             img_metas,
    #         )
    #     return x ,img_feature
    
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

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

        if self.with_bbox_head:
            outputs = self.bbox_head.predict(feats, batch_input_metas)

        res = self.add_pred_to_datasample(batch_data_samples, outputs)

        return res

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
        **kwargs,
    ):
        imgs = batch_inputs_dict.get('img_original', None)
        points = batch_inputs_dict.get('points_original', None)
        
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

            # # ############## input display ##########################
            # display_depth_maps(imgs,dense_depth_img_color_mis,sbs_img)
            # print("input dispaly end")
        
        return sbs_img, points
    
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
            multi_cam_2d_anns = sample.metainfo['ann_info_2d_per_cam']
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

            # 4. 정답(Ground Truth) 바운딩 박스 그리기 (녹색)
            if 'bboxes' in gt_sample.gt_instances:
                gt_bboxes = gt_sample.gt_instances.bboxes.cpu().numpy()
                gt_labels = gt_sample.gt_instances.labels.cpu().numpy()
                for box, label_idx in zip(gt_bboxes, gt_labels):
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    rect = patches.Rectangle(
                        (x1, y1), w, h, linewidth=2, edgecolor='g', facecolor='none')
                    ax.add_patch(rect)
                    ax.text(
                        x1, y1 + h + 20,
                        f'{self.class_names[int(label_idx)]}',
                        bbox=dict(facecolor='g', alpha=0.5),
                        fontsize=10, color='white')
            
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
            orig_images = batch_inputs_dict['img_original']
            N, V, C, H, W = orig_images.shape
            orig_images_reshaped = orig_images.view(N * V, C, H, W)
            
            self.display_2d_results(
                images=orig_images_reshaped,
                predictions=detections_2d,
                ground_truths=reshaped_data_samples
            )
            
        return detections_2d

    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats,img_feats,lidar2imag,camera_intrinsics,camera2lidar = self.extract_feat(batch_inputs_dict, batch_input_metas)
        sbs_img, pertubed_points = self.extract_sbs_img(batch_inputs_dict, batch_input_metas)

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

        # ✨ 2. 헬퍼 함수를 호출하여 2D 탐지 결과 생성, 처리, 시각화를 한 번에 수행
        #    시각화가 필요할 때 visualize=True로 설정
        detections_2d = self._generate_and_process_2d_dets(
            reshaped_img_feats, 
            reshaped_data_samples, 
            batch_inputs_dict, 
            visualize=True # 디버깅 시 True, 평소에는 False
        )

        if self.with_bbox_head:
            bbox_loss = self.bbox_head.loss(feats, batch_data_samples)

        losses.update(bbox_loss)

        return losses

