# modify from https://github.com/mit-han-lab/bevfusion
from typing import Any, Dict

import numpy as np
import torch
from mmcv.transforms import BaseTransform
from PIL import Image

from mmdet3d.datasets import GlobalRotScaleTrans
from mmdet3d.structures import points_cam2img
from mmdet3d.registry import TRANSFORMS
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import datetime
from pyquaternion import Quaternion
from mmdet3d.structures import LiDARInstance3DBoxes


@TRANSFORMS.register_module()
class ImageAug3D(BaseTransform):

    def __init__(self, final_dim, resize_lim, bot_pct_lim, rot_lim, rand_flip,
                 is_train):
        self.final_dim = final_dim
        self.resize_lim = resize_lim
        self.bot_pct_lim = bot_pct_lim
        self.rand_flip = rand_flip
        self.rot_lim = rot_lim
        self.is_train = is_train

    def sample_augmentation(self, results):
        H, W = results['ori_shape']
        fH, fW = self.final_dim
        if self.is_train:
            resize = np.random.uniform(*self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.random.uniform(*self.bot_pct_lim)) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.rand_flip and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.rot_lim)
        else:
            resize = np.mean(self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.bot_pct_lim)) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform(self, img, rotation, translation, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        img = Image.fromarray(img.astype('uint8'), mode='RGB')
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        rotation *= resize
        translation -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
        theta = rotate / 180 * np.pi
        A = torch.Tensor([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)],
        ])
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        rotation = A.matmul(rotation)
        translation = A.matmul(translation) + b

        return img, rotation, translation

    # def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
    #     imgs = data['img']
    #     new_imgs = []
    #     transforms = []
    #     for img in imgs:
    #         resize, resize_dims, crop, flip, rotate = self.sample_augmentation(
    #             data)
    #         post_rot = torch.eye(2)
    #         post_tran = torch.zeros(2)
    #         new_img, rotation, translation = self.img_transform(
    #             img,
    #             post_rot,
    #             post_tran,
    #             resize=resize,
    #             resize_dims=resize_dims,
    #             crop=crop,
    #             flip=flip,
    #             rotate=rotate,
    #         )
    #         transform = torch.eye(4)
    #         transform[:2, :2] = rotation
    #         transform[:2, 3] = translation
    #         new_imgs.append(np.array(new_img).astype(np.float32))
    #         transforms.append(transform.numpy())
    #     data['img'] = new_imgs
    #     # update the calibration matrices
    #     data['img_aug_matrix'] = transforms
    #     return data
    
    # ====================================================================
    # ===== 아래 transform 함수에 파라미터를 저장하는 로직이 추가되었습니다 =====
    # ====================================================================
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        imgs = data['img']
        new_imgs = []
        transforms = []
        
        # BBox 역변환에 사용할 파라미터들을 저장할 리스트
        img_aug_params = []

        for img in imgs:
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(data)
            
            # 역변환에 필요한 파라미터들을 딕셔너리 형태로 저장
            params = {
                'resize': resize,
                'crop': crop,
                'flip': flip,
                'rotate': rotate,
                'final_dim': self.final_dim,
            }
            img_aug_params.append(params)
            
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            new_img, rotation, translation = self.img_transform(
                img,
                post_rot,
                post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            new_imgs.append(np.array(new_img).astype(np.float32))
            transforms.append(transform.numpy())
        
        data['img'] = new_imgs
        data['img_aug_matrix'] = transforms
        
        # ===== 저장된 파라미터를 data 딕셔너리에 추가 =====
        data['img_aug_params'] = img_aug_params
        
        return data

@TRANSFORMS.register_module()
class CustomImageAug3D(ImageAug3D):
    """
    기존 ImageAug3D를 실행하면서 'img_original' 키를 보존하는 래퍼 클래스.
    """
    def transform(self, results: dict) -> dict:
        # 1. 'img_original' 키가 있다면 잠시 빼서 보관합니다.
        img_original = results.get('img_original', None)

        # 2. 부모 클래스(ImageAug3D)의 증강 로직을 그대로 실행합니다.
        augmented_results = super().transform(results)

        # 3. 증강된 결과에 보관해두었던 'img_original'을 다시 넣어줍니다.
        if img_original is not None:
            augmented_results['img_original'] = img_original
        
        return augmented_results
    
@TRANSFORMS.register_module()
class GenerateUpdated2DAnnotations(BaseTransform):
    """
    [ValueError 및 원근 나누기 오류 최종 해결 버전]
    1. 2D 좌표를 올바른 4D 동차 좌표계로 변환하여 행렬 차원 문제를 해결.
    2. 원근 나누기 시 올바른 값(w')을 사용하도록 수정.
    """

    def __init__(self, classes=None, visualize=False, vis_dir='vis_outputs'):
        super().__init__()
        self.classes = classes if classes is not None else [
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
            'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]
        self.visualize = visualize
        self.vis_dir = vis_dir
        self.camera_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
            'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

    def _create_empty_annotations(self) -> dict:
        return {'gt_bboxes': np.zeros((0, 4), dtype=np.float32), 'gt_labels': []}

    def transform(self, results: dict) -> dict:
        gt_bboxes_3d = results.get('gt_bboxes_3d', None)

        if gt_bboxes_3d is None or len(gt_bboxes_3d.tensor) == 0:
            ann_info_aug_2d_per_cam = {}
            for cam_name in self.camera_types:
                ann_info_aug_2d_per_cam[cam_name] = self._create_empty_annotations()
            results['ann_info_aug_2d_per_cam'] = ann_info_aug_2d_per_cam
            return results

        corners_3d_augmented = gt_bboxes_3d.corners.cpu().numpy()
        num_gt = corners_3d_augmented.shape[0]
        # final_img_shape = results['img'].shape[1:3]
        final_img_shape = (256,704)
        lidar2img_matrices = results['lidar2img']
        img_aug_matrices = results['img_aug_matrix']
        lidar_aug_mat = results['lidar_aug_matrix']

        ann_info_aug_2d_per_cam = {}

        for cam_idx, cam_name in enumerate(self.camera_types):
            lidar2img_mat = lidar2img_matrices[cam_idx]
            img_aug_mat = img_aug_matrices[cam_idx]

            if isinstance(lidar2img_mat, torch.Tensor): lidar2img_mat = lidar2img_mat.cpu().numpy()
            if isinstance(img_aug_mat, torch.Tensor): img_aug_mat = img_aug_mat.cpu().numpy()
            if isinstance(lidar_aug_mat, torch.Tensor): lidar_aug_mat = lidar_aug_mat.cpu().numpy()

            # 1단계: 3D 증강 되돌리기
            lidar_aug_inv = np.linalg.inv(lidar_aug_mat)
            corners_3d_augmented_flat = corners_3d_augmented.reshape(-1, 3)
            corners_3d_augmented_hom = np.concatenate([corners_3d_augmented_flat, np.ones((corners_3d_augmented_flat.shape[0], 1))], axis=1)
            corners_3d_original_hom = corners_3d_augmented_hom @ lidar_aug_inv.T
            corners_3d_original = corners_3d_original_hom[:, :3]

            # 2단계: 3D -> 2D 투영 (라이브러리 함수 사용)
            corners_2d_original = points_cam2img(corners_3d_original, lidar2img_mat)

            # 3단계: 2D 증강 적용
            # [수정 1] 4x4 행렬과 곱하기 위해 4D 동차 좌표 [u, v, 0, 1]로 변환
            corners_2d_original_hom = np.concatenate(
                [corners_2d_original, 
                 np.zeros((corners_2d_original.shape[0], 1)),
                 np.ones((corners_2d_original.shape[0], 1))], 
                axis=1)

            # (N, 4) @ (4, 4) -> (N, 4) 곱셈
            corners_2d_final_hom = corners_2d_original_hom @ img_aug_mat.T
            
            eps = 1e-5
            
            # [수정 2] 원근 나누기를 위해 네 번째 값(w')을 사용
            depth = corners_2d_final_hom[:, 3]

            corners_2d_final = corners_2d_final_hom[:, :2] / (depth[:, np.newaxis] + eps)
            corners_2d_final = corners_2d_final.reshape(num_gt, 8, 2)
            
            # 이하 로직은 동일
            on_img = (corners_2d_final[..., 0] >= 0) & (corners_2d_final[..., 0] < final_img_shape[1]) & \
                     (corners_2d_final[..., 1] >= 0) & (corners_2d_final[..., 1] < final_img_shape[0])
            valid_mask = on_img.any(axis=1)

            if np.any(valid_mask):
                valid_corners = corners_2d_final[valid_mask]
                valid_corners[..., 0] = np.clip(valid_corners[..., 0], 0, final_img_shape[1])
                valid_corners[..., 1] = np.clip(valid_corners[..., 1], 0, final_img_shape[0])
                min_uv = np.min(valid_corners, axis=1)
                max_uv = np.max(valid_corners, axis=1)
                bboxes_2d = np.concatenate([min_uv, max_uv], axis=1).astype(np.float32)
                labels_3d_np = results['gt_labels_3d']
                string_labels = [self.classes[l] for l in labels_3d_np[valid_mask]]
                ann_info_aug_2d_per_cam[cam_name] = {'gt_bboxes': bboxes_2d, 'gt_labels': string_labels}
            else:
                ann_info_aug_2d_per_cam[cam_name] = self._create_empty_annotations()
        
        results['ann_info_aug_2d_per_cam'] = ann_info_aug_2d_per_cam
        
        if self.visualize:
            # (이하 시각화 코드는 동일)
            os.makedirs(self.vis_dir, exist_ok=True)
            img_array = results['img']
            fig, axes = plt.subplots(2, 3, figsize=(24, 8))
            axes = axes.flatten()
            fig.suptitle(f"Sample IDX: {results.get('sample_idx', 'N/A')}", fontsize=16)
            for i, cam_name in enumerate(self.camera_types):
                ax = axes[i]
                cam_img_hwc = img_array[i]
                cam_img_display = np.clip(cam_img_hwc, 0, 255).astype(np.uint8)
                # cam_img_display = cam_img_display[..., ::-1]
                ax.imshow(cam_img_display)
                ax.set_title(cam_name)
                ax.axis('off')
                annotations = ann_info_aug_2d_per_cam[cam_name]
                for bbox, label in zip(annotations['gt_bboxes'], annotations['gt_labels']):
                    x1, y1, x2, y2 = bbox
                    width, height = x2 - x1, y2 - y1
                    rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='lime', facecolor='none')
                    ax.add_patch(rect)
                    ax.text(x1, y1 - 5, label, bbox=dict(facecolor='lime', alpha=0.8), fontsize=8, color='black')
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            sample_id = results.get('sample_idx', 'sample')
            save_path = os.path.join(self.vis_dir, f'{sample_id}_{timestamp}.jpg')
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close(fig)
        
        results['gt_bboxes'] = np.zeros((0, 4), dtype=np.float32)
        results['gt_labels'] = np.zeros((0,), dtype=np.int64)

        return results

@TRANSFORMS.register_module()
class BEVFusionRandomFlip3D:
    """Compared with `RandomFlip3D`, this class directly records the lidar
    augmentation matrix in the `data`."""

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        flip_horizontal = np.random.choice([0, 1])
        flip_vertical = np.random.choice([0, 1])

        rotation = np.eye(3)
        if flip_horizontal:
            rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ rotation
            if 'points' in data:
                data['points'].flip('horizontal')
            if 'gt_bboxes_3d' in data:
                data['gt_bboxes_3d'].flip('horizontal')
            if 'gt_masks_bev' in data:
                data['gt_masks_bev'] = data['gt_masks_bev'][:, :, ::-1].copy()

        if flip_vertical:
            rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ rotation
            if 'points' in data:
                data['points'].flip('vertical')
            if 'gt_bboxes_3d' in data:
                data['gt_bboxes_3d'].flip('vertical')
            if 'gt_masks_bev' in data:
                data['gt_masks_bev'] = data['gt_masks_bev'][:, ::-1, :].copy()

        if 'lidar_aug_matrix' not in data:
            data['lidar_aug_matrix'] = np.eye(4)
        data['lidar_aug_matrix'][:3, :] = rotation @ data[
            'lidar_aug_matrix'][:3, :]
        return data


@TRANSFORMS.register_module()
class BEVFusionGlobalRotScaleTrans(GlobalRotScaleTrans):
    """Compared with `GlobalRotScaleTrans`, the augmentation order in this
    class is rotation, translation and scaling (RTS)."""

    def transform(self, input_dict: dict) -> dict:
        """Private function to rotate, scale and translate bounding boxes and
        points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans' and `gt_bboxes_3d` are updated
            in the result dict.
        """
        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        self._rot_bbox_points(input_dict)

        if 'pcd_scale_factor' not in input_dict:
            self._random_scale(input_dict)
        self._trans_bbox_points(input_dict)
        self._scale_bbox_points(input_dict)

        input_dict['transformation_3d_flow'].extend(['R', 'T', 'S'])

        lidar_augs = np.eye(4)
        lidar_augs[:3, :3] = input_dict['pcd_rotation'].T * input_dict[
            'pcd_scale_factor']
        lidar_augs[:3, 3] = input_dict['pcd_trans'] * \
            input_dict['pcd_scale_factor']

        if 'lidar_aug_matrix' not in input_dict:
            input_dict['lidar_aug_matrix'] = np.eye(4)
        input_dict[
            'lidar_aug_matrix'] = lidar_augs @ input_dict['lidar_aug_matrix']

        return input_dict


@TRANSFORMS.register_module()
class GridMask(BaseTransform):

    def __init__(
        self,
        use_h,
        use_w,
        max_epoch,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=0,
        prob=1.0,
        fixed_prob=False,
    ):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
        self.epoch = None
        self.max_epoch = max_epoch
        self.fixed_prob = fixed_prob

    def set_epoch(self, epoch):
        self.epoch = epoch
        if not self.fixed_prob:
            self.set_prob(self.epoch, self.max_epoch)

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * self.epoch / self.max_epoch

    def transform(self, results):
        if np.random.rand() > self.prob:
            return results
        imgs = results['img']
        h = imgs[0].shape[0]
        w = imgs[0].shape[1]
        self.d1 = 2
        self.d2 = min(h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(self.d1, self.d2)
        if self.ratio == 1:
            self.length = np.random.randint(1, d)
        else:
            self.length = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.length, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.length, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h,
                    (ww - w) // 2:(ww - w) // 2 + w]

        mask = mask.astype(np.float32)
        mask = mask[:, :, None]
        if self.mode == 1:
            mask = 1 - mask

        # mask = mask.expand_as(imgs[0])
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float()
            offset = (1 - mask) * offset
            imgs = [x * mask + offset for x in imgs]
        else:
            imgs = [x * mask for x in imgs]

        results.update(img=imgs)
        return results