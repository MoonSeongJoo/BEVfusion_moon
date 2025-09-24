import numpy as np
import torch
# from mmdet3d.structures.ops import limit_period
from mmdet3d.registry import TRANSFORMS
# from mmdet3d.structures import points_cam2img

@TRANSFORMS.register_module()
class BEVDetProject3DBoxToImage:
    def __call__(self, results):
        gt_bboxes_3d = results['gt_bboxes_3d']
        gt_labels_3d = results['gt_labels_3d']
        lidar2img = np.stack(results['lidar2img']) # (N_views, 4, 4)
        img_shape = results['img_shape'][0] # (H, W, C), 모든 뷰가 같다고 가정
        
        # LiDARInstance3DBoxes를 코너 좌표로 변환
        corners_3d = gt_bboxes_3d.corners.numpy()  # (N_boxes, 8, 3)
        num_boxes = corners_3d.shape[0]
        
        if num_boxes == 0:
            results['gt_bboxes'] = torch.zeros((0, 4), dtype=torch.float32)
            results['gt_labels'] = torch.zeros((0,), dtype=torch.long)
            return results

        corners_3d_hom = np.concatenate(
            [corners_3d, np.ones((num_boxes, 8, 1))], axis=-1)

        all_corners_2d = []
        for i in range(num_boxes):
            box_corners = corners_3d_hom[i] # (8, 4)
            # 모든 뷰에 대해 투영
            corners_2d_views = box_corners @ lidar2img.transpose(0, 2, 1) # (N_views, 8, 4)
            
            # 정규화
            corners_2d_views[..., :2] /= corners_2d_views[..., 2:3]
            all_corners_2d.append(corners_2d_views)

        all_corners_2d = np.stack(all_corners_2d, axis=0) # (N_boxes, N_views, 8, 4)

        # 이미지 경계 내에 있는지 확인
        on_screen = (
            (all_corners_2d[..., 0] > 0) &
            (all_corners_2d[..., 0] < img_shape[1]) &
            (all_corners_2d[..., 1] > 0) &
            (all_corners_2d[..., 1] < img_shape[0]) &
            (all_corners_2d[..., 2] > 0) # 카메라 앞에 있는지 확인
        )
        
        # 각 박스가 보이는 뷰의 수
        visible_views_per_box = on_screen.all(axis=2).sum(axis=1)
        visible_mask = visible_views_per_box > 0
        
        if not np.any(visible_mask):
            results['gt_bboxes'] = torch.zeros((0, 4), dtype=torch.float32)
            results['gt_labels'] = torch.zeros((0,), dtype=torch.long)
            return results

        # 보이는 박스만 필터링
        visible_corners_2d = all_corners_2d[visible_mask]
        visible_labels = gt_labels_3d[visible_mask]

        # 2D BBox 생성 (min/max)
        min_coords = visible_corners_2d[..., :2].min(axis=2) # (N_visible_boxes, N_views, 2)
        max_coords = visible_corners_2d[..., :2].max(axis=2) # (N_visible_boxes, N_views, 2)
        
        # 여기서는 모든 뷰를 통합하여 하나의 2D BBox 세트를 만듭니다.
        # 가장 처음 보이는 뷰의 박스를 대표로 사용하겠습니다.
        final_bboxes = []
        final_labels = []

        for i in range(len(visible_labels)):
            box_min_coords = min_coords[i] # (N_views, 2)
            box_max_coords = max_coords[i] # (N_views, 2)
            box_on_screen_views = on_screen[visible_mask][i].all(axis=1) # (N_views,)
            
            # 이 박스가 보이는 첫 번째 뷰의 인덱스
            first_visible_view_idx = np.where(box_on_screen_views)[0][0]
            
            x1, y1 = box_min_coords[first_visible_view_idx]
            x2, y2 = box_max_coords[first_visible_view_idx]

            # 이미지 경계에 맞게 클리핑
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_shape[1], x2)
            y2 = min(img_shape[0], y2)
            
            # 너무 작은 박스는 제외
            if x2 - x1 > 1 and y2 - y1 > 1:
                final_bboxes.append([x1, y1, x2, y2])
                final_labels.append(visible_labels[i])

        if not final_bboxes:
            results['gt_bboxes'] = torch.zeros((0, 4), dtype=torch.float32)
            results['gt_labels'] = torch.zeros((0,), dtype=torch.long)
        else:
            results['gt_bboxes'] = torch.tensor(final_bboxes, dtype=torch.float32)
            results['gt_labels'] = torch.tensor(final_labels, dtype=torch.long)
            
        return results