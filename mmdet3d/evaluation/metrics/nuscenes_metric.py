# nuscenes_metric.py 파일 전체를 아래 내용으로 교체하세요.
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pyquaternion
import torch
import os.path as osp
import tempfile
import mmengine
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box as NuScenesBox
from mmdet3d.registry import METRICS
from mmdet3d.structures import LiDARInstance3DBoxes, CameraInstance3DBoxes
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box

# ===================================================================
# ✨ 1. 수정된 output_to_nusc_box_global 함수
# ===================================================================
def ensure_orthogonal(mat):
    """Make rotation matrix orthogonal (safe for Quaternion)."""
    u, _, vh = np.linalg.svd(mat)
    return np.dot(u, vh)

def output_to_nusc_box_global(detection, info_dict, min_score=0.01):
    """
    Convert detection results to NuScenes global frame Boxes.
    Handles size, velocity, and attributes from the model's output.
    """
    boxes_list = []
    
    # --- Prediction tensors ---
    bboxes3d = detection['bboxes_3d']
    
    # ✨ 수정: .cpu()를 호출하여 GPU 텐서를 CPU로 복사한 뒤 NumPy로 변환합니다.
    bboxes_tensor = bboxes3d.tensor if hasattr(bboxes3d, 'tensor') else bboxes3d
    bboxes = bboxes_tensor.cpu().numpy()
    scores = detection['scores_3d'].cpu().numpy()
    labels = detection['labels_3d'].cpu().numpy()

    # --- Score threshold filtering ---
    if len(bboxes) == 0:
        return []
    keep = scores > min_score
    bboxes = bboxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    
    # Also filter attributes and velocity if they exist
    if 'attr_labels' in detection:
        # ✨ 수정: .cpu() 호출 추가
        attrs = detection['attr_labels'].cpu().numpy()[keep]
    else:
        attrs = -np.ones_like(labels) # Use -1 as a placeholder for no attribute

    # --- Transformation matrices ---
    # lidar2ego와 ego2global은 data_infos에서 오므로 이미 CPU 데이터(NumPy)입니다.
    lidar2ego = np.array(info_dict['lidar_points']['lidar2ego'])
    lidar2ego_rot = ensure_orthogonal(lidar2ego[:3, :3])
    lidar2ego_trans = lidar2ego[:3, 3]

    ego2global = np.array(info_dict['ego2global'])
    ego2global_rot = ensure_orthogonal(ego2global[:3, :3])
    ego2global_trans = ego2global[:3, 3]

    for i in range(len(bboxes)):
        center = bboxes[i, :3]
        
        # mASE 수정: (l, w, h) -> (w, l, h) 순서로 변경
        l, w, h = bboxes[i, 3:6]
        size = [w, l, h]

        yaw = bboxes[i, 6]
        score = scores[i]
        label = labels[i]
        attr = attrs[i]
        
        # mAVE 수정: 모델 출력에서 velocity 가져오기
        if bboxes.shape[1] > 7:
            velocity = (bboxes[i, 7], bboxes[i, 8], 0.0)
        else:
            velocity = (0.0, 0.0)

        box = Box(
            center=center, 
            size=size,
            orientation=Quaternion(axis=[0, 0, 1], radians=yaw),
            score=score, 
            label=label, 
            velocity=velocity
        )
        
        # LiDAR -> Ego -> Global 변환
        box.rotate(Quaternion(matrix=lidar2ego_rot))
        box.translate(lidar2ego_trans)
        box.rotate(Quaternion(matrix=ego2global_rot))
        box.translate(ego2global_trans)
        
        # mAAE 수정: box 객체에 임시로 attribute label 저장
        box.attr = attr

        boxes_list.append(box)
        
    return boxes_list

# ===================================================================
# ✨ 2. 수정된 NuScenesMetric 클래스
# ===================================================================
@METRICS.register_module()
class NuScenesMetric(BaseMetric):
    def __init__(self,
                 ann_file: str,
                 metric: Union[str, List[str]] = 'bbox',
                 data_root: Optional[str] = None,
                 version: str = 'v1.0-trainval',
                 use_gt_for_debug: bool = False,
                 **kwargs) -> None:
        
        kwargs.pop('backend_args', None)
        super().__init__(**kwargs)
        
        self.ann_file = ann_file
        self.metric = metric
        self.data_root = data_root
        self.version = version
        self.use_gt_for_debug = use_gt_for_debug

        data = mmengine.load(ann_file)
        self.data_infos = data['data_list'] if isinstance(data, dict) and 'data_list' in data else data
        
        self.nusc = NuScenes(version=self.version, dataroot=self.data_root, verbose=False)
        from nuscenes.eval.detection.config import config_factory
        self.eval_detection_configs = config_factory('detection_cvpr_2019')
        self.eval_set_map = {'v1.0-mini': 'mini_val', 'v1.0-trainval': 'val'}
        self.ErrNameMapping = {
            'trans_err': 'mATE', 'scale_err': 'mASE', 'orient_err': 'mAOE',
            'vel_err': 'mAVE', 'attr_err': 'mAAE'
        }
        
        # ✨ 1. 공식 코드의 DefaultAttribute 딕셔너리 추가
        self.DefaultAttribute = {
            'car': 'vehicle.parked',
            'pedestrian': 'pedestrian.standing',
            'trailer': 'vehicle.parked',
            'truck': 'vehicle.parked',
            'bus': 'vehicle.parked',
            'motorcycle': 'cycle.without_rider',
            'construction_vehicle': 'vehicle.parked',
            'bicycle': 'cycle.without_rider',
            'barrier': '',
            'traffic_cone': '',
        }
        
        if self.use_gt_for_debug:
            logger = MMLogger.get_current_instance()
            logger.warning('\n\n!!! DEBUG MODE: Using GT for submission !!!\n')

    def process(self, data_batch: dict, data_samples: list) -> None:
        """Collect results from each batch."""
        for data_sample in data_samples:
            result = dict()
            pred_3d = data_sample['pred_instances_3d']
            
            result['pred_instances_3d'] = {
                'bboxes_3d': pred_3d['bboxes_3d'], # .tensor.cpu().tolist()는 format_results에서 처리
                'scores_3d': pred_3d['scores_3d'],
                'labels_3d': pred_3d['labels_3d'],
            }
            # ✨ mAAE 수정: attr_labels가 있으면 결과에 포함
            if 'attr_labels' in pred_3d:
                result['pred_instances_3d']['attr_labels'] = pred_3d['attr_labels']

            result['sample_idx'] = data_sample['sample_idx']
            self.results.append(result)

    def format_results(self, results: list) -> dict:
        logger = MMLogger.get_current_instance()
        logger.info('Formatting results for nuScenes evaluation...')
        
        nusc_annos = {}
        mapped_class_names = self.dataset_meta['classes']

        for i, res in enumerate(results):
            sample_idx = res['sample_idx']
            sample_token = self.data_infos[sample_idx]['token']
            annos = []
            
            # ✨ 1. 누락된 GT 제출 로직 추가
            if self.use_gt_for_debug:
                # --- [DEBUG PATH] GT 데이터를 사용하여 제출 파일 생성 ---
                ann_tokens = self.nusc.get('sample', sample_token)['anns']
                for ann_token in ann_tokens:
                    ann_metadata = self.nusc.get('sample_annotation', ann_token)

                    if ann_metadata['attribute_tokens']:
                        attr_token = ann_metadata['attribute_tokens'][0]
                        attribute_name = self.nusc.get('attribute', attr_token)['name']
                    else:
                        attribute_name = ''

                    nusc_anno = dict(
                        sample_token=sample_token,
                        translation=ann_metadata['translation'],
                        size=ann_metadata['size'],
                        rotation=ann_metadata['rotation'],
                        velocity=self.nusc.box_velocity(ann_token)[:2].tolist(),
                        detection_name=ann_metadata['category_name'],
                        detection_score=1.0,
                        attribute_name=attribute_name
                    )
                    annos.append(nusc_anno)
            else:
                # --- [NORMAL PATH] 기존처럼 모델 예측값을 사용 ---
                pred_instances_3d = res['pred_instances_3d']
                
                # GPU 텐서를 CPU로 옮기기
                for k, v in pred_instances_3d.items():
                    pred_instances_3d[k] = v.cpu()

                formatted_det = {
                    'bboxes_3d': pred_instances_3d['bboxes_3d'],
                    'scores_3d': pred_instances_3d['scores_3d'],
                    'labels_3d': pred_instances_3d['labels_3d'],
                }
                
                boxes = output_to_nusc_box_global(formatted_det, self.data_infos[sample_idx])
                
                for j, box in enumerate(boxes):
                    name = mapped_class_names[box.label]
                    
                    # 속도 기반 속성 추론 로직
                    if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                        if name in ['car', 'construction_vehicle', 'bus', 'truck', 'trailer']:
                            attribute_name = 'vehicle.moving'
                        elif name in ['motorcycle', 'bicycle']:
                            attribute_name = 'cycle.with_rider'
                        elif name in ['pedestrian']:
                            attribute_name = 'pedestrian.moving'
                        else:
                            attribute_name = self.DefaultAttribute.get(name, '')
                    else:
                        if name in ['pedestrian']:
                            attribute_name = 'pedestrian.standing'
                        elif name in ['motorcycle', 'bicycle']:
                            attribute_name = 'cycle.without_rider'
                        elif name in ['bus']:
                             attribute_name = 'vehicle.stopped'
                        else:
                            attribute_name = self.DefaultAttribute.get(name, '')

                    nusc_anno = dict(
                        sample_token=sample_token,
                        translation=box.center.tolist(),
                        size=box.wlh.tolist(),
                        rotation=box.orientation.elements.tolist(),
                        velocity=box.velocity[:2].tolist(),
                        detection_name=name,
                        detection_score=box.score,
                        attribute_name=attribute_name)
                    annos.append(nusc_anno)

            if sample_token in nusc_annos:
                nusc_annos[sample_token].extend(annos)
            else:
                nusc_annos[sample_token] = annos

        # ✨ 2. 누락된 빈 토큰 처리 로직 추가
        # nuScenes 평가는 모든 샘플 토큰에 대한 결과가 제출 파일에 포함되어야 함
        gt_tokens = {info['token'] for info in self.data_infos}
        for token in gt_tokens:
            if token not in nusc_annos:
                nusc_annos[token] = []

        final_submission = {'meta': self.dataset_meta, 'results': nusc_annos}
        return final_submission


    def compute_metrics(self, results: list):

        import os

        logger = MMLogger.get_current_instance()
        logger.info('Starting NuScenes evaluation...')

        # Raw model detections before evaluator formatting.
        raw_boxes = sum(
            len(r['pred_instances_3d']['scores_3d'])
            for r in results
        )

        raw_above_threshold = sum(
            int(
                (
                    r['pred_instances_3d']['scores_3d'] > 0.01
                ).sum().item()
            )
            for r in results
        )

        results_json = self.format_results(results)

        annotations = results_json['results']

        submitted_boxes = sum(
            len(boxes)
            for boxes in annotations.values()
        )

        nonempty_samples = sum(
            len(boxes) > 0
            for boxes in annotations.values()
        )

        logger.info(
            '[A EVAL DEBUG] '
            f'raw_boxes={raw_boxes}, '
            f'raw_above_0.01={raw_above_threshold}, '
            f'submitted_boxes={submitted_boxes}, '
            f'nonempty_samples={nonempty_samples}/'
            f'{len(annotations)}'
        )

        # Optional: retain nuScenes submission for inspection.
        export_path = os.environ.get('RRRF_DUMP_NUSC_JSON')

        if export_path:
            os.makedirs(
                os.path.dirname(export_path) or '.',
                exist_ok=True,
            )
            mmengine.dump(results_json, export_path)

            logger.info(
                '[A EVAL DEBUG] Saved submission: '
                + export_path
            )

        metric_dict = self.nus_evaluate(results_json)

        return metric_dict

    def nus_evaluate(self, results_json: dict) -> dict:
        """기존 코드와 동일 (수정 없음)"""
        from nuscenes.eval.detection.evaluate import NuScenesEval
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = osp.join(tmp_dir, 'results_nusc.json')
            mmengine.dump(results_json, result_path)

            nusc_eval = NuScenesEval(
                self.nusc,
                config=self.eval_detection_configs,
                result_path=result_path,
                eval_set=self.eval_set_map[self.version],
                output_dir=tmp_dir,
                verbose=False)
            nusc_eval.main(plot_examples=0, render_curves=False)

            metrics_summary = mmengine.load(osp.join(tmp_dir, 'metrics_summary.json'))
        
        metric_dict = {}
        for k, v in metrics_summary['mean_dist_aps'].items():
            metric_dict[f'{k}_mAP'] = v
        for k, v in metrics_summary['tp_errors'].items():
            metric_dict[self.ErrNameMapping[k]] = v
        metric_dict['NDS'] = metrics_summary['nd_score']
        metric_dict['mAP'] = metrics_summary['mean_ap']
        
        return metric_dict