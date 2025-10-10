# # Copyright (c) OpenMMLab. All rights reserved.
# import tempfile
# from os import path as osp
# from typing import Dict, List, Optional, Sequence, Tuple, Union

# import mmengine
# import numpy as np
# import pyquaternion
# import torch
# from mmengine import Config, load
# from mmengine.evaluator import BaseMetric
# from mmengine.logging import MMLogger
# from nuscenes.eval.detection.config import config_factory
# from nuscenes.eval.detection.data_classes import DetectionConfig
# from nuscenes.utils.data_classes import Box as NuScenesBox

# from mmdet3d.models.layers import box3d_multiclass_nms
# from mmdet3d.registry import METRICS
# from mmdet3d.structures import (CameraInstance3DBoxes, LiDARInstance3DBoxes,
#                                 bbox3d2result, xywhr2xyxyr)
# import pickle
# from typing import Dict, Any


# @METRICS.register_module()
# class NuScenesMetric(BaseMetric):
#     """Nuscenes evaluation metric.

#     Args:
#         data_root (str): Path of dataset root.
#         ann_file (str): Path of annotation file.
#         metric (str or List[str]): Metrics to be evaluated. Defaults to 'bbox'.
#         modality (dict): Modality to specify the sensor data used as input.
#             Defaults to dict(use_camera=False, use_lidar=True).
#         prefix (str, optional): The prefix that will be added in the metric
#             names to disambiguate homonymous metrics of different evaluators.
#             If prefix is not provided in the argument, self.default_prefix will
#             be used instead. Defaults to None.
#         format_only (bool): Format the output results without perform
#             evaluation. It is useful when you want to format the result to a
#             specific format and submit it to the test server.
#             Defaults to False.
#         jsonfile_prefix (str, optional): The prefix of json files including the
#             file path and the prefix of filename, e.g., "a/b/prefix".
#             If not specified, a temp file will be created. Defaults to None.
#         eval_version (str): Configuration version of evaluation.
#             Defaults to 'detection_cvpr_2019'.
#         collect_device (str): Device name used for collecting results from
#             different ranks during distributed training. Must be 'cpu' or
#             'gpu'. Defaults to 'cpu'.
#         backend_args (dict, optional): Arguments to instantiate the
#             corresponding backend. Defaults to None.
#     """
#     NameMapping = {
#         'movable_object.barrier': 'barrier',
#         'vehicle.bicycle': 'bicycle',
#         'vehicle.bus.bendy': 'bus',
#         'vehicle.bus.rigid': 'bus',
#         'vehicle.car': 'car',
#         'vehicle.construction': 'construction_vehicle',
#         'vehicle.motorcycle': 'motorcycle',
#         'human.pedestrian.adult': 'pedestrian',
#         'human.pedestrian.child': 'pedestrian',
#         'human.pedestrian.construction_worker': 'pedestrian',
#         'human.pedestrian.police_officer': 'pedestrian',
#         'movable_object.trafficcone': 'traffic_cone',
#         'vehicle.trailer': 'trailer',
#         'vehicle.truck': 'truck'
#     }
#     DefaultAttribute = {
#         'car': 'vehicle.parked',
#         'pedestrian': 'pedestrian.moving',
#         'trailer': 'vehicle.parked',
#         'truck': 'vehicle.parked',
#         'bus': 'vehicle.moving',
#         'motorcycle': 'cycle.without_rider',
#         'construction_vehicle': 'vehicle.parked',
#         'bicycle': 'cycle.without_rider',
#         'barrier': '',
#         'traffic_cone': '',
#     }
#     # https://github.com/nutonomy/nuscenes-devkit/blob/57889ff20678577025326cfc24e57424a829be0a/python-sdk/nuscenes/eval/detection/evaluate.py#L222 # noqa
#     ErrNameMapping = {
#         'trans_err': 'mATE',
#         'scale_err': 'mASE',
#         'orient_err': 'mAOE',
#         'vel_err': 'mAVE',
#         'attr_err': 'mAAE'
#     }

#     def __init__(self,
#                  data_root: str,
#                  ann_file: str,
#                  metric: Union[str, List[str]] = 'bbox',
#                  modality: dict = dict(use_camera=False, use_lidar=True),
#                  prefix: Optional[str] = None,
#                  format_only: bool = False,
#                  jsonfile_prefix: Optional[str] = None,
#                  eval_version: str = 'detection_cvpr_2019',
#                  collect_device: str = 'cpu',
#                  backend_args: Optional[dict] = None,
#                  version: str = 'v1.0-trainval',) -> None:
#         self.default_prefix = 'NuScenes metric'
#         super(NuScenesMetric, self).__init__(
#             collect_device=collect_device, prefix=prefix)
#         if modality is None:
#             modality = dict(
#                 use_camera=False,
#                 use_lidar=True,
#             )
#         self.ann_file = ann_file
#         self.data_root = data_root
#         self.modality = modality
#         self.format_only = format_only
#         if self.format_only:
#             assert jsonfile_prefix is not None, 'jsonfile_prefix must be not '
#             'None when format_only is True, otherwise the result files will '
#             'be saved to a temp directory which will be cleanup at the end.'

#         self.jsonfile_prefix = jsonfile_prefix
#         self.backend_args = backend_args

#         self.metrics = metric if isinstance(metric, list) else [metric]

#         self.eval_version = eval_version
#         self.eval_detection_configs = config_factory(self.eval_version)
#         self.version = version

#         # ann_file(.pkl)을 로드합니다.
#         self.data_infos = mmengine.load(ann_file)['data_list'] 

#     def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
#         """Process one batch of data samples and predictions.

#         The processed results should be stored in ``self.results``, which will
#         be used to compute the metrics when all batches have been processed.

#         Args:
#             data_batch (dict): A batch of data from the dataloader.
#             data_samples (Sequence[dict]): A batch of outputs from the model.
#         """
#         for data_sample in data_samples:
#             result = dict()
#             pred_3d = data_sample['pred_instances_3d']
#             pred_2d = data_sample['pred_instances']
#             for attr_name in pred_3d:
#                 pred_3d[attr_name] = pred_3d[attr_name].to('cpu')
#             result['pred_instances_3d'] = pred_3d
#             for attr_name in pred_2d:
#                 pred_2d[attr_name] = pred_2d[attr_name].to('cpu')
#             result['pred_instances'] = pred_2d
#             sample_idx = data_sample['sample_idx']
#             result['sample_idx'] = sample_idx
#             self.results.append(result)

#     # ##### original version #####
#     # def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
#     #     """Compute the metrics from processed results.

#     #     Args:
#     #         results (List[dict]): The processed results of each batch.

#     #     Returns:
#     #         Dict[str, float]: The computed metrics. The keys are the names of
#     #         the metrics, and the values are corresponding results.
#     #     """
#     #     """
#     #     [임시 디버깅 함수]
#     #     이 함수는 results 리스트의 실제 구조를 확인하고 즉시 종료됩니다.
#     #     """
#     #     # # ==================== 구조 확인을 위한 임시 코드 ====================
#     #     # print("\n" + "="*20 + " 결과 구조 확인 " + "="*20)
#     #     # if results:
#     #     #     first_result = results[0]
#     #     #     print(f"[INSPECT] 첫 번째 결과의 타입: {type(first_result)}")
#     #     #     print(f"[INSPECT] 첫 번째 결과의 키 목록: {first_result.keys()}")
            
#     #     #     # 만약 'pred_instances_3d' 키가 있다면 그 내부도 확인
#     #     #     if 'pred_instances_3d' in first_result:
#     #     #         print(f"[INSPECT] pred_instances_3d의 타입: {type(first_result['pred_instances_3d'])}")
#     #     #         # .keys()가 있다면 키 목록 출력
#     #     #         if hasattr(first_result['pred_instances_3d'], 'keys'):
#     #     #             print(f"[INSPECT] pred_instances_3d의 키 목록: {first_result['pred_instances_3d'].keys()}")

#     #     # else:
#     #     #     print("[INSPECT] 결과 리스트가 비어있습니다.")
        
#     #     # import sys
#     #     # sys.exit("--> 구조 확인 완료. 위의 [INSPECT] 결과를 알려주세요.")
#     #     # # =================================================================
        
#     #     logger: MMLogger = MMLogger.get_current_instance()

#     #     classes = self.dataset_meta['classes']
#     #     self.version = self.dataset_meta['version']
#     #     # load annotations
#     #     self.data_infos = load(
#     #         self.ann_file, backend_args=self.backend_args)['data_list']
#     #     result_dict, tmp_dir = self.format_results(results, classes,
#     #                                                self.jsonfile_prefix)

#     #     metric_dict = {}

#     #     if self.format_only:
#     #         logger.info(
#     #             f'results are saved in {osp.basename(self.jsonfile_prefix)}')
#     #         return metric_dict

#     #     for metric in self.metrics:
#     #         ap_dict = self.nus_evaluate(
#     #             result_dict, classes=classes, metric=metric, logger=logger)
#     #         for result in ap_dict:
#     #             metric_dict[result] = ap_dict[result]

#     #     if tmp_dir is not None:
#     #         tmp_dir.cleanup()
#     #     return metric_dict

#     # def compute_metrics(self, results: list) :
#     #     """
#     #     [최종 수정본 v2]
#     #     1. .pkl 파일 구조에 맞게 수정
#     #     2. sample_idx -> sample_token 변환
#     #     3. 누락된 샘플에 대한 빈 결과 추가
#     #     """
#     #     logger = MMLogger.get_current_instance()
#     #     logger.info("[FINAL PATCH] Applying the definitive patch to the results list...")

#     #     # --- 1단계: ann_file에서 idx -> token 변환 맵과 전체 GT 토큰 목록 생성 ---
#     #     with open(self.ann_file, 'rb') as f:
#     #         nuscenes_infos = pickle.load(f)
        
#     #     # .pkl 파일 자체가 정보 리스트이므로 ['infos'] 접근을 삭제합니다.
#     #     all_gt_infos = nuscenes_infos
        
#     #     gt_tokens = {info['token'] for info in all_gt_infos}
#     #     idx_to_token_map = {info['sample_idx']: info['token'] for info in all_gt_infos}
        
#     #     logger.info(f"Loaded {len(gt_tokens)} GT tokens and created idx-to-token map.")

#     #     # --- 2단계: 현재 예측 결과(results)의 'sample_idx'를 'sample_token'으로 변환 ---
#     #     pred_tokens = set()
#     #     for res in results:
#     #         idx = res['sample_idx']
#     #         if idx in idx_to_token_map:
#     #             token = idx_to_token_map[idx]
#     #             res['sample_token'] = token
#     #             pred_tokens.add(token)
#     #         else:
#     #             logger.warning(f"sample_idx {idx} not found in the annotation file map. Skipping.")

#     #     logger.info(f"Converted {len(pred_tokens)} predictions from sample_idx to sample_token.")

#     #     # --- 3단계: 누락된 토큰을 찾아 빈(dummy) 결과로 추가 ---
#     #     missing_tokens = gt_tokens - pred_tokens
#     #     if missing_tokens:
#     #         logger.info(f"[PATCH] Found {len(missing_tokens)} missing samples. Adding empty predictions.")
#     #         for token in missing_tokens:
#     #             dummy_result = {
#     #                 'pred_instances_3d': {
#     #                     'bboxes_3d': [], 'scores_3d': [], 'labels_3d': [],
#     #                 },
#     #                 'sample_token': token 
#     #             }
#     #             results.append(dummy_result)
        
#     #     logger.info(f"[PATCH] Patch complete. Total samples for evaluation: {len(results)}")

#     #     # --- 이제 패치가 완료된 results 리스트로 원래 평가 로직 수행 ---
#     #     self.classes = self.dataset_meta['classes']
#     #     ap_dict = self.nus_evaluate(results) 

#     #     metric_dict = dict()
#     #     for iou, ap in ap_dict.items():
#     #         metric_dict[iou] = ap

#     #     return metric_dict
    
#     # def compute_metrics(self, results: list) -> Dict[str, Any]:
#     #     """
#     #     [최종 검사용 임시 함수]
#     #     self.ann_file(.pkl)의 실제 구조를 확인하고 즉시 종료됩니다.
#     #     """
#     #     logger = MMLogger.get_current_instance()

#     #     # --- PKL 파일 구조 확인을 위한 임시 코드 ---
#     #     logger.info("--- Inspecting the structure of the .pkl annotation file ---")
        
#     #     with open(self.ann_file, 'rb') as f:
#     #         nuscenes_infos = pickle.load(f)
        
#     #     print("\n" + "="*20 + " PKL 파일 구조 확인 " + "="*20)
#     #     print(f"[INSPECT] 불러온 데이터의 타입: {type(nuscenes_infos)}")

#     #     if isinstance(nuscenes_infos, dict):
#     #         print(f"[INSPECT] 딕셔너리입니다. 포함된 키: {nuscenes_infos.keys()}")
#     #         if 'infos' in nuscenes_infos:
#     #             print(f"[INSPECT] 'infos' 키 내용물의 타입: {type(nuscenes_infos['infos'])}")
#     #         if 'metadata' in nuscenes_infos:
#     #             print(f"[INSPECT] 'metadata' 키가 존재합니다.")

#     #     elif isinstance(nuscenes_infos, list):
#     #         print(f"[INSPECT] 리스트입니다. 총 {len(nuscenes_infos)}개의 항목이 있습니다.")
#     #         if nuscenes_infos:
#     #             print(f"[INSPECT] 리스트 첫 번째 항목의 타입: {type(nuscenes_infos[0])}")
#     #             if isinstance(nuscenes_infos[0], dict):
#     #                 print(f"[INSPECT] 첫 번째 항목의 키: {nuscenes_infos[0].keys()}")
        
#     #     print("="*55)
        
#     #     import sys
#     #     sys.exit("--> PKL 구조 확인 완료. 위의 [INSPECT] 결과를 알려주세요.")
#     #     # --- 코드 끝 ---

#     # def compute_metrics(self, results: list) -> Dict[str, Any]:
#     #     """
#     #     [진짜 최종 수정본]
#     #     1. 정확한 .pkl 구조('data_list') 반영
#     #     2. sample_idx -> sample_token 변환
#     #     3. 누락된 샘플에 대한 빈 결과 추가
#     #     """
#     #     # --- 이 디버깅 코드를 함수 맨 처음에 추가하고 다시 실행하세요 ---
#     #     if results: # results 리스트가 비어있지 않은지 확인
#     #         print("\n--- DEBUG: Content of the FIRST prediction result ---")
#     #         # 첫 번째 예측 결과 딕셔너리의 모든 키와 값의 타입을 출력합니다.
#     #         for key, value in results[0].items():
#     #             print(f"Key: '{key}', Value Type: {type(value)}")
#     #         print("----------------------------------------------------\n")
#     #     # --- 여기까지 추가 ---

#     #     logger = MMLogger.get_current_instance()
#     #     logger.info("[FINAL PATCH] Applying the definitive patch based on the correct .pkl structure...")

#     #     # --- 1단계: ann_file에서 정확한 키('data_list')를 사용하여 데이터 로드 ---
#     #     with open(self.ann_file, 'rb') as f:
#     #         nuscenes_infos = pickle.load(f)
        
#     #     # 'infos'가 아닌 'data_list' 키를 사용합니다.
#     #     all_gt_infos = nuscenes_infos['data_list']
        
#     #     gt_tokens = {info['token'] for info in all_gt_infos}
#     #     idx_to_token_map = {info['sample_idx']: info['token'] for info in all_gt_infos}
        
#     #     logger.info(f"Loaded {len(gt_tokens)} GT tokens and created idx-to-token map.")

#     #     # --- 2단계: 현재 예측 결과(results)의 'sample_idx'를 'sample_token'으로 변환 ---
#     #     pred_tokens = set()
#     #     for res in results:
#     #         if 'sample_idx' in res:
#     #             idx = res['sample_idx']
#     #             if idx in idx_to_token_map:
#     #                 token = idx_to_token_map[idx]
#     #                 res['sample_token'] = token
#     #                 pred_tokens.add(token)
#     #             else:
#     #                 logger.warning(f"sample_idx {idx} not found in the annotation file map. Skipping.")
#     #         else:
#     #             logger.warning(f"A result in the list is missing 'sample_idx'. Skipping conversion for this item.")


#     #     logger.info(f"Converted {len(pred_tokens)} predictions from sample_idx to sample_token.")

#     #     # --- 3단계: 누락된 토큰을 찾아 빈(dummy) 결과로 추가 ---
#     #     missing_tokens = gt_tokens - pred_tokens
#     #     if missing_tokens:
#     #         logger.info(f"[PATCH] Found {len(missing_tokens)} missing samples. Adding empty predictions.")
#     #         for token in missing_tokens:
#     #             dummy_result = {
#     #                 'pred_instances_3d': {
#     #                     'bboxes_3d': [], 'scores_3d': [], 'labels_3d': [],
#     #                 },
#     #                 'sample_token': token 
#     #             }
#     #             results.append(dummy_result)
        
#     #     logger.info(f"[PATCH] Patch complete. Total samples for evaluation: {len(results)}")

#     #     # --- 이제 패치가 완료된 results 리스트로 원래 평가 로직 수행 ---
#     #     self.classes = self.dataset_meta['classes']
#     #     # results 리스트를 'pred_instances_3d' 키를 가진 딕셔너리에 담아서 전달합니다.
#     #         # 예측 결과가 순서대로 왔다고 가정하고, 각 결과에 'sample_idx'를 수동으로 추가합니다.
#     #     for i, result in enumerate(results):
#     #         result['sample_idx'] = i
#     #     ap_dict = self.nus_evaluate({'pred_instances_3d': results})

#     #     metric_dict = dict()
#     #     for iou, ap in ap_dict.items():
#     #         metric_dict[iou] = ap

#     #     return metric_dict
    
#     def compute_metrics(self, results: list) -> Dict[str, Any]:
#         """ FINAL FIX #4: compute_metrics를 가장 단순하고 올바른 형태로 되돌립니다. """
        
#         logger = MMLogger.get_current_instance()
#         logger.info('Starting NuScenes evaluation...')

#         # 모든 복잡한 로직은 하위 함수에 위임하고, 여기서는 단순히 호출만 합니다.
#         # 이전에 추가했던 모든 수동 패치, enumerate 루프를 제거합니다.
#         ap_dict = self.nus_evaluate({'pred_instances_3d': results})
        
#         metric_dict = {}
#         for metric, val in ap_dict.items():
#             metric_dict[metric] = val
            
#         return metric_dict

#     def nus_evaluate(self,
#                      result_dict: dict,
#                      metric: str = 'bbox',
#                      classes: Optional[List[str]] = None,
#                      logger: Optional[MMLogger] = None) -> Dict[str, float]:
#         """Evaluation in Nuscenes protocol.

#         Args:
#             result_dict (dict): Formatted results of the dataset.
#             metric (str): Metrics to be evaluated. Defaults to 'bbox'.
#             classes (List[str], optional): A list of class name.
#                 Defaults to None.
#             logger (MMLogger, optional): Logger used for printing related
#                 information during evaluation. Defaults to None.

#         Returns:
#             Dict[str, float]: Results of each evaluation metric.
#         """
#         metric_dict = dict()
#         for name in result_dict:
#             print(f'Evaluating bboxes of {name}')
#             ret_dict = self._evaluate_single(
#                 result_dict[name], classes=classes, result_name=name)
#             metric_dict.update(ret_dict)
#         return metric_dict

#     # def _evaluate_single(
#     #         self,
#     #         result_path: str,
#     #         classes: Optional[List[str]] = None,
#     #         result_name: str = 'pred_instances_3d') -> Dict[str, float]:
#     #     """Evaluation for a single model in nuScenes protocol.

#     #     Args:
#     #         result_path (str): Path of the result file.
#     #         classes (List[str], optional): A list of class name.
#     #             Defaults to None.
#     #         result_name (str): Result name in the metric prefix.
#     #             Defaults to 'pred_instances_3d'.

#     #     Returns:
#     #         Dict[str, float]: Dictionary of evaluation details.
#     #     """
#     #     from nuscenes import NuScenes
#     #     from nuscenes.eval.detection.evaluate import NuScenesEval

#     #     # --- 수정 코드 (아래 내용으로 교체) ---
#     #     # result_path[0]의 타입이 문자열(str)인지 확인합니다.
#     #     if isinstance(result_path[0], str):
#     #         # 만약 문자열이라면, 기존 로직대로 디렉토리 경로를 추출합니다.
#     #         output_dir = osp.join(*osp.split(result_path[0])[:-1])
#     #     else:
#     #         # --- 수정 코드 (아래 내용으로 교체) ---
#     #         if isinstance(result_path[0], str):
#     #             output_dir = osp.join(*osp.split(result_path[0])[:-1])
#     #         else:
#     #             # self.jsonfile_prefix가 설정되어 있다면 그 경로를 사용합니다.
#     #             if self.jsonfile_prefix: # <--- 올바른 이름으로 수정
#     #                 output_dir = osp.dirname(self.jsonfile_prefix) # <--- 올바른 이름으로 수정
#     #             else:
#     #                 output_dir = './'
#     #     nusc = NuScenes(
#     #         version=self.version, dataroot=self.data_root, verbose=False)
#     #     eval_set_map = {
#     #         'v1.0-mini': 'mini_val',
#     #         'v1.0-trainval': 'val',
#     #     }
#     #     nusc_eval = NuScenesEval(
#     #         nusc,
#     #         config=self.eval_detection_configs,
#     #         result_path=result_path,
#     #         eval_set=eval_set_map[self.version],
#     #         output_dir=output_dir,
#     #         verbose=False)
#     #     nusc_eval.main(render_curves=False)

#     #     # record metrics
#     #     metrics = mmengine.load(osp.join(output_dir, 'metrics_summary.json'))
#     #     detail = dict()
#     #     metric_prefix = f'{result_name}_NuScenes'
#     #     for name in classes:
#     #         for k, v in metrics['label_aps'][name].items():
#     #             val = float(f'{v:.4f}')
#     #             detail[f'{metric_prefix}/{name}_AP_dist_{k}'] = val
#     #         for k, v in metrics['label_tp_errors'][name].items():
#     #             val = float(f'{v:.4f}')
#     #             detail[f'{metric_prefix}/{name}_{k}'] = val
#     #         for k, v in metrics['tp_errors'].items():
#     #             val = float(f'{v:.4f}')
#     #             detail[f'{metric_prefix}/{self.ErrNameMapping[k]}'] = val

#     #     detail[f'{metric_prefix}/NDS'] = metrics['nd_score']
#     #     detail[f'{metric_prefix}/mAP'] = metrics['mean_ap']
#     #     return detail
    
#     def _evaluate_single(
#         self,
#         result_path: list,  # 실제 타입은 list이므로 명확하게 변경
#         classes: Optional[List[str]] = None,
#         result_name: str = 'pred_instances_3d') -> Dict[str, float]:
#         # """nuScenes protocol에 따라 단일 모델을 평가합니다."""
#         # # 필요한 라이브러리 임포트
#         from nuscenes import NuScenes
#         from nuscenes.eval.detection.evaluate import NuScenesEval
#         # import mmengine
#         # import os.path as osp
#         # import tempfile

#         # (1) output_dir 결정 로직을 깔끔하게 정리합니다.
#         # result_path는 리스트이므로, 첫 번째 요소가 문자열인지 확인하는 로직은 불필요할 수 있습니다.
#         # 만약 항상 메모리 내 결과라면, 아래처럼 더 단순화할 수 있습니다.
#         if self.jsonfile_prefix:
#             output_dir = osp.dirname(self.jsonfile_prefix)
#         else:
#             # jsonfile_prefix가 없다면, 임시 디렉토리를 사용하거나 현재 디렉토리를 사용합니다.
#             # 여기서는 NuScenesEval이 결과를 저장할 경로로 현재 디렉토리를 지정합니다.
#             output_dir = './'
        
#         # (2) NuScenes API 객체와 버전 맵을 준비합니다.
#         nusc = NuScenes(
#             version=self.version, dataroot=self.data_root, verbose=False)
#         eval_set_map = {
#             'v1.0-mini': 'mini_val',
#             'v1.0-trainval': 'val',
#         }

#         # (3) 단일 with 블록 내에서 파일 저장, 평가, 결과 처리를 모두 수행합니다.
#         metrics_summary = None
#         with tempfile.TemporaryDirectory() as tmp_dir:
#             tmp_json_path = osp.join(tmp_dir, 'temp_results.json')
            
#             # 수정 코드 (After)
#             # (1) 예측 결과를 nuScenes 제출 형식에 맞게 변환합니다.
#             # 이 과정에서 모든 Tensor가 Python 리스트/숫자로 변환됩니다.
#             results_json = self.format_results(result_path)

#             # (2) 변환이 완료된 깨끗한 데이터를 임시 JSON 파일에 씁니다.
#             mmengine.dump(results_json, tmp_json_path)

#             # NuScenesEval을 '한 번만' 올바르게 호출합니다.
#             nusc_eval = NuScenesEval(
#                 nusc=nusc,  # self.nusc 대신 방금 생성한 nusc 객체 사용
#                 config=self.eval_detection_configs,
#                 result_path=tmp_json_path,
#                 eval_set=eval_set_map[self.version], # self.eval_set 대신 맵 사용
#                 output_dir=output_dir, # 결과를 저장할 디렉토리
#                 verbose=False)
            
#             # 평가를 실행하고, 결과를 파일로 읽는 대신 변수로 직접 받습니다.
#             metrics_summary = nusc_eval.main(plot_examples=0, render_curves=False)

#         # (4) 받아온 metrics_summary 변수를 사용해 최종 결과 딕셔너리를 만듭니다.
#         detail = dict()
#         metric_prefix = f'{result_name}_NuScenes'
#         for name in classes:
#             for k, v in metrics_summary['label_aps'][name].items():
#                 val = float(f'{v:.4f}')
#                 detail[f'{metric_prefix}/{name}_AP_dist_{k}'] = val
#             for k, v in metrics_summary['label_tp_errors'][name].items():
#                 val = float(f'{v:.4f}')
#                 detail[f'{metric_prefix}/{name}_{k}'] = val

#         # tp_errors는 클래스와 무관하므로 별도로 처리합니다.
#         for k, v in metrics_summary['tp_errors'].items():
#             val = float(f'{v:.4f}')
#             detail[f'{metric_prefix}/{self.ErrNameMapping[k]}'] = val

#         detail[f'{metric_prefix}/NDS'] = metrics_summary['nd_score']
#         detail[f'{metric_prefix}/mAP'] = metrics_summary['mean_ap']
        
#         return detail

#     def format_results(
#         self,
#         results: List[dict],
#         classes: Optional[List[str]] = None,
#         jsonfile_prefix: Optional[str] = None
#     ) -> Tuple[dict, Union[tempfile.TemporaryDirectory, None]]:
#         """Format the mmdet3d results to standard NuScenes json file.

#         Args:
#             results (List[dict]): Testing results of the dataset.
#             classes (List[str], optional): A list of class name.
#                 Defaults to None.
#             jsonfile_prefix (str, optional): The prefix of json files. It
#                 includes the file path and the prefix of filename, e.g.,
#                 "a/b/prefix". If not specified, a temp file will be created.
#                 Defaults to None.

#         Returns:
#             tuple: Returns (result_dict, tmp_dir), where ``result_dict`` is a
#             dict containing the json filepaths, ``tmp_dir`` is the temporal
#             directory created for saving json files when ``jsonfile_prefix`` is
#             not specified.
#         """
#         classes = self.dataset_meta['classes']
#         assert isinstance(results, list), 'results must be a list'

#         if jsonfile_prefix is None:
#             tmp_dir = tempfile.TemporaryDirectory()
#             jsonfile_prefix = osp.join(tmp_dir.name, 'results')
#         else:
#             tmp_dir = None
#         result_dict = dict()
#         sample_idx_list = [result['sample_idx'] for result in results]

#         for name in results[0]:
#             if 'pred' in name and '3d' in name and name[0] != '_':
#                 print(f'\nFormating bboxes of {name}')
#                 results_ = [out[name] for out in results]
#                 tmp_file_ = osp.join(jsonfile_prefix, name)
#                 box_type_3d = type(results_[0]['bboxes_3d'])
#                 if box_type_3d == LiDARInstance3DBoxes:
#                     result_dict[name] = self._format_lidar_bbox(
#                         results_, sample_idx_list, classes, tmp_file_)
#                 elif box_type_3d == CameraInstance3DBoxes:
#                     result_dict[name] = self._format_camera_bbox(
#                         results_, sample_idx_list, classes, tmp_file_)

#         return result_dict, tmp_dir

#     def get_attr_name(self, attr_idx: int, label_name: str) -> str:
#         """Get attribute from predicted index.

#         This is a workaround to predict attribute when the predicted velocity
#         is not reliable. We map the predicted attribute index to the one in the
#         attribute set. If it is consistent with the category, we will keep it.
#         Otherwise, we will use the default attribute.

#         Args:
#             attr_idx (int): Attribute index.
#             label_name (str): Predicted category name.

#         Returns:
#             str: Predicted attribute name.
#         """
#         # TODO: Simplify the variable name
#         AttrMapping_rev2 = [
#             'cycle.with_rider', 'cycle.without_rider', 'pedestrian.moving',
#             'pedestrian.standing', 'pedestrian.sitting_lying_down',
#             'vehicle.moving', 'vehicle.parked', 'vehicle.stopped', 'None'
#         ]
#         if label_name == 'car' or label_name == 'bus' \
#             or label_name == 'truck' or label_name == 'trailer' \
#                 or label_name == 'construction_vehicle':
#             if AttrMapping_rev2[attr_idx] == 'vehicle.moving' or \
#                 AttrMapping_rev2[attr_idx] == 'vehicle.parked' or \
#                     AttrMapping_rev2[attr_idx] == 'vehicle.stopped':
#                 return AttrMapping_rev2[attr_idx]
#             else:
#                 return self.DefaultAttribute[label_name]
#         elif label_name == 'pedestrian':
#             if AttrMapping_rev2[attr_idx] == 'pedestrian.moving' or \
#                 AttrMapping_rev2[attr_idx] == 'pedestrian.standing' or \
#                     AttrMapping_rev2[attr_idx] == \
#                     'pedestrian.sitting_lying_down':
#                 return AttrMapping_rev2[attr_idx]
#             else:
#                 return self.DefaultAttribute[label_name]
#         elif label_name == 'bicycle' or label_name == 'motorcycle':
#             if AttrMapping_rev2[attr_idx] == 'cycle.with_rider' or \
#                     AttrMapping_rev2[attr_idx] == 'cycle.without_rider':
#                 return AttrMapping_rev2[attr_idx]
#             else:
#                 return self.DefaultAttribute[label_name]
#         else:
#             return self.DefaultAttribute[label_name]

#     def _format_camera_bbox(self,
#                             results: List[dict],
#                             sample_idx_list: List[int],
#                             classes: Optional[List[str]] = None,
#                             jsonfile_prefix: Optional[str] = None) -> str:
#         """Convert the results to the standard format.

#         Args:
#             results (List[dict]): Testing results of the dataset.
#             sample_idx_list (List[int]): List of result sample idx.
#             classes (List[str], optional): A list of class name.
#                 Defaults to None.
#             jsonfile_prefix (str, optional): The prefix of the output jsonfile.
#                 You can specify the output directory/filename by modifying the
#                 jsonfile_prefix. Defaults to None.

#         Returns:
#             str: Path of the output json file.
#         """
#         nusc_annos = {}

#         print('Start to convert detection format...')

#         # Camera types in Nuscenes datasets
#         camera_types = [
#             'CAM_FRONT',
#             'CAM_FRONT_RIGHT',
#             'CAM_FRONT_LEFT',
#             'CAM_BACK',
#             'CAM_BACK_LEFT',
#             'CAM_BACK_RIGHT',
#         ]

#         CAM_NUM = 6

#         for i, det in enumerate(mmengine.track_iter_progress(results)):

#             sample_idx = sample_idx_list[i]

#             frame_sample_idx = sample_idx // CAM_NUM
#             camera_type_id = sample_idx % CAM_NUM

#             if camera_type_id == 0:
#                 boxes_per_frame = []
#                 attrs_per_frame = []

#             # need to merge results from images of the same sample
#             annos = []
#             boxes, attrs = output_to_nusc_box(det)
#             sample_token = self.data_infos[frame_sample_idx]['token']
#             camera_type = camera_types[camera_type_id]
#             boxes, attrs = cam_nusc_box_to_global(
#                 self.data_infos[frame_sample_idx], boxes, attrs, classes,
#                 self.eval_detection_configs, camera_type)
#             boxes_per_frame.extend(boxes)
#             attrs_per_frame.extend(attrs)
#             # Remove redundant predictions caused by overlap of images
#             if (sample_idx + 1) % CAM_NUM != 0:
#                 continue
#             boxes = global_nusc_box_to_cam(self.data_infos[frame_sample_idx],
#                                            boxes_per_frame, classes,
#                                            self.eval_detection_configs)
#             cam_boxes3d, scores, labels = nusc_box_to_cam_box3d(boxes)
#             # box nms 3d over 6 images in a frame
#             # TODO: move this global setting into config
#             nms_cfg = dict(
#                 use_rotate_nms=True,
#                 nms_across_levels=False,
#                 nms_pre=4096,
#                 nms_thr=0.05,
#                 score_thr=0.01,
#                 min_bbox_size=0,
#                 max_per_frame=500)
#             nms_cfg = Config(nms_cfg)
#             cam_boxes3d_for_nms = xywhr2xyxyr(cam_boxes3d.bev)
#             boxes3d = cam_boxes3d.tensor
#             # generate attr scores from attr labels
#             attrs = labels.new_tensor([attr for attr in attrs_per_frame])
#             boxes3d, scores, labels, attrs = box3d_multiclass_nms(
#                 boxes3d,
#                 cam_boxes3d_for_nms,
#                 scores,
#                 nms_cfg.score_thr,
#                 nms_cfg.max_per_frame,
#                 nms_cfg,
#                 mlvl_attr_scores=attrs)
#             cam_boxes3d = CameraInstance3DBoxes(boxes3d, box_dim=9)
#             det = bbox3d2result(cam_boxes3d, scores, labels, attrs)
#             boxes, attrs = output_to_nusc_box(det)
#             boxes, attrs = cam_nusc_box_to_global(
#                 self.data_infos[frame_sample_idx], boxes, attrs, classes,
#                 self.eval_detection_configs)

#             for i, box in enumerate(boxes):
#                 name = classes[box.label]
#                 attr = self.get_attr_name(attrs[i], name)
#                 nusc_anno = dict(
#                     sample_token=sample_token,
#                     translation=box.center.tolist(),
#                     size=box.wlh.tolist(),
#                     rotation=box.orientation.elements.tolist(),
#                     velocity=box.velocity[:2].tolist(),
#                     detection_name=name,
#                     detection_score=box.score,
#                     attribute_name=attr)
#                 annos.append(nusc_anno)
#             # other views results of the same frame should be concatenated
#             if sample_token in nusc_annos:
#                 nusc_annos[sample_token].extend(annos)
#             else:
#                 nusc_annos[sample_token] = annos

#         nusc_submissions = {
#             'meta': self.modality,
#             'results': nusc_annos,
#         }

#         mmengine.mkdir_or_exist(jsonfile_prefix)
#         res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
#         print(f'Results writes to {res_path}')
#         mmengine.dump(nusc_submissions, res_path)
#         return res_path

#     def _format_lidar_bbox(self,
#                            results: List[dict],
#                            sample_idx_list: List[int],
#                            classes: Optional[List[str]] = None,
#                            jsonfile_prefix: Optional[str] = None) -> str:
#         """Convert the results to the standard format.

#         Args:
#             results (List[dict]): Testing results of the dataset.
#             sample_idx_list (List[int]): List of result sample idx.
#             classes (List[str], optional): A list of class name.
#                 Defaults to None.
#             jsonfile_prefix (str, optional): The prefix of the output jsonfile.
#                 You can specify the output directory/filename by modifying the
#                 jsonfile_prefix. Defaults to None.

#         Returns:
#             str: Path of the output json file.
#         """
#         nusc_annos = {}

#         print('Start to convert detection format...')
#         for i, det in enumerate(mmengine.track_iter_progress(results)):
#             annos = []
#             boxes, attrs = output_to_nusc_box(det)
#             sample_idx = sample_idx_list[i]
#             sample_token = self.data_infos[sample_idx]['token']
#             boxes = lidar_nusc_box_to_global(self.data_infos[sample_idx],
#                                              boxes, classes,
#                                              self.eval_detection_configs)
#             for i, box in enumerate(boxes):
#                 name = classes[box.label]
#                 if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
#                     if name in [
#                             'car',
#                             'construction_vehicle',
#                             'bus',
#                             'truck',
#                             'trailer',
#                     ]:
#                         attr = 'vehicle.moving'
#                     elif name in ['bicycle', 'motorcycle']:
#                         attr = 'cycle.with_rider'
#                     else:
#                         attr = self.DefaultAttribute[name]
#                 else:
#                     if name in ['pedestrian']:
#                         attr = 'pedestrian.standing'
#                     elif name in ['bus']:
#                         attr = 'vehicle.stopped'
#                     else:
#                         attr = self.DefaultAttribute[name]

#                 nusc_anno = dict(
#                     sample_token=sample_token,
#                     translation=box.center.tolist(),
#                     size=box.wlh.tolist(),
#                     rotation=box.orientation.elements.tolist(),
#                     velocity=box.velocity[:2].tolist(),
#                     detection_name=name,
#                     detection_score=box.score,
#                     attribute_name=attr)
#                 annos.append(nusc_anno)
#             nusc_annos[sample_token] = annos
#         nusc_submissions = {
#             'meta': self.modality,
#             'results': nusc_annos,
#         }
#         mmengine.mkdir_or_exist(jsonfile_prefix)
#         res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
#         print(f'Results writes to {res_path}')
#         mmengine.dump(nusc_submissions, res_path)
#         return res_path


# # def output_to_nusc_box(
# #         detection: dict) -> Tuple[List[NuScenesBox], Union[np.ndarray, None]]:
# #     """Convert the output to the box class in the nuScenes.

# #     Args:
# #         detection (dict): Detection results.

# #             - bboxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
# #             - scores_3d (torch.Tensor): Detection scores.
# #             - labels_3d (torch.Tensor): Predicted box labels.

# #     Returns:
# #         Tuple[List[:obj:`NuScenesBox`], np.ndarray or None]: List of standard
# #         NuScenesBoxes and attribute labels.
# #     """
# #     # --- 디버깅을 위해 이 코드를 함수 맨 처음에 추가하고 다시 실행하세요 ---
# #     print("\n--- DEBUG: Available keys in 'detection' dictionary ---")
# #     print(detection.keys())
# #     print("--------------------------------------------------------\n")
# #     # --- 여기까지 추가 ---

# #     # --- 이 코드를 함수 맨 처음에 추가하세요 ---
# #     # 'boxes_3d'가 이미 list로 변환되었다면, 다시 Box 객체로 재조립합니다.
# #     if isinstance(detection['bboxes_3d'], list):
# #         # box_type_3d 정보가 없을 경우를 대비해 기본값을 사용합니다.
# #         box_type_3d = detection.get('box_type_3d', LiDARInstance3DBoxes)
# #         # box_dim 정보가 없을 경우를 대비해 기본값을 사용합니다. (보통 7 또는 9)
# #         box_dim = detection.get('box_dim', 9) 
        
# #         # 빈 리스트일 경우 에러가 나지 않도록 처리합니다.
# #         if len(detection['bboxes_3d']) == 0:
# #             detection['bboxes_3d'] = box_type_3d(torch.empty((0, box_dim)))
# #         else:
# #             detection['bboxes_3d'] = box_type_3d(torch.tensor(detection['bboxes_3d']), box_dim=box_dim)
# #     # --- 여기까지 추가 ---
# #     bbox3d = detection['bboxes_3d']
# #     # scores = detection['scores_3d'].numpy()
# #     # labels = detection['labels_3d'].numpy()
# #     # 수정 후 코드 (After)
# #     # .numpy() 대신 np.array()를 사용하여 리스트를 NumPy 배열로 변환합니다.
# #     scores = np.array(detection['scores_3d'])
# #     labels = np.array(detection['labels_3d'])
# #     attrs = None
# #     if 'attr_labels' in detection:
# #         attrs = detection['attr_labels'].numpy()

# #     box_gravity_center = bbox3d.gravity_center.numpy()
# #     box_dims = bbox3d.dims.numpy()
# #     box_yaw = bbox3d.yaw.numpy()

# #     box_list = []

# #     if isinstance(bbox3d, LiDARInstance3DBoxes):
# #         # our LiDAR coordinate system -> nuScenes box coordinate system
# #         nus_box_dims = box_dims[:, [1, 0, 2]]
# #         for i in range(len(bbox3d)):
# #             quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
# #             velocity = (*bbox3d.tensor[i, 7:9], 0.0)
# #             # velo_val = np.linalg.norm(box3d[i, 7:9])
# #             # velo_ori = box3d[i, 6]
# #             # velocity = (
# #             # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
# #             box = NuScenesBox(
# #                 box_gravity_center[i],
# #                 nus_box_dims[i],
# #                 quat,
# #                 label=labels[i],
# #                 score=scores[i],
# #                 velocity=velocity)
# #             box_list.append(box)
# #     elif isinstance(bbox3d, CameraInstance3DBoxes):
# #         # our Camera coordinate system -> nuScenes box coordinate system
# #         # convert the dim/rot to nuscbox convention
# #         nus_box_dims = box_dims[:, [2, 0, 1]]
# #         nus_box_yaw = -box_yaw
# #         for i in range(len(bbox3d)):
# #             q1 = pyquaternion.Quaternion(
# #                 axis=[0, 0, 1], radians=nus_box_yaw[i])
# #             q2 = pyquaternion.Quaternion(axis=[1, 0, 0], radians=np.pi / 2)
# #             quat = q2 * q1
# #             velocity = (bbox3d.tensor[i, 7], 0.0, bbox3d.tensor[i, 8])
# #             box = NuScenesBox(
# #                 box_gravity_center[i],
# #                 nus_box_dims[i],
# #                 quat,
# #                 label=labels[i],
# #                 score=scores[i],
# #                 velocity=velocity)
# #             box_list.append(box)
# #     else:
# #         raise NotImplementedError(
# #             f'Do not support convert {type(bbox3d)} bboxes '
# #             'to standard NuScenesBoxes.')

# #     return box_list, attrs

# def output_to_nusc_box(
#         detection: dict) -> Tuple[List[NuScenesBox], Union[np.ndarray, None]]:
#     """Convert the output to the box class in the nuScenes."""

#     # 1. 올바른 키('bboxes_3d')로 데이터를 추출하고 기본형으로 변환합니다.
#     # 이전 디버깅에서 확인한 모델 출력 키를 사용합니다.
#     bboxes_3d_data = detection['bboxes_3d']
#     scores = np.array(detection['scores_3d'])
#     labels = np.array(detection['labels_3d'])
#     attrs = None
#     if 'attr_labels' in detection:
#         attrs = np.array(detection['attr_labels'])

#     # 2. (가장 중요) 'list'로 단순화된 박스 데이터를 다시 LiDARInstance3DBoxes 객체로 '복원'합니다.
#     # 이렇게 해야 .gravity_center, .dims 등의 기능을 사용할 수 있습니다.
#     if isinstance(bboxes_3d_data, list):
#         if len(bboxes_3d_data) == 0:
#             # 예측된 박스가 없는 경우, 비어있는 Box 객체를 생성합니다.
#             box_dim = detection.get('box_dim', 9) 
#             bbox3d = LiDARInstance3DBoxes(torch.empty((0, box_dim)), box_dim=box_dim)
#         else:
#             # 리스트를 텐서로 변환하여 Box 객체를 재조립합니다.
#             box_dim = len(bboxes_3d_data[0])
#             bbox3d = LiDARInstance3DBoxes(torch.tensor(bboxes_3d_data), box_dim=box_dim)
#     else:
#         # 데이터가 이미 Box 객체인 경우 그대로 사용합니다.
#         bbox3d = bboxes_3d_data
            
#     # 3. '복원된' 객체의 기능을 사용하여 필요한 계산을 수행합니다.
#     box_gravity_center = bbox3d.gravity_center.numpy()
#     box_dims = bbox3d.dims.numpy()
#     box_yaw = bbox3d.yaw.numpy()

#     # 4. 계산된 결과를 nuScenes 표준 Box 형식으로 변환합니다.
#     box_list = []
#     if isinstance(bbox3d, LiDARInstance3DBoxes):
#         # LiDAR 좌표계 -> nuScenes 박스 좌표계 변환
#         nus_box_dims = box_dims[:, [1, 0, 2]]
#         for i in range(len(bbox3d)):
#             quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
#             # 속도 정보가 있다면 사용하고, 없다면 0으로 처리합니다.
#             velocity = (*bbox3d.tensor[i, 7:9], 0.0) if bbox3d.tensor.shape[1] == 9 else (0.0, 0.0, 0.0)
#             box = NuScenesBox(
#                 box_gravity_center[i],
#                 nus_box_dims[i],
#                 quat,
#                 label=labels[i],
#                 score=scores[i],
#                 velocity=velocity)
#             box_list.append(box)
#     elif isinstance(bbox3d, CameraInstance3DBoxes):
#         # 카메라 좌표계 -> nuScenes 박스 좌표계 변환
#         nus_box_dims = box_dims[:, [2, 0, 1]]
#         nus_box_yaw = -box_yaw
#         for i in range(len(bbox3d)):
#             q1 = pyquaternion.Quaternion(
#                 axis=[0, 0, 1], radians=nus_box_yaw[i])
#             q2 = pyquaternion.Quaternion(axis=[1, 0, 0], radians=np.pi / 2)
#             quat = q2 * q1
#             velocity = (bbox3d.tensor[i, 7], 0.0, bbox3d.tensor[i, 8])
#             box = NuScenesBox(
#                 box_gravity_center[i],
#                 nus_box_dims[i],
#                 quat,
#                 label=labels[i],
#                 score=scores[i],
#                 velocity=velocity)
#             box_list.append(box)
#     else:
#         # 지원하지 않는 박스 타입에 대한 에러 처리
#         if len(bbox3d) > 0:
#              raise NotImplementedError(
#                 f'Do not support convert {type(bbox3d)} bboxes '
#                 'to standard NuScenesBoxes.')

#     return box_list, attrs

# def lidar_nusc_box_to_global(
#         info: dict, boxes: List[NuScenesBox], classes: List[str],
#         eval_configs: DetectionConfig) -> List[NuScenesBox]:
#     """Convert the box from ego to global coordinate.

#     Args:
#         info (dict): Info for a specific sample data, including the calibration
#             information.
#         boxes (List[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
#         classes (List[str]): Mapped classes in the evaluation.
#         eval_configs (:obj:`DetectionConfig`): Evaluation configuration object.

#     Returns:
#         List[:obj:`DetectionConfig`]: List of standard NuScenesBoxes in the
#         global coordinate.
#     """
#     box_list = []
#     for box in boxes:
#         # Move box to ego vehicle coord system
#         lidar2ego = np.array(info['lidar_points']['lidar2ego'])
#         box.rotate(
#             pyquaternion.Quaternion(matrix=lidar2ego, rtol=1e-05, atol=1e-07))
#         box.translate(lidar2ego[:3, 3])
#         # filter det in ego.
#         cls_range_map = eval_configs.class_range
#         radius = np.linalg.norm(box.center[:2], 2)
#         det_range = cls_range_map[classes[box.label]]
#         if radius > det_range:
#             continue
#         # Move box to global coord system
#         ego2global = np.array(info['ego2global'])
#         box.rotate(
#             pyquaternion.Quaternion(matrix=ego2global, rtol=1e-05, atol=1e-07))
#         box.translate(ego2global[:3, 3])
#         box_list.append(box)
#     return box_list


# def cam_nusc_box_to_global(
#     info: dict,
#     boxes: List[NuScenesBox],
#     attrs: np.ndarray,
#     classes: List[str],
#     eval_configs: DetectionConfig,
#     camera_type: str = 'CAM_FRONT',
# ) -> Tuple[List[NuScenesBox], List[int]]:
#     """Convert the box from camera to global coordinate.

#     Args:
#         info (dict): Info for a specific sample data, including the calibration
#             information.
#         boxes (List[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
#         attrs (np.ndarray): Predicted attributes.
#         classes (List[str]): Mapped classes in the evaluation.
#         eval_configs (:obj:`DetectionConfig`): Evaluation configuration object.
#         camera_type (str): Type of camera. Defaults to 'CAM_FRONT'.

#     Returns:
#         Tuple[List[:obj:`NuScenesBox`], List[int]]: List of standard
#         NuScenesBoxes in the global coordinate and attribute label.
#     """
#     box_list = []
#     attr_list = []
#     for (box, attr) in zip(boxes, attrs):
#         # Move box to ego vehicle coord system
#         cam2ego = np.array(info['images'][camera_type]['cam2ego'])
#         box.rotate(
#             pyquaternion.Quaternion(matrix=cam2ego, rtol=1e-05, atol=1e-07))
#         box.translate(cam2ego[:3, 3])
#         # filter det in ego.
#         cls_range_map = eval_configs.class_range
#         radius = np.linalg.norm(box.center[:2], 2)
#         det_range = cls_range_map[classes[box.label]]
#         if radius > det_range:
#             continue
#         # Move box to global coord system
#         ego2global = np.array(info['ego2global'])
#         box.rotate(
#             pyquaternion.Quaternion(matrix=ego2global, rtol=1e-05, atol=1e-07))
#         box.translate(ego2global[:3, 3])
#         box_list.append(box)
#         attr_list.append(attr)
#     return box_list, attr_list


# def global_nusc_box_to_cam(info: dict, boxes: List[NuScenesBox],
#                            classes: List[str],
#                            eval_configs: DetectionConfig) -> List[NuScenesBox]:
#     """Convert the box from global to camera coordinate.

#     Args:
#         info (dict): Info for a specific sample data, including the calibration
#             information.
#         boxes (List[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
#         classes (List[str]): Mapped classes in the evaluation.
#         eval_configs (:obj:`DetectionConfig`): Evaluation configuration object.

#     Returns:
#         List[:obj:`NuScenesBox`]: List of standard NuScenesBoxes in camera
#         coordinate.
#     """
#     box_list = []
#     for box in boxes:
#         # Move box to ego vehicle coord system
#         ego2global = np.array(info['ego2global'])
#         box.translate(-ego2global[:3, 3])
#         box.rotate(
#             pyquaternion.Quaternion(matrix=ego2global, rtol=1e-05,
#                                     atol=1e-07).inverse)
#         # filter det in ego.
#         cls_range_map = eval_configs.class_range
#         radius = np.linalg.norm(box.center[:2], 2)
#         det_range = cls_range_map[classes[box.label]]
#         if radius > det_range:
#             continue
#         # Move box to camera coord system
#         cam2ego = np.array(info['images']['CAM_FRONT']['cam2ego'])
#         box.translate(-cam2ego[:3, 3])
#         box.rotate(
#             pyquaternion.Quaternion(matrix=cam2ego, rtol=1e-05,
#                                     atol=1e-07).inverse)
#         box_list.append(box)
#     return box_list


# def nusc_box_to_cam_box3d(
#     boxes: List[NuScenesBox]
# ) -> Tuple[CameraInstance3DBoxes, torch.Tensor, torch.Tensor]:
#     """Convert boxes from :obj:`NuScenesBox` to :obj:`CameraInstance3DBoxes`.

#     Args:
#         boxes (:obj:`List[NuScenesBox]`): List of predicted NuScenesBoxes.

#     Returns:
#         Tuple[:obj:`CameraInstance3DBoxes`, torch.Tensor, torch.Tensor]:
#         Converted 3D bounding boxes, scores and labels.
#     """
#     locs = torch.Tensor([b.center for b in boxes]).view(-1, 3)
#     dims = torch.Tensor([b.wlh for b in boxes]).view(-1, 3)
#     rots = torch.Tensor([b.orientation.yaw_pitch_roll[0]
#                          for b in boxes]).view(-1, 1)
#     velocity = torch.Tensor([b.velocity[0::2] for b in boxes]).view(-1, 2)

#     # convert nusbox to cambox convention
#     dims[:, [0, 1, 2]] = dims[:, [1, 2, 0]]
#     rots = -rots

#     boxes_3d = torch.cat([locs, dims, rots, velocity], dim=1).cuda()
#     cam_boxes3d = CameraInstance3DBoxes(
#         boxes_3d, box_dim=9, origin=(0.5, 0.5, 0.5))
#     scores = torch.Tensor([b.score for b in boxes]).cuda()
#     labels = torch.LongTensor([b.label for b in boxes]).cuda()
#     nms_scores = scores.new_zeros(scores.shape[0], 10 + 1)
#     indices = labels.new_tensor(list(range(scores.shape[0])))
#     nms_scores[indices, labels] = scores
#     return cam_boxes3d, nms_scores, labels

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

# ----- HELPER FUNCTION -----

# def output_to_nusc_box(
#         detection: dict) -> Tuple[List[NuScenesBox], Union[np.ndarray, None]]:
#     """Helper function to convert detection dictionary to NuScenesBox objects."""
#     bboxes_3d_data = detection['bboxes_3d']
#     scores = np.array(detection['scores_3d'])
#     labels = np.array(detection['labels_3d'])
#     attrs = None
#     if 'attr_labels' in detection:
#         attrs = np.array(detection['attr_labels'])

#     if isinstance(bboxes_3d_data, list):
#         if len(bboxes_3d_data) == 0:
#             box_dim = detection.get('box_dim', 9)
#             bbox3d = LiDARInstance3DBoxes(torch.empty((0, box_dim)), box_dim=box_dim)
#         else:
#             box_dim = len(bboxes_3d_data[0])
#             bbox3d = LiDARInstance3DBoxes(torch.tensor(bboxes_3d_data), box_dim=box_dim)
#     else:
#         bbox3d = bboxes_3d_data
            
#     box_gravity_center = bbox3d.gravity_center.numpy()
#     box_dims = bbox3d.dims.numpy()
#     box_yaw = bbox3d.yaw.numpy()

#     box_list = []
#     if isinstance(bbox3d, LiDARInstance3DBoxes):
#         nus_box_dims = box_dims[:, [1, 0, 2]]
#         for i in range(len(bbox3d)):
#             quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
#             velocity = (*bbox3d.tensor[i, 7:9], 0.0) if bbox3d.tensor.shape[1] == 9 else (0.0, 0.0, 0.0)
#             box = NuScenesBox(
#                 box_gravity_center[i],
#                 nus_box_dims[i],
#                 quat,
#                 label=labels[i],
#                 score=scores[i],
#                 velocity=velocity)
#             box_list.append(box)
#     else:
#         if len(bbox3d) > 0:
#              raise NotImplementedError(f'Do not support convert {type(bbox3d)} bboxes.')

#     return box_list, attrs

# # 기존 output_to_nusc_box 함수를 아래 코드로 완전히 교체하세요.
# def output_to_nusc_box(
#         detection: dict,
#         info: dict  # 변환 정보를 담은 info 딕셔너리
# ) -> Tuple[List[NuScenesBox], Union[np.ndarray, None]]:
#     """모든 수정 사항이 반영된 최종 헬퍼 함수"""
    
#     # 1. 데이터 추출 및 객체 복원 (이전과 동일)
#     bboxes_3d_data = detection['bboxes_3d']
#     scores = np.array(detection['scores_3d'])
#     labels = np.array(detection['labels_3d'])
#     attrs = None
#     if 'attr_labels' in detection:
#         attrs = np.array(detection['attr_labels'])

#     if isinstance(bboxes_3d_data, list):
#         if len(bboxes_3d_data) == 0:
#             box_dim = detection.get('box_dim', 9) 
#             bbox3d = LiDARInstance3DBoxes(torch.empty((0, box_dim)), box_dim=box_dim)
#         else:
#             box_dim = len(bboxes_3d_data[0])
#             bbox3d = LiDARInstance3DBoxes(torch.tensor(bboxes_3d_data), box_dim=box_dim)
#     else:
#         bbox3d = bboxes_3d_data

#     # --- ▼▼▼▼▼ [핵심 수정] 실제 데이터 구조에 맞는 정확한 키 경로 사용 ▼▼▼▼▼ ---
    
#     bbox3d.tensor[:, 6] += np.pi / 2  # yaw 보정 (필요시)
#     # 2. info 딕셔너리에서 올바른 경로로 4x4 변환 행렬을 가져옵니다.
#     lidar2ego_mat = np.array(info['lidar_points']['lidar2ego']) # 경로 수정!
#     ego2global_mat = np.array(info['ego2global'])
    
#     # 3. 각 행렬에서 3x3 회전 행렬과 3x1 이동 벡터를 추출합니다.
#     lidar2ego_rot = lidar2ego_mat[:3, :3]
#     lidar2ego_trans = lidar2ego_mat[:3, 3]
    
#     ego2global_rot = ego2global_mat[:3, :3]
#     ego2global_trans = ego2global_mat[:3, 3]

#     # 4. 박스를 라이다 -> 자차(Ego) -> 글로벌(Global) 좌표계로 순차적으로 변환합니다.
#     bbox3d.rotate(lidar2ego_rot)
#     bbox3d.translate(lidar2ego_trans)
#     bbox3d.rotate(ego2global_rot)
#     bbox3d.translate(ego2global_trans)
#     # --- ▲▲▲▲▲ [핵심 수정] 여기까지 ▲▲▲▲▲ ---
            
#     # 5. '글로벌 좌표계'로 변환된 객체에서 정보를 추출합니다.
#     box_gravity_center = bbox3d.gravity_center.numpy()
#     box_dims = bbox3d.dims.numpy()
#     box_yaw = bbox3d.yaw.numpy()

#     # 6. nuScenes 표준 Box 형식으로 변환합니다 (이후 로직은 동일).
#     box_list = []
#     if isinstance(bbox3d, LiDARInstance3DBoxes):
#         nus_box_dims = box_dims[:, [1, 0, 2]]
#         for i in range(len(bbox3d)):
#             quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
#             velocity = (*bbox3d.tensor[i, 7:9], 0.0) if bbox3d.tensor.shape[1] == 9 else (0.0, 0.0, 0.0)
#             box = NuScenesBox(
#                 box_gravity_center[i],
#                 nus_box_dims[i],
#                 quat,
#                 label=labels[i],
#                 score=scores[i],
#                 velocity=velocity)
#             box_list.append(box)
#     else:
#         if len(bbox3d) > 0:
#              raise NotImplementedError(
#                 f'Do not support convert {type(bbox3d)} bboxes '
#                 'to standard NuScenesBoxes.')

#     return box_list, attrs

# # nuscenes_metric.py 파일 상단에 이 코드를 추가해주세요.
# _DEBUG_PRINTED_ONCE = False

# # 기존 output_to_nusc_box 함수를 아래 코드로 완전히 교체하세요.
# def output_to_nusc_box(
#         detection: dict,
#         info: dict,
#         nusc: NuScenes  # <-- nusc 객체를 받도록 추가
# ) -> Tuple[List[NuScenesBox], Union[np.ndarray, None]]:
#     """좌표 변환의 모든 단계를 기록하는 최종 디버깅 함수"""
    
#     global _DEBUG_PRINTED_ONCE

#     # (데이터 추출 및 객체 복원 로직은 동일)
#     bboxes_3d_data = detection['bboxes_3d']
#     scores = np.array(detection['scores_3d'])
#     labels = np.array(detection['labels_3d'])
#     attrs = None
#     if 'attr_labels' in detection:
#         attrs = np.array(detection['attr_labels'])

#     if isinstance(bboxes_3d_data, list):
#         if len(bboxes_3d_data) == 0:
#             box_dim = detection.get('box_dim', 9)
#             bbox3d = LiDARInstance3DBoxes(torch.empty((0, box_dim)), box_dim=box_dim)
#         else:
#             box_dim = len(bboxes_3d_data[0])
#             bbox3d = LiDARInstance3DBoxes(torch.tensor(bboxes_3d_data), box_dim=box_dim)
#     else:
#         bbox3d = bboxes_3d_data

#     # --- ▼▼▼▼▼ '블랙박스 기록기' 시작 ▼▼▼▼▼ ---
#     if not _DEBUG_PRINTED_ONCE and len(bbox3d) > 0:
#         logger = MMLogger.get_current_instance()
#         logger.info("\n\n--- BLACK BOX RECORDER ACTIVATED ---\n")
        
#         lidar2ego_mat = np.array(info['lidar_points']['lidar2ego'])
#         ego2global_mat = np.array(info['ego2global'])
#         logger.info(f"[BLACK BOX] lidar2ego matrix:\n{lidar2ego_mat}")
#         logger.info(f"[BLACK BOX] ego2global matrix:\n{ego2global_mat}")

#         initial_center = bbox3d.tensor[0, :3].cpu().numpy()
#         logger.info(f"[BLACK BOX] 1. Initial Center (Model Output Frame): {initial_center}")
        
#         temp_bbox3d = bbox3d.clone()
#         temp_bbox3d.rotate(lidar2ego_mat[:3, :3])
#         temp_bbox3d.translate(lidar2ego_mat[:3, 3])
#         center_after_ego = temp_bbox3d.tensor[0, :3].cpu().numpy()
#         logger.info(f"[BLACK BOX] 2. Center after Lidar->Ego Transform: {center_after_ego}")

#         temp_bbox3d.rotate(ego2global_mat[:3, :3])
#         temp_bbox3d.translate(ego2global_mat[:3, 3])
#         final_center = temp_bbox3d.tensor[0, :3].cpu().numpy()
#         logger.info(f"[BLACK BOX] 3. Final Center after Ego->Global Transform: {final_center}")

#         sample_token = info['token']
#         ann_tokens = nusc.get('sample', sample_token)['anns']
#         if ann_tokens:
#             first_ann_metadata = nusc.get('sample_annotation', ann_tokens[0])
#             logger.info(f"[BLACK BOX] 4. Ground Truth Center (Global Frame): {first_ann_metadata['translation']}")
        
#         logger.info("\n--- BLACK BOX RECORDER END ---\n\n")
#         _DEBUG_PRINTED_ONCE = True
#     # --- ▲▲▲▲▲ '블랙박스 기록기' 끝 ▲▲▲▲▲ ---

#     # 실제 변환 로직
#     lidar2ego_mat = np.array(info['lidar_points']['lidar2ego'])
#     ego2global_mat = np.array(info['ego2global'])
#     bbox3d.rotate(lidar2ego_mat[:3, :3])
#     bbox3d.translate(lidar2ego_mat[:3, 3])
#     bbox3d.rotate(ego2global_mat[:3, :3])
#     bbox3d.translate(ego2global_mat[:3, 3])
            
#     box_gravity_center = bbox3d.gravity_center.numpy()
#     box_dims = bbox3d.dims.numpy()
#     box_yaw = bbox3d.yaw.numpy()

#     # (이후 nuScenesBox 생성 로직은 기존과 동일)
#     box_list = []
#     # ... (rest of the function is the same)
#     if isinstance(bbox3d, LiDARInstance3DBoxes):
#         nus_box_dims = box_dims[:, [1, 0, 2]]
#         for i in range(len(bbox3d)):
#             quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
#             velocity = (*bbox3d.tensor[i, 7:9], 0.0) if bbox3d.tensor.shape[1] == 9 else (0.0, 0.0, 0.0)
#             box = NuScenesBox(
#                 box_gravity_center[i],
#                 nus_box_dims[i],
#                 quat,
#                 label=labels[i],
#                 score=scores[i],
#                 velocity=velocity)
#             box_list.append(box)
#     else:
#         if len(bbox3d) > 0:
#              raise NotImplementedError(f'Do not support convert {type(bbox3d)} bboxes.')

#     return box_list, attrs

# # nuscenes_metric.py 파일 상단에 이 코드를 추가해주세요.
# _DEBUG_PRINTED_ONCE = False

# # 기존 output_to_nusc_box 함수를 아래 코드로 완전히 교체하세요.
# def output_to_nusc_box(
#         detection: dict,
#         info: dict,
#         nusc: NuScenes  # nusc 객체를 받도록 유지
# ) -> Tuple[List[NuScenesBox], Union[np.ndarray, None]]:
#     """
#     [진짜 최종본] 라이브러리 버전 문제 해결 및 블랙박스 디버깅 포함
#     """
    
#     global _DEBUG_PRINTED_ONCE

#     # 1. 데이터 추출 및 객체 복원
#     bboxes_3d_data = detection['bboxes_3d']
#     scores = np.array(detection['scores_3d'])
#     labels = np.array(detection['labels_3d'])
#     attrs = None
#     if 'attr_labels' in detection:
#         attrs = np.array(detection['attr_labels'])

#     if isinstance(bboxes_3d_data, list):
#         if len(bboxes_3d_data) == 0:
#             return [], None # 예측된 박스가 없으면 즉시 종료
#         box_dim = len(bboxes_3d_data[0])
#         bbox3d = LiDARInstance3DBoxes(torch.tensor(bboxes_3d_data), box_dim=box_dim)
#     else:
#         bbox3d = bboxes_3d_data
    
#     if len(bbox3d) == 0:
#         return [], None

#     # --- ▼▼▼▼▼ '블랙박스 기록기' 시작 ▼▼▼▼▼ ---
#     if not _DEBUG_PRINTED_ONCE:
#         logger = MMLogger.get_current_instance()
#         logger.info("\n\n--- BLACK BOX RECORDER (v2) ACTIVATED ---\n")
        
#         lidar2ego_mat = np.array(info['lidar_points']['lidar2ego'])
#         ego2global_mat = np.array(info['ego2global'])
#         logger.info(f"[BLACK BOX] lidar2ego matrix:\n{lidar2ego_mat}")
#         logger.info(f"[BLACK BOX] ego2global matrix:\n{ego2global_mat}")

#         initial_center = bbox3d.tensor[0, :3].cpu().numpy()
#         logger.info(f"[BLACK BOX] 1. Initial Center (Model Output Frame): {initial_center}")
        
#         # 임시 복사본으로 변환 과정 추적
#         temp_bbox3d = bbox3d.clone()
#         temp_bbox3d.rotate(lidar2ego_mat[:3, :3])
#         temp_bbox3d.translate(lidar2ego_mat[:3, 3])
#         center_after_ego = temp_bbox3d.tensor[0, :3].cpu().numpy()
#         logger.info(f"[BLACK BOX] 2. Center after Lidar->Ego Transform: {center_after_ego}")

#         temp_bbox3d.rotate(ego2global_mat[:3, :3])
#         temp_bbox3d.translate(ego2global_mat[:3, 3])
#         final_center = temp_bbox3d.tensor[0, :3].cpu().numpy()
#         logger.info(f"[BLACK BOX] 3. Final Center after Ego->Global Transform: {final_center}")

#         sample_token = info['token']
#         ann_tokens = nusc.get('sample', sample_token)['anns']
#         if ann_tokens:
#             first_ann_metadata = nusc.get('sample_annotation', ann_tokens[0])
#             logger.info(f"[BLACK BOX] 4. Ground Truth Center (Global Frame): {first_ann_metadata['translation']}")
        
#         logger.info("\n--- BLACK BOX RECORDER END ---\n\n")
#         _DEBUG_PRINTED_ONCE = True
#     # --- ▲▲▲▲▲ '블랙박스 기록기' 끝 ▲▲▲▲▲ ---

#     # 2. 실제 변환 로직: bbox3d 객체에 직접 변환 적용
#     lidar2ego_mat = np.array(info['lidar_points']['lidar2ego'])
#     ego2global_mat = np.array(info['ego2global'])
#     bbox3d.rotate(lidar2ego_mat[:3, :3])
#     bbox3d.translate(lidar2ego_mat[:3, 3])
#     bbox3d.rotate(ego2global_mat[:3, :3])
#     bbox3d.translate(ego2global_mat[:3, 3])
            
#     # 3. '글로벌 좌표계'로 변환된 최종 객체에서 속성 추출
#     box_gravity_center = bbox3d.gravity_center.numpy()
#     box_dims = bbox3d.dims.numpy()
#     box_yaw = bbox3d.yaw.numpy()

#     # 4. nuScenesBox 생성
#     box_list = []
#     nus_box_dims = box_dims[:, [1, 0, 2]]
#     for i in range(len(bbox3d)):
#         quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
#         velocity = (*bbox3d.tensor[i, 7:9], 0.0) if bbox3d.tensor.shape[1] == 9 else (0.0, 0.0, 0.0)
#         box = NuScenesBox(
#             box_gravity_center[i],
#             nus_box_dims[i],
#             quat,
#             label=labels[i],
#             score=scores[i],
#             velocity=velocity)
#         box_list.append(box)

#     return box_list, attrs


# def ensure_orthogonal(mat):
#     """Make rotation matrix orthogonal (safe for Quaternion)."""
#     u, _, vh = np.linalg.svd(mat)
#     return np.dot(u, vh)

# def output_to_nusc_box_global(detection, info_dict, min_score=0.01):
#     """
#     Convert detection results from LiDAR frame -> ego frame -> NuScenes global frame.
#     """
#     boxes_list = []  # Renamed from 'boxes' to avoid confusion with the numpy array name
#     attrs = []

#     # --- Prediction tensors ---
#     bboxes3d = detection['bboxes_3d']
#     bboxes = np.array(bboxes3d.tensor) if hasattr(bboxes3d, 'tensor') else np.array(bboxes3d)
#     scores = np.array(detection['scores_3d'])
#     labels = np.array(detection['labels_3d'])

#     # --- Score threshold filtering ---
#     if len(bboxes) == 0:
#         return [], []
#     keep = scores > min_score
#     bboxes = bboxes[keep]
#     scores = scores[keep]
#     labels = labels[keep]

#     # --- Extract transformation matrices from info_dict ---
#     # 1. LiDAR sensor to Ego vehicle frame transformation
#     lidar2ego = np.array(info_dict['lidar_points']['lidar2ego'])
#     lidar2ego_rot = ensure_orthogonal(lidar2ego[:3, :3])
#     lidar2ego_trans = lidar2ego[:3, 3]

#     # 2. Ego vehicle to Global frame transformation
#     ego2global = np.array(info_dict['ego2global'])
#     ego2global_rot = ensure_orthogonal(ego2global[:3, :3])
#     ego2global_trans = ego2global[:3, 3]

#     for i in range(len(bboxes)):
#         center = bboxes[i, :3]
#         size = bboxes[i, 3:6]
#         yaw = bboxes[i, 6]
#         score = scores[i]
#         label = labels[i]

#         # Create Box in LiDAR frame
#         box = Box(center=center, size=size,
#                   orientation=Quaternion(axis=[0, 0, 1], radians=yaw),
#                   score=score, label=label)

#         # Step 1: Apply LiDAR -> Ego transformation
#         box.rotate(Quaternion(matrix=lidar2ego_rot))
#         box.translate(lidar2ego_trans)

#         # Step 2: Apply Ego -> Global transformation
#         box.rotate(Quaternion(matrix=ego2global_rot))
#         box.translate(ego2global_trans)

#         boxes_list.append(box)
#         attrs.append(None)
    
#     # 디버깅 코드는 유지해도 좋습니다.
#     # 하지만 변환 후 좌표를 확인하는 것이 중요합니다.
#     # if boxes_list:
#     #     global_coords = np.array([b.center for b in boxes_list])
#         # print(f"[DEBUG] Sample processed. Final global X range: {global_coords[:,0].min():.2f} ~ {global_coords[:,0].max():.2f}")
        
#     return boxes_list, attrs

# # ----- MAIN METRIC CLASS -----
# @METRICS.register_module()
# class NuScenesMetric(BaseMetric):

#     def __init__(self,
#                  ann_file: str,
#                  metric: Union[str, List[str]] = 'bbox',
#                  data_root: Optional[str] = None,
#                  version: str = 'v1.0-trainval',
#                  **kwargs) -> None:
        
#         kwargs.pop('backend_args', None)
#         # kwargs.pop('format_only', None)
#         # kwargs.pop('outfile_prefix', None)
#         super().__init__(**kwargs)
        
#         self.ann_file = ann_file
#         self.metric = metric
#         self.data_root = data_root
#         self.version = version

#         data = mmengine.load(ann_file)
#         if isinstance(data, dict) and 'data_list' in data:
#             self.data_infos = data['data_list']
#         elif isinstance(data, list):
#             self.data_infos = data
#         else:
#             raise ValueError('Unsupported annotation file format.')
        
#         self.nusc = NuScenes(version=self.version, dataroot=self.data_root, verbose=False)
#         from nuscenes.eval.detection.config import config_factory
#         self.eval_detection_configs = config_factory('detection_cvpr_2019')
#         self.eval_set_map = {'v1.0-mini': 'mini_val', 'v1.0-trainval': 'val'}
#         self.ErrNameMapping = {
#             'trans_err': 'mATE', 'scale_err': 'mASE', 'orient_err': 'mAOE',
#             'vel_err': 'mAVE', 'attr_err': 'mAAE'
#         }

#     def process(self, data_batch: dict, data_samples: list) -> None:
#         """Collect results from each batch."""
#         for data_sample in data_samples:
#             result = dict()
#             pred_3d = data_sample['pred_instances_3d']
            
#             result['pred_instances_3d'] = {
#                 'bboxes_3d': pred_3d['bboxes_3d'].tensor.cpu().tolist(),
#                 'scores_3d': pred_3d['scores_3d'].cpu().tolist(),
#                 'labels_3d': pred_3d['labels_3d'].cpu().tolist(),
#             }
#             if 'attr_labels' in pred_3d:
#                 result['pred_instances_3d']['attr_labels'] = pred_3d['attr_labels'].cpu().tolist()

#             if 'sample_idx' in data_sample:
#                  result['sample_idx'] = data_sample['sample_idx']
            
#             self.results.append(result)

#     def compute_metrics(self, results: list) -> Dict[str, Any]:
#         """Compute metrics from the collected results."""
#         logger = MMLogger.get_current_instance()
#         logger.info('Starting NuScenes evaluation...')
        
#         # Format the results into the required JSON structure.
#         results_json = self.format_results(results)
        
#         # Perform evaluation using the formatted results.
#         metric_dict = self.nus_evaluate(results_json)
        
#         return metric_dict
        
#     # 기존 format_results 함수를 아래의 디버깅 코드가 추가된 버전으로 교체하세요.
#     def format_results(self, results: list) -> dict:
#         """
#         우리가 이전에 수정했던 로직을 통합하여
#         BaseNuScenesMetric의 format_results와 유사하게 만듭니다.
#         """
#         logger = MMLogger.get_current_instance()
#         logger.info('Formatting results for nuScenes evaluation...')
        
#         nusc_annos = {}
#         mapped_class_names = self.dataset_meta['classes']

#         for i, res in enumerate(results):
#             sample_idx = res['sample_idx']
#             pred_instances_3d = res['pred_instances_3d']
            
#             # --- ▼▼▼▼▼ 디버깅 블록 시작 ▼▼▼▼▼ ---
#             if i == 0: # 첫 번째 샘플에 대해서만 상세 로그 출력
#                 sample_token_debug = self.data_infos[sample_idx]['token']
#                 logger.info("\n\n--- STARTING PRECISION DEBUG FOR THE FIRST SAMPLE ---\n")
#                 logger.info(f"[DEBUG] Processing sample_idx: {sample_idx}, sample_token: {sample_token_debug}")

#                 # 1. 원본 예측 좌표 출력 (자차 좌표계일 가능성이 높음)
#                 raw_pred_boxes = pred_instances_3d.get('bboxes_3d', [])
#                 if raw_pred_boxes:
#                     logger.info(f"[DEBUG] Raw Predicted Box Center (Ego Frame?): {raw_pred_boxes[0][:3]}")

#                 # 2. 정답(GT) 좌표 출력 (글로벌 좌표계)
#                 ann_tokens = self.nusc.get('sample', sample_token_debug)['anns']
#                 if ann_tokens:
#                     first_ann_metadata = self.nusc.get('sample_annotation', ann_tokens[0])
#                     logger.info(f"[DEBUG] Ground Truth Box Center (Global Frame): {first_ann_metadata['translation']}")
#             # --- ▲▲▲▲▲ 디버깅 블록 끝 ▲▲▲▲▲ ---
            
#             formatted_det = {
#                 'bboxes_3d': pred_instances_3d['bboxes_3d'],
#                 'scores_3d': pred_instances_3d['scores_3d'],
#                 'labels_3d': pred_instances_3d['labels_3d'],
#             }
#             if 'attr_labels' in pred_instances_3d:
#                 formatted_det['attr_labels'] = pred_instances_3d['attr_labels']

#             # boxes, attrs = output_to_nusc_box(formatted_det)
#             boxes, attrs = output_to_nusc_box_global(formatted_det, self.data_infos[sample_idx])
#             # boxes, attrs = output_to_nusc_box_ego(formatted_det, self.data_infos[sample_idx], self.nusc)
#             sample_token = self.data_infos[sample_idx]['token']
            
#             annos = []
#             for j, box in enumerate(boxes):
#                 name = mapped_class_names[box.label]
#                 # ▼▼▼▼▼ 디버깅 코드 추가 ▼▼▼▼▼
#                 if i == 0 and j < 5: # 첫 번째 샘플의 예측 5개에 대해서만 로그 출력
#                     logger = MMLogger.get_current_instance()
#                     logger.info(f"[DEBUG] Raw Label ID: {box.label}, Mapped Name: '{name}', Score: {box.score:.2f}")
#                 # ▲▲▲▲▲ 디버깅 코드 끝 ▲▲▲▲▲
#                 velocity = (0.0, 0.0)
#                 if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
#                     if name in ['car', 'construction_vehicle', 'bus', 'trailer', 'truck']:
#                         velocity = box.velocity[:2].tolist()

#                 nusc_anno = dict(
#                     sample_token=sample_token,
#                     translation=box.center.tolist(),
#                     size=box.wlh.tolist(),
#                     rotation=box.orientation.elements.tolist(),
#                     velocity=velocity,
#                     detection_name=name,
#                     detection_score=box.score,
#                     attribute_name='')
#                 annos.append(nusc_anno)
            
#             # --- ▼▼▼▼▼ 디버깅 블록 2 시작 ▼▼▼▼▼ ---
#             if i == 0 and annos:
#                 # 3. 최종 변환된 예측 좌표 출력 (글로벌 좌표계여야 함)
#                 logger.info(f"[DEBUG] Final Formatted Box Center (Global Frame?): {annos[0]['translation']}")
#                 logger.info("\n--- END OF PRECISION DEBUG ---\n\n")
#             # --- ▲▲▲▲▲ 디버깅 블록 2 끝 ▲▲▲▲▲ ---
            
#             if sample_token in nusc_annos:
#                 nusc_annos[sample_token].extend(annos)
#             else:
#                 nusc_annos[sample_token] = annos

#         # (함수의 나머지 부분은 그대로)
#         gt_tokens = {info['token'] for info in self.data_infos}
#         for token in gt_tokens:
#             if token not in nusc_annos:
#                 nusc_annos[token] = []

#         final_submission = {
#             'meta': self.dataset_meta,
#             'results': nusc_annos,
#         }
        
#         return final_submission

#     def nus_evaluate(self, results_json: dict) -> dict:
#         """Evaluate the results using nuScenes-devkit."""
#         from nuscenes.eval.detection.evaluate import NuScenesEval
        
#         # Create a temporary directory to save the submission file
#         with tempfile.TemporaryDirectory() as tmp_dir:
#             result_path = osp.join(tmp_dir, 'results_nusc.json')
#             mmengine.dump(results_json, result_path)

#             nusc_eval = NuScenesEval(
#                 self.nusc,
#                 config=self.eval_detection_configs,
#                 result_path=result_path,
#                 eval_set=self.eval_set_map[self.version],
#                 output_dir=tmp_dir,
#                 verbose=False,)
#             nusc_eval.main(plot_examples=0, render_curves=False)

#             metrics_summary = mmengine.load(osp.join(tmp_dir, 'metrics_summary.json'))
        
#         metric_dict = {}
#         for k, v in metrics_summary['mean_dist_aps'].items():
#             metric_dict[f'{k}_mAP'] = v
#         for k, v in metrics_summary['tp_errors'].items():
#             metric_dict[self.ErrNameMapping[k]] = v
#         metric_dict['NDS'] = metrics_summary['nd_score']
#         metric_dict['mAP'] = metrics_summary['mean_ap']
        
#         return metric_dict
    

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

    def compute_metrics(self, results: list) -> Dict[str, Any]:
        """기존 코드와 동일 (수정 없음)"""
        logger = MMLogger.get_current_instance()
        logger.info('Starting NuScenes evaluation...')
        results_json = self.format_results(results)
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