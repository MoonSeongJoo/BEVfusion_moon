# 파일 경로: my_project/datasets/my_collate.py

import torch
from typing import List, Dict, Any
from mmengine.registry import FUNCTIONS
from torch.utils.data.dataloader import default_collate
from mmengine.model import stack_batch

@FUNCTIONS.register_module()
def custom_collate(data_batch: list) -> dict:
    """
    A robust custom collate function that explicitly handles different
    data types ('points', 'imgs', 'img_original', etc.) to ensure
    correct batching.
    """
    data_samples = [d.pop('data_samples') for d in data_batch]
    inputs_list = [d['inputs'] for d in data_batch]
    
    # 모든 inputs 딕셔너리에 있는 키들을 모음
    all_keys = set().union(*[d.keys() for d in inputs_list])
    
    batched_inputs = {}
    for key in all_keys:
        data_for_key = [d[key] for d in inputs_list]
        
        if key in ['points', 'points_original' ,'perturbed_points']:
            # 포인트 클라우드는 항상 리스트로 유지
            batched_inputs[key] = data_for_key
            
        elif key == 'imgs':
            # 'imgs'는 파이프라인에서 이미 크기가 통일되었으므로, torch.stack으로 합침
            batched_inputs[key] = torch.stack(data_for_key, dim=0)
            
        elif key == 'img_original':
            # 'img_original'은 크기가 다를 수 있으므로, stack_batch로 동적 패딩 후 합침
            # pad_value는 이미지 정규화 상태에 따라 0 또는 다른 값으로 조정할 수 있습니다.
            batched_inputs[key] = stack_batch(data_for_key, pad_size_divisor=1, pad_value=0)
            
        else:
            # 그 외 다른 모든 키들(행렬, GT 좌표 등)은 기본 collate 방식을 시도
            try:
                batched_inputs[key] = default_collate(data_for_key)
            except Exception:
                # default_collate가 실패할 경우, 리스트로 유지
                batched_inputs[key] = data_for_key
                
    return {'inputs': batched_inputs, 'data_samples': data_samples}