import torch
import os
from collections import OrderedDict

# --- 1. 설정: 이 부분만 수정하세요 ---

# 불러올 원본 BEVFusion 체크포인트 파일 경로
FULL_CHECKPOINT_PATH = "data/work_dirs/bevfusion/20251118_corr_e2e_1st_stage_v2.0/iter_176000.pth"

# 추출한 파라미터를 저장할 디렉토리
OUTPUT_DIR = "data/work_dirs/extracted_backbones6"

# config 파일에 정의된 모듈 이름과 정확히 일치해야 합니다.
MODULE_PREFIXES_TO_EXTRACT = [
    # "pts_voxel_encoder",
    # "pts_middle_encoder",
    # "pts_backbone",
    # "pts_neck",
    "img_neck",
    "img_bbox_head",
    "img_backbone",
    # "pts_voxel_layer"는 일반적으로 학습 가능한 파라미터가 없습니다.
    # 만약 커스텀 VoxelLayer에 파라미터가 있다면 리스트에 추가하세요.
    # "pts_voxel_layer", 
    "z_estimator",
    "corr",
    "calib_head",
]

# -------------------------------------

def extract_weights(full_checkpoint_path, output_dir, module_prefixes):
    """
    거대 체크포인트에서 특정 모듈의 가중치(state_dict)만 추출하여
    개별 .pth 파일로 저장합니다.
    """
    print(f"Loading full checkpoint from: {full_checkpoint_path}")
    # CPU로 로드하여 GPU 메모리 문제 방지
    try:
        checkpoint = torch.load(full_checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading checkpoint file: {e}")
        print("Please check if the file path is correct and the file is not corrupted.")
        return

    # MMDetection 체크포인트는 'state_dict' 키 안에 가중치를 저장합니다.
    if 'state_dict' not in checkpoint:
        print("Error: 'state_dict' key not found in the checkpoint.")
        print("This might be a lightweight checkpoint or in a different format.")
        return
        
    full_state_dict = checkpoint['state_dict']
    print(f"Total keys in full checkpoint: {len(full_state_dict)}")

    # 모듈별로 가중치를 저장할 딕셔너리 준비
    extracted_weights = {prefix: OrderedDict() for prefix in module_prefixes}
    
    found_keys = 0
    
    # 전체 state_dict를 순회하며 모듈별로 분리
    for k, v in full_state_dict.items():
        
        # DDP (Distributed Data Parallel) 학습 시 'module.' 접두사 제거
        if k.startswith('module.'):
            k_clean = k[7:]
        else:
            k_clean = k
            
        # 등록된 접두사와 일치하는지 확인
        for prefix in module_prefixes:
            if k_clean.startswith(prefix + '.'):
                # 새 state_dict에 저장할 키 (접두사 제거)
                # 예: 'pts_backbone.conv1.weight' -> 'conv1.weight'
                new_key = k_clean[len(prefix) + 1:]
                extracted_weights[prefix][new_key] = v
                found_keys += 1
                break # 다음 키로 이동
                
    print(f"Successfully matched and processed {found_keys} keys.")

    # 추출한 가중치를 파일로 저장
    os.makedirs(output_dir, exist_ok=True)
    
    for prefix, state_dict in extracted_weights.items():
        if not state_dict:
            print(f"⚠️ Warning: No weights found for prefix '{prefix}'. Skipping file creation.")
            continue
            
        output_path = os.path.join(output_dir, f"{prefix}_pretrained.pth")
        
        # MMDetection의 init_cfg가 인식할 수 있도록 'state_dict' 키로 감싸서 저장
        try:
            torch.save({'state_dict': state_dict}, output_path)
            print(f"✅ Successfully extracted {len(state_dict)} keys and saved to: {output_path}")
        except Exception as e:
            print(f"❌ Failed to save file for '{prefix}': {e}")

    print("\nExtraction complete.")

if __name__ == "__main__":
    extract_weights(FULL_CHECKPOINT_PATH, OUTPUT_DIR, MODULE_PREFIXES_TO_EXTRACT)