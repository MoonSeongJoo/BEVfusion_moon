import pickle
import numpy as np

# -------------------------------------------------------------------
# 👇👇👇 방금 생성된 'train.pkl' 파일의 전체 경로를 여기에 입력하세요 👇👇👇
# -------------------------------------------------------------------
pkl_path = 'data/nuscenes/nuscenes_infos_train_new_with_2d.pkl' 
# -------------------------------------------------------------------

print(f"\n--- '{pkl_path}' 파일 검사를 시작합니다 ---\n")

try:
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
except FileNotFoundError:
    print(f"🚨 에러: 파일을 찾을 수 없습니다! 경로를 다시 확인해주세요.")
    exit()

infos = data.get('data_list')
if infos is None:
    print("🚨 에러: pkl 파일에서 'infos' 키를 찾을 수 없습니다.")
    exit()

print(f"총 샘플 개수: {len(infos)}")
print("-" * 40)

# 파일의 첫 3개 샘플 구조만 상세히 확인
for i in range(min(3, len(infos))):
    print(f"\n🔍 샘플 #{i} 분석")
    info = infos[i]
    
    print(f"  - 최상위 레벨에 있는 키 목록: {list(info.keys())}")
    
    # 1. 최상위 레벨에 GT 정보가 있는지 확인
    if 'gt_names' in info and len(info['gt_names']) > 0:
        print(f"  ✅ [최상위] 'gt_names' 발견! (개수: {len(info['gt_names'])}) -> 예시: {info['gt_names'][:3]}")
    else:
        print("  ❌ [최상위] 'gt_names' 없음 또는 비어있음.")

    if 'gt_bboxes_3d' in info:
        print(f"  ✅ [최상위] 'gt_bboxes_3d' 발견! (Shape: {info['gt_bboxes_3d'].shape})")
    else:
        print("  ❌ [최상위] 'gt_bboxes_3d' 없음.")

    # 2. 'ann_info' 내부에 GT 정보가 있는지 확인
    if 'ann_info' in info:
        print("  ✅ [내부] 'ann_info' 키 발견!")
        ann_info = info['ann_info']
        print(f"    - 'ann_info' 내부 키: {list(ann_info.keys())}")
        if 'gt_labels_3d' in ann_info and len(ann_info['gt_labels_3d']) > 0:
            print(f"    ✅ 'gt_labels_3d' 발견! (개수: {len(ann_info['gt_labels_3d'])})")
        else:
            print("    ❌ 'gt_labels_3d' 없음 또는 비어있음.")
    else:
        print("  ❌ [내부] 'ann_info' 키 없음.")

print("\n" + "-" * 40)
print("--- 검사 완료 ---\n")