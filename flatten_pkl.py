import pickle
import os
from tqdm import tqdm

# -------------------------------------------------------------------
# 👇👇👇 1. 원본 pkl 파일 경로를 입력하세요 👇👇👇
# -------------------------------------------------------------------
original_pkl_path = 'data/nuscenes/nuscenes_bevfusion_2d_gt_insert_infos_train.pkl'
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 👇👇👇 2. 저장할 새로운 파일 경로와 이름을 직접 입력하세요 👇👇👇
# -------------------------------------------------------------------
new_pkl_path = 'data/nuscenes/nuscenes_bevfusion_2d_gt_insert_infos_train_flattened1.pkl'
# -------------------------------------------------------------------


print(f"1. 원본 파일 로딩 중...: '{original_pkl_path}'")
if not os.path.exists(original_pkl_path):
    print(f"🚨 에러: 원본 파일을 찾을 수 없습니다! 경로를 다시 확인해주세요: {original_pkl_path}")
    exit()

with open(original_pkl_path, 'rb') as f:
    data = pickle.load(f)

print("2. 데이터 구조 변경 시작...")

original_data_list = data['data_list']
new_data_list = []

flatten_success_count = 0

for info in tqdm(original_data_list, desc="  - 샘플 처리 중"):
    # 'instances' 키가 있고 내용물이 비어있지 않은지 확인
    if 'instances' in info and info['instances']:
        instance_data = info['instances']

        data_to_merge = None
        # 타입에 따라 유연하게 처리
        if isinstance(instance_data, list) and len(instance_data) > 0:
            # 리스트인 경우, 첫 번째 딕셔너리를 사용 (일반적인 구조)
            data_to_merge = instance_data[0]
        elif isinstance(instance_data, dict):
            # 이미 딕셔너리인 경우, 그대로 사용
            data_to_merge = instance_data
        
        if data_to_merge is not None:
            info.update(data_to_merge)
            del info['instances']
            flatten_success_count += 1
            
    new_data_list.append(info)

# 새 데이터 구조 생성
new_data = {
    'metainfo': data['metainfo'],
    'data_list': new_data_list
}

print(f"3. 변경된 데이터 저장 중...: '{new_pkl_path}'")
with open(new_pkl_path, 'wb') as f:
    pickle.dump(new_data, f)

print("\n" + "="*50)
print("🎉 작업 완료! 🎉")
print(f"  - 원본 파일: {original_pkl_path}")
print(f"  - 생성된 파일: {new_pkl_path}")
print("\n🔍 최종 결과 검증:")
print(f"  - 총 샘플 개수: {len(new_data_list)}")
print(f"  - ✅ 구조가 성공적으로 변경된 샘플(GT가 있던 샘플) 수: {flatten_success_count}")
print(f"  - ❌ 구조 변경이 필요 없었던 샘플(GT가 없던 샘플) 수: {len(new_data_list) - flatten_success_count}")
print("="*50)

if flatten_success_count == 0:
    print("\n🚨 경고: Ground Truth를 포함한 샘플이 하나도 없어 구조 변경이 일어나지 않았습니다.")
    print("   원본 데이터 생성 과정에 문제가 없는지 확인해 보세요.")
else:
    print("\n이제 config 파일에서 pkl 파일 경로를 위의 '생성된 파일' 경로로 변경하고,")
    print("이전에 추가했던 ann_file_key='instances' 라인은 반드시 삭제해주세요!")