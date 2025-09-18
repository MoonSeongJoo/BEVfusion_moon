import pickle
import random
from collections import defaultdict
from tqdm import tqdm

def create_mini_dataset_stratified(original_path, new_path, num_samples, class_names):
    """
    pkl 파일에서 각 클래스별로 최소 1개의 샘플을 보장하며,
    지정된 샘플 수만큼 미니 데이터셋을 생성합니다. (층화 샘플링)
    """
    print(f"'{original_path}' 파일을 로딩 중입니다...")
    with open(original_path, 'rb') as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise TypeError("pkl 파일이 딕셔너리 형태가 아닙니다.")

    info_list_key = 'data_list' if 'data_list' in data else 'infos'
    if info_list_key not in data:
        raise ValueError(f"파일에서 데이터 목록 키('data_list' 또는 'infos')를 찾을 수 없습니다.")
    
    original_data_list = data[info_list_key]
    
    # 1. 클래스별로 어떤 샘플에 포함되어 있는지 인덱스 맵핑
    print("데이터셋 스캔 중: 클래스-샘플 인덱스 매핑...")
    class_to_indices = defaultdict(list)
    for i, info in enumerate(tqdm(original_data_list)):
        # ann_info에 3D 라벨이 있는지 확인 (구조에 따라 키 변경 필요 가능성 있음)
        if 'ann_info' in info and 'gt_labels_3d' in info['ann_info']:
            # MMDetection은 라벨을 숫자로 저장하므로, 숫자-이름 매핑 필요
            # 여기서는 gt_names 또는 gt_labels_3d가 클래스 이름 문자열이라고 가정
            labels = info['ann_info']['gt_labels_3d']
            unique_labels = set(labels)
            for label in unique_labels:
                class_to_indices[label].append(i)

    # 2. 각 클래스별로 최소 1개의 샘플 인덱스 추출 (중복 없이)
    final_indices = set()
    print("최소 샘플 보장: 각 클래스에서 샘플 1개씩 추출...")
    for class_name in class_names:
        if class_name in class_to_indices and class_to_indices[class_name]:
            # 해당 클래스가 포함된 샘플 중 하나를 무작위로 선택
            chosen_index = random.choice(class_to_indices[class_name])
            final_indices.add(chosen_index)
        else:
            print(f"경고: 클래스 '{class_name}'가 데이터셋에 없어 샘플을 추출할 수 없습니다.")

    # 3. 목표 샘플 수에서 부족한 만큼 나머지 샘플에서 무작위 추출
    num_already_chosen = len(final_indices)
    num_remaining_needed = num_samples - num_already_chosen

    if num_remaining_needed > 0:
        print(f"나머지 샘플 추출: {num_remaining_needed}개 무작위 추출...")
        all_indices = set(range(len(original_data_list)))
        # 이미 선택된 인덱스들을 제외한 나머지 인덱스 풀
        pool_of_remaining_indices = list(all_indices - final_indices)
        
        # 만약 남은 풀이 필요한 샘플 수보다 적으면, 그냥 풀 전체를 사용
        if len(pool_of_remaining_indices) < num_remaining_needed:
            print(f"경고: 남은 샘플 수가 부족하여 {len(pool_of_remaining_indices)}개만 추가합니다.")
            random_indices = pool_of_remaining_indices
        else:
            random_indices = random.sample(pool_of_remaining_indices, num_remaining_needed)
        
        final_indices.update(random_indices)

    # 4. 최종 선택된 인덱스로 새로운 데이터 리스트 생성
    final_indices_list = sorted(list(final_indices))
    new_data_list = [original_data_list[i] for i in final_indices_list]
    
    # 5. 메타데이터와 함께 새로운 pkl 파일 저장
    new_data = {}
    for key, value in data.items():
        if key != info_list_key:
            new_data[key] = value
    new_data[info_list_key] = new_data_list

    with open(new_path, 'wb') as f:
        pickle.dump(new_data, f)
    
    print(f"'{new_path}' 파일 생성 완료! (총 샘플 {len(new_data_list)}개)\n")

# --- 설정 ---
# pkl 파일에 저장된 클래스 이름 목록 (설정 파일의 class_names와 동일해야 함)
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

# 원본 학습/검증 pkl 파일 경로
original_train_info_path = 'data/nuscenes/nuscenes_infos_train_with_2d.pkl'
original_val_info_path = 'data/nuscenes/nuscenes_infos_val_with_2d.pkl'

# 새로 만들 미니 pkl 파일 경로
debug_train_info_path = 'data/nuscenes/debug_infos_train_with_2d.pkl'
debug_val_info_path = 'data/nuscenes/debug_infos_val_with_2d.pkl'

# 디버깅에 사용할 샘플 수
num_debug_samples = 100

# --- 실행 ---
create_mini_dataset_stratified(original_train_info_path, debug_train_info_path, num_debug_samples, class_names)
create_mini_dataset_stratified(original_val_info_path, debug_val_info_path, num_debug_samples, class_names)