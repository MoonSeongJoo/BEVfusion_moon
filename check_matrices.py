import mmengine
import numpy as np

# -------------------------------------------------------------------
# 1. 여기에 검증하고 싶은 .pkl 파일의 정확한 경로를 입력하세요.
# 예시: 'data/nuscenes/debug_infos_val_with_2d.pkl'
#      'data/nuscenes/debug_infos_train_with_2d.pkl'
# PKL_FILE_PATH = 'data/nuscenes/debug_infos_val_with_2d.pkl'
# PKL_FILE_PATH = 'data/nuscenes/debug_infos_train_with_2d.pkl'
# PKL_FILE_PATH = 'data/nuscenes/nuscenes_infos_val_new_with_2d.pkl'
PKL_FILE_PATH = 'data/nuscenes/nuscenes_infos_train_new_with_2d.pkl'
# -------------------------------------------------------------------

def verify_matrices(file_path):
    """지정된 .pkl 파일의 변환 행렬을 로드하고 검증합니다."""
    
    print(f"\n{'='*50}")
    print(f"--- Verifying file: {file_path} ---")
    print(f"{'='*50}")

    try:
        # 2. mmengine을 사용하여 .pkl 파일 로드
        data = mmengine.load(file_path)

        # 3. 'data_list' 키에서 실제 데이터 정보 리스트를 가져옵니다.
        if isinstance(data, dict) and 'data_list' in data:
            data_infos = data['data_list']
        elif isinstance(data, list):
            data_infos = data
        else:
            raise ValueError("Could not find 'data_list' key or a list in the .pkl file.")

        print(f"[INFO] Successfully loaded {len(data_infos)} data samples.")

        # 4. 검증을 위해 첫 번째 샘플(index 0)을 선택합니다.
        if not data_infos:
            print("[ERROR] The data list is empty. Cannot verify.")
            return

        first_sample_info = data_infos[0]
        sample_token = first_sample_info.get('token', 'N/A')
        print(f"[INFO] Inspecting first sample with token: {sample_token}\n")

        # 5. lidar2ego 행렬 추출 및 검증
        try:
            lidar2ego_mat = np.array(first_sample_info['lidar_points']['lidar2ego'])
            print(f"[OK] Found 'lidar2ego' matrix (shape: {lidar2ego_mat.shape}):")
            print(lidar2ego_mat)
            if lidar2ego_mat.shape != (4, 4):
                print("\n[CRITICAL WARNING] 'lidar2ego' is NOT a 4x4 matrix!")
        except KeyError:
            print("\n[ERROR] Could not find the key path: 'lidar_points' -> 'lidar2ego'.")

        # 6. ego2global 행렬 추출 및 검증
        try:
            ego2global_mat = np.array(first_sample_info['ego2global'])
            print(f"\n[OK] Found 'ego2global' matrix (shape: {ego2global_mat.shape}):")
            print(ego2global_mat)
            if ego2global_mat.shape != (4, 4):
                print("\n[CRITICAL WARNING] 'ego2global' is NOT a 4x4 matrix!")
        except KeyError:
            print("\n[ERROR] Could not find the key: 'ego2global'.")

    except FileNotFoundError:
        print(f"\n[FATAL ERROR] File not found at: {file_path}")
    except Exception as e:
        print(f"\n[FATAL ERROR] An unexpected error occurred: {e}")

if __name__ == '__main__':
    verify_matrices(PKL_FILE_PATH)