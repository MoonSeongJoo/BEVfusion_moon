import mmengine
import numpy as np
from tqdm import tqdm

# --- 설정 부분 ---
# 학습용과 검증용 .pkl 파일의 경로를 정확하게 입력해주세요.
TRAIN_PKL_PATH = 'data/nuscenes/nuscenes_infos_train_new_with_2d.pkl'
VAL_PKL_PATH = 'data/nuscenes/nuscenes_infos_val_new_with_2d.pkl'
# ------------------

def find_common_sample_discrepancies(train_path, val_path):
    """
    train과 val .pkl 파일을 로드하여, 공통된 sample_token을 찾고,
    그 샘플들의 'ego2global' 행렬이 다른지 비교합니다.
    """
    print(f"\n{'='*60}")
    print("--- 데이터 불일치 검증 스크립트 시작 ---")
    print(f"학습 파일: {train_path}")
    print(f"검증 파일: {val_path}")
    print(f"{'='*60}")

    try:
        # 1. 두 .pkl 파일을 모두 로드합니다.
        print("\n[단계 1] 데이터 파일을 로드하는 중... (시간이 걸릴 수 있습니다)")
        train_data = mmengine.load(train_path)
        val_data = mmengine.load(val_path)

        train_infos = train_data.get('data_list', train_data if isinstance(train_data, list) else [])
        val_infos = val_data.get('data_list', val_data if isinstance(val_data, list) else [])

        if not train_infos or not val_infos:
            print("\n[오류] 하나 또는 두 파일 모두 비어있습니다. 스크립트를 종료합니다.")
            return

        # 2. 빠른 비교를 위해 검증 데이터를 딕셔너리로 변환합니다 (key: token, value: sample info).
        print("[단계 2] 빠른 조회를 위해 검증 데이터 맵 생성 중...")
        val_lookup = {sample['token']: sample for sample in val_infos}
        print(f"  - 학습 데이터 샘플 수: {len(train_infos)}")
        print(f"  - 검증 데이터 샘플 수: {len(val_infos)}")

        # 3. 공통 토큰을 찾고, 행렬을 비교합니다.
        print("[단계 3] 학습 데이터와 검증 데이터 간 공통 샘플을 찾고 행렬을 비교합니다...")
        discrepancy_count = 0
        common_tokens_found = 0

        for train_sample in tqdm(train_infos, desc="비교 진행 중"):
            token = train_sample.get('token')
            if token in val_lookup:
                common_tokens_found += 1
                val_sample = val_lookup[token]

                try:
                    train_matrix = np.array(train_sample['ego2global'])
                    val_matrix = np.array(val_sample['ego2global'])

                    # 4. 두 행렬이 다른지 확인합니다.
                    if not np.allclose(train_matrix, val_matrix, atol=1e-6):
                        discrepancy_count += 1
                        if discrepancy_count == 1: # 첫 번째 불일치만 상세히 출력
                            print(f"\n\n[!!!] 결정적 불일치 발견! Token: {token}")
                            with np.printoptions(precision=6, suppress=True):
                                print("--- 학습(.pkl) 파일의 'ego2global' 행렬 ---")
                                print(train_matrix)
                                print("--- 검증(.pkl) 파일의 'ego2global' 행렬 ---")
                                print(val_matrix)
                            print("\n" + "-"*40)
                except KeyError:
                    print(f"\n[오류] Token {token}의 'ego2global' 키가 파일 중 하나에 없습니다.")
                    discrepancy_count += 1
        
        # 5. 최종 결과를 보고합니다.
        print(f"\n{'='*60}")
        print("--- 최종 검증 결과 ---")
        if common_tokens_found == 0:
            print("[결론] 두 파일 간에 공통된 샘플이 없습니다.")
            print("       이는 데이터가 정상적으로 분할되었다는 의미일 수 있습니다.")
            print("       하지만 mAP=0 현상과 종합해 볼 때, 'val' 데이터 생성 로직 자체의 오류일 가능성은 여전히 높습니다.")
        elif discrepancy_count > 0:
            print(f"[결론] 총 {common_tokens_found}개의 공통 샘플 중 {discrepancy_count}개에서 'ego2global' 행렬 불일치가 발견되었습니다.")
            print("\n      이것은 데이터 전처리 스크립트가 학습 데이터와 검증 데이터를 다르게 처리한다는 '결정적인 증거'입니다.")
        else:
            print(f"[결론] 총 {common_tokens_found}개의 공통 샘플 모두 'ego2global' 행렬이 일치합니다.")
            print("       데이터 분할 방식에 문제가 있을 수 있으나, 행렬 생성 로직 자체는 동일한 것으로 보입니다.")
        print(f"{'='*60}")

    except Exception as e:
        print(f"\n[치명적 오류] 스크립트 실행 중 예외가 발생했습니다: {e}")

if __name__ == '__main__':
    find_common_sample_discrepancies(TRAIN_PKL_PATH, VAL_PKL_PATH)