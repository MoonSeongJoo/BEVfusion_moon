import pickle

# 생성된 pkl 파일 경로 (train, val 중 하나로 확인)
# --extra-tag를 nuscenes_bevfusion으로 지정했으므로 파일 이름은 아래와 같을 것입니다.
pkl_path = './data/nuscenes/debug_infos_train.pkl'

print(f"Checking file: {pkl_path}")

try:
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # 첫 번째 데이터 샘플의 annotation 정보 확인
    first_sample_annotations = data['data_list'][0]['instances']

    if first_sample_annotations:
        print("첫 번째 annotation instance에 포함된 키:")
        print(first_sample_annotations[0].keys())

        if 'bbox' in first_sample_annotations[0]:
            print("\n성공! 2D 박스 키인 'bbox'를 찾았습니다. ✅")
        else:
            print("\n실패: 'bbox' 키를 찾지 못했습니다. 스크립트가 2D 박스를 생성하지 않았습니다.")
    else:
        print("첫 번째 샘플에 확인할 annotation이 없습니다.")

except FileNotFoundError:
    print(f"오류: 파일을 찾을 수 없습니다. 경로를 확인하세요: {pkl_path}")
except Exception as e:
    print(f"오류 발생: {e}")