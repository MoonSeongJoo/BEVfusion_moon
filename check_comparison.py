import json

# --- 1. 본인의 로그에 나온 JSON 파일 경로로 수정하세요 ---
json_path = './work_dirs/bevfusion_nus_results/pred_instances_3d/results_nusc.json'

# --- 2. Ground Truth 토큰 로드 ---
try:
    with open('gt_tokens.txt', 'r') as f:
        gt_tokens = set(line.strip() for line in f)
    print(f"✅ 정답(GT) 토큰 로드 성공: {len(gt_tokens)}개")
except FileNotFoundError:
    print("❌ 에러: gt_tokens.txt 파일을 찾을 수 없습니다. 1단계가 정상적으로 완료되었는지 확인하세요.")
    exit()

# --- 3. 예측 결과 JSON 파일에서 토큰 추출 ---
pred_tokens = set()
try:
    with open(json_path, 'r') as f:
        results_data = json.load(f)
    
    # nuScenes 결과 JSON 형식에 따라 'results' 딕셔너리의 키가 sample_token 입니다.
    pred_tokens = set(results_data['results'].keys())
    print(f"✅ 예측(Prediction) 토큰 로드 성공: {len(pred_tokens)}개")
except FileNotFoundError:
    print(f"❌ 에러: '{json_path}' 파일을 찾을 수 없습니다. 경로를 다시 확인해주세요.")
    exit()
except (KeyError, json.JSONDecodeError):
    print(f"❌ 에러: JSON 파일을 분석할 수 없거나 'results' 키를 찾을 수 없습니다. 파일이 손상되었을 수 있습니다.")
    exit()

# --- 4. 두 토큰 목록 비교 및 결과 분석 ---
print("\n" + "="*20 + " 분석 결과 " + "="*20)
print(f"정답(GT) 토큰 개수: {len(gt_tokens)}")
print(f"예측(Pred) 토큰 개수:   {len(pred_tokens)}")

missing_in_pred = gt_tokens - pred_tokens
extra_in_pred = pred_tokens - gt_tokens

if missing_in_pred:
    print(f"\n[원인] 예측 파일에 누락된 토큰이 {len(missing_in_pred)}개 있습니다.")
    print("누락된 토큰 예시 (최대 5개):")
    for token in list(missing_in_pred)[:5]:
        print(f"  - {token}")

if extra_in_pred:
    print(f"\n[원인] 예측 파일에 정답셋에 없는 추가 토큰이 {len(extra_in_pred)}개 있습니다.")
    print("추가된 토큰 예시 (최대 5개):")
    for token in list(extra_in_pred)[:5]:
        print(f"  - {token}")

if not missing_in_pred and not extra_in_pred:
    if len(gt_tokens) > 0:
         print("\n[분석] 토큰 목록은 일치합니다. 예측 파일에 중복된 토큰이 있는지 확인해야 합니다.")
    else:
         print("\n[분석] 차이점을 찾지 못했습니다.")

print("="*53)

from mmdet3d.evaluation.metrics import NuScenesMetric