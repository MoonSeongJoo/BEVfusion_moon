import pickle

# -------------------------------------------------------------------
# 👇👇👇 확인하고 싶은 dbinfos.pkl 파일 경로를 입력하세요 👇👇👇
# -------------------------------------------------------------------
db_path = 'data/nuscenes/new_infos_retry_dbinfos_train.pkl'
# -------------------------------------------------------------------

print(f"\n--- 🗂️ '{db_path}' 파일 내용물을 검사합니다 ---\n")

try:
    with open(db_path, 'rb') as f:
        db_infos = pickle.load(f)
except FileNotFoundError:
    print(f"🚨 파일을 찾을 수 없습니다: {db_path}")
    exit()

if not db_infos:
    print("❌ 결과: 데이터베이스 파일이 완전히 비어있습니다!")
    total_objects = 0
else:
    print("✅ 데이터베이스에 저장된 객체(스티커) 수:")
    total_objects = 0
    # 클래스 이름 순서대로 정렬해서 보기 좋게 출력
    for cat in sorted(db_infos.keys()):
        instances = db_infos[cat]
        count = len(instances)
        print(f"  - {cat:<15}: {count} 개")
        total_objects += count
    print(f"\n총 {total_objects}개의 객체가 데이터베이스에 저장되어 있습니다.")

print("\n" + "="*50)
if total_objects == 0:
    print("🚨 진단: GT 데이터베이스가 비어있습니다.")
    print("   '스티커 북'이 비어있어 최종 pkl 파일에 GT를 추가할 수 없었습니다.")
    print("\n   [다음 단계] 데이터 생성 스크립트(create_data.py) 실행 시")
    print("   1. 원본 nuScenes 데이터를 읽는 경로가 올바른지 확인하세요.")
    print("   2. 스크립트 내부의 필터링 조건(예: min_points_in_gt)이 너무 엄격하지 않은지 확인하세요.")
else:
    print("🎉 진단: GT 데이터베이스는 정상적으로 생성되었습니다.")
    print("   만약 그래도 문제가 발생한다면, 데이터 증강(Augmentation) 파이프라인 설정 자체의 문제일 수 있습니다.")
print("="*50)