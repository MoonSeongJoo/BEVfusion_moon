from nuscenes.nuscenes import NuScenes
import os

# 1. nuScenes 객체 초기화
nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes', verbose=True)

# 2. 파일명에서 추출한 타임스탬프 설정
target_timestamp = 1533151275012404

# 3. 해당 타임스탬프와 채널(CAM_FRONT)에 맞는 sample_data 레코드 찾기
try:
    sd_record = [sd for sd in nusc.sample_data if sd['timestamp'] == target_timestamp and sd['channel'] == 'CAM_FRONT'][0]
    sd_token = sd_record['token']

    # --- 저장 경로 설정 ---
    output_dir = './nuscenes_visualized'
    os.makedirs(output_dir, exist_ok=True)

    # 4. 이미지 전용 (Original Image Only) 추출
    # with_anns=False 옵션을 사용하여 Bbox를 제거합니다.
    nusc.render_sample_data(sd_token, 
                            with_anns=False, 
                            out_path=os.path.join(output_dir, f'{target_timestamp}_original.png'))

    # 5. 3D Bbox 오버랩 (GT Overlap) 추출
    # with_anns=True (기본값) 옵션을 사용하여 3D Bbox를 그립니다.
    nusc.render_sample_data(sd_token, 
                            with_anns=True, 
                            out_path=os.path.join(output_dir, f'{target_timestamp}_with_bbox.png'))

    print(f"작업 완료: {output_dir} 폴더에 두 종류의 이미지가 저장되었습니다.")

except IndexError:
    print(f"에러: 타임스탬프 {target_timestamp}에 해당하는 데이터를 찾을 수 없습니다.")