# merge_coco_to_pkl.py (ann_info 구조 생성 최종 버전)
import pickle
import json
import argparse
from tqdm import tqdm
import numpy as np
from nuscenes.nuscenes import NuScenes

def parse_args():
    parser = argparse.ArgumentParser(description='Merge COCO format 2D annotations into a nuScenes pkl file.')
    parser.add_argument('pkl_path', help='Path to the input nuScenes pkl file (e.g., nuscenes_infos_train.pkl)')
    parser.add_argument('json_path', help='Path to the COCO format json file (e.g., nuscenes_infos_train_mono3d.coco.json)')
    parser.add_argument('out_path', help='Path to save the new pkl file')
    parser.add_argument('--dataroot', default='./data/nuscenes', help='Root path of the nuScenes dataset for the SDK')
    args = parser.parse_args()
    return args

# merge_coco_to_pkl.py 파일의 main 함수를 교체하세요.

def main():
    args = parse_args()

    print(f"Loading pkl file from {args.pkl_path}...")
    with open(args.pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loading json file from {args.json_path}...")
    with open(args.json_path, 'r') as f:
        coco_data = json.load(f)

    print("Loading NuScenes SDK...")
    nusc = NuScenes(version='v1.0-trainval', dataroot=args.dataroot, verbose=False)

    print("Creating annotation map from COCO json...")
    img_id_to_anns = {}
    for ann in tqdm(coco_data['annotations']):
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)

    print("Merging multi-view 2D annotations into a separate key...")
    info_list_key = 'data_list' if 'data_list' in data else 'infos'
    
    # ✨ 1. 모든 카메라 타입을 리스트로 정의
    camera_types = [
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ]
    
    for info in tqdm(data[info_list_key]):
        sample_token = info['token']
        sample_rec = nusc.get('sample', sample_token)
        
        # ✨ 2. 현재 샘플의 모든 카메라 뷰에 대한 2D GT를 저장할 딕셔너리 생성
        all_cam_2d_anns = {}

        # ✨ 3. 모든 카메라 타입을 순회하며 2D GT 추출
        for cam_name in camera_types:
            cam_token = sample_rec['data'][cam_name]
            
            gt_bboxes_2d = []
            gt_labels_2d = []

            if cam_token in img_id_to_anns:
                for ann in img_id_to_anns[cam_token]:
                    x, y, w, h = ann['bbox']
                    bbox = [x, y, x + w, y + h]
                    gt_bboxes_2d.append(bbox)
                    gt_labels_2d.append(ann['category_name'])
            
            # ✨ 4. 현재 카메라 이름(cam_name)을 키로 하여 2D GT 저장
            all_cam_2d_anns[cam_name] = {
                'gt_bboxes': np.array(gt_bboxes_2d, dtype=np.float32),
                'gt_labels': np.array(gt_labels_2d)
            }
        
        # ✨ 5. 6개 뷰 전체의 2D GT가 담긴 딕셔너리를 새로운 최상위 키에 저장
        info['ann_info_2d_per_cam'] = all_cam_2d_anns

    print(f"Saving new pkl file to {args.out_path}...")
    with open(args.out_path, 'wb') as f:
        pickle.dump(data, f)
    
    print("Done! New pkl file created with multi-view 2D annotations.")

if __name__ == '__main__':
    main()