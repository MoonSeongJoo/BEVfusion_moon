import argparse
import os
import cv2
import numpy as np
from mmengine.config import Config
from mmdet3d.registry import DATASETS, TRANSFORMS
from mmengine.registry import init_default_scope

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize 2D Ground Truth boxes by manually running the loader.'
    )
    parser.add_argument('config', help='Path to the model config file.')
    parser.add_argument(
        '--count', type=int, default=5, help='Number of samples to visualize.'
    )
    parser.add_argument(
        '--output-dir',
        default='./annotation_debug_output',
        help='Directory to save the visualization images.'
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output images will be saved to: {args.output_dir}")

    # 데이터셋 설정을 가져옵니다.
    if 'dataset' in cfg.train_dataloader.dataset:
        dataset_cfg = cfg.train_dataloader.dataset.dataset
    else:
        dataset_cfg = cfg.train_dataloader.dataset

    # 데이터 정보에 접근하기 위해 데이터셋 객체를 빌드합니다.
    dataset = DATASETS.build(dataset_cfg)
    print(f"Loaded dataset with {len(dataset)} samples.")

    # ========================= [ 핵심 수정 부분 ] =========================
    # 파이프라인에서 이미지 로더만 수동으로 생성합니다.
    # 이렇게 하면 dataset[i]를 호출할 필요 없이 필요한 데이터만 로드할 수 있습니다.
    image_loader_pipeline = TRANSFORMS.build(
        dict(type='BEVLoadMultiViewImageFromFiles', to_float32=True, color_type='color')
    )
    # ====================================================================

    for i in range(min(args.count, len(dataset))):
        print(f"\nProcessing sample index: {i}")
        try:
            # 1. 파이프라인을 실행하지 않고 원본 데이터 정보를 가져옵니다.
            data_info = dataset.get_data_info(i)

            # 2. 이미지 로더만 수동으로 실행하여 이미지 데이터를 가져옵니다.
            #    data_info의 복사본을 넘겨 원본이 변경되지 않도록 합니다.
            processed_data = image_loader_pipeline(data_info.copy())
            
            # 3. 시각화에 필요한 데이터를 추출합니다.
            images = processed_data['img']                  # 로드된 이미지
            ann_info_2d = data_info.get('ann_info_2d_per_cam', {})  # 원본 어노테이션

            camera_types = [
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ]
            
            for cam_idx, cam_name in enumerate(camera_types):
                if cam_idx >= len(images): continue

                img = images[cam_idx].transpose(1, 2, 0).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
                
                cam_anns = ann_info_2d.get(cam_name, {})
                gt_bboxes = cam_anns.get('gt_bboxes', np.array([]))
                gt_labels = cam_anns.get('gt_labels', [])

                for box, label_name in zip(gt_bboxes, gt_labels):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_text = f'{label_name}'
                    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img, (x1, y1 - text_h - 4), (x1 + text_w, y1), (0, 255, 0), -1)
                    cv2.putText(img, label_text, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                output_path = os.path.join(args.output_dir, f'sample_{i:03d}_{cam_name}.png')
                cv2.imwrite(output_path, img)

        except Exception as e:
            print(f"An unexpected error occurred while processing sample {i}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n✅ Visualization finished. Check the images in '{args.output_dir}'.")

if __name__ == '__main__':
    main()