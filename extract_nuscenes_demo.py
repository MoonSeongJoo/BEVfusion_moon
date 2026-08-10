import os
import shutil
from typing import List, Tuple

import numpy as np
from PIL import Image

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, BoxVisibility


# ============================================================
# User configuration
# ============================================================

DATAROOT = "./data/nuscenes"
VERSION = "v1.0-trainval"

OUT_IMG = "./data/nuscenes/demo/demo_frames_src"
OUT_LAB = "./data/nuscenes/demo/demo_labels_src"

NUM_FRAMES = 100

# 현재 C 코드의 HASEE_MAX_OBJS 값과 맞추는 것을 추천
MAX_OBJECTS_PER_FRAME = 4

CAM_NAME = "CAM_FRONT"

# 너무 작은 bbox는 제거
MIN_BBOX_WIDTH = 8
MIN_BBOX_HEIGHT = 8

# nuScenes category -> demo class mapping
CATEGORY_MAP = {
    "vehicle.car": "Car",
    "vehicle.truck": "Car",
    "vehicle.bus.bendy": "Car",
    "vehicle.bus.rigid": "Car",
    "vehicle.construction": "Car",
    "vehicle.emergency.ambulance": "Car",
    "vehicle.emergency.police": "Car",
    "vehicle.trailer": "Car",

    "human.pedestrian.adult": "Pedestrian",
    "human.pedestrian.child": "Pedestrian",
    "human.pedestrian.construction_worker": "Pedestrian",
    "human.pedestrian.police_officer": "Pedestrian",

    "movable_object.trafficcone": "Cone",
}


# ============================================================
# Utility functions
# ============================================================

def map_category(nuscenes_name: str):
    """
    nuScenes category name을 데모용 class name으로 변환.
    매핑되지 않는 category는 None으로 버림.
    """
    return CATEGORY_MAP.get(nuscenes_name, None)


def project_box_to_2d(box, camera_intrinsic, image_w: int, image_h: int):
    """
    nuScenes 3D Box를 camera image plane에 projection해서
    2D bbox [x1, y1, x2, y2]로 변환한다.

    nusc.get_sample_data()가 반환하는 camera boxes는 이미 camera coordinate로
    변환된 상태이므로 box.corners()를 camera_intrinsic으로 projection하면 된다.
    """
    corners_3d = box.corners()  # shape: 3 x 8

    # camera 앞쪽에 있는 corner만 사용
    # z <= 0인 corner는 camera 뒤쪽이므로 projection에 부적합
    depths = corners_3d[2, :]
    valid = depths > 0.1

    if np.sum(valid) < 1:
        return None

    corners_3d_valid = corners_3d[:, valid]

    projected = view_points(
        corners_3d_valid,
        np.array(camera_intrinsic),
        normalize=True
    )

    xs = projected[0, :]
    ys = projected[1, :]

    x1 = float(np.min(xs))
    y1 = float(np.min(ys))
    x2 = float(np.max(xs))
    y2 = float(np.max(ys))

    # image boundary로 clipping
    x1 = max(0.0, min(x1, image_w - 1.0))
    y1 = max(0.0, min(y1, image_h - 1.0))
    x2 = max(0.0, min(x2, image_w - 1.0))
    y2 = max(0.0, min(y2, image_h - 1.0))

    bw = x2 - x1
    bh = y2 - y1

    if bw < MIN_BBOX_WIDTH or bh < MIN_BBOX_HEIGHT:
        return None

    return int(x1), int(y1), int(x2), int(y2)


def bbox_area(b):
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)


def clean_output_dirs():
    os.makedirs(OUT_IMG, exist_ok=True)
    os.makedirs(OUT_LAB, exist_ok=True)

    for d in [OUT_IMG, OUT_LAB]:
        for name in os.listdir(d):
            path = os.path.join(d, name)
            if os.path.isfile(path):
                os.remove(path)


# ============================================================
# Main extraction
# ============================================================

def main():
    print("[INFO] DATAROOT:", DATAROOT)
    print("[INFO] VERSION :", VERSION)
    print("[INFO] OUT_IMG :", OUT_IMG)
    print("[INFO] OUT_LAB :", OUT_LAB)
    print("[INFO] NUM_FRAMES:", NUM_FRAMES)

    clean_output_dirs()

    nusc = NuScenes(
        version=VERSION,
        dataroot=DATAROOT,
        verbose=True
    )

    picked_count = 0
    scanned_count = 0

    manifest_path = os.path.join(os.path.dirname(OUT_IMG), "demo_manifest.csv")

    with open(manifest_path, "w") as manifest:
        manifest.write(
            "demo_idx,source_sample_token,source_sample_data_token,"
            "source_image,output_image,output_label,num_boxes\n"
        )

        for sample in nusc.sample:
            if picked_count >= NUM_FRAMES:
                break

            scanned_count += 1

            cam_token = sample["data"].get(CAM_NAME)
            if cam_token is None:
                continue

            sd = nusc.get("sample_data", cam_token)
            img_path = os.path.join(DATAROOT, sd["filename"])

            if not os.path.exists(img_path):
                continue

            try:
                # boxes는 camera coordinate로 변환된 3D box 리스트
                # camera_intrinsic은 projection에 사용
                data_path, boxes, camera_intrinsic = nusc.get_sample_data(
                    cam_token,
                    box_vis_level=BoxVisibility.ANY
                )
            except Exception as e:
                print("[WARN] get_sample_data failed:", e)
                continue

            with Image.open(img_path) as im:
                image_w, image_h = im.size

            objects: List[Tuple[str, Tuple[int, int, int, int], float]] = []

            for box in boxes:
                demo_cls = map_category(box.name)
                if demo_cls is None:
                    continue

                bbox2d = project_box_to_2d(
                    box,
                    camera_intrinsic,
                    image_w,
                    image_h
                )

                if bbox2d is None:
                    continue

                # GT label이므로 confidence는 1.00으로 둔다.
                conf = 1.00
                objects.append((demo_cls, bbox2d, conf))

            if len(objects) == 0:
                continue

            # 큰 bbox 우선으로 정렬해서 데모 화면에서 잘 보이게 함
            objects.sort(key=lambda x: bbox_area(x[1]), reverse=True)

            # 현재 VM2 C packet 구조가 최대 4 object이므로 4개만 저장
            objects = objects[:MAX_OBJECTS_PER_FRAME]

            dst_img = os.path.join(OUT_IMG, f"frame_{picked_count:04d}.jpg")
            dst_lab = os.path.join(OUT_LAB, f"frame_{picked_count:04d}.txt")

            shutil.copy(img_path, dst_img)

            with open(dst_lab, "w") as f:
                for demo_cls, bbox, conf in objects:
                    x1, y1, x2, y2 = bbox
                    f.write(
                        f"{demo_cls} {x1} {y1} {x2} {y2} {conf:.2f}\n"
                    )

            manifest.write(
                f"{picked_count},"
                f"{sample['token']},"
                f"{cam_token},"
                f"{sd['filename']},"
                f"{dst_img},"
                f"{dst_lab},"
                f"{len(objects)}\n"
            )

            print(f"[{picked_count:04d}]")
            print(" source image:", img_path)
            print(" output image:", dst_img)
            print(" output label:", dst_lab)
            print(" boxes:", len(objects))
            for demo_cls, bbox, conf in objects:
                print("  ", demo_cls, bbox, conf)

            picked_count += 1

    print("")
    print("[DONE]")
    print("scanned samples:", scanned_count)
    print("picked frames  :", picked_count)
    print("manifest       :", manifest_path)

    if picked_count < NUM_FRAMES:
        print("[WARN] Requested", NUM_FRAMES, "frames but only extracted", picked_count)


if __name__ == "__main__":
    main()