import argparse
import json
import os
from collections import Counter

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet3d.registry import DATASETS


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=100)

    args = parser.parse_args()

    # Only ONE project's config is imported
    # in this Python process.
    cfg = Config.fromfile(args.config)

    init_default_scope(
        cfg.get("default_scope", "mmdet3d")
    )

    dataset = DATASETS.build(
        cfg.test_dataloader.dataset
    )

    count = min(args.count, len(dataset))
    frames = []

    for i in range(count):

        info = dataset.get_data_info(i)

        lidar_path = info.get("lidar_path")

        if lidar_path is None:
            lidar_path = info.get(
                "lidar_points", {}
            ).get("lidar_path")

        if lidar_path is None:
            raise RuntimeError(
                f"Missing lidar_path at index {i}"
            )

        lidar_path = str(lidar_path)

        frame_key = os.path.basename(
            lidar_path.replace("\\", "/")
        )

        frames.append({
            "index": i,
            "sample_idx": str(
                info.get("sample_idx")
            ),
            "token": str(
                info.get("token")
            ),
            "lidar_path": lidar_path,
            "frame_key": frame_key,
        })

    frame_keys = [
        f["frame_key"] for f in frames
    ]

    duplicates = [
        key
        for key, n in Counter(frame_keys).items()
        if n > 1
    ]

    if duplicates:
        raise RuntimeError(
            f"Duplicate LiDAR frames: {duplicates[:10]}"
        )

    os.makedirs(
        os.path.dirname(
            os.path.abspath(args.output)
        ),
        exist_ok=True,
    )

    with open(args.output, "w") as f:
        json.dump({
            "dataset_length": len(dataset),
            "num_frames": len(frames),
            "frames": frames,
        }, f, indent=2)

    print("=================================")
    print("Config:", args.config)
    print("Dataset length:", len(dataset))
    print("Exported frames:", len(frames))
    print("Unique LiDAR frames:", len(set(frame_keys)))
    print("Output:", args.output)
    print("=================================")


if __name__ == "__main__":
    main()