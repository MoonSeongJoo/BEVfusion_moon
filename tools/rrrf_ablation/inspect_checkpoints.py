
import torch

PGFR = (
    "/workspace/mmdetection3d/data/weights/"
    "PGFR_Det_final_model.pth"
)

OFFICIAL = (
    "/workspace/mmdetection3d/data/weights/"
    "bevfusion_lidar-cam_voxel0075_second_secfpn_"
    "8xb4-cyclic-20e_nus-3d-5239b1af.pth"
)


def load_sd(path):
    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    sd = ckpt.get("state_dict", ckpt)
    return {
        k.removeprefix("module."): v
        for k, v in sd.items()
    }


pgfr = load_sd(PGFR)
official = load_sd(OFFICIAL)

prefixes = (
    "bbox_head.shared_conv.",
    "bbox_head.heatmap_head.",
    "bbox_head.class_encoding.",
    "bbox_head.decoder.",
    "bbox_head.prediction_heads.",
    "corr.",
    "z_estimator.",
    "calib_head.",
    "feat_projector.",
    "bbox_head.fusion_cross_attention.",
    "bbox_head.refined_attention.",
    "bbox_head.rrrf_query_selector.",
)

print("\n========== MODULE CHECK ==========")

for prefix in prefixes:
    pgfr_keys = [
        k for k in pgfr if k.startswith(prefix)
    ]
    off_keys = [
        k for k in official if k.startswith(prefix)
    ]
    print(
        f"{prefix:45s} "
        f"PGFR={len(pgfr_keys):4d} "
        f"Official={len(off_keys):4d}"
    )

print("\n====== OFFICIAL DECODER CHECK ======")

for prefix in (
    "bbox_head.decoder.",
    "bbox_head.prediction_heads.",
):
    keys = [k for k in pgfr if k.startswith(prefix)]

    missing = [k for k in keys if k not in official]
    mismatch = [
        k for k in keys
        if k in official
        and pgfr[k].shape != official[k].shape
    ]

    print(f"\n{prefix}")
    print("PGFR key count:", len(keys))
    print("Missing official keys:", len(missing))
    print("Shape mismatches:", len(mismatch))

    for k in missing[:10]:
        print("  MISSING:", k)

    for k in mismatch[:10]:
        print(
            "  MISMATCH:",
            k,
            tuple(pgfr[k].shape),
            tuple(official[k].shape),
        )
