# ==========================================================
# LGPC -> Official BEVFusion
#
# Official BEVFusion checkpoint
# + Offline LGPC predicted extrinsics
#
# NO RRRF
# NO fine-tuning
# ==========================================================

_base_ = [
    "./bevfusion_v110_reproduce.py"
]

# ----------------------------------------------------------
# Official model
# ----------------------------------------------------------

model = dict(

    calibration_mode="lgpc_manifest",

    lgpc_manifest_path=(
        "work_dirs/lgpc_official/lgpc_manifest.json"
    ),

)

# ----------------------------------------------------------
# Evaluation dataloader
# ----------------------------------------------------------

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
)

# ----------------------------------------------------------
# NuScenes Metric
# ----------------------------------------------------------

test_evaluator = dict(

    type="NuScenesMetric",

    data_root="data/nuscenes/",

    ann_file=(
        "data/nuscenes/v110_infos/"
        "nuscenes_infos_val.pkl"
    ),

    metric="bbox",

    version="v1.0-trainval",

    collect_dir=(
        "work_dirs/lgpc_official/"
        "nuscenes_lgpc_collect"
    ),
)

# ----------------------------------------------------------
# Runtime
# ----------------------------------------------------------

test_cfg = dict(
    _delete_=True,
    type="TestLoop",
)

load_from = None
resume = False