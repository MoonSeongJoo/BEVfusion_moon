# ============================================================
# LGPC -> Official BEVFusion
# Prediction Debug Evaluation
#
# No NuScenes mAP / NDS computation.
# Only predicted 3D boxes and confidence statistics.
# ============================================================

_base_ = [
    "./bevfusion_v110_lgpc_manifest_eval.py"
]

# Register Official BEVFusion and debug metric.
# Do NOT import PCC BEVFusion in this config.

custom_imports = dict(
    imports=[
        "projects.OfficialBEVFusion.bevfusion",
        "projects.OfficialBEVFusion.bevfusion.prediction_debug_metric",
    ],
    allow_failed_imports=False,
)

model = dict(
    calibration_mode="lgpc_manifest",
    lgpc_manifest_path=(
        "work_dirs/lgpc_official/lgpc_manifest_100.json"
    ),
)

test_evaluator = dict(
    _delete_=True,
    type="NuScenesPredictionDebugMetric",
)

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
)

load_from = None
resume = False