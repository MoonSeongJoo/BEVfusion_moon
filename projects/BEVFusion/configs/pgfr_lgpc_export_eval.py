# ==========================================================
# PGFR-Det LGPC Stage-1 Calibration Export
#
# - Load existing PGFR checkpoint
# - Execute LGPC Stage-1
# - Bypass RRRF
# - Export corrected camera extrinsics
# ==========================================================

_base_ = [
    "./pcc_bevfusion_rrrf_exp4_joint_fresh_eval.py"
]

# ----------------------------------------------------------
# Register custom metric and PCC modules
# ----------------------------------------------------------

custom_imports = dict(
    imports=[
        "projects.BEVFusion.bevfusion.my_collate",
        "projects.BEVFusion.bevfusion.cotr",
        "projects.BEVFusion.bevfusion.my_hooks",
        "projects.BEVFusion.bevfusion.lgpc_manifest_metric",
    ],
    allow_failed_imports=False,
)

# ----------------------------------------------------------
# Model
# ----------------------------------------------------------

model = dict(
    calibration_mode="lgpc_export",
    enable_selective_freezing=False,
)

# ----------------------------------------------------------
# Evaluation metric: calibration export only
# ----------------------------------------------------------

test_evaluator = dict(
    _delete_=True,
    type="LGPCManifestMetric",
    output_path="work_dirs/lgpc_official/lgpc_manifest.json",
)

# ----------------------------------------------------------
# Evaluation dataloader
# ----------------------------------------------------------

# ==========================================================
# LGPC Calibration Export - Evaluation Dataloader
#
# IMPORTANT:
# Do not randomly replace evaluation frames.
# Preserve ann_info required by existing PCC pipeline.
# ==========================================================

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,

    sampler=dict(
        type="DefaultSampler",
        shuffle=False,
    ),

    dataset=dict(
        test_mode=False,
        filter_empty_gt=False,
    ),
)

# ----------------------------------------------------------
# Test loop
# ----------------------------------------------------------

test_cfg = dict(
    _delete_=True,
    type="TestLoop",
)

load_from = None
resume = False