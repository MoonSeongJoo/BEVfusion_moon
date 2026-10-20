
# ============================================================
# PCC EXP4 - Joint Fine-Tuning Evaluation
#
# Geometry-Selective Attention
# + Residual Feature Adapter
# + Transformer Decoder Tail
# + Detection Prediction Head
#
# Evaluation target:
#   Geometry Attention 20K
#   -> Joint Fine-Tuning 81K
#
# Metrics:
#   1. CalibLCCNetMetric
#   2. NuScenesMetric (mAP / NDS)
#
# All learned modules are enabled during inference.
# ============================================================


# ------------------------------------------------------------
# 1. Base Configuration
# ------------------------------------------------------------

# Inherit the actual EXP4 training model and
# dataset/evaluator configuration.
#
# This avoids differences in dataset transforms,
# calibration settings and model architecture.

_base_ = [
    './pcc_bevfusion_rrrf_exp4_joint_fresh_train.py'
]


# ------------------------------------------------------------
# 2. Model Evaluation Settings
# ------------------------------------------------------------

model = dict(

    # Full LGPC + RRRF + BEVFusion.
    calibration_mode='pcc_full',

    # Do not use legacy selective-freezing behavior.
    enable_selective_freezing=False,

    # Match the EXP4 Joint Training architecture.
    lgpc_train_stage='rrrf_joint_fresh',

    bbox_head=dict(

        # RRRF V2
        rrrf_mode='residual_se3_v2',

        rrrf_num_cams=6,

        rrrf_detach_geometry=True,

        query_source='fused',

        # --------------------------------------------
        # Residual Feature Adapter
        # --------------------------------------------

        rrrf_adapter_enabled=True,

        # Full ON
        rrrf_adapter_scale=1.0,

        # --------------------------------------------
        # Geometry-Selective Attention
        # --------------------------------------------

        rrrf_query_select_enabled=True,

        # Full ON
        rrrf_query_select_scale=1.0,

        # Geometry selection parameters
        rrrf_query_select_topk=16,

        rrrf_query_select_radius_m=4.0,

        rrrf_query_select_sigma_m=2.0,

    ),

)


# ------------------------------------------------------------
# 3. Evaluation Loop
# ------------------------------------------------------------

test_cfg = dict(
    _delete_=True,
    type='TestLoop',
)


# ------------------------------------------------------------
# 4. Evaluation Checkpoint
# ------------------------------------------------------------

# The checkpoint is supplied as the second
# argument to tools/test.py.
#
# Example:
# work_dirs/pcc_exp4_joint_fresh_400k/iter_81000.pth

load_from = None

resume = False


# ------------------------------------------------------------
# 5. Visualization
# ------------------------------------------------------------

default_hooks = dict(

    visualization=dict(
        _delete_=True,
        type='Det3DVisualizationHook',
        draw=False,
    ),

)


# ------------------------------------------------------------
# 6. Evaluation Summary
# ------------------------------------------------------------

# Geometry Attention     : ON (scale=1.0)
# Residual Adapter       : ON (scale=1.0)
# Transformer Decoder    : Pretrained EXP4 weights
# Detection Head         : Pretrained EXP4 weights
# LGPC Calibration       : Pretrained frozen weights
# RRRF V2 Calibration    : Pretrained frozen weights
#
# Evaluators and test_dataloader are inherited
# from the existing EXP4 configuration chain.
# ============================================================
