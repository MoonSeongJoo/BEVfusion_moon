# ============================================================
# PCC EXP1 V2 - Residual Feature Adapter
#
# Base checkpoint:
#   LGPC_RRRF_2nd_nds70_44_ORIGINAL.pth
#
# Assumption:
#   ORIGINAL checkpoint contains trained RRRF V2 weights.
#
# Frozen:
#   LGPC Stage1
#   RRRF V2 Per-Camera Predictor
#   Coarse / Refined Fusion
#   3D Decoder
#
# Train:
#   bbox_head.rrrf_adapter ONLY
# ============================================================

_base_ = [
    './pcc_bevfusion_rrrf_exp1_adapter_train.py'
]


# ============================================================
# Model
# ============================================================

model = dict(

    enable_selective_freezing=False,

    calibration_mode='pcc_full',

    # IMPORTANT:
    # Do not use rrrf_v2 here.
    # Only Adapter must be trainable.
    lgpc_train_stage='rrrf_adapter',

    # Candidate-B setting used by the historical V2 config.
    calib_head=dict(
        trans_refine_mode='geo_feature_residual',
        init_cfg=None,
    ),

    bbox_head=dict(

        # V2: per-camera Stage2
        rrrf_mode='residual_se3_v2',

        rrrf_num_cams=6,

        # Detection gradients do not modify
        # Stage2 geometry predictor.
        rrrf_detach_geometry=True,

        query_source='fused',

        # Enable new residual adapter
        rrrf_adapter_enabled=True,

        rrrf_adapter_scale=1.0,
    ),

)


# ============================================================
# Checkpoint
# ============================================================

load_from = (
    'data/weights/'
    'LGPC_RRRF_2nd_nds70_44_ORIGINAL.pth'
)

# Start a new optimizer.
resume = False


# ============================================================
# Training remains inherited:
#
# 3000 iterations
# AdamW
# LR = 1e-4
# WD = 0.01
# Gradient clipping = 10
# Save every 500 iterations
# ============================================================