
# ============================================================
# LGPC-B 246k + RRRF V2 adaptation
#
# Frozen:
#   - Camera backbone / detector
#   - LGPC CorrNet
#   - LGPC ZEstimator
#   - LGPC CalibHead including Candidate-B
#   - LiDAR / BEVFusion backbone
#   - Main detection head
#
# Train:
#   - RRRF feature projector
#   - Coarse fusion
#   - RRRF V2 residual SE(3)
#   - Refined fusion
# ============================================================

_base_ = [
    './pcc_bevfusion_rrrf_feature_refine_full.py'
]

model = dict(

    # Actual LGPC correction is physically applied,
    # then RRRF + BEVFusion continue.
    calibration_mode='pcc_full',

    # RRRF V2 adaptation only.
    lgpc_train_stage='rrrf_v2',

    # Dedicated RRRF freeze policy is used.
    enable_selective_freezing=False,

    # --------------------------------------------------------
    # Keep Candidate-B LGPC active but FROZEN.
    # --------------------------------------------------------
    calib_head=dict(
        trans_refine_mode='geo_feature_residual',
    ),

    # --------------------------------------------------------
    # RRRF V2
    # --------------------------------------------------------
    bbox_head=dict(
        rrrf_mode='residual_se3_v2',
        rrrf_num_cams=6,
        rrrf_detach_geometry=True,
        query_source='fused',
    ),
)


# ============================================================
# Pilot training
# ============================================================

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=5000,
    val_interval=999999,
)


# Small constant LR for adaptation.
optim_wrapper = dict(
    optimizer=dict(
        lr=1.0e-5,
    ),
)


# Existing weights are already mature.
# Keep LR constant during this short pilot.
param_scheduler = []


default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=1000,
        max_keep_ckpts=10,
    ),
)

