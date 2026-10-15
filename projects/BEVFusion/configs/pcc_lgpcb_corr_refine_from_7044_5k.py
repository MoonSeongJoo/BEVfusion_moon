# ============================================================
# LocalCorrRefiner Pilot
#
# Base weights:
#   LGPC_RRRF_2nd_nds70_44.pth
#
# TRAIN:
#   ONLY corr.local_refiner
#
# FROZEN:
#   Base CorrNet
#   ZEstimator
#   CalibHead
#   Candidate-B
#   RRRF
#   BEVFusion
#   2D Detector
# ============================================================

_base_ = [
    './pcc_bevfusion_rrrf_feature_refine_full.py'
]


model = dict(

    # Stage-1 standalone training behavior
    calibration_mode='pcc_calib_only',

    # ========================================================
    # IMPORTANT
    # ========================================================
    lgpc_train_stage='corr_refine',

    enable_selective_freezing=False,

    # --------------------------------------------------------
    # Local correspondence refinement ON
    # --------------------------------------------------------
    corr=dict(

        enable_local_refine=True,

        local_refine_patch_embed_dim=32,

        local_refine_patch_grid_size=3,

        local_refine_patch_radius_l2_px=24.0,

        local_refine_patch_radius_high_px=48.0,

        # Existing
        local_refine_candidate_loss_weight=0.01,

        # ====================================================
        # NEW
        # ====================================================
        local_refine_coarse_loss_weight=0.2,
    ),

    # Keep current Candidate-B definition.
    # It is frozen during corr_refine training.
    calib_head=dict(
        trans_refine_mode='geo_feature_residual',
    ),

    # ========================================================
    # CRITICAL
    #
    # This makes BEVFusion enter the LGPC Stage-1
    # dedicated freeze policy.
    # ========================================================
    bbox_head=dict(
        rrrf_mode='lgpc_only',
    ),
)


# ============================================================
# Load mature PCC checkpoint as INITIAL WEIGHTS.
#
# DO NOT resume optimizer / iteration state.
# ============================================================

load_from = (
    'data/weights/'
    'LGPC_RRRF_2nd_nds70_44.pth'
)

resume = False


# ============================================================
# LocalRefiner pilot
# ============================================================

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=5000,
    val_interval=999999,
)


# ============================================================
# LocalRefiner is RANDOMLY INITIALIZED.
#
# Do NOT use the RRRF adaptation LR=1e-5.
# Start with 1e-4 for the new small head.
# ============================================================

optim_wrapper = dict(
    optimizer=dict(
        lr=1.0e-4,
    ),
)


param_scheduler = []


default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=1000,
        max_keep_ckpts=10,
    ),
)


work_dir = (
    './work_dirs/'
    'pcc_lgpcb_corr_refine_from_7044_5k'
)