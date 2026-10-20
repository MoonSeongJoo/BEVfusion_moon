
# ============================================================
# PCC EXP4 - Joint Fresh Training
#
# Geometry-Selective Attention
# + Residual Feature Adapter
# + Transformer Decoder Tail
# + Detection Prediction Head
#
# Starting point:
#   Pre-Decoder-20K RRRF V2 Reference
#
# Frozen:
#   LGPC CorrNet
#   ZEstimator
#   Calibration Head
#   RRRF V2 Calibration Predictor
#   Backbone / Existing Fusion
#
# Trainable:
#   1. Geometry-Selective Attention
#   2. Residual Adapter
#   3. Transformer Decoder Tail
#   4. Detection Prediction Head
#
# Loss:
#   layer_-1_loss_cls
#   layer_-1_loss_bbox
# ============================================================


# ------------------------------------------------------------
# 1. Base Configuration
# ------------------------------------------------------------

_base_ = [
    './pcc_bevfusion_rrrf_exp3_query_select_train.py'
]


# ------------------------------------------------------------
# 2. Model Configuration
# ------------------------------------------------------------

model = dict(

    # Disable legacy selective-freezing logic.
    # Exp4 trainability is controlled by lgpc_train_stage.
    enable_selective_freezing=False,

    # Full PCC calibration + detection pipeline.
    calibration_mode='pcc_full',

    # New EXP4 stage.
    lgpc_train_stage='rrrf_joint_fresh',

    # --------------------------------------------------------
    # Prevent independent pretrained initialization.
    # Load weights from a single prepared checkpoint.
    # --------------------------------------------------------

    corr=dict(
        init_cfg=None,
    ),

    z_estimator=dict(
        init_cfg=None,
    ),

    calib_head=dict(
        init_cfg=None,
        trans_refine_mode='geo_feature_residual',
    ),

    # --------------------------------------------------------
    # RRRF / Detection Head
    # --------------------------------------------------------

    bbox_head=dict(

        # RRRF V2 calibration path.
        rrrf_mode='residual_se3_v2',

        rrrf_num_cams=6,

        rrrf_detach_geometry=True,

        # Fused BEV query.
        query_source='fused',

        # ----------------------------------------------------
        # EXP4 Module 1: Residual Feature Adapter
        # ----------------------------------------------------

        rrrf_adapter_enabled=True,

        # Full contribution.
        # The clean initial checkpoint must not contain
        # previously trained Adapter weights.
        rrrf_adapter_scale=1.0,

        # ----------------------------------------------------
        # EXP4 Module 2: Geometry-Selective Attention
        # ----------------------------------------------------

        rrrf_query_select_enabled=True,

        rrrf_query_select_scale=1.0,

        # Maximum camera proposals per query.
        rrrf_query_select_topk=16,

        # Neighbor search radius in meters.
        rrrf_query_select_radius_m=4.0,

        # Geometry distance bias sigma in meters.
        rrrf_query_select_sigma_m=2.0,

    ),
)


# ------------------------------------------------------------
# 3. Training Loop
# ------------------------------------------------------------

train_cfg = dict(

    _delete_=True,

    type='IterBasedTrainLoop',

    # EXP4 pilot training.
    max_iters=400000,

    # Skip full validation during training.
    val_interval=999999,

)


# ------------------------------------------------------------
# 4. Optimizer
# ------------------------------------------------------------

optim_wrapper = dict(

    _delete_=True,

    type='OptimWrapper',

    optimizer=dict(

        type='AdamW',

        # Base learning rate.
        lr=1.0e-5,

        weight_decay=0.01,

    ),

    # --------------------------------------------------------
    # Per-module learning rates
    #
    # Geometry Selector : 2e-5
    # Residual Adapter  : 2e-5
    # Decoder Tail      : 1e-5
    # Prediction Head   : 1e-5
    # --------------------------------------------------------

    paramwise_cfg=dict(

        custom_keys={

            # Geometry-Selective Attention.
            'bbox_head.rrrf_query_selector': dict(
                lr_mult=2.0,
            ),

            # Residual Feature Adapter.
            'bbox_head.rrrf_adapter': dict(
                lr_mult=2.0,
            ),

            # Final Transformer Decoder.
            'bbox_head.decoder.0': dict(
                lr_mult=1.0,
            ),

            # Final Detection Prediction Head.
            'bbox_head.prediction_heads.0': dict(
                lr_mult=1.0,
            ),

        },

    ),

    # Gradient clipping.
    clip_grad=dict(

        max_norm=10.0,

        norm_type=2,

    ),

)


# ------------------------------------------------------------
# 5. Learning Rate Scheduler
# ------------------------------------------------------------

# Constant learning rate during the 3K pilot.
param_scheduler = []


# ------------------------------------------------------------
# 6. Automatic LR Scaling
# ------------------------------------------------------------

auto_scale_lr = dict(

    enable=False,

    base_batch_size=32,

)


# ------------------------------------------------------------
# 7. Logging and Checkpoint
# ------------------------------------------------------------

default_hooks = dict(

    logger=dict(

        type='LoggerHook',

        interval=50,

    ),

    checkpoint=dict(

        _delete_=True,

        type='CheckpointHook',

        by_epoch=False,

        # Save every 500 iterations.
        interval=1000,

        # Keep up to 8 checkpoints.
        max_keep_ckpts=8,

        save_last=True,

    ),

)


# ------------------------------------------------------------
# 8. Initial Checkpoint
# ------------------------------------------------------------

# IMPORTANT:
# This checkpoint must be prepared from a pre-Decoder-20K
# RRRF V2 reference, with old Adapter and Selector weights
# excluded.
#
# Existing trained LGPC and RRRF V2 weights are retained.
# Decoder and Prediction Head are loaded from the reference.

load_from = (
    'work_dirs/pcc_exp4_joint_fresh_3k_1/iter_3000.pth'
)


# ------------------------------------------------------------
# 9. Resume Policy
# ------------------------------------------------------------

# Start EXP4 training from iteration 0.
# Load pretrained model weights only.
# Do not restore the previous optimizer or iteration number.

resume = False
