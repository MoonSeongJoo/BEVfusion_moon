# ============================================================
# PCC Repair V1
# Full LGPC Stage-1 Training
#
# IMPORTANT
# - Base: repaired PCC implementation
# - Full nuScenes training dataset
# - Train corruption: RANDOM
# - Validation corruption: deterministic frozen O-3
# - LGPC Stage1 only
# - 24 epochs
# ============================================================

_base_ = [
    './pcc_bevfusion_repair_v1.py'
]


# ============================================================
# MODEL
#
# Keep CorrNet correspondence-learning path trainable.
# Do NOT enable the old selective freezing path because that
# freezes self.corr as a whole.
# ============================================================

model = dict(

    enable_selective_freezing=False,
    lgpc_train_stage='corr',

    bbox_head=dict(
        rrrf_mode='lgpc_only',
    ),

    corr=dict(
        init_cfg=None,
        enable_cycle=False,
    ),
    
    corr_loss=dict(
        type='CorrelationCycleLoss',

        corr_weight=1.0,

        # Cycle is auxiliary.
        # Start conservatively.
        cycle_weight=0.1,
    ),

    z_estimator=dict(
        init_cfg=None,
    ),

    calib_head=dict(
        init_cfg=None,
    ),
)


# ============================================================
# DISTRIBUTED TRAINING
#
# lgpc_only does not necessarily exercise every parameter on
# every iteration. find_unused_parameters=True avoids DDP
# reduction failures from unused branches.
# ============================================================

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True,
)


# ============================================================
# TRAINING SCHEDULE
#
# Paper protocol:
# LGPC standalone pretraining = 24 epochs
#
# Full 6019-frame validation is expensive, so validate at
# epochs 6 / 12 / 18 / 24.
# ============================================================

train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=24,
    val_interval=6,
)

val_cfg = dict(
    _delete_=True,
    type='ValLoop',
)

test_cfg = dict(
    _delete_=True,
    type='TestLoop',
)


# ============================================================
# VALIDATION
#
# For Stage-1 training we only need physical calibration
# residuals. Do not run NuScenes mAP/NDS every validation.
# ============================================================

val_evaluator = [
    dict(
        type='CalibLCCNetMetric',
        collect_device='cpu',
        prefix='calib_lccnet',
        debug=True,
        debug_n=3,
    ),
]

test_evaluator = val_evaluator


# ============================================================
# HOOKS
#
# Save periodically. Avoid inherited save_best='train/loss',
# which previously caused KeyError after validation.
# ============================================================

default_hooks = dict(

    logger=dict(
        type='LoggerHook',
        interval=20,
    ),

    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        interval=2,
        by_epoch=True,
        max_keep_ckpts=12,
        save_last=True,
    ),

    visualization=dict(
        _delete_=True,
        type='Det3DVisualizationHook',
        draw=False,
    ),
)

load_from = (
    'data/weights/'
    'bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth'
)

resume = False