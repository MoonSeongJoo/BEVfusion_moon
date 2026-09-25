# ============================================================
# PCC / Hybrid IROS-70.11
# RRRF Feature-Refine Adaptation
# ============================================================

_base_ = [
    './pcc_bevfusion_repair_v1.py'
]

model = dict(
    enable_selective_freezing=False,
    calibration_mode='pcc_full',
    lgpc_train_stage='rrrf',

    bbox_head=dict(
        rrrf_mode='feature_refine',
        rrrf_num_cams=6,
        rrrf_detach_geometry=True,
        query_source='fused',
    ),

    corr=dict(
        init_cfg=None,
        enable_cycle=False,
    ),

    z_estimator=dict(
        init_cfg=None,
    ),

    calib_head=dict(
        type='CalibrationCorrectionHead',
        in_channels=312,
        num_kp=200,
        dropout_p=0.1,
        local_dim=32,
        point_dim=64,
        global_dim=64,
        rot_condition_dim=16,
        rot_condition_scale_deg=10.0,
        camera_pose_dim=16,
        init_cfg=None,
    ),
)

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True,
)

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=5000,
    val_interval=999999,
)

val_cfg = dict(
    _delete_=True,
    type='ValLoop',
)

test_cfg = dict(
    _delete_=True,
    type='TestLoop',
)

log_processor = dict(
    type='LogProcessor',
    window_size=50,
    by_epoch=False,
)

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

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=5.0e-5,
        weight_decay=0.01,
    ),
    clip_grad=dict(
        max_norm=35,
        norm_type=2,
    ),
)

param_scheduler = []

auto_scale_lr = dict(
    enable=False,
    base_batch_size=32,
)

default_hooks = dict(
    logger=dict(
        type='LoggerHook',
        interval=20,
    ),

    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        by_epoch=False,
        interval=500,
        max_keep_ckpts=11,
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
    'hybrid_iros_nds7011_lgpc_iter123000.pth'
)

resume = False
