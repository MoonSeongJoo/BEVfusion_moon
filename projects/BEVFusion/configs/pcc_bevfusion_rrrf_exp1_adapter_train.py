# ============================================================
# PCC EXP1 - RRRF Residual Feature Adapter
#
# Baseline:
#   LGPC_RRRF_2nd_nds70_44_ORIGINAL.pth
#
# Train:
#   ONLY bbox_head.rrrf_adapter
#
# Frozen:
#   LGPC
#   LiDAR / Camera Backbone
#   RRRF Coarse / 2nd Predictor / Refined
#   BEVFusion Decoder
# ============================================================

_base_ = [
    './pcc_bevfusion_hybrid_iros7011_eval.py'
]


# ============================================================
# 1. MODEL
# ============================================================

model = dict(

    enable_selective_freezing=False,

    calibration_mode='pcc_full',

    lgpc_train_stage='rrrf_adapter',

    bbox_head=dict(

        # 기존 RRRF V1 유지
        rrrf_mode='residual_se3',

        query_source='fused',

        # 신규 Adapter 활성화
        rrrf_adapter_enabled=True,

        rrrf_adapter_scale=1.0,
    ),

    # 내부 pretrained 중복 로딩 방지
    # ORIGINAL checkpoint에서 전체 가중치 로딩
    corr=dict(init_cfg=None),

    z_estimator=dict(init_cfg=None),

    calib_head=dict(init_cfg=None),
)


# ============================================================
# 2. DDP
# ============================================================

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True,
)


# ============================================================
# 3. TRAINING LOOP
# ============================================================

train_cfg = dict(
    _delete_=True,

    type='IterBasedTrainLoop',

    max_iters=3000,

    # 중간 Validation은 별도 tools/test.py 사용
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


# ============================================================
# 4. OPTIMIZER
# ============================================================

optim_wrapper = dict(
    _delete_=True,

    type='OptimWrapper',

    optimizer=dict(
        type='AdamW',
        lr=1.0e-4,
        weight_decay=0.01,
    ),

    clip_grad=dict(
        max_norm=10.0,
        norm_type=2,
    ),
)


# ============================================================
# 5. LR SCHEDULER
# ============================================================

# 첫 번째 Exp1에서는 고정 LR
param_scheduler = []

auto_scale_lr = dict(
    enable=False,
    base_batch_size=32,
)


# ============================================================
# 6. LOGGING
# ============================================================

log_processor = dict(
    type='LogProcessor',
    window_size=20,
    by_epoch=False,
)


# ============================================================
# 7. HOOKS
# ============================================================

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

        max_keep_ckpts=8,

        save_last=True,
    ),

    visualization=dict(
        _delete_=True,
        type='Det3DVisualizationHook',
        draw=False,
    ),
)


# ============================================================
# 8. CHECKPOINT
# ============================================================

load_from = (
    'data/weights/'
    'LGPC_RRRF_2nd_nds70_44_ORIGINAL.pth'
)

# optimizer는 새로 시작
resume = False