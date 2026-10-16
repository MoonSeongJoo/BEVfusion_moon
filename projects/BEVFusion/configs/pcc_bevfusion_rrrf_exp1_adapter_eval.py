# ============================================================
# PCC EXP1 - Residual Adapter Evaluation
#
# Same dataset / evaluator as ORIGINAL baseline
# ============================================================

_base_ = [
    './pcc_bevfusion_hybrid_iros7011_eval.py'
]


model = dict(

    enable_selective_freezing=False,

    calibration_mode='pcc_full',

    lgpc_train_stage='rrrf_adapter',

    bbox_head=dict(

        rrrf_mode='residual_se3',

        query_source='fused',

        rrrf_adapter_enabled=True,

        rrrf_adapter_scale=1.0,
    ),

    corr=dict(init_cfg=None),

    z_estimator=dict(init_cfg=None),

    calib_head=dict(init_cfg=None),
)


load_from = None

resume = False