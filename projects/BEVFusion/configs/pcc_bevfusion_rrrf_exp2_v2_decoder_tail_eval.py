
# ============================================================
# EXP2: V2 Decoder Tail Evaluation
# ============================================================

_base_ = [
    './pcc_bevfusion_rrrf_exp1_adapter_eval.py'
]

model = dict(
    enable_selective_freezing=False,

    calibration_mode='pcc_full',

    lgpc_train_stage='rrrf_decoder_tail',

    calib_head=dict(
        trans_refine_mode='geo_feature_residual',
        init_cfg=None,
    ),

    bbox_head=dict(
        rrrf_mode='residual_se3_v2',
        rrrf_num_cams=6,
        rrrf_detach_geometry=True,
        query_source='fused',

        rrrf_adapter_enabled=True,
        rrrf_adapter_scale=0.0,
    ),
)

load_from = None

resume = False
