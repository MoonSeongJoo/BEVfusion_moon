# ============================================================
# EXP3: Query-Selective Attention Evaluation
# ============================================================

_base_ = [
    './pcc_bevfusion_rrrf_exp2_v2_decoder_tail_eval.py'
]

model = dict(

    enable_selective_freezing=False,

    calibration_mode='pcc_full',

    lgpc_train_stage='rrrf_query_select',

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

        rrrf_query_select_enabled=True,
        rrrf_query_select_scale=1.0,

        rrrf_query_select_topk=16,
        rrrf_query_select_radius_m=4.0,
        rrrf_query_select_sigma_m=2.0,
    ),
)

load_from = None

resume = False