# ============================================================
# PCC EXP1 V2 Adapter Evaluation
#
# NuScenes NDS + LCCNet-compatible Calibration Metric
# ============================================================

_base_ = [
    './pcc_bevfusion_rrrf_exp1_adapter_eval.py'
]


model = dict(

    enable_selective_freezing=False,

    calibration_mode='pcc_full',

    lgpc_train_stage='rrrf_adapter',

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

        rrrf_adapter_scale=1.0,
    ),
)


# tools/test.py passes checkpoint as argument.
load_from = None
resume = False