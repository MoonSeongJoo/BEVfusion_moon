# ============================================================
# LGPC-B 246k + OLD RRRF + BEVFusion
# Evaluation-only configuration
#
# Purpose:
#   New LGPC-B Stage1
#        +
#   Existing RRRF Stage2
#        +
#   Existing BEVFusion
#
# No training.
# ============================================================

_base_ = [
    './pcc_bevfusion_rrrf_feature_refine_full.py'
]


model = dict(

    # --------------------------------------------------------
    # IMPORTANT:
    # Use actual predicted LGPC correction,
    # then continue through RRRF + BEVFusion.
    # --------------------------------------------------------
    calibration_mode='pcc_full',

    # --------------------------------------------------------
    # RRRF V2 evaluation path.
    #
    # This mainly selects the RRRF configuration/freeze policy.
    # Inference branch itself is selected by bbox_head.rrrf_mode.
    # --------------------------------------------------------
    lgpc_train_stage='rrrf_v2',

    # Dedicated RRRF policy requires this False.
    enable_selective_freezing=False,

    # --------------------------------------------------------
    # Stage-1 LGPC-B
    # --------------------------------------------------------
    calib_head=dict(
        trans_refine_mode='geo_feature_residual',
    ),

    # --------------------------------------------------------
    # Existing RRRF V2
    # --------------------------------------------------------
    bbox_head=dict(

        rrrf_mode='residual_se3_v2',

        rrrf_num_cams=6,

        rrrf_detach_geometry=True,

        # Keep same query source as previous RRRF.
        query_source='fused',
    ),
)


# ============================================================
# Evaluators
#
# Measure BOTH:
#   1) physical calibration
#   2) nuScenes detection performance
# ============================================================

data_root = '/workspace/mmdetection3d/data/nuscenes/'

test_evaluator = [

    dict(
        type='CalibLCCNetMetric',
        prefix='calib_lccnet',
        debug=False,
        debug_n=3,
    ),

    dict(
        type='NuScenesMetric',

        data_root=data_root,

        ann_file=(
            data_root
            + 'nuscenes_infos_val_new_with_2d.pkl'
        ),

        metric='bbox',

        version='v1.0-trainval',

        collect_dir=(
            'test_results_tmp_lgpcb246k_oldrrrf'
        ),
    ),
]