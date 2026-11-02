
# ================================================
# PGFR-Det RRRF Ablation Evaluation
# Same checkpoint / same data / same perturbation
# ================================================

_base_ = [
    './pcc_bevfusion_rrrf_exp4_joint_fresh_eval.py'
]

# -----------------------------------------------
# Full model by default
# -----------------------------------------------
model = dict(
    calibration_mode='pcc_full',

    bbox_head=dict(
        rrrf_mode='residual_se3_v2',
        query_source='fused',

        rrrf_ablation_identity_geometry=False,
        rrrf_ablation_skip_refined_attention=False,

        rrrf_adapter_enabled=True,
        rrrf_adapter_scale=1.0,

        rrrf_query_select_enabled=True,
        rrrf_query_select_scale=1.0,
    ),
)

# -----------------------------------------------
# Detection-only metric
# -----------------------------------------------
test_evaluator = [
    dict(
        type='NuScenesMetric',

        data_root=(
            '/workspace/mmdetection3d/data/nuscenes/'
        ),

        ann_file=(
            '/workspace/mmdetection3d/data/nuscenes/'
            'nuscenes_infos_val_new_with_2d.pkl'
        ),

        metric='bbox',
        version='v1.0-trainval',
        collect_dir='test_results_tmp_pgfr_ablation',
    ),
]

test_cfg = dict(
    _delete_=True,
    type='TestLoop',
)

load_from = None
resume = False
