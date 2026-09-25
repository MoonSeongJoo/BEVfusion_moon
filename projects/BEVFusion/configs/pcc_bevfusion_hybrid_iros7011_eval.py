# projects/BEVFusion/configs/
# pcc_bevfusion_hybrid_iros7011_eval.py

_base_ = [
    './pcc_bevfusion_repair_v1.py'
]

model = dict(
    calibration_mode='pcc_full',

    bbox_head=dict(
        rrrf_mode='residual_se3',
        query_source='fused',
    ),
)

# Detection + physical calibration metric만 유지
test_evaluator = [
    dict(
        type='CalibLCCNetMetric',
        prefix='calib_lccnet',
        debug=False,
        debug_n=3,
    ),

    dict(
        type='NuScenesMetric',
        data_root='/workspace/mmdetection3d/data/nuscenes/',
        ann_file=(
            '/workspace/mmdetection3d/data/nuscenes/'
            'nuscenes_infos_val_new_with_2d.pkl'
        ),
        metric='bbox',
        version='v1.0-trainval',
        collect_dir='test_results_tmp',
    ),
]

val_evaluator = test_evaluator

# 평가 중 visualization 불필요
default_hooks = dict(
    visualization=dict(
        _delete_=True,
        type='Det3DVisualizationHook',
        draw=False,
    ),
)

# checkpoint는 tools/test.py argument로 넘기므로 여기서는 의미 없음
load_from = None
resume = False