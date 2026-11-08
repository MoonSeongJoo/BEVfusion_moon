_base_ = ['./pgfr_feature_refine_local_5k.py']

model = dict(
    enable_selective_freezing=False,
    calibration_mode='pcc_full',
    lgpc_train_stage='rrrf_joint_fresh',
    bbox_head=dict(
        rrrf_mode='feature_refine',
        query_source='fused',
        rrrf_use_coarse=True,
        rrrf_use_refined=True,
        rrrf_query_select_enabled=True,
        rrrf_query_select_scale=1.0,
        rrrf_query_select_topk=16,
        rrrf_query_select_radius_m=4.0,
        rrrf_query_select_sigma_m=2.0,
        rrrf_adapter_enabled=False,
        rrrf_adapter_scale=0.0,
    ),
)

load_from = '/workspace/mmdetection3d/data/weights/ablation_v2/PGFR_Ablation_P1.pth'
resume = False
work_dir = '/workspace/mmdetection3d/work_dirs/p1_local_gate_target_5k'
train_cfg = dict(max_iters=5000, val_interval=999999)
val_cfg = None
val_dataloader = None
val_evaluator = None
