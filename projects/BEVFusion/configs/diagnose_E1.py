_base_ = ['./ablation_A_eval.py']

model = dict(
    calibration_mode='pcc_full',
    bbox_head=dict(
        rrrf_use_coarse=False,
        rrrf_use_refined=False,
        rrrf_query_select_enabled=False,
        rrrf_query_select_scale=0.0,
        rrrf_adapter_enabled=False,
        rrrf_adapter_scale=0.0,
    ),
)

load_from = None
resume = False
test_cfg = dict(_delete_=True, type='TestLoop')

work_dir = 'work_dirs/diagnose_E123/E1'
