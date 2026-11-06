# ablation_C_eval.py

_base_ = ['./ablation_C_train.py']

load_from = None
resume = False

work_dir = 'work_dirs/ablation_v2_C_eval'

test_cfg = dict(_delete_=True, type='TestLoop')