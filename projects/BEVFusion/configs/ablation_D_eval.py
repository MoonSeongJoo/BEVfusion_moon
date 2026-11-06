# ablation_D_eval.py

_base_ = ['./ablation_D_train.py']

load_from = None
resume = False

work_dir = 'work_dirs/ablation_v2_D_eval'

test_cfg = dict(_delete_=True, type='TestLoop')