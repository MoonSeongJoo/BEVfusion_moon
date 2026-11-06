# ablation_B_eval.py

_base_ = ['./ablation_B_train.py']

load_from = None
resume = False

work_dir = 'work_dirs/ablation_v2_B_eval'

test_cfg = dict(_delete_=True, type='TestLoop')