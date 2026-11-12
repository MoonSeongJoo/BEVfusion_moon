_base_ = ['./diagnose_E1.py']

model = dict(
    calibration_mode='pcc_calib_only',
)

work_dir = 'work_dirs/diagnose_E123/E2'
