import sys
import torch

from mmengine.config import Config
from mmengine.runner import Runner


config_path = sys.argv[1]

cfg = Config.fromfile(config_path)

runner = Runner.from_cfg(cfg)
model = runner.model

if hasattr(model, 'module'):
    model = model.module

model.train()


# ============================================================
# 1. Check Trainable Parameters
# ============================================================

allowed = (
    'corr_bev_lifter.',
    'bbox_head.coarse_bev_adapter.',
    'bbox_head.coarse_bev_gate.',
    'bbox_head.coarse_bev_alpha',
    'bbox_head.fusion_cross_attention.',
    'bbox_head.fusion_ffn.',
    'bbox_head.fusion_norm1.',
    'bbox_head.fusion_norm2.',
)

trainable_names = []

for name, p in model.named_parameters():

    if p.requires_grad:

        trainable_names.append(name)

        assert name.startswith(allowed), (
            f'Unexpected trainable: {name}'
        )

for prefix in allowed:
    assert any(
        name.startswith(prefix)
        for name in trainable_names
    ), f'Missing trainable module: {prefix}'

print('[FREEZE PASS]')

for name in trainable_names:
    print('  ', name)


# ============================================================
# 2. Check Module Training Modes
# ============================================================

head = model.bbox_head

assert not head.training
assert not head.decoder[0].training
assert not head.prediction_heads[0].training

assert model.corr_bev_lifter.training
assert head.fusion_cross_attention.training

print('[TRAIN/EVAL MODE PASS]')


# ============================================================
# 3. Check Checkpoint Tensor Shapes
# ============================================================

path = cfg.load_from

ckpt = torch.load(
    path,
    map_location='cpu',
    weights_only=False,
)

state = ckpt.get('state_dict', ckpt)

state = {
    k[7:] if k.startswith('module.') else k: v
    for k, v in state.items()
}

expected = model.state_dict()

mismatched = []

for name, value in state.items():

    if name not in expected:
        continue

    if value.shape != expected[name].shape:

        mismatched.append(
            (name, value.shape, expected[name].shape)
        )

for name, old_shape, new_shape in mismatched:

    print(
        '[SIZE MISMATCH]',
        name,
        old_shape,
        new_shape,
    )

if mismatched:
    raise RuntimeError('Checkpoint shape mismatch')

missing_frozen = [
    name
    for name, p in model.named_parameters()
    if not p.requires_grad and name not in state
]

if missing_frozen:
    print('[WARNING] Missing frozen parameters:')
    for name in missing_frozen:
        print('  ', name)

print('[CHECKPOINT SHAPE PASS]')
print('[VERIFICATION COMPLETED]')