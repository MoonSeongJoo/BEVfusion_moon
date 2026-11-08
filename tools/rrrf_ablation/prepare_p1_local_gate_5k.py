#!/usr/bin/env python3
"""Prepare a source-audited P1 local-gate 5000-iteration train/test experiment.

No checkpoint weights are modified. Patches only two source files, with backups:
- geometry_selective_attention.py: differentiable gate-open auxiliary quantity
- transfusion_head.py: add weighted gate-open loss to existing detection loss
Creates train/eval MMEngine configs and a train-then-test shell runner.
Requires ZIP-compatible BEVFusion code; refuses mismatched code anchors.
"""
from __future__ import annotations
import argparse
import ast
import datetime
import os
from pathlib import Path
import shutil
import sys

GEOM_MARKER = 'PGFR_GATE_OPEN_AUX_V1'
LOSS_MARKER = 'PGFR_GATE_OPEN_LOSS_V1'

GEOM_INSERT = '''        # PGFR_GATE_OPEN_AUX_V1: differentiable training-only gate target.
        # This is CONDITIONAL on having a valid geometric neighbor; it
        # cannot change query coverage, validity masks, or selection radius.
        self._pgfr_gate_aux_loss = None
        self._pgfr_gate_valid_train = None
        if self.training:
            gate_target = float(os.environ.get("PGFR_LOCAL_GATE_TARGET", "0.99"))
            if not (0.0 < gate_target < 1.0):
                raise ValueError("PGFR_LOCAL_GATE_TARGET must be in (0, 1)")
            gate_values = gate.squeeze(-1)
            active_float = has_neighbor.to(dtype=gate_values.dtype)
            active_count_safe = active_float.sum().clamp_min(1.0)
            self._pgfr_gate_valid_train = (
                (gate_values * active_float).sum() / active_count_safe
            ).detach()
            # One-sided penalty: only gates smaller than target are penalized.
            # If no valid neighbors exist, the penalty is exactly zero.
            self._pgfr_gate_aux_loss = (
                ((gate_target - gate_values).clamp_min(0.0).square())
                * active_float
            ).sum() / active_count_safe

'''

LOSS_INSERT = '''        # PGFR_GATE_OPEN_LOSS_V1: selective gate supervision plus detection losses.
        # The parent TransFusionHead is intentionally in eval() mode during
        # rrrf_joint_fresh training. Check selector.training, NOT self.training.
        selector = getattr(self, 'rrrf_query_selector', None)
        if selector is not None and selector.training:
            gate_aux = getattr(selector, '_pgfr_gate_aux_loss', None)
            gate_valid_train = getattr(selector, '_pgfr_gate_valid_train', None)
            gate_weight = float(os.environ.get('PGFR_LOCAL_GATE_AUX_WEIGHT', '0.0'))
            if gate_weight < 0.0:
                raise ValueError('PGFR_LOCAL_GATE_AUX_WEIGHT must be >= 0')
            if gate_weight > 0.0:
                if gate_aux is None or gate_valid_train is None:
                    raise RuntimeError('Gate auxiliary loss missing: selector not executed')
                loss_dict['loss_local_gate_open'] = gate_weight * gate_aux
                loss_dict['local_gate_valid_train'] = gate_valid_train

'''

def patch_source(geom_src: str, head_src: str):
    if GEOM_MARKER not in geom_src:
        # Handle the no-camera-proposal path as well; otherwise the head could
        # accidentally reuse the prior batch's gate-loss tensor.
        empty_anchor = '        if Np == 0:\n            return torch.zeros_like(query_chw)'
        empty_replacement = (
            '        if Np == 0:\n'
            '            self._pgfr_gate_aux_loss = query_chw.sum() * 0.0\n'
            '            self._pgfr_gate_valid_train = query_chw.new_zeros(())\n'
            '            return torch.zeros_like(query_chw)'
        )
        if geom_src.count(empty_anchor) != 1:
            raise RuntimeError('No-proposal source anchor missing or duplicated')
        geom_src = geom_src.replace(empty_anchor, empty_replacement, 1)
        anchor = '        # --------------------------------------------\n        # STEP G: Residual output\n'
        if geom_src.count(anchor) != 1:
            raise RuntimeError('Geometry source anchor missing or duplicated; inspect source before patching')
        geom_src = geom_src.replace(anchor, GEOM_INSERT + anchor, 1)
    if LOSS_MARKER not in head_src:
        anchor = "        loss_dict['matched_ious'] = layer_loss_cls.new_tensor(matched_ious)"
        if head_src.count(anchor) != 1:
            raise RuntimeError('TransFusionHead loss anchor missing or duplicated; inspect source before patching')
        if 'import os\n' not in head_src:
            h = 'import copy\n'
            if head_src.count(h) != 1:
                raise RuntimeError('TransFusionHead import anchor missing')
            head_src = head_src.replace(h, h + 'import os\n', 1)
        head_src = head_src.replace(anchor, LOSS_INSERT + anchor, 1)
    ast.parse(geom_src)
    ast.parse(head_src)
    return geom_src, head_src


def write_script(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
    ast.parse(content) if path.suffix == '.py' else None


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',default='/workspace/mmdetection3d')
    ap.add_argument('--p1',default='data/weights/ablation_v2/PGFR_Ablation_P1.pth')
    ap.add_argument('--output-dir',default='work_dirs/p1_local_gate_target_5k')
    ap.add_argument('--iterations',type=int,default=5000)
    ap.add_argument('--gate-target',type=float,default=0.99)
    ap.add_argument('--gate-aux-weight',type=float,default=0.05)
    ap.add_argument('--gate-logit-offset',type=float,default=2.0)
    ap.add_argument('--train-port',type=int,default=29680)
    ap.add_argument('--test-port',type=int,default=29681)
    ap.add_argument('--check-only',action='store_true',help='Check patchability and inputs without writing anything')
    args=ap.parse_args()
    if args.iterations < 1 or not (0 < args.gate_target < 1) or args.gate_aux_weight <= 0:
        ap.error('iterations >=1, target in (0,1), aux-weight > 0 required')
    repo=Path(args.repo).resolve()
    if not repo.is_dir():
        raise SystemExit(f'[ERROR] Repo does not exist: {repo}')
    p1=Path(args.p1)
    if not p1.is_absolute():p1=repo/p1
    if not p1.is_file():
        raise SystemExit(f'[ERROR] P1 checkpoint not found: {p1}')
    rel_dir=Path(args.output_dir)
    outdir=rel_dir if rel_dir.is_absolute() else repo/rel_dir
    cfgdir=repo/'projects/BEVFusion/configs'
    base_train=cfgdir/'pgfr_feature_refine_local_5k.py'
    base_eval=cfgdir/'pgfr_feature_refine_local_eval.py'
    geom=repo/'projects/BEVFusion/bevfusion/geometry_selective_attention.py'
    head=repo/'projects/BEVFusion/bevfusion/transfusion_head.py'
    for p in (base_train,base_eval,geom,head,repo/'tools/dist_train.sh',repo/'tools/dist_test.sh'):
        if not p.is_file():raise SystemExit(f'[ERROR] Required file missing: {p}')
    orig_g=geom.read_text(encoding='utf-8')
    orig_h=head.read_text(encoding='utf-8')
    new_g,new_h=patch_source(orig_g,orig_h)
    print('[PASS] Source anchors and Python syntax validated')
    print(f'[P1] {p1}')
    print(f'[Gate] target={args.gate_target} weight={args.gate_aux_weight} logit_offset={args.gate_logit_offset:+.2f}')
    if args.check_only:
        print('[CHECK ONLY] No files were changed')
        return

    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    for file,old,new in ((geom,orig_g,new_g),(head,orig_h,new_h)):
        if new != old:
            backup=file.with_name(file.name+f'.before_gate_aux_{stamp}.bak')
            shutil.copy2(file,backup)
            file.write_text(new,encoding='utf-8')
            print(f'[PATCHED] {file}\n  [BACKUP] {backup}')
        else: print(f'[ALREADY PATCHED] {file}')

    outdir.mkdir(parents=True,exist_ok=True)
    # Prepare train/eval configs; both extend known-good originals.
    train_cfg=cfgdir/'pgfr_p1_gate_target_5k_train.py'
    test_cfg=cfgdir/'pgfr_p1_gate_target_5k_eval.py'
    local_str = repr(str(p1))
    output_str = repr(str(outdir))
    model_str = '''model = dict(
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
'''
    tr=f"_base_ = ['./pgfr_feature_refine_local_5k.py']\n\n{model_str}\nload_from = {local_str}\nresume = False\nwork_dir = {output_str}\ntrain_cfg = dict(max_iters={args.iterations}, val_interval=999999)\nval_cfg = None\nval_dataloader = None\nval_evaluator = None\n"
    ev=f"_base_ = ['./pgfr_feature_refine_local_eval.py']\n\n{model_str}\nload_from = None\nresume = False\nwork_dir = {repr(str(outdir/'evaluation'))}\n"
    write_script(train_cfg,tr)
    write_script(test_cfg,ev)
    print(f'[CONFIG] TRAIN: {train_cfg}')
    print(f'[CONFIG] EVAL : {test_cfg}')

    ckpt=outdir/f'iter_{args.iterations}.pth'
    runner=outdir/'run_train_then_test.sh'
    body=f'''#!/usr/bin/env bash
set -u
cd {str(repo)!r} || exit 1
export CUDA_VISIBLE_DEVICES=0,1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PGFR_GATE_LOGIT_OFFSET={str(args.gate_logit_offset)!r}
export PGFR_LOCAL_GATE_AUX_WEIGHT={str(args.gate_aux_weight)!r}
export PGFR_LOCAL_GATE_TARGET={str(args.gate_target)!r}
RUN_DIR={str(outdir)!r}
TRAIN_CFG={str(train_cfg)!r}
TEST_CFG={str(test_cfg)!r}
BASE_P1={str(p1)!r}
EVAL_CKPT={str(ckpt)!r}
mkdir -p "$RUN_DIR/evaluation"
rm -f "$RUN_DIR/train_exit_code.txt" "$RUN_DIR/test_exit_code.txt" "$RUN_DIR/exit_code.txt"
echo "[START] $(date -Is) P1 gate-training 5000 -> test"
echo "[GPUS] $CUDA_VISIBLE_DEVICES"
echo "[P1] $BASE_P1"
echo "[GATE] offset=$PGFR_GATE_LOGIT_OFFSET weight=$PGFR_LOCAL_GATE_AUX_WEIGHT target=$PGFR_LOCAL_GATE_TARGET"
echo "[TRAIN CFG] $TRAIN_CFG"
echo "[TEST CFG] $TEST_CFG"
if [ ! -s "$BASE_P1" ]; then echo '[ERROR] Base checkpoint missing or empty'; echo 2 > "$RUN_DIR/exit_code.txt"; exit 2; fi

PORT={args.train_port} bash tools/dist_train.sh "$TRAIN_CFG" 2 \\
  --work-dir "$RUN_DIR" \\
  > "$RUN_DIR/train.log" 2>&1
TRAIN_CODE=$?
echo "$TRAIN_CODE" > "$RUN_DIR/train_exit_code.txt"
echo "[TRAIN END] $(date -Is) exit=$TRAIN_CODE"
if [ "$TRAIN_CODE" -ne 0 ]; then
  tail -80 "$RUN_DIR/train.log"
  echo "$TRAIN_CODE" > "$RUN_DIR/exit_code.txt"
  exit "$TRAIN_CODE"
fi
if [ ! -s "$EVAL_CKPT" ]; then
  echo "[ERROR] Expected checkpoint missing: $EVAL_CKPT"
  find "$RUN_DIR" -maxdepth 1 -name 'iter_*.pth' -printf '%f\\n' | sort -V
  echo 3 > "$RUN_DIR/exit_code.txt"
  exit 3
fi

echo "[TEST START] $(date -Is) checkpoint=$EVAL_CKPT"
PORT={args.test_port} bash tools/dist_test.sh "$TEST_CFG" "$EVAL_CKPT" 2 \\
  --work-dir "$RUN_DIR/evaluation" \\
  > "$RUN_DIR/test.log" 2>&1
TEST_CODE=$?
echo "$TEST_CODE" > "$RUN_DIR/test_exit_code.txt"
echo "[TEST END] $(date -Is) exit=$TEST_CODE"
if [ "$TEST_CODE" -eq 0 ]; then
  grep -E '^mAP:|^NDS:|^mATE:|^mASE:|^mAOE:|^mAVE:|^mAAE:|\\[EXP3-EVAL\\]' "$RUN_DIR/test.log" | tail -20 || true
else
  tail -100 "$RUN_DIR/test.log"
fi
echo "$TEST_CODE" > "$RUN_DIR/exit_code.txt"
exit "$TEST_CODE"
'''
    runner.write_text(body,encoding='utf-8')
    runner.chmod(0o755)
    print(f'[RUNNER] {runner}')
    print('[PASS] Preparation finished. Validate configs with mmengine Config.fromfile on server before launch.')
    print(f"[LAUNCH] nohup setsid bash {runner} > {outdir/'runner.log'} 2>&1 < /dev/null &")

if __name__=='__main__':
    main()