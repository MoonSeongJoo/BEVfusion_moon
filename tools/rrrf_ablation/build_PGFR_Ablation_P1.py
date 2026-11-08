#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build PGFR RRRF P2 checkpoint: Official Decoder + PGFR Prediction Heads.

P2 tensor provenance (including buffers):
    PGFR    : ALL model keys except OFFICIAL_PREFIXES, including feat_projector,
              Coarse/Refined/Local RRRF, calibration, encoders, BEV, query modules
    Official: bbox_head.decoder.* ONLY (prediction_heads remain PGFR)
    Fresh   : NONE

V6: Handles an mmdet3d/spconv sparse-kernel load-layout incompatibility by
checking strict load first and (only if needed) inverse-permuting the 5-D sparse
kernels under pts_middle_encoder. The saved checkpoint is validated against the
same runtime loader and checked to reconstruct the EXACT PGFR source values.

No training, optimizer state, or source-code changes. Strict coverage: any missing
or incompatible target tensor aborts without saving a P2 checkpoint.

IMPORTANT: Matching an RRRF tensor in PGFR proves provenance, not that it was
updated by training. Verify training configuration/logs or compare against the
actual original pre-training initialization if that distinction matters.

If ablation_A_eval.py disables Coarse, Refined or Local Proposal, this script
turns on all three for a complete P2 model (NOT the original A ablation).
Stage scaling also needs valid nonzero values. The Local Proposal scale is
EXACTLY bbox_head.rrrf_query_select_scale (not a *local* named key). Prefer
--trained-config from the original PGFR run; if unavailable, explicitly supply
--local-scale 1.0, which is NOT proof of exact original PGFR inference.
The actual Coarse and Refined config-flag names are determined by inspecting
the installed TransFusionHead class. No guessed flag is silently injected. The effective config is exported
next to P2. The source config is not modified.

Run from the mmdetection3d repository root in its MMEngine environment.
"""

import argparse
import ast
import inspect
import textwrap
import random
from collections.abc import Mapping
from numbers import Real
from collections import Counter
from pathlib import Path

import numpy as np
import torch

OFFICIAL_PREFIXES = (
    'bbox_head.decoder.',
)

# These ALWAYS come from the PGFR checkpoint for P2 (NEVER initialize fresh).
PGFR_RRRF_PREFIXES = (
    'feat_projector.',
    'bbox_head.fusion_cross_attention.',
    'bbox_head.fusion_ffn.',
    'bbox_head.fusion_norm1.',
    'bbox_head.fusion_norm2.',
    'bbox_head.camera_proposal_pos_embedding.',
    'bbox_head.bev_query_pos_embedding.',
    'bbox_head.refined_attention.',
    'bbox_head.refined_ffn.',
    'bbox_head.refined_norm1.',
    'bbox_head.refined_norm2.',
    'bbox_head.rrrf_query_selector.',
    'bbox_head.calibration_predictor.',
    'bbox_head.residual_',
    'bbox_head.refined_fusion_aux_head.',
)

# Modules that must exist in the selected P2 model config. Additional keys are
# covered automatically by the all-PGFR-except-Official policy.
REQUIRED_PREFIXES = (
    'corr.', 'z_estimator.', 'calib_head.',
    'bbox_head.shared_conv.', 'bbox_head.heatmap_head.',
    'bbox_head.class_encoding.',
    'bbox_head.decoder.', 'bbox_head.prediction_heads.',
    'feat_projector.',
    'bbox_head.fusion_cross_attention.',
    'bbox_head.refined_attention.',
    'bbox_head.rrrf_query_selector.',
)


# Resolve dependency guards from the installed bbox_head constructor instead
# of guessing whether this fork calls the switch e.g. use_coarse_fusion,
# rrrf_coarse_enabled, or something else.
_GUARDS = {
    'coarse': 'Refined Fusion requires Coarse Fusion',
    'refined': 'Local Proposal Fusion requires Refined Fusion',
}
_EXCLUDE = (
    'loss', 'aux', 'dim', 'num', 'layer', 'drop', 'norm', 'ffn', 'weight',
    'head', 'iter', 'lr', 'threshold', 'score', 'kernel', 'position',
)


def _expr_names(expr):
    return ({n.id for n in ast.walk(expr) if isinstance(n, ast.Name)}
            | {n.attr for n in ast.walk(expr) if isinstance(n, ast.Attribute)})


def _negative_expr_names(expr):
    """Names inside `not X`, and straightforward `X is False` tests."""
    result = set()
    for n in ast.walk(expr):
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            result |= _expr_names(n.operand)
        elif isinstance(n, ast.Compare):
            if any(isinstance(op, (ast.Is, ast.Eq)) for op in n.ops):
                if any(isinstance(c, ast.Constant) and c.value is False
                       for c in n.comparators):
                    result |= _expr_names(n.left)
    return result


def _constructor_aliases(init_tree):
    """Map `self.flag = flag_arg` (and `self.flag = bool(flag_arg)`)."""
    aliases = {}
    for node in ast.walk(init_tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) \
                and value.func.id == 'bool' and len(value.args) == 1:
            value = value.args[0]
        if not isinstance(value, ast.Name):
            continue
        for target in targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) \
                    and target.value.id == 'self':
                aliases[target.attr] = value.id
    return aliases


def _guard_details(head_cls, message):
    try:
        source = textwrap.dedent(inspect.getsource(head_cls.__init__))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError) as exc:
        raise RuntimeError('Cannot inspect TransFusionHead guard source. '
                           'Use --coarse-flag and --refined-flag explicitly.') from exc
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Raise) and message in ast.unparse(inner):
                span = (getattr(node, 'end_lineno', node.lineno) - node.lineno)
                matches.append((span, node.test))
                break
    if not matches:
        return set(), set(), _constructor_aliases(tree)
    test = min(matches, key=lambda pair: pair[0])[1]
    return _expr_names(test), _negative_expr_names(test), _constructor_aliases(tree)


def _discover_stage_flag(head_cls, head_cfg, stage, override=None):
    try:
        params = inspect.signature(head_cls.__init__).parameters
    except (TypeError, ValueError) as exc:
        raise RuntimeError('Cannot inspect bbox_head constructor signature') from exc

    if override:
        if override not in params:
            raise ValueError('--{}-flag {!r} is not a bbox_head __init__ argument'
                             .format(stage, override))
        return override

    guard_tokens, negated_tokens, aliases = _guard_details(
        head_cls, _GUARDS[stage])
    # Guard may refer to a `self.foo` property rather than the actual
    # constructor argument `foo`; recognize the simple assignment alias.
    expanded_guard = guard_tokens | {aliases[t] for t in guard_tokens if t in aliases}
    expanded_negative = negated_tokens | {aliases[t] for t in negated_tokens if t in aliases}

    boolean_args = [name for name, param in params.items()
                    if isinstance(param.default, bool)
                    or isinstance(head_cfg.get(name, None), bool)]
    selected = [name for name in boolean_args
                if ('refin' in name.lower() if stage == 'refined'
                    else 'coarse' in name.lower())
                and not any(tag in name.lower() for tag in _EXCLUDE)]
    if stage == 'coarse' and not selected:
        selected = [name for name in boolean_args
                    if 'fusion' in name.lower()
                    and 'refin' not in name.lower()
                    and 'local' not in name.lower()
                    and 'query' not in name.lower()
                    and not any(tag in name.lower() for tag in _EXCLUDE)]

    print('[CONFIG] {} dependency guard tokens: {}'.format(stage, sorted(guard_tokens)))
    print('[CONFIG] {} negated guard tokens: {}'.format(stage, sorted(negated_tokens)))
    print('[CONFIG] {} candidate constructor flags: {}'.format(stage, selected))

    # Prefer the boolean setting that the guard requires to be enabled.
    neg_match = [k for k in selected if k in expanded_negative]
    guard_match = [k for k in selected if k in expanded_guard]
    for options, description in ((neg_match, 'negated guard'),
                                 (guard_match, 'guard'),
                                 (selected, 'constructor name')):
        if len(options) == 1:
            print('[CONFIG] Auto-resolved {} via {}: {}'.format(stage, description, options[0]))
            return options[0]
        if len(options) > 1:
            # Don't guess when more than one flag could alter the architecture.
            break

    raise RuntimeError(
        'Cannot safely identify the {} Fusion switch. Candidates={}, guard={}, '
        'negated_guard={}. Inspect transfusion_head.py near `{}`, then pass '
        '--{}-flag ACTUAL_CONSTRUCTOR_ARGUMENT.'.format(
            stage, selected, sorted(guard_tokens), sorted(negated_tokens),
            _GUARDS[stage], stage))




_STAGE_WORDS = ('coarse', 'refined', 'local')


def _iter_scale_fields(tree, path=()):
    """Yield stage-related scale values, allowing {'scales': {'local': 0}}."""
    if isinstance(tree, Mapping):
        for key, val in tree.items():
            current = path + (str(key),)
            label = '.'.join(current).lower()
            if 'scale' in label and any(stage in label for stage in _STAGE_WORDS) \
                    and isinstance(val, Real) and not isinstance(val, bool):
                yield current, val
            if isinstance(val, (Mapping, list, tuple)):
                yield from _iter_scale_fields(val, current)
    elif isinstance(tree, (list, tuple)):
        for i, val in enumerate(tree):
            if isinstance(val, (Mapping, list, tuple)):
                yield from _iter_scale_fields(val, path + (str(i),))


def _get_config_path(cfg, path):
    obj = cfg
    for key in path:
        if isinstance(obj, (list, tuple)) and key.isdigit():
            obj = obj[int(key)]
        else:
            obj = obj[key]
    return obj


def _set_config_path(cfg, path, value):
    obj = cfg
    for key in path[:-1]:
        obj = obj[int(key)] if isinstance(obj, (list, tuple)) and key.isdigit() else obj[key]
    final = path[-1]
    if isinstance(obj, tuple):
        raise RuntimeError('Cannot modify tuple-valued scale field: ' + '.'.join(path))
    if isinstance(obj, list) and final.isdigit():
        obj[int(final)] = value
    else:
        obj[final] = value


def _repair_rrrf_scales(model_cfg, *, donor_model=None, replacement=None):
    """Resolve all *configured* zero Coarse/Refined/Local scales before build.

    Donor's nonzero scale is preferred; otherwise explicit replacement is used.
    No implicit guesses: absent/ambiguous fields fail with diagnostic output.
    """
    located = list(_iter_scale_fields(model_cfg))
    donor_paths = dict(_iter_scale_fields(donor_model)) if donor_model is not None else {}
    print('[SCALE] Found stage scale fields:',
          [(".".join(path), float(value)) for path, value in located])
    changes = []
    for path, old in located:
        if float(old) != 0.0:
            continue
        new = donor_paths.get(path)
        provenance = 'trained-config'
        if not isinstance(new, Real) or isinstance(new, bool) or float(new) == 0.0:
            new = replacement
            provenance = 'explicit CLI fallback'
        if new is None:
            raise RuntimeError(
                'Zero scale {} detected. Provide --trained-config ORIGINAL_PGFR_CONFIG '
                'or --repair-zero-scales 1.0 (fallback changes inference).'
                .format('.'.join(path)))
        if not isinstance(new, Real) or isinstance(new, bool) or float(new) == 0.0:
            raise ValueError('Replacement scale must be a nonzero number')
        _set_config_path(model_cfg, path, float(new))
        changes.append({'field': '.'.join(path), 'previous': float(old),
                        'value': float(new), 'source': provenance})
        print('[SCALE] {}: {} -> {} ({})'.format(
            '.'.join(path), old, new, provenance))
    return changes, located


def _configure_exact_local_scale(model_cfg, trained_model=None,
                                 local_scale=None, fallback=None):
    """Set the *actual* constructor field used by the Local scale guard.

    The previous heuristic looked for 'local' in the config-key name and
    missed rrrf_query_select_scale entirely. MMEngine config often omits this
    key, leaving a zero-valued constructor default.
    """
    key = 'rrrf_query_select_scale'
    head = model_cfg['bbox_head']
    original = head.get(key, None)
    print('[SCALE] Effective source bbox_head.{} = {!r}{}'.format(
        key, original, ' (not present; constructor default may be 0.0)'
        if original is None else ''))

    if original is not None:
        if not isinstance(original, Real) or isinstance(original, bool):
            raise TypeError('bbox_head.{} must be numeric, got {!r}'.format(key, original))
        if float(original) != 0.0:
            print('[SCALE] Local selector already has nonzero config scale: {}'.format(original))
            return []

    trained_value = None
    if trained_model is not None:
        trained_value = trained_model.get('bbox_head', {}).get(key, None)
    choices = (
        ('original PGFR training config', trained_value),
        ('--local-scale', local_scale),
        ('--repair-zero-scales (fallback)', fallback),
    )
    for source, candidate in choices:
        if candidate is None:
            continue
        if not isinstance(candidate, Real) or isinstance(candidate, bool):
            raise TypeError('{} must be numeric, got {!r}'.format(source, candidate))
        if not np.isfinite(float(candidate)) or float(candidate) == 0.0:
            # An original trained config may intentionally have Local disabled.
            # In that case a nonzero CLI override is necessary for this full P2.
            if source == 'original PGFR training config':
                print('[SCALE] PGFR training config has zero Local scale; ignoring it.')
                continue
            raise ValueError('{} must be a finite nonzero value, got {!r}'.format(
                source, candidate))
        value = float(candidate)
        head[key] = value
        print('[SCALE] Set bbox_head.{}: {!r} -> {} ({})'.format(
            key, original, value, source))
        return [{'field': 'bbox_head.' + key, 'previous': original,
                 'value': value, 'source': source}]

    raise RuntimeError(
        'Missing/nonzero Local selector scale: '
        'bbox_head.rrrf_query_select_scale. '
        'Pass --trained-config ACTUAL_PGFR_CONFIG.py or '
        '--local-scale 1.0 (explicit fallback; not verified as trained value).')


def _get_local_guard_excerpt(model_cls):
    """Show source around Local scale guard if it couldn't be resolved."""
    try:
        source, line = inspect.getsourcelines(model_cls._configure_rrrf_stage)
    except (AttributeError, OSError, TypeError):
        return '(unable to inspect _configure_rrrf_stage)'
    hits = [i for i, value in enumerate(source) if 'Local scale must be nonzero' in value]
    if not hits:
        return '(no literal Local scale guard found)'
    i = hits[0]
    return ''.join('{}: {}'.format(line + j, source[j])
                   for j in range(max(0, i-12), min(len(source), i+3)))


def load_state_dict(path):
    # Load only checkpoints whose source you trust (weights_only=False unpickles).
    obj = torch.load(str(path), map_location='cpu', weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError('Checkpoint is not a dict: {}'.format(path))
    state = obj.get('state_dict', obj)
    if not isinstance(state, dict):
        raise TypeError('Invalid state_dict in {}'.format(path))
    normalized = {}
    for name, value in state.items():
        if not isinstance(name, str):
            raise TypeError('Non-string state_dict key in {}'.format(path))
        key = name[7:] if name.startswith('module.') else name
        if key in normalized:
            raise ValueError('Duplicate normalized state_dict key: {}'.format(key))
        if not isinstance(value, torch.Tensor):
            raise TypeError('Expected tensor for key: {}'.format(key))
        normalized[key] = value
    return normalized



def _is_spconv_weight(model, key):
    """Conservatively accept only actual spconv 3D weight modules, not Conv3d."""
    if not key.startswith('pts_middle_encoder.') or not key.endswith('.weight'):
        return False
    module_path = key.rsplit('.', 1)[0]
    module = dict(model.named_modules()).get(module_path)
    if module is None:
        return False
    return type(module).__module__.startswith('spconv.') and getattr(module, 'weight', None) is not None


def _runtime_tensors(model):
    """Read actual model tensors, not state_dict hooks' serialized views."""
    tensors = dict(model.named_parameters())
    tensors.update(dict(model.named_buffers()))
    return tensors


def _assert_loaded_exactly(model, original_merged):
    """Physical tensors after strict load must exactly match both trained donors."""
    live = _runtime_tensors(model)
    checked = 0
    problems = []
    for key, canonical in original_merged.items():
        if key not in live:
            # Rare state_dict keys may be created by hooks; they were checked
            # by strict load. But sparse conv weights MUST be physical params.
            if key.startswith('pts_middle_encoder.') and key.endswith('.weight'):
                problems.append('MISSING physical sparse weight ' + key)
            continue
        current = live[key].detach().cpu()
        if not torch.equal(current, canonical):
            problems.append('LOADED TENSOR DIFFERS FROM DONOR {}: '
                            'live={} donor={}'.format(key, tuple(current.shape),
                                                      tuple(canonical.shape)))
        checked += 1
    if problems:
        raise RuntimeError('Post-load provenance check failed ({}):\n{}'.format(
            len(problems), '\n'.join(problems[:25])))
    print('[PASS] Physical model parameters/buffers match donor values: {}'.format(checked))


def strict_load_with_spconv_fallback(model, merged):
    """Return checkpoint layout accepted by THIS runtime, never guessed silently.

    In the affected mmdet3d fork, `_load_from_state_dict` rotates checkpoint
    sparse kernels with permute(4,0,1,2,3). Consequently, a donor tensor in
    physical KRSC shape [K,R,S,T,C] is incorrectly rotated on strict load,
    yielding e.g. [C,K,R,S,T]. An inverse pre-permutation of the on-disk
    tensor:  [K,R,S,T,C] -> [R,S,T,C,K] via (1,2,3,4,0), is needed for
    this loader to yield the original [K,R,S,T,C] tensor.

    Guardrails:
      - Trigger ONLY on strict errors confined to actual spconv 5D weights.
      - Never touch any LiDAR dense layers, RRRF, Official, or calibration.
      - Check loaded physical parameter values against ORIGINAL PGFR/Official.
      - Return only after full strict loading and exact equality succeeds.
    """
    import re
    try:
        model.load_state_dict(merged, strict=True)
    except RuntimeError as exc:
        message = str(exc)
        mismatch_keys = re.findall(r'size mismatch for ([^:\s]+):', message)
        unique_keys = sorted(set(mismatch_keys))
        if not unique_keys:
            raise RuntimeError('Strict load failed for a non-shape reason; '
                               'spconv conversion is not applicable.\n' + message) from exc
        # Validate every listed failure is a sparse 3D convolution, to prevent
        # a false fix for architecture mismatch or accidental key typo.
        non_sparse = [k for k in unique_keys
                      if k not in merged or merged[k].ndim != 5 or
                      not _is_spconv_weight(model, k)]
        if non_sparse:
            raise RuntimeError('Strict load contains non-spconv mismatches: {}. '
                               'Automatic conversion REFUSED.\n{}'.format(
                                   non_sparse[:15], message)) from exc

        # This failure is only caused by a *known* checkpoint loader problem if
        # the error presents a rotation (last dim -> first). Validate every
        # observed pair with the actual live parameter shape.
        live = _runtime_tensors(model)
        candidate = dict(merged)
        for key in unique_keys:
            original = merged[key]
            if key not in live:
                raise RuntimeError('Cannot identify runtime parameter: ' + key) from exc
            physical_shape = tuple(live[key].shape)
            if tuple(original.shape) != physical_shape:
                raise RuntimeError('Source tensor for {} has shape {}; actual '
                                   'model parameter has shape {}. Not the known '
                                   'spconv loader issue; refusing conversion.'.format(
                                       key, tuple(original.shape), physical_shape)) from exc
            # See stack trace: checkpoint [16,3,3,3,5] wrongly appears inside
            # _load_from_state_dict as [5,16,3,3,3]. The pre-converted saved
            # tensor should be [3,3,3,5,16], so loader rotation restores it.
            save_layout = original.permute(1, 2, 3, 4, 0).contiguous()
            reconstructed = save_layout.permute(4, 0, 1, 2, 3).contiguous()
            if not torch.equal(reconstructed, original):
                raise AssertionError('Round-trip sparse layout error: ' + key)
            candidate[key] = save_layout
            print('[SPCONV] {}: PGFR physical {} -> checkpoint {} '
                  '(loader rotates last axis to first)'.format(
                      key, tuple(original.shape), tuple(save_layout.shape)))

        print('[SPCONV] Retry strict load with {} inverse-permuted sparse tensors.'
              .format(len(unique_keys)))
        try:
            model.load_state_dict(candidate, strict=True)
        except RuntimeError as retry_exc:
            raise RuntimeError('Sparse loader-layout conversion failed in THIS '
                               'environment. No P2 checkpoint saved. Inspect '
                               'the installed spconv/mmdet3d load hooks; DO NOT '
                               'use strict=False.\n' + str(retry_exc)) from retry_exc
        _assert_loaded_exactly(model, merged)
        return candidate, unique_keys, 'inverse-permutation-for-installed-loader'

    _assert_loaded_exactly(model, merged)
    print('[SPCONV] Native strict load accepted: no sparse conversion needed.')
    return merged, [], 'native'

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pgfr', required=True, help='PGFR_Det_final_model.pth')
    ap.add_argument('--official', required=True, help='Official BEVFusion checkpoint')
    ap.add_argument('--config', required=True,
                    help='P2 feature_refine config (Refined/Local may be disabled)')
    ap.add_argument('--selector-mode', choices=('auto', 'require'), default='auto',
                    help='auto: activate Local Proposal in an effective P2 config; '
                         'require: fail if input config disables it (default: auto)')
    ap.add_argument('--coarse-flag', default=None,
                    help='optional exact TransFusionHead __init__ argument controlling '
                         'Coarse Fusion (auto-discovered by default)')
    ap.add_argument('--refined-flag', default=None,
                    help='optional exact TransFusionHead __init__ argument controlling '
                         'Refined Fusion (auto-discovered by default)')
    ap.add_argument('--config-out', default=None,
                    help='path to save effective P2 evaluation config when selector is enabled '
                         '(default: <output-stem>_eval_config.py)')
    ap.add_argument('--trained-config', default=None,
                    help='Original full PGFR model config, to copy authentic nonzero stage scales')
    ap.add_argument('--local-scale', type=float, default=None,
                    help='explicit nonzero bbox_head.rrrf_query_select_scale, '
                         'e.g. 1.0. Does not establish original trained scale')
    ap.add_argument('--repair-zero-scales', type=float, default=None,
                    help='Explicit nonzero fallback for disabled stage scales; e.g. 1.0. '
                         'Not guaranteed to reproduce original PGFR inference')
    ap.add_argument('--output', required=True,
                    help='e.g. data/weights/ablation_v2/PGFR_Ablation_P2.pth')
    ap.add_argument('--seed', type=int, default=20261106,
                    help='model-construction seed only; NO parameters saved from initialization')
    ap.add_argument('--p0', default=None,
                    help='optional P0 checkpoint: report tensor differences for RRRF modules')
    ap.add_argument('--expected-official-tensors', type=int, default=None,
                    help='optional expected Official Decoder tensor count; '
                         'by default check automatically against model Decoder keys')
    args = ap.parse_args()

    from mmengine.config import Config
    from mmdet3d.registry import MODELS
    from mmdet3d.utils import register_all_modules
    import projects.BEVFusion.bevfusion  # noqa: F401; registers project components

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    register_all_modules(init_default_scope=True)

    config = Config.fromfile(args.config)
    head_config = config.model.get('bbox_head', {})
    if head_config.get('rrrf_mode') != 'feature_refine':
        raise ValueError('P2 config must have rrrf_mode=feature_refine')
    if head_config.get('rrrf_adapter_enabled', False):
        raise ValueError('P2 config must disable the residual feature adapter')

    # Enable Coarse -> Refined -> Local together BEFORE MODELS.build().
    # Enabling only Local or only Refined triggers the two dependency guards.
    head_type = head_config.get('type')
    head_cls = MODELS.get(head_type)
    if head_cls is None:
        raise RuntimeError('bbox_head type {!r} is not registered'.format(head_type))

    coarse_flag = _discover_stage_flag(head_cls, head_config, 'coarse', args.coarse_flag)
    refined_flag = _discover_stage_flag(head_cls, head_config, 'refined', args.refined_flag)
    if coarse_flag == refined_flag or coarse_flag == 'rrrf_query_select_enabled' \
            or refined_flag == 'rrrf_query_select_enabled':
        raise RuntimeError('Coarse/Refined/Local flags must be distinct: '
                           'coarse={!r}, refined={!r}'.format(coarse_flag, refined_flag))

    coarse_was_enabled = bool(head_config.get(coarse_flag, False))
    refined_was_enabled = bool(head_config.get(refined_flag, False))
    selector_was_enabled = bool(head_config.get('rrrf_query_select_enabled', False))
    if args.selector_mode == 'require' and not selector_was_enabled:
        raise ValueError('Input config disables rrrf_query_select_enabled; '
                         'use --selector-mode auto (default)')
    head_config[coarse_flag] = True
    head_config[refined_flag] = True
    head_config['rrrf_query_select_enabled'] = True
    print('[CONFIG] Source: {}={}, {}={}, Local={}'.format(
        coarse_flag, coarse_was_enabled, refined_flag, refined_was_enabled,
        selector_was_enabled))
    print('[CONFIG] Effective P2: Coarse=True, Refined=True, Local=True')
    print('[CONFIG] Original config is NOT changed on disk.')

    # Ablation A typically has stage scale = 0 as well as stage enable = False.
    # Enabling stages without fixing those values causes
    #   RuntimeError: [ABLATION] Local scale must be nonzero
    # Copy original trained PGFR scale values where possible. Otherwise a
    # nonzero fallback requires explicit opt-in, and is logged in checkpoint meta.
    model_cls = MODELS.get(config.model.get('type'))
    trained_cfg = Config.fromfile(args.trained_config) if args.trained_config else None
    trained_model = trained_cfg.model if trained_cfg is not None else None
    scale_changes, scale_fields = _repair_rrrf_scales(
        config.model, donor_model=trained_model,
        replacement=args.repair_zero_scales)
    # Required by bevfusion.py::_configure_rrrf_stage():
    #   head.rrrf_query_select_enabled and head.rrrf_query_select_scale == 0.0
    # A 'local' keyword search misses rrrf_query_select_scale.
    scale_changes.extend(_configure_exact_local_scale(
        config.model, trained_model=trained_model,
        local_scale=args.local_scale,
        fallback=args.repair_zero_scales))
    print('[SCALE] Effective bbox_head.rrrf_query_select_scale =',
          config.model['bbox_head']['rrrf_query_select_scale'])

    # Fail early if PGFR checkpoint does not contain trained-source RRRF keys.
    print('[CHECK] Inspecting PGFR RRRF coverage BEFORE model construction')
    pgfr_sd = load_state_dict(args.pgfr)
    for prefix in ('bbox_head.fusion_cross_attention.',
                   'bbox_head.refined_attention.',
                   'bbox_head.rrrf_query_selector.'):
        n = sum(key.startswith(prefix) for key in pgfr_sd)
        print('[CHECK] PGFR {} -> {} tensors'.format(prefix, n))
        if n == 0:
            raise RuntimeError('PGFR checkpoint has no {} tensors. A trained PGFR '
                               'Local/Refined/Coarse P2 cannot be created by '
                               'tensor merging without a trained source.'
                               .format(prefix))

    print('[BUILD] Constructing target architecture on CPU')
    try:
        model = MODELS.build(config.model)
    except (ValueError, RuntimeError) as exc:
        print('[DIAGNOSTIC] Model config RRRF stage scales:',
              [(".".join(p), v) for p, v in _iter_scale_fields(config.model)])
        if model_cls:
            print('[DIAGNOSTIC] Local guard:\n' + _get_local_guard_excerpt(model_cls))
        raise RuntimeError('P2 architecture construction failed. See the '
                           'scale settings and source guard printed above: '
                           + str(exc)) from exc
    target_sd = model.state_dict()
    official_sd = load_state_dict(args.official)

    problems = []
    for prefix in REQUIRED_PREFIXES:
        if not any(key.startswith(prefix) for key in target_sd):
            problems.append('MISSING TARGET MODEL PREFIX: ' + prefix)

    merged = {}
    owners = {}
    counts = Counter()
    rrrf_counts = Counter()
    for key, expected in target_sd.items():
        origin = 'Official' if key.startswith(OFFICIAL_PREFIXES) else 'PGFR'
        donor = official_sd if origin == 'Official' else pgfr_sd

        if key not in donor:
            problems.append('MISSING [{}] {}'.format(origin, key))
            continue
        value = donor[key]
        if expected.shape != value.shape:
            problems.append('SHAPE [{}] {} target={} source={}'.format(
                origin, key, tuple(expected.shape), tuple(value.shape)))
            continue
        if expected.dtype != value.dtype:
            problems.append('DTYPE [{}] {} target={} source={}'.format(
                origin, key, expected.dtype, value.dtype))
            continue
        merged[key] = value.detach().cpu().clone()
        owners[key] = origin
        counts[origin] += 1
        if key.startswith(PGFR_RRRF_PREFIXES):
            rrrf_counts[origin] += 1

    for key in official_sd:
        if key.startswith(OFFICIAL_PREFIXES) and key not in target_sd:
            problems.append('EXTRA Official Decoder key: ' + key)

    # P2: do not retain P1's hard-coded Decoder+Head total (84).
    # Require full Official Decoder coverage, and full PGFR Head coverage.
    decoder_keys = [key for key in target_sd
                    if key.startswith('bbox_head.decoder.')]
    head_keys = [key for key in target_sd
                 if key.startswith('bbox_head.prediction_heads.')]
    if not decoder_keys or not head_keys:
        problems.append('P2 requires both Decoder and Prediction Head in model')
    if counts['Official'] != len(decoder_keys):
        problems.append('Official Decoder coverage: copied {} / {}'.format(
            counts['Official'], len(decoder_keys)))
    if any(owners.get(k) != 'Official' for k in decoder_keys):
        problems.append('P2 Decoder is not fully sourced from Official')
    if any(owners.get(k) != 'PGFR' for k in head_keys):
        problems.append('P2 Prediction Heads are not fully sourced from PGFR')
    if args.expected_official_tensors is not None and (
            counts['Official'] != args.expected_official_tensors):
        problems.append('Expected {} Official Decoder tensors, got {}'.format(
            args.expected_official_tensors, counts['Official']))
    print('[P2] Official Decoder tensors: {} / {}'.format(
        counts['Official'], len(decoder_keys)))
    print('[P2] PGFR Prediction Head tensors: {} / {}'.format(
        sum(owners.get(k) == 'PGFR' for k in head_keys), len(head_keys)))

    if rrrf_counts['PGFR'] == 0:
        problems.append('No RRRF/projector tensors copied from PGFR')
    # A full P2 must include the local selector. Checking actual model keys
    # prevents a config flag that is ignored by an incompatible implementation.
    local_keys = [k for k in target_sd
                  if k.startswith('bbox_head.rrrf_query_selector.')]
    if not local_keys:
        problems.append('Local selector enabled in config but model has no '
                        'bbox_head.rrrf_query_selector.* state_dict keys')
    elif not all(owners.get(k) == 'PGFR' for k in local_keys):
        problems.append('Some Local selector tensors were not copied from PGFR')
    if rrrf_counts['Official'] != 0:
        problems.append('RRRF tensors must not come from Official')
    if len(merged) != len(target_sd):
        problems.append('Incomplete target coverage: copied {} / {}'.format(
            len(merged), len(target_sd)))

    if problems:
        print('\n'.join('[ERROR] ' + p for p in problems[:120]))
        print('[FAIL] P2 NOT saved ({} problems).'.format(len(problems)))
        print('[HINT] PGFR_Det_final_model.pth must contain every RRRF tensor '
              'required by this target architecture, including the Local selector.')
        print('[HINT] If the selector is new and was not trained in PGFR, '
              'a fully trained PGFR-sourced Local P2 cannot be built by merging.')
        raise RuntimeError('P2 build failed: incompatible/missing checkpoint tensors')

    # Verify provenance of EVERY copied model tensor, not only an RRRF subset.
    for key, origin in owners.items():
        src = official_sd if origin == 'Official' else pgfr_sd
        if not torch.equal(merged[key], src[key]):
            raise AssertionError('Provenance mismatch for {}: {}'.format(origin, key))

    # Strict-load the checkpoint as tools/test.py will. Some custom mmdet3d
    # spconv loaders rotate sparse kernels even though model.state_dict and
    # donor checkpoint shapes are equal. Save in that loader's expected input
    # layout ONLY after exact physical tensor equality has been confirmed.
    checkpoint_sd, converted_spconv_keys, spconv_layout = strict_load_with_spconv_fallback(
        model, merged)
    del model

    if args.p0:
        p0_sd = load_state_dict(args.p0)
        matched, changed, identical, missing = 0, 0, 0, 0
        for key, tensor in merged.items():
            if not key.startswith(PGFR_RRRF_PREFIXES):
                continue
            if key not in p0_sd:
                missing += 1
                continue
            matched += 1
            if torch.equal(tensor, p0_sd[key]):
                identical += 1
            else:
                changed += 1
        print('[COMPARE P0] RRRF/projector keys present in both: {}'.format(matched))
        print('[COMPARE P0] changed={} identical={} missing_in_P0={}'.format(
            changed, identical, missing))
        print('[NOTE] Difference from P0 does not by itself prove RRRF was trained.')

    output = Path(args.output)
    config_out = (Path(args.config_out) if args.config_out else
                  output.with_name(output.stem + '_eval_config.py'))
    if config_out.resolve() == Path(args.config).resolve():
        raise ValueError('Refusing to overwrite source --config. '
                         'Choose another --config-out path.')
    if config_out.suffix != '.py':
        raise ValueError('--config-out must end in .py')
    output.parent.mkdir(parents=True, exist_ok=True)
    config_out.parent.mkdir(parents=True, exist_ok=True)

    # Store the EXACT model config used to construct P2. Testing with the
    # original ablation_A_eval.py (selector=False) would not evaluate the same
    # architecture. MMEngine writes a flattened standalone Python config.
    config.dump(str(config_out))
    print('[PASS] Effective P2 evaluation config saved to:', config_out)

    checkpoint = {
        'meta': {
            'purpose': 'RRRF A-E common P2 with PGFR RRRF weights; no training',
            'pgfr_source': str(Path(args.pgfr).resolve()),
            'official_source': str(Path(args.official).resolve()),
            'source_model_config': str(Path(args.config).resolve()),
            'effective_model_config': str(config_out.resolve()),
            'local_selector_enabled_in_source_config': selector_was_enabled,
            'coarse_flag': coarse_flag,
            'coarse_enabled_in_source_config': coarse_was_enabled,
            'coarse_enabled_for_P2': True,
            'refined_flag': refined_flag,
            'refined_enabled_in_source_config': refined_was_enabled,
            'refined_enabled_for_P2': True,
            'local_selector_enabled_for_P2': True,
            'seed_for_model_construction_only': args.seed,
            'source_tensor_counts': dict(counts),
            'rrrf_tensor_counts': dict(rrrf_counts),
            'official_prefixes': OFFICIAL_PREFIXES,
            'pgfr_rrrf_prefixes': PGFR_RRRF_PREFIXES,
            'fresh_tensor_count': 0,
            'scale_changes': scale_changes,
            'trained_config_for_scales': str(Path(args.trained_config).resolve()) if args.trained_config else None,
            'scale_fallback': args.repair_zero_scales,
            'explicit_local_scale': args.local_scale,
            'strict_loading_validated': True,
            'spconv_loader_layout': spconv_layout,
            'spconv_converted_keys': converted_spconv_keys,
            'spconv_weight_value_provenance': 'PGFR exactly, after runtime loading',
            'checkpoint_runtime_note': 'Use the same mmdet3d/spconv loading implementation',
        },
        'state_dict': checkpoint_sd,
    }
    torch.save(checkpoint, str(output))

    print('[PASS] P2 saved to:', output)
    if any(c['source'] in ('explicit CLI fallback', '--local-scale',
                                 '--repair-zero-scales (fallback)') for c in scale_changes):
        print('[WARNING] P2 stage scales used manual fallback. This is NOT proven to match PGFR training/inference.')
    print('[PASS] Tensor sources:', dict(counts))
    print('[PASS] RRRF/projector tensor sources:', dict(rrrf_counts))
    print('[PASS] Fresh tensors: 0')
    print('[PASS] spconv checkpoint layout: {} ({} converted keys)'.format(
        spconv_layout, len(converted_spconv_keys)))
    print('[PASS] All {} model keys compatible and copied without missing keys.'.format(
        len(merged)))
    print('[NEXT] TEST with this config, NOT the original ablation_A_eval.py:')
    print('       python tools/test.py {} {}'.format(config_out, output))
    print('[NOTE] The provenance check proves tensor copying, not training history.')


if __name__ == '__main__':
    main()