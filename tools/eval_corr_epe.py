import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint
from mmengine.utils import import_modules_from_strings

from mmdet3d.registry import MODELS
from mmdet3d.utils import register_all_modules


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            'Evaluate CorrNet correspondence EPE '
            'using BEVFusion.loss() without training.'
        )
    )

    parser.add_argument(
        'config',
        type=str,
    )

    parser.add_argument(
        'work_dir',
        type=str,
        help=(
            'Training work directory containing '
            'last_checkpoint.'
        ),
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help=(
            'Optional explicit checkpoint. '
            'If omitted, last_checkpoint is used.'
        ),
    )

    parser.add_argument(
        '--max-samples',
        type=int,
        default=200,
        help=(
            'Number of validation samples. '
            'Use 0 for full validation set.'
        ),
    )

    parser.add_argument(
        '--print-every',
        type=int,
        default=20,
    )

    parser.add_argument(
        '--out',
        type=str,
        default='corr_epe_eval.csv',
    )

    return parser.parse_args()


def resolve_checkpoint(
    work_dir,
    explicit_checkpoint=None,
):

    if explicit_checkpoint is not None:

        p = Path(
            explicit_checkpoint
        )

        if not p.exists():

            raise FileNotFoundError(
                f'Checkpoint not found: {p}'
            )

        return p.resolve()


    work_dir = Path(
        work_dir
    )

    last_file = (
        work_dir
        / 'last_checkpoint'
    )

    if not last_file.exists():

        raise FileNotFoundError(
            f'last_checkpoint not found: '
            f'{last_file}'
        )


    text = (
        last_file
        .read_text()
        .strip()
    )

    p = Path(text)


    if p.is_absolute() and p.exists():

        return p


    # MMEngine may write a path relative to cwd.
    if p.exists():

        return p.resolve()


    # Fallback:
    # use checkpoint basename inside work_dir.
    candidate = (
        work_dir
        / p.name
    )

    if candidate.exists():

        return candidate.resolve()


    raise FileNotFoundError(
        'Could not resolve checkpoint from '
        f'{last_file}: {text}'
    )


def tensor_to_float(x):

    if torch.is_tensor(x):

        return float(
            x
            .detach()
            .cpu()
            .item()
        )

    return float(x)


def main():

    args = parse_args()


    # ============================================================
    # 1. Load configuration
    # ============================================================

    cfg = Config.fromfile(
        args.config
    )


    if cfg.get(
        'custom_imports',
        None,
    ) is not None:

        import_modules_from_strings(
            **cfg.custom_imports
        )


    # Explicit project import:
    # registers BEVFusion/COTR/custom modules.
    import projects.BEVFusion.bevfusion  # noqa: F401


    register_all_modules(
        init_default_scope=True
    )


    # ============================================================
    # 2. Safety Gate
    # ============================================================

    stage = (
        cfg.model.get(
            'lgpc_train_stage',
            None,
        )
    )

    rrrf_mode = (
        cfg.model[
            'bbox_head'
        ].get(
            'rrrf_mode',
            None,
        )
    )

    enable_cycle = (
        cfg.model[
            'corr'
        ].get(
            'enable_cycle',
            None,
        )
    )


    print()
    print(
        '============================================'
    )
    print(
        '[CORR EPE CONFIG GATE]'
    )
    print(
        '============================================'
    )
    print(
        'lgpc_train_stage =',
        stage,
    )
    print(
        'rrrf_mode        =',
        rrrf_mode,
    )
    print(
        'enable_cycle     =',
        enable_cycle,
    )
    print(
        '============================================'
    )


    if stage != 'corr':

        raise RuntimeError(
            'Expected lgpc_train_stage="corr", '
            f'got {stage}'
        )


    if rrrf_mode != 'lgpc_only':

        raise RuntimeError(
            'Expected rrrf_mode="lgpc_only", '
            f'got {rrrf_mode}'
        )


    # ============================================================
    # 3. Resolve latest checkpoint
    # ============================================================

    checkpoint_path = (
        resolve_checkpoint(
            args.work_dir,
            args.checkpoint,
        )
    )


    print()
    print(
        '[CHECKPOINT]'
    )
    print(
        checkpoint_path
    )


    # ============================================================
    # 4. Build model
    # ============================================================

    model = MODELS.build(
        cfg.model
    )


    load_checkpoint(
        model,
        str(checkpoint_path),
        map_location='cpu',
        strict=False,
    )


    device = torch.device(
        'cuda:0'
        if torch.cuda.is_available()
        else 'cpu'
    )


    model.to(device)

    # Important:
    # evaluation mode + no gradient.
    model.eval()


    # ============================================================
    # 5. Build deterministic O-3 validation dataloader
    # ============================================================

    val_loader = (
        Runner.build_dataloader(
            cfg.val_dataloader,
            seed=0,
        )
    )


    print()
    print(
        '[VAL DATASET]'
    )
    print(
        'num_samples =',
        len(
            val_loader.dataset
        ),
    )


    # ============================================================
    # 6. Accumulators
    # ============================================================

    rows = []

    total_valid = 0.0

    weighted_corr_sum = 0.0
    weighted_identity_sum = 0.0

    sample_corr_values = []
    sample_identity_values = []
    sample_valid_ratios = []


    # ============================================================
    # 7. Evaluation
    # ============================================================

    with torch.no_grad():

        for idx, data_batch in enumerate(
            val_loader
        ):

            if (
                args.max_samples > 0
                and idx >= args.max_samples
            ):

                break


            # ----------------------------------------
            # MMDetection3D preprocessing
            # ----------------------------------------

            processed = (
                model.data_preprocessor(
                    data_batch,
                    training=False,
                )
            )


            batch_inputs = (
                processed[
                    'inputs'
                ]
            )

            batch_samples = (
                processed[
                    'data_samples'
                ]
            )


            # ----------------------------------------
            # IMPORTANT:
            #
            # Do NOT call predict().
            #
            # Corr EPE diagnostics live in loss().
            # No gradients are generated because
            # torch.no_grad() is active.
            # ----------------------------------------

            metrics = model.loss(
                batch_inputs,
                batch_samples,
            )


            required = [
                'corr_epe_px',
                'identity_epe_px',
                'corr_recovery',
                'corr_valid_ratio',
            ]


            for key in required:

                if key not in metrics:

                    raise RuntimeError(
                        f'Missing diagnostic '
                        f'"{key}" in model.loss().\n'
                        'Check that updated '
                        'bevfusion.py is being used.'
                    )


            corr_epe = tensor_to_float(
                metrics[
                    'corr_epe_px'
                ]
            )

            identity_epe = (
                tensor_to_float(
                    metrics[
                        'identity_epe_px'
                    ]
                )
            )

            recovery = (
                tensor_to_float(
                    metrics[
                        'corr_recovery'
                    ]
                )
            )

            valid_ratio = (
                tensor_to_float(
                    metrics[
                        'corr_valid_ratio'
                    ]
                )
            )


            # ----------------------------------------
            # Prefer exact valid count when available.
            # ----------------------------------------

            if (
                'corr_valid_count'
                in metrics
            ):

                valid_count = (
                    tensor_to_float(
                        metrics[
                            'corr_valid_count'
                        ]
                    )
                )

            else:

                # Fallback only.
                valid_count = 1.0


            total_valid += (
                valid_count
            )


            weighted_corr_sum += (
                corr_epe
                * valid_count
            )

            weighted_identity_sum += (
                identity_epe
                * valid_count
            )


            sample_corr_values.append(
                corr_epe
            )

            sample_identity_values.append(
                identity_epe
            )

            sample_valid_ratios.append(
                valid_ratio
            )


            rows.append({
                'sample_index':
                    idx,

                'identity_epe_px':
                    identity_epe,

                'corr_epe_px':
                    corr_epe,

                'corr_recovery':
                    recovery,

                'corr_valid_ratio':
                    valid_ratio,

                'corr_valid_count':
                    valid_count,
            })


            if (
                (idx + 1)
                % args.print_every
                == 0
            ):

                current_corr = (
                    weighted_corr_sum
                    / max(
                        total_valid,
                        1.0,
                    )
                )

                current_identity = (
                    weighted_identity_sum
                    / max(
                        total_valid,
                        1.0,
                    )
                )

                current_recovery = (
                    1.0
                    - current_corr
                    / max(
                        current_identity,
                        1e-6,
                    )
                )


                print(
                    f'[EVAL] '
                    f'{idx + 1:5d} '
                    f'identity='
                    f'{current_identity:8.2f}px '
                    f'corr='
                    f'{current_corr:8.2f}px '
                    f'recovery='
                    f'{100.0 * current_recovery:7.2f}%'
                )


    # ============================================================
    # 8. Final statistics
    # ============================================================

    if len(rows) == 0:

        raise RuntimeError(
            'No validation samples evaluated.'
        )


    global_corr = (
        weighted_corr_sum
        / max(
            total_valid,
            1.0,
        )
    )

    global_identity = (
        weighted_identity_sum
        / max(
            total_valid,
            1.0,
        )
    )

    global_recovery = (
        1.0
        - global_corr
        / max(
            global_identity,
            1e-6,
        )
    )


    corr_np = np.asarray(
        sample_corr_values,
        dtype=np.float64,
    )

    identity_np = np.asarray(
        sample_identity_values,
        dtype=np.float64,
    )

    valid_np = np.asarray(
        sample_valid_ratios,
        dtype=np.float64,
    )


    print()
    print(
        '========================================================'
    )
    print(
        '[CORR PHYSICAL EPE SUMMARY]'
    )
    print(
        '========================================================'
    )

    print(
        f'checkpoint        : '
        f'{checkpoint_path}'
    )

    print(
        f'samples           : '
        f'{len(rows)}'
    )

    print(
        f'valid corr count  : '
        f'{total_valid:.0f}'
    )

    print(
        f'identity EPE      : '
        f'{global_identity:.3f} px'
    )

    print(
        f'CorrNet EPE       : '
        f'{global_corr:.3f} px'
    )

    print(
        f'Corr recovery     : '
        f'{global_recovery * 100.0:.2f} %'
    )

    print(
        f'mean valid ratio  : '
        f'{valid_np.mean():.4f}'
    )

    print(
        f'Corr sample median: '
        f'{np.median(corr_np):.3f} px'
    )

    print(
        f'Corr sample P90   : '
        f'{np.percentile(corr_np, 90):.3f} px'
    )

    print(
        f'Identity median   : '
        f'{np.median(identity_np):.3f} px'
    )

    print(
        '========================================================'
    )


    # ============================================================
    # 9. Save per-sample CSV
    # ============================================================

    out_path = Path(
        args.out
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    with out_path.open(
        'w',
        newline='',
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                'sample_index',
                'identity_epe_px',
                'corr_epe_px',
                'corr_recovery',
                'corr_valid_ratio',
                'corr_valid_count',
            ],
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


    print()
    print(
        '[CSV SAVED]'
    )
    print(
        out_path.resolve()
    )


if __name__ == '__main__':

    main()