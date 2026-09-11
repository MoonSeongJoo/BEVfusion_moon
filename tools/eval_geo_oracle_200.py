import argparse
import copy
import csv

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
            'Deterministic 200-frame '
            'Geometric Oracle evaluation'
        )
    )

    parser.add_argument(
        'config',
        help='Model config'
    )

    parser.add_argument(
        'checkpoint',
        help='Checkpoint'
    )

    parser.add_argument(
        '--num-samples',
        type=int,
        default=200,
    )

    parser.add_argument(
        '--csv',
        type=str,
        default=None,
    )

    return parser.parse_args()


def print_case(
    name,
    mae,
    l2,
    valid,
    residual,
):

    print(
        f'\n[{name}]'
    )

    print(
        f'MAE mean       = '
        f'{np.mean(mae):.6f} m'
    )

    print(
        f'MAE median     = '
        f'{np.median(mae):.6f} m'
    )

    print(
        f'MAE P90        = '
        f'{np.quantile(mae, 0.90):.6f} m'
    )

    print(
        f'L2 mean        = '
        f'{np.mean(l2):.6f} m'
    )

    print(
        f'L2 median      = '
        f'{np.median(l2):.6f} m'
    )

    print(
        f'L2 P90         = '
        f'{np.quantile(l2, 0.90):.6f} m'
    )

    print(
        f'Valid mean     = '
        f'{np.mean(valid):.2f}'
    )

    print(
        f'Residual mean  = '
        f'{np.mean(residual):.6f}'
    )


def main():

    args = parse_args()

    # ============================================================
    # 1. Register modules
    # ============================================================

    register_all_modules(
        init_default_scope=True
    )

    cfg = Config.fromfile(
        args.config
    )

    if cfg.get(
        'custom_imports',
        None
    ) is not None:

        import_modules_from_strings(
            **cfg.custom_imports
        )


    # ============================================================
    # 2. IMPORTANT:
    # Use the EXACT model architecture from the Corr-R1 config,
    # but force inference mode to geometric oracle.
    #
    # Do NOT use a different old config because CalibHead / other
    # module dimensions may differ from the checkpoint.
    # ============================================================

    cfg.model[
        'calibration_mode'
    ] = 'geo_oracle_gtrot'


    # ============================================================
    # 3. Use same deterministic validation loader as Corr-200
    # ============================================================

    dataloader_cfg = copy.deepcopy(
        cfg.val_dataloader
    )

    dataloader_cfg[
        'batch_size'
    ] = 1

    dataloader_cfg[
        'sampler'
    ][
        'shuffle'
    ] = False


    val_loader = Runner.build_dataloader(
        dataloader_cfg,
        seed=20260811,
    )


    # ============================================================
    # 4. Build exact model
    # ============================================================

    model = MODELS.build(
        cfg.model
    )


    # ============================================================
    # 5. Load requested checkpoint
    # ============================================================

    print(
        '\n'
        '====================================================\n'
        '[GEO ORACLE 200 EVAL]\n'
        f'checkpoint  = {args.checkpoint}\n'
        f'num_samples = {args.num_samples}\n'
        '====================================================\n'
    )


    load_checkpoint(
        model,
        args.checkpoint,
        map_location='cpu',
        strict=False,
    )


    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )

    model.to(
        device
    )

    model.eval()


    # ============================================================
    # 6. Determinism
    # ============================================================

    torch.manual_seed(
        20260811
    )

    np.random.seed(
        20260811
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            20260811
        )

        torch.backends.cudnn.benchmark = False

        torch.backends.cudnn.deterministic = True


    # ============================================================
    # 7. Result containers
    # ============================================================

    cases = [
        'A',
        'B',
        'C',
        'D',
        'D_common',
    ]

    results = {}

    for case in cases:

        results[case] = {
            'mae': [],
            'l2': [],
            'valid': [],
            'residual': [],
        }


    rows = []


    # ============================================================
    # 8. Run first 200 deterministic frames
    # ============================================================

    with torch.no_grad():

        for frame_idx, data_batch in enumerate(
            val_loader
        ):

            if frame_idx >= args.num_samples:
                break


            # BaseModel.test_step() runs:
            #
            # data_preprocessor
            #        ↓
            # predict()
            #
            # calibration_mode was forced to geo_oracle_gtrot.
            outputs = model.test_step(
                data_batch
            )


            if len(outputs) != 1:

                raise RuntimeError(
                    'Expected batch_size=1, '
                    f'got {len(outputs)} outputs.'
                )


            sample = outputs[0]

            mi = sample.metainfo


            # ----------------------------------------------------
            # Read metainfo produced by geo_oracle_summary
            # ----------------------------------------------------

            row = {
                'frame': frame_idx,
            }


            for case in cases:

                prefix = (
                    f'geo_{case}'
                )

                mae_key = (
                    prefix
                    + '_mae_m'
                )

                l2_key = (
                    prefix
                    + '_l2_m'
                )

                valid_key = (
                    prefix
                    + '_valid_count'
                )

                residual_key = (
                    prefix
                    + '_residual_rmse'
                )


                required = [
                    mae_key,
                    l2_key,
                    valid_key,
                    residual_key,
                ]

                for k in required:

                    if k not in mi:

                        raise RuntimeError(
                            f'Missing metainfo key: {k}'
                        )


                mae = float(
                    mi[mae_key]
                )

                l2 = float(
                    mi[l2_key]
                )

                valid = float(
                    mi[valid_key]
                )

                residual = float(
                    mi[residual_key]
                )


                results[case][
                    'mae'
                ].append(mae)

                results[case][
                    'l2'
                ].append(l2)

                results[case][
                    'valid'
                ].append(valid)

                results[case][
                    'residual'
                ].append(residual)


                row[
                    f'{case}_mae'
                ] = mae

                row[
                    f'{case}_l2'
                ] = l2

                row[
                    f'{case}_valid'
                ] = valid

                row[
                    f'{case}_residual'
                ] = residual


            rows.append(
                row
            )


            if (
                (frame_idx + 1)
                % 20
                == 0
            ):

                print(
                    '[GEO-200] '
                    f'frames='
                    f'{frame_idx + 1}'
                )


    # ============================================================
    # 9. Final aggregated result
    # ============================================================

    print(
        '\n'
        '====================================================\n'
        '[GEO ORACLE DETERMINISTIC 200-VAL RESULT]\n'
        '===================================================='
    )

    print(
        f'Frames = {len(rows)}'
    )


    for case in cases:

        print_case(
            case,
            results[case]['mae'],
            results[case]['l2'],
            results[case]['valid'],
            results[case]['residual'],
        )


    print(
        '\n'
        '===================================================='
    )


    # ============================================================
    # 10. Optional CSV
    # ============================================================

    if (
        args.csv is not None
        and len(rows) > 0
    ):

        with open(
            args.csv,
            'w',
            newline='',
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=
                    list(
                        rows[0].keys()
                    ),
            )

            writer.writeheader()

            writer.writerows(
                rows
            )

        print(
            f'\nSaved CSV: {args.csv}'
        )


if __name__ == '__main__':

    main()