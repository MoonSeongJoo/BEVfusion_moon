import argparse
import copy

import numpy as np
import torch

from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint
from mmengine.utils import import_modules_from_strings

from mmdet3d.registry import MODELS
from mmdet3d.utils import register_all_modules


def parse_args():

    parser = argparse.ArgumentParser(
        description='Deterministic CorrNet 200-frame evaluation'
    )

    parser.add_argument(
        'config',
        help='training/evaluation config'
    )

    parser.add_argument(
        'checkpoint',
        help='checkpoint to evaluate'
    )

    parser.add_argument(
        '--num-samples',
        type=int,
        default=200,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    # ============================================================
    # 1. Register MMDetection3D modules
    # ============================================================

    register_all_modules(
        init_default_scope=True
    )

    cfg = Config.fromfile(
        args.config
    )

    # Load project custom modules.
    if cfg.get(
        'custom_imports',
        None
    ) is not None:

        import_modules_from_strings(
            **cfg.custom_imports
        )


    # ============================================================
    # 2. Validation loader
    #
    # IMPORTANT:
    # shuffle=False
    #
    # We simply stop after first 200 frames.
    # This avoids modifying dataset indices.
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
    # 3. Build model
    # ============================================================

    model = MODELS.build(
        cfg.model
    )


    # ============================================================
    # 4. Load checkpoint
    # ============================================================

    print(
        '\n'
        '====================================================\n'
        '[CORR-200 EVAL]\n'
        f'checkpoint = {args.checkpoint}\n'
        f'num_samples = {args.num_samples}\n'
        '====================================================\n'
    )


    load_checkpoint(
        model,
        args.checkpoint,
        map_location='cpu',
        strict=False,
    )


    # ============================================================
    # 5. Evaluation mode
    # ============================================================

    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )

    model.to(
        device
    )

    model.eval()


    # Deterministic runtime as much as practical.
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
    # 6. Global accumulators
    # ============================================================

    all_epe = []
    all_identity = []

    all_u = []
    all_v = []

    num_frames = 0


    # ============================================================
    # 7. Run model.loss() in EVAL mode
    #
    # Why loss()?
    #
    # Corr GT generation and pixel EPE diagnostics currently
    # exist in BEVFusion.loss(), not predict().
    #
    # torch.no_grad() means NO optimization occurs.
    # ============================================================

    with torch.no_grad():

        for batch_idx, data_batch in enumerate(
            val_loader
        ):

            if num_frames >= args.num_samples:

                break


            # ---------------------------------------------
            # Same data preprocessor used by model runtime
            # ---------------------------------------------

            data = model.data_preprocessor(
                data_batch,
                training=False,
            )


            # ---------------------------------------------
            # Run forward in LOSS mode only for diagnostics.
            #
            # No backward.
            # No optimizer.
            # model remains eval().
            # ---------------------------------------------

            _ = model._run_forward(
                data,
                mode='loss',
            )


            # ---------------------------------------------
            # Read point-wise Corr evaluation cache.
            # ---------------------------------------------

            cache = getattr(
                model,
                '_corr_eval_cache',
                None,
            )


            if cache is None:

                raise RuntimeError(
                    '\n'
                    '[CORR-200] _corr_eval_cache not found.\n'
                    'Check that the cache block was inserted '
                    'inside the Corr pixel diagnostic section.'
                )


            if cache[
                'epe_px'
            ].numel() > 0:

                all_epe.append(
                    cache[
                        'epe_px'
                    ]
                )

                all_identity.append(
                    cache[
                        'identity_epe_px'
                    ]
                )

                all_u.append(
                    cache[
                        'u_abs_px'
                    ]
                )

                all_v.append(
                    cache[
                        'v_abs_px'
                    ]
                )


            num_frames += 1


            if (
                num_frames % 20
                == 0
            ):

                print(
                    '[CORR-200] '
                    f'frames={num_frames}'
                )


    # ============================================================
    # 8. Aggregate
    #
    # IMPORTANT:
    # These statistics are over ALL valid UNIQUE correspondences
    # from the 200 frames, not averages of batch averages.
    # ============================================================

    if len(all_epe) == 0:

        raise RuntimeError(
            'No valid Corr correspondences collected.'
        )


    epe = torch.cat(
        all_epe
    ).float()

    identity = torch.cat(
        all_identity
    ).float()

    u_abs = torch.cat(
        all_u
    ).float()

    v_abs = torch.cat(
        all_v
    ).float()


    mean_epe = (
        epe.mean().item()
    )

    median_epe = (
        torch.quantile(
            epe,
            0.50,
        ).item()
    )

    p90_epe = (
        torch.quantile(
            epe,
            0.90,
        ).item()
    )


    identity_mean = (
        identity.mean().item()
    )

    identity_median = (
        torch.quantile(
            identity,
            0.50,
        ).item()
    )

    identity_p90 = (
        torch.quantile(
            identity,
            0.90,
        ).item()
    )


    u_mae = (
        u_abs.mean().item()
    )

    v_mae = (
        v_abs.mean().item()
    )


    recovery = (
        1.0
        -
        mean_epe
        /
        max(
            identity_mean,
            1e-8,
        )
    ) * 100.0


    # ============================================================
    # 9. Final report
    # ============================================================

    print(
        '\n'
        '====================================================\n'
        '[CORR DETERMINISTIC 200-VAL RESULT]\n'
        '====================================================\n'
        f'Frames              = {num_frames}\n'
        f'Valid Corr points   = {epe.numel()}\n'
        '\n'
        '[CORR]\n'
        f'Mean EPE            = {mean_epe:.6f} px\n'
        f'Median EPE          = {median_epe:.6f} px\n'
        f'P90 EPE             = {p90_epe:.6f} px\n'
        f'U MAE               = {u_mae:.6f} px\n'
        f'V MAE               = {v_mae:.6f} px\n'
        '\n'
        '[IDENTITY / NO CORRECTION]\n'
        f'Mean EPE            = {identity_mean:.6f} px\n'
        f'Median EPE          = {identity_median:.6f} px\n'
        f'P90 EPE             = {identity_p90:.6f} px\n'
        '\n'
        f'Corr Recovery       = {recovery:.3f}%\n'
        '====================================================\n'
    )


if __name__ == '__main__':

    main()