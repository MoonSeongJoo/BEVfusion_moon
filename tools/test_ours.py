# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, ConfigDict, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet3d.utils import replace_ceph_backend


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet3D test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--ceph', action='store_true', help='Use ceph as data storage backend')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--score-thr', type=float, default=0.1, help='bbox score threshold')
    parser.add_argument(
        '--task',
        type=str,
        choices=[
            'mono_det', 'multi-view_det', 'lidar_det', 'lidar_seg',
            'multi-modality_det'
        ],
        help='Determine the visualization method depending on the task.')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/test.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualization_hook['test_out_dir'] = args.show_dir
        all_task_choices = [
            'mono_det', 'multi-view_det', 'lidar_det', 'lidar_seg',
            'multi-modality_det'
        ]
        assert args.task in all_task_choices, 'You must set '\
            f"'--task' in {all_task_choices} in the command " \
            'if you want to use visualization hook'
        visualization_hook['vis_task'] = args.task
        visualization_hook['score_thr'] = args.score_thr
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)

    # TODO: We will unify the ceph support approach with other OpenMMLab repos
    if args.ceph:
        cfg = replace_ceph_backend(cfg)

    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:
        # Currently, we only support tta for 3D segmentation
        # TODO: Support tta for 3D detection
        assert 'tta_model' in cfg, 'Cannot find ``tta_model`` in config.'
        assert 'tta_pipeline' in cfg, 'Cannot find ``tta_pipeline`` in config.'
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)
    
    # --- ▼▼▼▼▼ 여기에 아래 코드를 추가하세요 ▼▼▼▼▼ ---

    # ===============================================================
    # ============== 시각화 강제 활성화 디버깅 코드 =================
    print("\n\n" + "="*50)
    print("      DEBUG: FORCING VISUALIZATION HOOK ACTIVATION")
    print("="*50 + "\n")

    # 시각화 훅을 직접 찾아서 속성을 강제로 덮어씁니다.
    for hook in runner.hooks:
        # 우리가 사용하는 시각화 훅의 정확한 클래스 이름을 찾습니다.
        if 'VisualizationHook' in hook.__class__.__name__:
            print(f"DEBUG: Found hook: {hook.__class__.__name__}")
            
            # 1. 시각화 스위치를 강제로 켭니다.
            hook.draw = True
            print(f"DEBUG: hook.draw forced to -> {hook.draw}")
            
            # 2. 출력 경로를 강제로 설정합니다.
            hook.test_out_dir = './visualization_forced'
            print(f"DEBUG: hook.test_out_dir forced to -> {hook.test_out_dir}")

            # 3. ▼▼▼▼▼ [중요] 작업 유형을 강제로 설정합니다. ▼▼▼▼▼
            # 이 값이 없으면 hook이 무엇을 그려야 할지 모릅니다.
            hook.vis_task = 'multi-modality_det'  # BEVFusion이므로 'multi-modality_det'
            print(f"DEBUG: hook.vis_task forced to -> {hook.vis_task}")
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
            
            # 4. 점수 커트라인을 강제로 설정합니다.
            if hasattr(hook, 'score_thr'):
                hook.score_thr = 0.01
                print(f"DEBUG: hook.score_thr forced to -> {hook.score_thr}")

    # ===============================================================
    # ===============================================================

    # start testing
    runner.test()


if __name__ == '__main__':
    main()
