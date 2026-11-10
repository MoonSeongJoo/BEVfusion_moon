# my_hooks.py
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

@HOOKS.register_module()
class ValidateBeforeTrainHook(Hook):
    """학습 시작 직전에 검증을 실행하는 훅"""
    def before_run(self, runner):
        runner.logger.info('Running validation before training begins...')
        runner.val_loop.run()


@HOOKS.register_module()
class StopOnNDSThresholdHook(Hook):
    """
    Stop training when validation NDS
    exceeds the configured threshold.
    """

    # CheckpointHook 저장 이후 실행
    priority = 'LOWEST'

    def __init__(
        self,
        threshold=0.7069,
        metric_key='NDS',
    ):
        self.threshold = float(threshold)
        self.metric_key = metric_key

    def after_val_epoch(self, runner, metrics):

        if self.metric_key not in metrics:
            raise KeyError(
                f'Missing {self.metric_key}. '
                f'Available metrics: {list(metrics.keys())}'
            )

        nds = float(metrics[self.metric_key])

        runner.logger.info(
            f'[NDS CHECK] Iter={runner.iter}, '
            f'NDS={nds:.6f}, '
            f'Target={self.threshold:.6f}'
        )

        if nds > self.threshold:

            runner.train_loop.stop_training = True

            runner.logger.info(
                '[NDS TARGET ACHIEVED] '
                'Best checkpoint saved. '
                'Stopping training.'
            )