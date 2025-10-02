# my_hooks.py
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

@HOOKS.register_module()
class ValidateBeforeTrainHook(Hook):
    """학습 시작 직전에 검증을 실행하는 훅"""
    def before_run(self, runner):
        runner.logger.info('Running validation before training begins...')
        runner.val_loop.run()