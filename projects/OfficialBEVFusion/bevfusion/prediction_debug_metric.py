import torch

from mmengine.evaluator import BaseMetric
from mmdet3d.registry import METRICS


@METRICS.register_module()
class NuScenesPredictionDebugMetric(BaseMetric):

    default_prefix = "pred_debug"

    def process(self, data_batch, data_samples):

        for sample in data_samples:

            if isinstance(sample, dict):
                pred = sample["pred_instances_3d"]
                sample_idx = sample.get(
                    "sample_idx", "unknown"
                )
            else:
                pred = sample.pred_instances_3d
                sample_idx = sample.metainfo.get(
                    "sample_idx", "unknown"
                )

            if isinstance(pred, dict):
                scores = pred["scores_3d"]
            else:
                scores = pred.scores_3d

            scores = (
                torch.as_tensor(scores)
                .detach()
                .float()
                .cpu()
                .reshape(-1)
            )

            result = dict(
                sample_idx=str(sample_idx),
                boxes=int(scores.numel()),
                score025=int(
                    (scores >= 0.25).sum().item()
                ),
                score050=int(
                    (scores >= 0.50).sum().item()
                ),
                max_score=(
                    float(scores.max().item())
                    if scores.numel() else 0.0
                ),
            )

            self.results.append(result)

            if len(self.results) <= 3:
                print(
                    "[PRED DEBUG]", result,
                    flush=True
                )

    def compute_metrics(self, results):

        if not results:
            raise RuntimeError("No predictions")

        n = len(results)
        total_boxes = sum(x["boxes"] for x in results)

        return dict(
            frames=float(n),
            avg_boxes=float(total_boxes / n),
            frames_with_boxes=float(
                sum(x["boxes"] > 0 for x in results)
            ),
            boxes_score025=float(
                sum(x["score025"] for x in results)
            ),
            boxes_score050=float(
                sum(x["score050"] for x in results)
            ),
            mean_max_score=float(
                sum(x["max_score"] for x in results) / n
            ),
        )