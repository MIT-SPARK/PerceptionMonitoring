from torchmetrics import Metric, ConfusionMatrix
from torchmetrics.functional.classification.confusion_matrix import _confusion_matrix_compute, _confusion_matrix_update
import torch
from torch_geometric.data import Batch
from typing import Tuple

# def extract_failure_modes_only_from_batch_prediction(preds: torch.Tensor, batch: Batch) -> Tuple[torch.Tensor]:
#     assert preds.shape == batch.y.shape
#     preds_out = []
#     targets_out = []
#     idx = 0
#     for i in range(batch.num_graphs):
#         data = batch.get_example(i)
#         target = data.y
#         failures_idx = list(data.failure_modes_idx.values())
#         pred = preds[idx : idx + len(target)]
#         preds_out.append(pred[failures_idx, :])
#         targets_out.append(target[failures_idx])
#         idx += len(target)
#     return torch.cat(preds_out), torch.cat(targets_out)

class FaultIdentificationAccuracy(Metric):
    def __init__(self, partial=True, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.partial = partial
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, batch: Batch):
        preds = torch.argmax(preds, dim=1)
        assert preds.shape == batch.y.shape
        batch_correct = torch.tensor(0)
        batch_total = torch.tensor(0)
        idx = 0
        for i in range(batch.num_graphs):
            data = batch.get_example(i)
            target = data.y
            failures_idx = list(data.failure_modes_idx.values())
            pred = preds[idx : idx + len(target)]
            if self.partial:
                batch_correct += torch.sum(pred[failures_idx] == target[failures_idx]).to("cpu")
                batch_total += target[failures_idx].numel()
            else:
                batch_correct += torch.all(pred[failures_idx] == target[failures_idx]).to("cpu")
                batch_total += torch.tensor(1).to("cpu")
            idx += len(target)
        self.correct += batch_correct
        self.total += batch_total
        return batch_correct.float() / batch_total

    def compute(self):
        return self.correct.float() / self.total
