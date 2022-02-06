from torchmetrics import Metric
import torch


class MRR(Metric):
    def __init__(self, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("cls", default=[], dist_reduce_fx="cat")
        self.add_state("labels", default=[], dist_reduce_fx="cat")

    def update(self, cls: torch.Tensor, labels: torch.Tensor):
        self.cls.append(cls)
        self.labels.append(labels)

    def compute(self):
        self.cls = torch.cat([torch.unsqueeze(x, 0) for _, x in sorted(zip(self.labels, self.cls), key=lambda pair: pair[0])], 0)
        num = self.cls.shape[0]
        self.cls /= self.cls.norm(2, dim=-1, keepdim=True)
        src_embeds = self.cls[: num // 2]
        trg_embeds = self.cls[num // 2:]
        # (1000, 1000)
        preds = src_embeds @ trg_embeds.T
        reciprocal_rank = get_reciprocal_rank(preds)
        # compute final result
        return reciprocal_rank


def get_reciprocal_rank(preds: torch.Tensor) -> torch.Tensor:
    """Compute MRR from row-aligned matrices of square query-document pairs.

    `mrr` is primarily intended for BLI or sentence-translation retrieval.

    Args:
        preds: square matrix of ranking scores

    Returns:
        torch.Tensor: mean reciprocal rank
    """
    N = preds.shape[0]
    if torch.cuda.is_available():
        torch_arange = torch.arange(N)[:, None].to(device='cuda')
    else:
        torch_arange = torch.arange(N)[:, None]
    rankings = preds.argsort(dim=-1, descending=True) == torch_arange
    reciprocal_rank = 1 / (1 + rankings.float().argmax(dim=-1))
    return reciprocal_rank
