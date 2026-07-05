"""Evaluation: run a model over one or more test datasets and average metrics.

Produces a ``{dataset_name: {metric_name: value}}`` table -- the core artifact
of the project (compare models across datasets across metrics).
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class Evaluator:
    def __init__(self, metrics: dict, device: str = "cuda", num_workers: int = 0):
        self.metrics = metrics
        self.device = device
        self.num_workers = num_workers

    @torch.no_grad()
    def run(self, model, dataset, dataset_name: str = "test", on_sample=None) -> dict:
        """Average each metric over ``dataset``. ``on_sample(name, lr, sr, scores)``,
        when given, is called once per image (used for per-image reports)."""
        model.eval()
        loader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=self.num_workers)
        totals = {name: 0.0 for name in self.metrics}
        for batch in tqdm(loader, desc=f"eval[{dataset_name}]", leave=False):
            lr = batch["lr"].to(self.device)
            hr = batch["hr"].to(self.device)
            # Diffusion models produce the SR image by iterative sampling
            # (super_resolve); feed-forward models by a plain forward pass.
            if hasattr(model, "super_resolve"):
                sr = model.super_resolve(lr).clamp(0, 1)
            else:
                sr = model(lr).clamp(0, 1)
            scores = {name: metric(sr, hr) for name, metric in self.metrics.items()}
            for name, value in scores.items():
                totals[name] += value
            if on_sample is not None:
                on_sample(batch["name"][0], lr[0], sr[0], scores)
        n = len(dataset)
        return {name: total / n for name, total in totals.items()}


def evaluate_model(model, datasets: dict, metrics: dict, device: str = "cuda",
                   num_workers: int = 0) -> dict:
    """``datasets``: {name: Dataset}. Returns {name: {metric: value}}."""
    evaluator = Evaluator(metrics, device, num_workers=num_workers)
    return {name: evaluator.run(model, ds, name) for name, ds in datasets.items()}
