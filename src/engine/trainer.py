"""SR trainer.

Minimal but complete: Adam + step LR, periodic validation via the Evaluator, and
early stopping. The per-step loss is dispatched on the model: diffusion models
(SR3) provide ``compute_loss(lr, hr)``; feed-forward models (SRCNN) use the
configured pixel loss on their output.

Checkpoints: ``best.pth`` is saved whenever the monitored validation metric
improves, ``last.pth`` at the end. Early stopping (``early_stop_patience > 0``)
halts training when the metric has not improved for that many validations.
"""

# Metrics where a lower value is better (everything else: higher is better).
_LOWER_IS_BETTER = {"lpips", "niqe"}

from pathlib import Path

import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..losses import build_loss
from .evaluator import evaluate_model


class Trainer:
    def __init__(self, model, train_set, cfg, device="cuda",
                 val_sets=None, metrics=None, out_dir="experiments/run"):
        self.device = device
        self.model = model.to(device)
        self._parallel = device.startswith("cuda") and torch.cuda.device_count() > 1
        if self._parallel:
            self.model = DataParallel(self.model)
        self.cfg = cfg
        self.val_sets = val_sets or {}
        self.metrics = metrics or {}
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.loader = DataLoader(
            train_set,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.get("num_workers", 4),
            drop_last=True,
            pin_memory=True,
        )
        # feed-forward criterion (diffusion models supply their own compute_loss)
        self.criterion = build_loss(cfg).to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=cfg.get("lr_step", 200),
            gamma=cfg.get("lr_gamma", 0.5),
        )
        self.best = {}

        # early stopping: monitor one validation metric, stop after `patience`
        # validations without improvement (patience 0 disables it).
        self.patience = cfg.get("early_stop_patience", 0)
        self.monitor = cfg.get("early_stop_metric", "psnr")
        self.monitor_mode = "min" if self.monitor in _LOWER_IS_BETTER else "max"
        self.best_score = None
        self.no_improve = 0
        self.should_stop = False

    @property
    def _base(self):
        return self.model.module if self._parallel else self.model

    def train(self):
        val_every = self.cfg.get("val_every", 10)
        for epoch in range(1, self.cfg.epochs + 1):
            self._train_epoch(epoch)
            self.scheduler.step()
            if epoch % val_every == 0 or epoch == self.cfg.epochs:
                self._validate(epoch)
                if self.should_stop:
                    print(f"early stop at epoch {epoch}: no {self.monitor} "
                          f"improvement for {self.patience} validations")
                    break
        self.save("last.pth")

    def _train_epoch(self, epoch):
        self.model.train()
        running = 0.0
        pbar = tqdm(self.loader, desc=f"train[{epoch}/{self.cfg.epochs}]")
        for batch in pbar:
            lr = batch["lr"].to(self.device)
            hr = batch["hr"].to(self.device)
            # Diffusion models (SR3) define their own loss on predicted noise;
            # feed-forward models use a pixel loss on the output.
            if hasattr(self._base, "compute_loss"):
                loss = self.model.compute_loss(lr, hr)
            else:
                loss = self.criterion(self.model(lr), hr)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        print(f"epoch {epoch}: avg loss {running / len(self.loader):.4f}")

    def _validate(self, epoch):
        if not self.val_sets or not self.metrics:
            return
        results = evaluate_model(self.model, self.val_sets, self.metrics, self.device)
        for ds_name, scores in results.items():
            line = ", ".join(f"{m}={v:.4f}" for m, v in scores.items())
            print(f"[val/{ds_name}] {line}")
        self.best = results
        self._update_early_stop(results)

    def _update_early_stop(self, results):
        # monitored score = mean of the metric across val datasets that report it
        vals = [s[self.monitor] for s in results.values() if self.monitor in s]
        if not vals:
            return
        score = sum(vals) / len(vals)
        if self.best_score is None or (
            score > self.best_score if self.monitor_mode == "max"
            else score < self.best_score
        ):
            self.best_score = score
            self.no_improve = 0
            self.save("best.pth")
            print(f"  new best {self.monitor}={score:.4f} -> saved best.pth")
        else:
            self.no_improve += 1
            if self.patience > 0 and self.no_improve >= self.patience:
                self.should_stop = True

    def save(self, filename):
        torch.save(
            {"model": self._base.state_dict(), "cfg": dict(self.cfg)},
            self.out_dir / filename,
        )
