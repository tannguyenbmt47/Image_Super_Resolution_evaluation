"""SR trainer.

Minimal but complete: Adam + step LR, periodic validation via the Evaluator, and
early stopping. The per-step loss is dispatched on the model: diffusion models
(SR3) provide ``compute_loss(lr, hr)``; feed-forward models (SRCNN) use the
configured pixel loss on their output; GAN models (SRGAN) return
``{"g_loss", "d_loss"}`` for adversarial training with separate optimizers.

Checkpoints: ``best.pth`` is saved whenever the monitored validation metric
improves, ``last.pth`` at the end. Early stopping (``early_stop_patience > 0``)
halts training when the metric has not improved for that many validations.
"""

# Metrics where a lower value is better (everything else: higher is better).
_LOWER_IS_BETTER = {"lpips", "niqe"}

import json
from pathlib import Path

import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..losses import build_loss
from .evaluator import evaluate_model


class Trainer:
    def __init__(
        self,
        model,
        train_set,
        cfg,
        device="cuda",
        val_sets=None,
        metrics=None,
        out_dir="experiments/run",
    ):
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

        use_pin = device.startswith("cuda")
        self.loader = DataLoader(
            train_set,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.get("num_workers", 4),
            drop_last=True,
            pin_memory=use_pin,
        )
        # feed-forward criterion (diffusion models supply their own compute_loss)
        self.criterion = build_loss(cfg).to(device)
        # Generator optimizer (covers all params for non-GAN models;
        # for GAN models, only generator params are used)
        gen_params = (
            self._base.generator.parameters()
            if hasattr(self._base, "generator")
            else model.parameters()
        )
        self.optimizer = torch.optim.Adam(gen_params, lr=cfg.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=cfg.get("lr_step", 200),
            gamma=cfg.get("lr_gamma", 0.5),
        )
        # Discriminator optimizer (GAN models only)
        self.d_optimizer = (
            self._build_d_optimizer(cfg)
            if hasattr(self._base, "discriminator")
            else None
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

        # per-epoch history, flushed to history.json each epoch (crash-safe)
        self.history = {"train_loss": [], "val": []}
        self.start_epoch = 1

    @property
    def _base(self):
        return self.model.module if self._parallel else self.model

    def _build_d_optimizer(self, cfg):
        """Build discriminator optimizer for GAN training."""
        d_cfg = cfg.get("discriminator", {}) or {}
        return torch.optim.Adam(
            self._base.discriminator.parameters(),
            lr=d_cfg.get("lr", cfg.get("d_lr", 1e-4)),
            betas=(d_cfg.get("beta1", 0.9), d_cfg.get("beta2", 0.999)),
        )

    def train(self):
        val_every = self.cfg.get("val_every", 10)
        for epoch in range(self.start_epoch, self.cfg.epochs + 1):
            self._train_epoch(epoch)
            self.scheduler.step()
            if epoch % val_every == 0 or epoch == self.cfg.epochs:
                self._validate(epoch)
            # full resumable state every epoch (crash-safe)
            self.save("last.pth", epoch=epoch)
            if self.should_stop:
                print(
                    f"early stop at epoch {epoch}: no {self.monitor} "
                    f"improvement for {self.patience} validations"
                )
                break

    def _train_epoch(self, epoch):
        self.model.train()
        running = 0.0
        # amp: true -> bf16 autocast (no GradScaler needed for bf16)
        import contextlib
        amp = self.cfg.get("amp", False) and self.device.startswith("cuda")
        cast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if amp \
            else contextlib.nullcontext
        pbar = tqdm(self.loader, desc=f"train[{epoch}/{self.cfg.epochs}]")
        for batch in pbar:
            lr = batch["lr"].to(self.device)
            hr = batch["hr"].to(self.device)
            # Diffusion models (SR3) define their own loss on predicted noise;
            # feed-forward models use a pixel loss on the output;
            # GAN models (SRGAN) return {"g_loss", "d_loss"}.
            if hasattr(self._base, "compute_loss"):
                with cast():
                    loss_out = self._base.compute_loss(lr, hr)
                if isinstance(loss_out, dict):
                    # GAN training: separate generator and discriminator steps
                    g_loss = loss_out["g_loss"]
                    d_loss = loss_out["d_loss"]
                    # Generator step
                    self.optimizer.zero_grad()
                    g_loss.backward(retain_graph=True)
                    self.optimizer.step()
                    # Discriminator step
                    if self.d_optimizer is not None:
                        self.d_optimizer.zero_grad()
                        d_loss.backward()
                        self.d_optimizer.step()
                    running += g_loss.item()
                    pbar.set_postfix(
                        g_loss=f"{g_loss.item():.4f}", d_loss=f"{d_loss.item():.4f}"
                    )
                else:
                    loss = loss_out
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    running += loss.item()
                    pbar.set_postfix(loss=f"{loss.item():.4f}")
            else:
                with cast():
                    loss = self.criterion(self.model(lr), hr)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg = running / len(self.loader)
        lr_now = self.scheduler.get_last_lr()[0]
        print(f"epoch {epoch}: avg loss {avg:.4f} (lr {lr_now:.2e})")
        self.history["train_loss"].append({"epoch": epoch, "loss": avg, "lr": lr_now})
        self._flush_history()

    def _validate(self, epoch):
        if not self.val_sets or not self.metrics:
            return
        results = evaluate_model(self._base, self.val_sets, self.metrics, self.device)
        for ds_name, scores in results.items():
            line = ", ".join(f"{m}={v:.4f}" for m, v in scores.items())
            print(f"[val/{ds_name}] {line}")
        self.best = results
        self.history["val"].append({"epoch": epoch, **{
            ds: dict(scores) for ds, scores in results.items()}})
        self._flush_history()
        self._update_early_stop(results)

    def _flush_history(self):
        with open(self.out_dir / "history.json", "w") as fh:
            json.dump(self.history, fh, indent=1)

    def _update_early_stop(self, results):
        # monitored score = mean of the metric across val datasets that report it
        vals = [s[self.monitor] for s in results.values() if self.monitor in s]
        if not vals:
            return
        score = sum(vals) / len(vals)
        if self.best_score is None or (
            score > self.best_score
            if self.monitor_mode == "max"
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

    def save(self, filename, epoch=None):
        state_dict = self._base.state_dict()
        state = {
            "model": {
                key: value
                for key, value in state_dict.items()
                if not key.startswith("_perceptual.")
            },
            "cfg": dict(self.cfg),
        }
        if hasattr(self._base, "discriminator"):
            state["discriminator"] = self._base.discriminator.state_dict()
        if epoch is not None:  # full resumable state (last.pth)
            import random

            state["resume"] = {
                "epoch": epoch,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "d_optimizer": self.d_optimizer.state_dict() if self.d_optimizer else None,
                "best_score": self.best_score,
                "no_improve": self.no_improve,
                "history": self.history,
                "rng": {
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "python": random.getstate(),
                },
            }
        torch.save(state, self.out_dir / filename)

    def load_resume(self, path):
        """Restore full training state from a last.pth written by save(epoch=...)."""
        import random

        state = torch.load(path, map_location=self.device, weights_only=False)
        self._base.load_state_dict(state["model"], strict=False)
        r = state.get("resume")
        if not r:
            print(f"resume: {path} has weights only; starting from epoch 1")
            return
        self.optimizer.load_state_dict(r["optimizer"])
        self.scheduler.load_state_dict(r["scheduler"])
        if self.d_optimizer and r.get("d_optimizer"):
            self.d_optimizer.load_state_dict(r["d_optimizer"])
        self.best_score = r["best_score"]
        self.no_improve = r["no_improve"]
        self.history = r["history"]
        torch.set_rng_state(r["rng"]["torch"].cpu())
        if r["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in r["rng"]["cuda"]])
        random.setstate(r["rng"]["python"])
        self.start_epoch = r["epoch"] + 1
        print(f"resume: restored epoch {r['epoch']} -> continuing at {self.start_epoch}")
