"""Training loop, adapted from archive/kd/model/SymbolicGPT/trainer.py.

Changes vs. the original:
- `device` used to be the string `'gpu'` (triggering a `torch.cuda.is_available()`
  check plus `nn.DataParallel` wrapping) or else silently fell back to the
  plain string `'cpu'` -- `Trainer.device` therefore ended up being either an
  `int` GPU index or the string `'cpu'`, and calling code had to
  `isinstance(device, int)`-sniff which one it was. Replaced with a plain
  required `torch.device` argument (resolved once by the caller, matching
  `kd_symbolicgpt.py`'s `_resolve_device()`), and dropped the
  multi-GPU `DataParallel` wrapping -- not needed for the small models this
  wrapper trains.
"""

import math
import logging

from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6
    final_tokens = 260e9
    # checkpoint settings
    ckpt_path = None
    num_workers = 0  # for DataLoader
    show_progress = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, device, best=None):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.device = device
        self.best_loss = best

    def save_checkpoint(self):
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(self.model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        optimizer = model.configure_optimizers(config)

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            use_pinned_memory = self.device.type == "cuda"
            loader = DataLoader(
                data,
                shuffle=True,
                pin_memory=use_pinned_memory,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                persistent_workers=config.num_workers > 0,
            )

            losses = []
            pbar = (
                tqdm(enumerate(loader), total=len(loader))
                if is_train and config.show_progress
                else enumerate(loader)
            )
            for it, (x, y, p, v) in pbar:

                x = x.to(self.device, non_blocking=use_pinned_memory)  # input equation
                y = y.to(self.device, non_blocking=use_pinned_memory)  # output equation
                p = p.to(self.device, non_blocking=use_pinned_memory)  # points
                v = v.to(self.device, non_blocking=use_pinned_memory)  # number of variables

                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x, y, p, v)
                    loss = loss.mean()
                    if not is_train:
                        losses.append(loss.item())

                if is_train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    if config.lr_decay:
                        self.tokens += (y >= 0).sum()
                        if self.tokens < config.warmup_tokens:
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    if isinstance(pbar, tqdm):
                        pbar.set_description(
                            f"epoch {epoch+1} iter {it}: "
                            f"train loss {loss.detach().item():.5f}. lr {lr:e}"
                        )

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        self.best_loss = float('inf') if self.best_loss is None else self.best_loss
        self.tokens = 0
        for epoch in range(config.max_epochs):

            run_epoch('train')
            if self.test_dataset is not None:
                test_loss = run_epoch('test')

            good_model = self.test_dataset is None or test_loss < self.best_loss
            if self.config.ckpt_path is not None and good_model:
                self.best_loss = test_loss
                self.save_checkpoint()
