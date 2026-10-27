# Copyright (c) Zixuan Liu et al, OCTCubeM group
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Lightning port of the 3D part of `train_one_epoch_joint` / `eval_one_epoch` in
# engine_pretrain.py. The 2D branch (and with it SPL) is not ported yet: in the 3D
# path, `pre_mask` is ignored by `MaskedAutoencoderViT.forward` and `frame_loss` only
# feeds SPL, so the 3D objective below is the same as the original one.

import math
import os

import lightning.pytorch as pl
import torch

import custom_util.lr_sched as lr_sched
import custom_util.misc as misc
import models_mae_joint_res_flash_attn as models_mae


class MAEPretrainModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = models_mae.__dict__[args.model](
            **vars(args),
        )

    def on_train_batch_start(self, batch, batch_idx):
        # we use a per iteration (instead of per epoch) lr scheduler
        if batch_idx % self.args.accum_iter == 0:
            lr_sched.adjust_learning_rate(
                self.trainer.optimizers[0],
                batch_idx / self.trainer.num_training_batches + self.current_epoch,
                self.args,
            )

    def training_step(self, batch, batch_idx):
        samples = batch["image"]
        loss, _, _ = self.model(
            samples,
            mask_ratio=self.args.mask_ratio,
        )

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise Exception("Loss is {}, stopping training".format(loss_value))

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=samples.shape[0])
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True, on_epoch=False)
        self.log("mask_ratio", self.args.mask_ratio, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        samples = batch["image"]
        loss, _, _ = self.model(
            samples,
            mask_ratio=self.args.mask_ratio,
        )

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise Exception("Loss is {}, stopping training".format(loss_value))

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=samples.shape[0])
        return loss

    def configure_optimizers(self):
        # following timm: set wd as 0 for bias and norm layers
        param_groups = misc.add_weight_decay(
            self.model,
            self.args.weight_decay,
            bias_wd=self.args.bias_wd,
        )
        if self.args.beta is None:
            beta = (0.9, 0.95)
        else:
            beta = self.args.beta
        # `torch.optim._multi_tensor.AdamW` of the original is gone in torch 2.x;
        # `foreach=True` is the same multi-tensor implementation
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.args.lr,
            betas=beta,
            foreach=True,
        )
        return optimizer

    def on_train_epoch_end(self):
        # besides Lightning's own checkpoints, keep the original `misc.save_model`
        # layout, which OCTCube's finetuning and inference code load from
        epoch = self.current_epoch
        if not self.args.output_dir or not self.trainer.is_global_zero:
            return
        if epoch % self.args.checkpoint_period == 0 or epoch + 1 == self.args.epochs:
            checkpoint_path = "{}/checkpoint-{:05d}.pth".format(self.args.output_dir, epoch)
            to_save = {
                "model": self.model.state_dict(),
                "epoch": epoch,
                "args": self.args,
            }
            torch.save(to_save, checkpoint_path)
            print("Saved checkpoint to {}".format(os.path.abspath(checkpoint_path)))
