# Copyright (c) Zixuan Liu et al, OCTCubeM group
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Lightning port of `train_one_epoch_joint` / `eval_one_epoch` in engine_pretrain.py.
# With `--joint_2d`, every step adds the MAE loss of a batch of single B-scans at
# `high_res_input_size` to the 3D loss, as the original does; SPL is not ported yet.
# In the 3D path, `pre_mask` is ignored by `MaskedAutoencoderViT.forward` and
# `frame_loss` only feeds SPL, so the 3D objective below is the same as the original one.

import math
import os

import lightning.pytorch as pl
import torch

import custom_util.lr_sched as lr_sched
import custom_util.misc as misc
import models_mae_joint_res_flash_attn as models_mae
from main_pretrain_oph_joint_2d512_flash_attn import mask_ratio_2d_scheduler
from octcubem.transforms.defaults import octcube_2d_input

# the 2D batch parts of a joint training batch, see `JointPretrainDataModule`
SOURCES_2D = ("2d_alleye", "2d_kermany")
# the 2D mask ratio of the original's evaluation
VAL_MASK_RATIO_2D = 0.75


class MAEPretrainModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = models_mae.__dict__[args.model](
            **vars(args),
        )
        self.mask_ratio_2d = args.mask_ratio_2d_min

    def on_train_epoch_start(self):
        # the 2D mask ratio is scheduled per epoch, as in the original
        self.mask_ratio_2d = mask_ratio_2d_scheduler(
            self.current_epoch,
            mask_ratio_max=self.args.mask_ratio_2d_max,
            mask_ratio_min=self.args.mask_ratio_2d_min,
            all_epoch=self.args.epochs,
            warmup_epochs=self.args.warmup_epochs,
            epoch_offset=self.args.epoch_offset,
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
        samples = batch["3d"]["image"] if self.args.joint_2d else batch["image"]
        loss, _, _ = self.model(
            samples,
            mask_ratio=self.args.mask_ratio,
        )
        # `train_loss` stays the 3D loss alone, comparable with 3D-only runs
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=samples.shape[0])

        if self.args.joint_2d:
            sample_2d = torch.cat([batch[name]["image"] for name in SOURCES_2D if name in batch])
            loss_2d, _, _ = self.model(
                self._to_2d_input(sample_2d),
                mask_ratio=self.mask_ratio_2d,
            )
            loss = loss + loss_2d
            self.log("train_loss_2d", loss_2d, on_step=True, on_epoch=True, sync_dist=True, batch_size=sample_2d.shape[0])
            self.log("train_loss_all", loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=samples.shape[0])
            self.log("mask_ratio_2d", self.mask_ratio_2d, on_step=True, on_epoch=False)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise Exception("Loss is {}, stopping training".format(loss_value))

        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True, on_epoch=False)
        self.log("mask_ratio", self.args.mask_ratio, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # a joint datamodule runs one val loader per source: 3D first, then the 2D ones
        source = self.trainer.datamodule.val_names[dataloader_idx] if self.args.joint_2d else "3d"
        samples = batch["image"]
        if source == "3d":
            name, mask_ratio = "val_loss", self.args.mask_ratio
        else:
            name, mask_ratio = "val_loss_{}".format(source), VAL_MASK_RATIO_2D
            samples = self._to_2d_input(samples)
        loss, _, _ = self.model(
            samples,
            mask_ratio=mask_ratio,
        )

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise Exception("Loss is {}, stopping training".format(loss_value))

        self.log(name, loss, on_step=False, on_epoch=True, prog_bar=source == "3d", sync_dist=True, batch_size=samples.shape[0], add_dataloader_idx=False)
        return loss

    def _to_2d_input(self, samples):
        return octcube_2d_input(samples, imagenet_normalize=self.args.norm_2d == "imagenet")

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
