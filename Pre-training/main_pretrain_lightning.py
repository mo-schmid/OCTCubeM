# Copyright (c) Zixuan Liu et al, OCTCubeM group
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Lightning version of main_pretrain_oph_joint_2d512_flash_attn.py, pre-training
# OCTCube on the alleye-macula webdataset. 3D-only by default; `--joint_2d` adds the
# original's 2D branch on single B-scans of the train volumes (plus Kermany with
# `--use_kermany`), without SPL so far. Arguments are those of the original script,
# plus the dataset/Trainer/W&B ones added in `get_args_parser` below.
#
# Launch with srun (one task per GPU); Lightning picks up the SLURM environment.

import argparse
import datetime
import os
from pathlib import Path

import lightning.pytorch as pl
import torch.backends.cudnn as cudnn
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy

import custom_util.misc as misc
from main_pretrain_oph_joint_2d512_flash_attn import get_args_parser as get_original_args_parser
from octcubem.datamodules.alleye_macula import AlleyeMaculaDataModule
from octcubem.datamodules.alleye_macula_bscans import AlleyeMaculaBscanDataModule
from octcubem.datamodules.joint_pretrain import JointPretrainDataModule
from octcubem.datamodules.kermany import Kermany2018DataModule
from octcubem.transforms.defaults import FlipAugment
from pretrain_lightning_module import MAEPretrainModule

OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "pretraining"


def get_args_parser():
    parser = argparse.ArgumentParser("OCTCube Lightning pre-training", parents=[get_original_args_parser()])
    # dataset
    parser.add_argument("--data_dir", default="../data/alleye-macula_n100", type=str, help="webdataset root holding overview.csv")
    parser.add_argument("--train_folds", default=[0, 1, 2, 3, 4, 5, 6, 7], type=int, nargs="+")
    parser.add_argument("--val_folds", default=[8], type=int, nargs="+")
    parser.add_argument("--test_folds", default=[9], type=int, nargs="+")
    parser.add_argument("--steps_per_epoch", default=None, type=int, help="per-rank steps per epoch; default one pass over the train folds")
    # 2D branch; the 2D batch size, input size, mask ratio schedule are the original's arguments
    parser.add_argument("--joint_2d", action="store_true", help="add the 2D loss on single B-scans of the train volumes")
    parser.add_argument("--use_kermany", action="store_true", help="also draw 2D samples from Kermany (needs --joint_2d)")
    parser.add_argument("--kermany_fraction", default=None, type=float, help="Kermany's share of the 2D batch; default its share of the pooled images")
    parser.add_argument("--norm_2d", default="imagenet", choices=["imagenet", "unit"], help="ImageNet-normalise the 2D input as the original, or keep it in [0, 1] like the 3D input")
    parser.add_argument("--num_workers_2d", default=None, type=int, help="workers per 2D loader; default --num_workers")
    parser.add_argument("--bscans_per_volume", default=None, type=int, help="B-scans drawn per volume and visit; default all")
    parser.add_argument("--bscan_shuffle_buffer", default=500, type=int)
    parser.add_argument("--kermany_cache_dir", default="auto", type=str, help="decoded-image cache; 'auto' is $TMPDIR under SLURM")
    # run directory
    parser.add_argument("--output_root", default=str(OUTPUT_ROOT), type=str, help="every run gets its own directory in here, unless --output_dir is given")
    parser.add_argument("--run_name", default="octcube_pretrain", type=str, help="the run directory is <run_name>_<timestamp>")
    # trainer
    parser.add_argument("--devices", default="auto", help="GPUs per node")
    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument("--ckpt_path", default=None, type=str, help="Lightning checkpoint to resume training from")
    # wandb
    parser.add_argument("--wandb_project", default="octcube-pretrain", type=str)
    parser.add_argument("--wandb_entity", default=None, type=str)
    parser.add_argument("--wandb_name", default=None, type=str, help="defaults to the run directory's name")
    parser.add_argument("--wandb_offline", action="store_true", help="log locally, upload later with `wandb sync`")
    parser.add_argument("--no_wandb", action="store_true")
    # depth 48 (divisible by t_patch_size 3), no depth resizing; RETFound init as in the paper
    parser.set_defaults(num_frames=48, pred_t_dim=48, resume_type="retfound_2_flash_attn", output_dir=None)
    # the folder holding Kermany's index.csv, not the original's ImageFolder root
    parser.set_defaults(kermany_data_dir="ucsd_kermany_2018")
    return parser


def resolve_run_timestamp():
    """The run's timestamp, the same in every process of one run.

    Each srun step's tasks start as separate processes, so they take the job's start
    time (plus the step id past the first step, as the steps of one job share it)
    rather than a clock reading of their own. A single process exports its timestamp,
    which the DDP workers Lightning spawns from it inherit.
    """
    if "OCTCUBE_RUN_TIMESTAMP" not in os.environ:
        if int(os.environ.get("SLURM_NTASKS", 1)) > 1:
            start = datetime.datetime.fromtimestamp(int(os.environ["SLURM_JOB_START_TIME"]))
            timestamp = start.strftime("%Y%m%d-%H%M%S")
            step = int(os.environ.get("SLURM_STEP_ID", 0))
            if step > 0:
                timestamp += "-step{}".format(step)
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        os.environ["OCTCUBE_RUN_TIMESTAMP"] = timestamp
    return os.environ["OCTCUBE_RUN_TIMESTAMP"]


def build_datamodule(args):
    volumes = AlleyeMaculaDataModule(
        data_dir=args.data_dir,
        train_folds=args.train_folds,
        val_folds=args.val_folds,
        test_folds=args.test_folds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.input_size,
        num_slices=args.num_frames,
        steps_per_epoch=args.steps_per_epoch,
        # OCTCube's 3D transforms: trilinear in-plane resize, random depth / left-right flips
        resize_mode="trilinear",
        train_augment=FlipAugment(),
    )
    if not args.joint_2d:
        if args.use_kermany:
            raise ValueError("--use_kermany needs --joint_2d")
        return volumes

    # OCTCube's 2D transform: bicubic resize to the high-res input size, no augmentation;
    # the 2D batch size is set by the joint datamodule
    num_workers_2d = args.num_workers if args.num_workers_2d is None else args.num_workers_2d
    bscans = AlleyeMaculaBscanDataModule(
        data_dir=args.data_dir,
        train_folds=args.train_folds,
        val_folds=args.val_folds,
        test_folds=args.test_folds,
        num_workers=num_workers_2d,
        image_size=args.high_res_input_size,
        bscans_per_volume=args.bscans_per_volume,
        bscan_shuffle_buffer=args.bscan_shuffle_buffer,
    )
    kermany = None
    if args.use_kermany:
        kermany = Kermany2018DataModule(
            data_dir=args.kermany_data_dir,
            num_workers=num_workers_2d,
            image_size=args.high_res_input_size,
            cache_dir=args.kermany_cache_dir,
        )
    return JointPretrainDataModule(
        volumes,
        bscans=bscans,
        kermany=kermany,
        batch_size_2d=args.batch_size_2d,
        kermany_fraction=args.kermany_fraction,
    )


def main(args):
    if args.output_dir is None:
        run_dir_name = "{}_{}".format(args.run_name, resolve_run_timestamp())
        args.output_dir = os.path.join(args.output_root, run_dir_name)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if not args.log_dir:
        args.log_dir = args.output_dir

    cudnn.benchmark = True

    datamodule = build_datamodule(args)

    if args.no_wandb:
        logger = False
    else:
        logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name or Path(args.output_dir).name,
            save_dir=args.log_dir,
            offline=args.wandb_offline,
            log_model=False,
        )

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, "lightning_ckpt"),
            every_n_epochs=args.checkpoint_period,
            save_last=True,
        ),
    ]

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        # the high-res (2D) patch embed of the joint model gets no gradient in 3D-only training
        strategy=DDPStrategy(find_unused_parameters=not args.joint_2d),
        # shards are split across ranks by the datamodule itself
        use_distributed_sampler=False,
        precision="32-true" if args.fp32 else "16-mixed",
        max_epochs=args.epochs,
        accumulate_grad_batches=args.accum_iter,
        gradient_clip_val=args.clip_grad,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=args.output_dir,
        log_every_n_steps=20,
    )

    # fix the seed for reproducibility
    pl.seed_everything(args.seed + trainer.global_rank)

    eff_batch_size = args.batch_size * args.accum_iter * trainer.world_size

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    print("{}".format(args).replace(", ", ",\n"))
    if args.output_dir and trainer.is_global_zero:
        with open(os.path.join(args.output_dir, "args.txt"), "a") as f:
            f.write("{}".format(args).replace(', ', ',\n'))
    if logger:
        logger.log_hyperparams(vars(args))

    # define the model
    model = MAEPretrainModule(args)
    print("Model = %s" % str(model.model))

    # a Lightning resume restores the weights itself
    if args.resume and not args.ckpt_path:
        print("Resuming from checkpoint")
        if args.resume_type == "retfound_2_flash_attn":
            misc.load_model_retfound_flash_attn(
                args=args,
                model_without_ddp=model.model,
                encoder_only=True,
            )
        else:
            raise NotImplementedError("resume_type {} is not ported to Lightning yet".format(args.resume_type))

    print(f"Start training for {args.epochs} epochs")
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
