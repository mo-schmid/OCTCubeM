# Copyright (c) Zixuan Liu et al, OCTCubeM group
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Lightning version of main_pretrain_oph_joint_2d512_flash_attn.py, pre-training
# OCTCube on the alleye-macula webdataset. 3D-only for now: the 2D branch (and SPL)
# of the original is not ported yet. Arguments are those of the original script,
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


def main(args):
    if args.output_dir is None:
        run_dir_name = "{}_{}".format(args.run_name, resolve_run_timestamp())
        args.output_dir = os.path.join(args.output_root, run_dir_name)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if not args.log_dir:
        args.log_dir = args.output_dir

    cudnn.benchmark = True

    datamodule = AlleyeMaculaDataModule(
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
        strategy=DDPStrategy(find_unused_parameters=True),
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
