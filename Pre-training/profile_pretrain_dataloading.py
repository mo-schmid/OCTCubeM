# Copyright (c) Zixuan Liu et al, OCTCubeM group
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Profiling of OCTCube pre-training with real data loading, on one GPU. Where
# profile_pretrain.py times the step on synthetic volumes, this runs the real thing:
# the datamodule of main_pretrain_lightning.py and `MAEPretrainModule` under a
# Lightning Trainer, for `--warmup_steps` + `--timed_steps` steps. It answers two
# questions:
#
# - Does the configuration fit? The run either finishes or hits a CUDA OOM, and the
#   peak memory is reported. The 2D mask ratio is pinned to `--mask_ratio_2d_min`, the
#   ratio that keeps the most patches, so the memory is the worst case of the schedule.
# - Is the data loading a bottleneck? Every step is split into the time the training
#   loop waited for its batch (from the end of the previous step to the start of this
#   one: fetch, pin, host-to-device copy) and the time it computed (synchronised on the
#   GPU). Afterwards the train loader is iterated on its own, without the GPU, to
#   measure how many batches per second the input pipeline can supply at most -- the
#   joint loader as a whole, and each source's loader alone.
#
# The Trainer phase runs first, so it reads the data cold, as the first epoch of a
# real run does; the Kermany image cache (in $TMPDIR under SLURM) starts empty here.
# Takes the arguments of main_pretrain_lightning.py, plus the profiling ones below.
# Results go to outputs/profiling/<timestamp>/.

import argparse
import datetime
import gc
import json
import os
import statistics
import time

import lightning.pytorch as pl
import torch
from lightning.fabric.plugins.environments import LightningEnvironment
from lightning.pytorch.strategies import DDPStrategy

from main_pretrain_lightning import build_datamodule
from main_pretrain_lightning import get_args_parser as get_pretrain_args_parser
from pretrain_lightning_module import MAEPretrainModule
from profile_pretrain import PROFILING_ROOT

# a step that waits longer than this fraction of its time for data is input-bound
IO_BOUND_FRACTION = 0.05


def get_args_parser():
    parser = argparse.ArgumentParser("OCTCube pre-training profiling with data loading", parents=[get_pretrain_args_parser()], conflict_handler="resolve")
    parser.add_argument("--warmup_steps", default=20, type=int, help="steps left out of the timings: worker start-up, shuffle buffers, cudnn autotuning")
    parser.add_argument("--timed_steps", default=100, type=int)
    parser.add_argument("--loader_steps", default=100, type=int, help="batches drawn from the train loader alone, after its warmup; 0 skips it")
    parser.add_argument("--train_volumes", default=24024, type=int, help="volumes per epoch, to estimate epoch time (alleye-macula_n30000, folds 0-7)")
    parser.add_argument("--note", default="", type=str, help="goal of the run, saved to results.json and NOTE.md")
    parser.set_defaults(model="flash_attn_mae_vit_large_patch16", mask_ratio=0.9, data_dir="../data/alleye-macula_n30000", no_wandb=True)
    return parser


class StepTimer(pl.Callback):
    """Splits every training step into its wait for data and its compute."""

    def __init__(self, mask_ratio_2d):
        self.mask_ratio_2d = mask_ratio_2d
        self.data_wait, self.compute = [], []
        self._start = self._last_end = self.first_start = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # after `on_train_epoch_start`, which sets the scheduled ratio
        pl_module.mask_ratio_2d = self.mask_ratio_2d
        self._start = time.perf_counter()
        if self.first_start is None:
            self.first_start = self._start
        if self._last_end is not None:
            self.data_wait.append(self._start - self._last_end)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        torch.cuda.synchronize()
        self._last_end = time.perf_counter()
        self.compute.append(self._last_end - self._start)


def summarize(seconds):
    ms = sorted(s * 1000 for s in seconds)
    return {
        "mean_ms": statistics.mean(ms),
        "median_ms": statistics.median(ms),
        "p90_ms": ms[int(0.9 * (len(ms) - 1))],
        "max_ms": ms[-1],
    }


def batch_sizes(args, datamodule):
    """Volumes and B-scans per training batch."""
    if not args.joint_2d:
        return args.batch_size, 0
    return args.batch_size, sum(datamodule.train_batch_sizes_2d.values())


def profile_training(args):
    datamodule = build_datamodule(args)
    model = MAEPretrainModule(args)
    timer = StepTimer(args.mask_ratio_2d_min)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        # a single process, whatever the SLURM allocation says
        plugins=[LightningEnvironment()],
        # the strategy of main_pretrain_lightning.py, so DDP's gradient buckets are counted
        strategy=DDPStrategy(find_unused_parameters=not args.joint_2d),
        use_distributed_sampler=False,
        precision="32-true" if args.fp32 else "16-mixed",
        max_steps=args.warmup_steps + args.timed_steps,
        accumulate_grad_batches=args.accum_iter,
        gradient_clip_val=args.clip_grad,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[timer],
    )
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        trainer.fit(model, datamodule=datamodule)
    except torch.cuda.OutOfMemoryError as error:
        print("out of memory after {} steps: {}".format(len(timer.compute), error))
        return {"fits": False, "steps_done": len(timer.compute), "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3}
    wall_s = time.perf_counter() - start

    batch_size, batch_size_2d = batch_sizes(args, datamodule)
    # the wait before step k is in `data_wait[k - 1]`
    data_wait = timer.data_wait[args.warmup_steps - 1:]
    compute = timer.compute[args.warmup_steps:]
    step = [w + c for w, c in zip(data_wait, compute)]
    step_s = statistics.mean(step)
    compute_s = statistics.mean(compute)
    _, total_mem = torch.cuda.mem_get_info()
    result = {
        "fits": True,
        "batch_size": batch_size,
        "batch_sizes_2d": datamodule.train_batch_sizes_2d if args.joint_2d else {},
        "wall_s": wall_s,
        # model setup, worker start-up and the first batch
        "time_to_first_step_s": timer.first_start - start,
        "step": summarize(step),
        "data_wait": summarize(data_wait),
        "compute": summarize(compute),
        "data_wait_fraction": sum(data_wait) / sum(step),
        "volumes_per_s": batch_size / step_s,
        "bscans_per_s": batch_size_2d / step_s,
        # what the GPU would take if the data were always ready
        "volumes_per_s_compute_only": batch_size / compute_s,
        "bscans_per_s_compute_only": batch_size_2d / compute_s,
        "epoch_h_1gpu": args.train_volumes / (batch_size / step_s) / 3600 if batch_size else None,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
        "gpu_mem_gb": total_mem / 1024**3,
    }
    result["io_bound"] = result["data_wait_fraction"] > IO_BOUND_FRACTION
    del trainer, model, datamodule
    gc.collect()
    torch.cuda.empty_cache()
    return result


def time_loader(loader, steps, warmup_steps):
    """Batches per second a loader supplies on its own, after `warmup_steps`."""
    iterator = iter(loader)
    start = time.perf_counter()
    next(iterator)
    first_batch_s = time.perf_counter() - start
    for _ in range(warmup_steps):
        next(iterator)
    start = time.perf_counter()
    for _ in range(steps):
        next(iterator)
    batches_per_s = steps / (time.perf_counter() - start)
    # shut its workers down, so they do not compete with the next loader's
    del iterator
    if hasattr(loader, "reset"):
        loader.reset()
    gc.collect()
    return {"first_batch_s": first_batch_s, "batches_per_s": batches_per_s}


def profile_loader(args):
    datamodule = build_datamodule(args)
    datamodule.setup("fit")
    loader = datamodule.train_dataloader()
    batch_size, batch_size_2d = batch_sizes(args, datamodule)

    results = {}
    if args.joint_2d:
        sources = {"joint": loader, **loader.iterables}
        sizes = {"joint": (batch_size, batch_size_2d), "3d": (batch_size, 0)}
        sizes.update({name: (0, size) for name, size in datamodule.train_batch_sizes_2d.items()})
    else:
        sources, sizes = {"3d": loader}, {"3d": (batch_size, 0)}
    for name, source in sources.items():
        result = time_loader(source, args.loader_steps, args.warmup_steps)
        result["volumes_per_s"] = sizes[name][0] * result["batches_per_s"]
        result["bscans_per_s"] = sizes[name][1] * result["batches_per_s"]
        print("loader {}: {batches_per_s:.2f} batches/s ({volumes_per_s:.1f} volumes/s, {bscans_per_s:.1f} B-scans/s), "
              "first batch after {first_batch_s:.1f} s".format(name, **result))
        results[name] = result
    return results


def main(args):
    out_dir = PROFILING_ROOT / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.note:
        (out_dir / "NOTE.md").write_text(args.note.strip() + "\n")

    if args.lr is None:
        args.lr = args.blr * args.batch_size * args.accum_iter / 256
    # no checkpoints, no run directory
    args.output_dir = None
    # one epoch must outlast the profiling, or the loader stream ends early
    args.steps_per_epoch = max(args.steps_per_epoch or 0, args.warmup_steps + max(args.timed_steps, args.loader_steps) + 1)
    torch.backends.cudnn.benchmark = True

    print("GPU:", torch.cuda.get_device_name(0))
    print("CPUs available: {}, workers per loader: 3D {}, 2D {}".format(
        len(os.sched_getaffinity(0)), args.num_workers, args.num_workers if args.num_workers_2d is None else args.num_workers_2d))

    training = profile_training(args)
    if training["fits"]:
        print(
            "fits: peak memory {peak_mem_gb:.1f} GB allocated / {peak_reserved_gb:.1f} GB reserved of {gpu_mem_gb:.1f} GB\n"
            "step {s[mean_ms]:.0f} ms (median {s[median_ms]:.0f}), of which data wait {w[mean_ms]:.0f} ms "
            "(median {w[median_ms]:.0f}, p90 {w[p90_ms]:.0f}, max {w[max_ms]:.0f}) = {data_wait_fraction:.1%}, compute {c[mean_ms]:.0f} ms\n"
            "{volumes_per_s:.2f} volumes/s + {bscans_per_s:.1f} B-scans/s (compute only: {volumes_per_s_compute_only:.2f} + "
            "{bscans_per_s_compute_only:.1f}), epoch ~{epoch_h_1gpu:.2f} h on 1 GPU -> {verdict}".format(
                s=training["step"], w=training["data_wait"], c=training["compute"],
                verdict="input-bound" if training["io_bound"] else "compute-bound", **training)
        )

    loader = profile_loader(args) if args.loader_steps > 0 else {}
    if training["fits"] and "3d" in loader:
        # how many times faster the pipeline supplies batches than the GPU consumes them
        consumed_per_s = 1000 / training["compute"]["mean_ms"]
        for result in loader.values():
            result["supply_over_demand"] = result["batches_per_s"] / consumed_per_s

    with open(out_dir / "results.json", "w") as f:
        json.dump({"gpu": torch.cuda.get_device_name(0), "cpus": len(os.sched_getaffinity(0)), "note": args.note,
                   "args": vars(args), "training": training, "loader": loader}, f, indent=2, default=str)
    print("Saved results to {}".format(out_dir))


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
