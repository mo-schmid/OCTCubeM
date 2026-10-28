# Copyright (c) Zixuan Liu et al, OCTCubeM group
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Short profiling of an OCTCube pre-training step on synthetic volumes, on one GPU.
# It runs the step `MAEPretrainModule` runs under `Trainer(precision="16-mixed")`:
# forward, loss, backward and AdamW step under fp16 autocast with a grad scaler. The
# data pipeline is left out on purpose -- random volumes are generated on the GPU.
# With `--batch_sizes_2d`, each step also runs the 2D branch of `--joint_2d` training
# on that many random B-scans at `--high_res_input_size`, at the lowest 2D mask ratio
# (`--mask_ratio_2d_min`), which keeps the most patches and so needs the most memory.
# A batch size of 0 leaves the 3D branch out, so `--batch_sizes 0` times pure 2D steps.
# `--find_max 3d 2d` instead searches the largest batch size that fits for pure 3D and
# pure 2D steps (doubling from `--find_max_start`, then bisecting), timing each probe.
#
# Takes the arguments of main_pretrain_lightning.py (model, input size, frames,
# mask ratio, ...), plus the profiling ones below. Results go to
# outputs/profiling/<timestamp>/.

import argparse
import datetime
import itertools
import json
import os
import time
from pathlib import Path

import torch

import custom_util.misc as misc
import models_mae_joint_res_flash_attn as models_mae
from main_pretrain_lightning import get_args_parser as get_pretrain_args_parser
from octcubem.transforms.defaults import octcube_2d_input

PROFILING_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "profiling"


def get_args_parser():
    parser = argparse.ArgumentParser("OCTCube pre-training profiling", parents=[get_pretrain_args_parser()], conflict_handler="resolve")
    parser.add_argument("--batch_sizes", default=[1, 2, 4], type=int, nargs="+", help="per-GPU batch sizes to time")
    parser.add_argument("--batch_sizes_2d", default=[0], type=int, nargs="+", help="per-GPU 2D batch sizes to time with each batch size; 0 is 3D only")
    parser.add_argument("--warmup_steps", default=5, type=int)
    parser.add_argument("--timed_steps", default=20, type=int)
    parser.add_argument("--trace_steps", default=3, type=int, help="steps recorded by torch.profiler at the first 3D and 2D batch sizes; 0 disables it")
    parser.add_argument("--train_volumes", default=24024, type=int, help="volumes per epoch, to estimate epoch time (alleye-macula_n30000, folds 0-7)")
    parser.add_argument("--find_max", default=[], choices=["3d", "2d"], nargs="+", help="search the max pure 3D / pure 2D batch size instead of timing --batch_sizes x --batch_sizes_2d")
    parser.add_argument("--find_max_start", default=8, type=int, help="first batch size of the --find_max search")
    parser.add_argument("--note", default="", type=str, help="goal of the run, saved to results.json and NOTE.md")
    # the model and mask ratio of run_pretrain_lightning_alleye.sh
    parser.set_defaults(model="flash_attn_mae_vit_large_patch16", mask_ratio=0.9)
    return parser


def synthetic_batch(args, batch_size, device):
    if batch_size == 0:
        return None
    input_size = args.input_size if isinstance(args.input_size, tuple) else (args.input_size, args.input_size)
    return torch.rand(batch_size, args.in_chans, args.num_frames, *input_size, device=device)


def synthetic_batch_2d(args, batch_size, device):
    """Random B-scans as `MAEPretrainModule` feeds them: 3 identical frames, one temporal patch."""
    if batch_size == 0:
        return None
    image = torch.rand(batch_size, args.in_chans, 1, args.high_res_input_size, args.high_res_input_size, device=device)
    return octcube_2d_input(image)


def train_step(model, optimizer, scaler, samples, args, timings=None, samples_2d=None):
    """One step as in `MAEPretrainModule.training_step`, with per-phase CUDA timings."""
    events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    # the labels group the ops of a torch.profiler trace by phase
    record = torch.profiler.record_function
    events[0].record()
    with torch.autocast("cuda", dtype=torch.float16, enabled=not args.fp32):
        loss = 0
        if samples is not None:
            with record("forward_3d"):
                loss, _, _ = model(samples, mask_ratio=args.mask_ratio)
        if samples_2d is not None:
            with record("forward_2d"):
                loss_2d, _, _ = model(samples_2d, mask_ratio=args.mask_ratio_2d_min)
            loss = loss + loss_2d
    events[1].record()
    with record("backward"):
        scaler.scale(loss).backward()
    events[2].record()
    with record("optimizer"):
        scaler.unscale_(optimizer)
        if args.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    events[3].record()

    if timings is not None:
        torch.cuda.synchronize()
        timings["forward"].append(events[0].elapsed_time(events[1]))
        timings["backward"].append(events[1].elapsed_time(events[2]))
        timings["optimizer"].append(events[2].elapsed_time(events[3]))
    return loss


def profile_batch_size(model, optimizer, scaler, args, batch_size, batch_size_2d, device):
    samples = synthetic_batch(args, batch_size, device)
    samples_2d = synthetic_batch_2d(args, batch_size_2d, device)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.warmup_steps):
        train_step(model, optimizer, scaler, samples, args, samples_2d=samples_2d)
    torch.cuda.synchronize()

    timings = {"forward": [], "backward": [], "optimizer": []}
    start = time.perf_counter()
    for _ in range(args.timed_steps):
        loss = train_step(model, optimizer, scaler, samples, args, timings, samples_2d=samples_2d)
    torch.cuda.synchronize()
    step_s = (time.perf_counter() - start) / args.timed_steps

    volumes_per_s = batch_size / step_s
    return {
        "batch_size": batch_size,
        "batch_size_2d": batch_size_2d,
        "step_ms": step_s * 1000,
        "forward_ms": sum(timings["forward"]) / len(timings["forward"]),
        "backward_ms": sum(timings["backward"]) / len(timings["backward"]),
        "optimizer_ms": sum(timings["optimizer"]) / len(timings["optimizer"]),
        "volumes_per_s": volumes_per_s,
        "bscans_per_s": batch_size_2d / step_s,
        "epoch_h_1gpu": args.train_volumes / volumes_per_s / 3600 if batch_size else None,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "loss": loss.item(),
    }


def try_profile(model, optimizer, scaler, args, batch_size, batch_size_2d, device):
    """`profile_batch_size`, or an OOM record if the step does not fit."""
    try:
        result = profile_batch_size(model, optimizer, scaler, args, batch_size, batch_size_2d, device)
    except torch.cuda.OutOfMemoryError:
        print("batch size {} + 2D {}: out of memory".format(batch_size, batch_size_2d))
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return {"batch_size": batch_size, "batch_size_2d": batch_size_2d, "oom": True}
    print(
        "batch size {batch_size} + 2D {batch_size_2d}: {step_ms:.0f} ms/step (fwd {forward_ms:.0f} / bwd {backward_ms:.0f} / "
        "optim {optimizer_ms:.0f}), {volumes_per_s:.2f} volumes/s, {bscans_per_s:.2f} B-scans/s, "
        "peak memory {peak_mem_gb:.1f} GB".format(**result)
    )
    return result


def find_max_batch_size(model, optimizer, scaler, args, mode, device):
    """Largest pure 3D or pure 2D batch size that fits: double until OOM, then bisect."""
    fits, oom = 0, None
    batch_size = args.find_max_start
    results = []
    while oom is None or oom - fits > 1:
        sizes = (batch_size, 0) if mode == "3d" else (0, batch_size)
        result = try_profile(model, optimizer, scaler, args, *sizes, device)
        results.append(result)
        if result.get("oom"):
            oom = batch_size
        else:
            fits = batch_size
        batch_size = fits * 2 if oom is None else (fits + oom) // 2
    best = next((r for r in results if not r.get("oom") and max(r["batch_size"], r["batch_size_2d"]) == fits), None)
    print("max pure {} batch size: {}".format(mode.upper(), fits))
    return {"max_batch_size": fits, "at_max": best, "probes": results}


def record_trace(model, optimizer, scaler, args, batch_size, batch_size_2d, device, out_dir):
    samples = synthetic_batch(args, batch_size, device)
    samples_2d = synthetic_batch_2d(args, batch_size_2d, device)
    # untraced steps first, so cudnn autotuning and the allocator's growth stay out of the trace
    for _ in range(2):
        train_step(model, optimizer, scaler, samples, args, samples_2d=samples_2d)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(args.trace_steps):
            train_step(model, optimizer, scaler, samples, args, samples_2d=samples_2d)
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(out_dir / "trace.json"))
    events = prof.key_averages()
    # by self time, so nested ops are not counted twice; the phase labels show the split
    table = events.table(sort_by="self_cuda_time_total", row_limit=40, max_name_column_width=80)
    (out_dir / "top_ops.txt").write_text(table)
    # the same ops per input shape, to tell the 3D and 2D calls of one op apart
    by_shape = prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=60, max_name_column_width=60, max_shapes_column_width=100)
    (out_dir / "top_ops_by_shape.txt").write_text(by_shape)
    return table


def main(args):
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    out_dir = PROFILING_ROOT / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.lr is None:
        args.lr = args.blr * args.batch_sizes[0] * args.accum_iter / 256

    model = models_mae.__dict__[args.model](**vars(args)).to(device)
    model.train(True)
    param_groups = misc.add_weight_decay(model, args.weight_decay, bias_wd=args.bias_wd)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=args.beta or (0.9, 0.95), foreach=True)
    scaler = torch.amp.GradScaler("cuda", enabled=not args.fp32)

    print("GPU:", torch.cuda.get_device_name(0))
    print("input: {} x {} frames x {}, mask ratio {}, {}".format(
        args.in_chans, args.num_frames, args.input_size, args.mask_ratio, "fp32" if args.fp32 else "fp16 autocast"))

    if args.note:
        (out_dir / "NOTE.md").write_text(args.note.strip() + "\n")

    results, max_batch_sizes = [], {}
    if args.find_max:
        for mode in args.find_max:
            max_batch_sizes[mode] = find_max_batch_size(model, optimizer, scaler, args, mode, device)
    else:
        for batch_size, batch_size_2d in itertools.product(args.batch_sizes, args.batch_sizes_2d):
            results.append(try_profile(model, optimizer, scaler, args, batch_size, batch_size_2d, device))

    if args.trace_steps > 0:
        table = record_trace(model, optimizer, scaler, args, args.batch_sizes[0], args.batch_sizes_2d[0], device, out_dir)
        print(table)

    with open(out_dir / "results.json", "w") as f:
        json.dump({"gpu": torch.cuda.get_device_name(0), "note": args.note, "args": vars(args), "results": results,
                   "max_batch_sizes": max_batch_sizes}, f, indent=2, default=str)
    print("Saved results to {}".format(out_dir))


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
