# Copyright (c) Zixuan Liu et al, OCTCubeM group
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Short profiling of an OCTCube pre-training step on synthetic volumes, on one GPU.
# It runs the step `MAEPretrainModule` runs under `Trainer(precision="16-mixed")`:
# forward, loss, backward and AdamW step under fp16 autocast with a grad scaler. The
# data pipeline is left out on purpose -- random volumes are generated on the GPU.
#
# Takes the arguments of main_pretrain_lightning.py (model, input size, frames,
# mask ratio, ...), plus the profiling ones below. Results go to
# outputs/profiling/<timestamp>/.

import argparse
import datetime
import json
import os
import time
from pathlib import Path

import torch

import custom_util.misc as misc
import models_mae_joint_res_flash_attn as models_mae
from main_pretrain_lightning import get_args_parser as get_pretrain_args_parser

PROFILING_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "profiling"


def get_args_parser():
    parser = argparse.ArgumentParser("OCTCube pre-training profiling", parents=[get_pretrain_args_parser()], conflict_handler="resolve")
    parser.add_argument("--batch_sizes", default=[1, 2, 4], type=int, nargs="+", help="per-GPU batch sizes to time")
    parser.add_argument("--warmup_steps", default=5, type=int)
    parser.add_argument("--timed_steps", default=20, type=int)
    parser.add_argument("--trace_steps", default=3, type=int, help="steps recorded by torch.profiler at the first batch size; 0 disables it")
    parser.add_argument("--train_volumes", default=24024, type=int, help="volumes per epoch, to estimate epoch time (alleye-macula_n30000, folds 0-7)")
    # the model and mask ratio of run_pretrain_lightning_alleye.sh
    parser.set_defaults(model="flash_attn_mae_vit_large_patch16", mask_ratio=0.9)
    return parser


def synthetic_batch(args, batch_size, device):
    input_size = args.input_size if isinstance(args.input_size, tuple) else (args.input_size, args.input_size)
    return torch.rand(batch_size, args.in_chans, args.num_frames, *input_size, device=device)


def train_step(model, optimizer, scaler, samples, args, timings=None):
    """One step as in `MAEPretrainModule.training_step`, with per-phase CUDA timings."""
    events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    events[0].record()
    with torch.autocast("cuda", dtype=torch.float16, enabled=not args.fp32):
        loss, _, _ = model(samples, mask_ratio=args.mask_ratio)
    events[1].record()
    scaler.scale(loss).backward()
    events[2].record()
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


def profile_batch_size(model, optimizer, scaler, args, batch_size, device):
    samples = synthetic_batch(args, batch_size, device)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.warmup_steps):
        train_step(model, optimizer, scaler, samples, args)
    torch.cuda.synchronize()

    timings = {"forward": [], "backward": [], "optimizer": []}
    start = time.perf_counter()
    for _ in range(args.timed_steps):
        loss = train_step(model, optimizer, scaler, samples, args, timings)
    torch.cuda.synchronize()
    step_s = (time.perf_counter() - start) / args.timed_steps

    volumes_per_s = batch_size / step_s
    return {
        "batch_size": batch_size,
        "step_ms": step_s * 1000,
        "forward_ms": sum(timings["forward"]) / len(timings["forward"]),
        "backward_ms": sum(timings["backward"]) / len(timings["backward"]),
        "optimizer_ms": sum(timings["optimizer"]) / len(timings["optimizer"]),
        "volumes_per_s": volumes_per_s,
        "epoch_h_1gpu": args.train_volumes / volumes_per_s / 3600,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "loss": loss.item(),
    }


def record_trace(model, optimizer, scaler, args, batch_size, device, out_dir):
    samples = synthetic_batch(args, batch_size, device)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(args.trace_steps):
            train_step(model, optimizer, scaler, samples, args)
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(out_dir / "trace.json"))
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=25)
    (out_dir / "top_ops.txt").write_text(table)
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

    results = []
    for batch_size in args.batch_sizes:
        try:
            result = profile_batch_size(model, optimizer, scaler, args, batch_size, device)
        except torch.cuda.OutOfMemoryError:
            print("batch size {}: out of memory".format(batch_size))
            results.append({"batch_size": batch_size, "oom": True})
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue
        results.append(result)
        print(
            "batch size {batch_size}: {step_ms:.0f} ms/step (fwd {forward_ms:.0f} / bwd {backward_ms:.0f} / "
            "optim {optimizer_ms:.0f}), {volumes_per_s:.2f} volumes/s, epoch of {n} volumes ~{epoch_h_1gpu:.2f} h "
            "on 1 GPU, peak memory {peak_mem_gb:.1f} GB".format(n=args.train_volumes, **result)
        )

    if args.trace_steps > 0:
        table = record_trace(model, optimizer, scaler, args, args.batch_sizes[0], device, out_dir)
        print(table)

    with open(out_dir / "results.json", "w") as f:
        json.dump({"gpu": torch.cuda.get_device_name(0), "args": vars(args), "results": results}, f, indent=2, default=str)
    print("Saved results to {}".format(out_dir))


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
