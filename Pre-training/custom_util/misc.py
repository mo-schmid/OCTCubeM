# Copyright (c) Zixuan Liu et al, OCTCubeM group
# All rights reserved.


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import datetime
import math
import os
import time
from collections import defaultdict, deque, OrderedDict

import custom_util.loggings as logging
import psutil
import torch
import torch.distributed as dist
#import torch.fb.rendezvous.zeus
from matplotlib import pyplot as plt
from iopath.common.file_io import g_pathmgr as pathmgr
from custom_util.loggings import master_print as print
from torch import inf
import torch.nn.functional as F
import numpy as np

logger = logging.get_logger(__name__)


IMG_MEAN = 45.79 / 255
IMG_STD = 76.03 / 255


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "cur:{value:.4f}, {median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )

                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print("[{}] ".format(now), end="")  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(state, path):
    if is_main_process():
        print(f"save path {path}")
        with pathmgr.open(path, "wb") as f:
            torch.save(state, f)


def init_distributed_mode(args):
    if args.no_env:
        pass
    elif args.dist_on_itp:
        args.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        args.world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        args.gpu = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        args.dist_url = "tcp://%s:%s" % (
            os.environ["MASTER_ADDR"],
            os.environ["MASTER_PORT"],
        )
        os.environ["LOCAL_RANK"] = str(args.gpu)
        os.environ["RANK"] = str(args.rank)
        os.environ["WORLD_SIZE"] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}): {}, gpu {}".format(
            args.rank, args.dist_url, args.gpu
        ),
        # flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def cancel_gradients_last_layer(epoch, named_parameters, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in named_parameters:
        if "last_layer" in n:
            p.grad = None


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self, fp32=False):
        self._scaler = torch.cuda.amp.GradScaler(enabled=not fp32)

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True,
        cancel_last_layer_grad=False,
        named_parameters=None,
        epoch_and_freeze_last_layer_gradient_epoch=None,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)

            if cancel_last_layer_grad:
                assert named_parameters and epoch_and_freeze_last_layer_gradient_epoch is not None
                epoch, freeze_last_layer_gradient_epoch = epoch_and_freeze_last_layer_gradient_epoch
                cancel_gradients_last_layer(epoch, named_parameters, freeze_last_layer_gradient_epoch)

            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
            ),
            norm_type,
        )
    return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
    checkpoint_path = "{}/checkpoint-{:05d}.pth".format(args.output_dir, epoch)
    to_save = {
        "model": model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "scaler": loss_scaler.state_dict(),
        "args": args,
    }

    save_on_master(to_save, checkpoint_path)
    return checkpoint_path


def save_model_w_dino(args, epoch, model, model_without_ddp, optimizer, loss_scaler, dino_loss, teacher_model):
    checkpoint_path = "{}/checkpoint-{:05d}.pth".format(args.output_dir, epoch)
    to_save = {
        "model": model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "scaler": loss_scaler.state_dict(),
        "args": args,
        "dino_loss": dino_loss.state_dict(),
        "teacher_model": teacher_model.state_dict(),
    }

    save_on_master(to_save, checkpoint_path)
    return checkpoint_path


def get_last_checkpoint(args):
    """
    Get the last checkpoint from the checkpointing folder.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    d = args.output_dir
    names = pathmgr.ls(d) if pathmgr.exists(d) else []
    names = [f for f in names if "checkpoint" in f]
    if len(names) == 0:
        print("No checkpoints found in '{}'.".format(d))
        return None
    else:
        # Sort the checkpoints by epoch.
        name = sorted(names)[-1]
        return os.path.join(d, name)
    

def convert_released_checkpoint(checkpoint: dict):

    # replace pred_head.transforms.0 with decoder_blocks
    checkpoint_ = {k.replace("pred_head.transforms.0", "decoder_blocks"): v for k, v in checkpoint.items()}
    
    checkpoint_.pop("decoder_pos_embed")
    checkpoint_.pop("pos_embed_spatial")

    for k in list(checkpoint_.keys()):
        if "patch_embed" in k:
            checkpoint_.pop(k)

    return checkpoint_


def load_model(args, model_without_ddp, optimizer, loss_scaler, load_from_same_pretrain=True, not_load_optim=False, convert_pos_embed=False, high_res_model=False, dino_loss=None, teacher_model=None, load_dino=False):
    print('args.resume:', args.resume)
    if not args.resume:
        args.resume = get_last_checkpoint(args)
    if args.init_ckpt:
        print('go into here')
        checkpoint = torch.load(args.init_ckpt, map_location="cpu")
        msg, msg1 = model_without_ddp.load_state_dict_to_backbone(checkpoint["model"])
        print('Missing keys: ', msg, 'unexpected keys: ', msg1)
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            with pathmgr.open(args.resume, "rb") as f:
                checkpoint = torch.load(f, map_location="cpu")
        if not load_from_same_pretrain:
            # load the checkpoint from the released model trained on Kinetics
            model_checkpoint = convert_released_checkpoint(checkpoint["model_state"])
        else:
            model_checkpoint = checkpoint["model"]
        if convert_pos_embed:
            convert_spatial_pos_embed(model_without_ddp, model_checkpoint, high_res_model=high_res_model)
        if high_res_model:
            checkpoint['model']['high_res_patch_embed.proj.weight'] = checkpoint['model']['patch_embed.proj.weight']
            checkpoint['model']['high_res_patch_embed.proj.bias'] = checkpoint['model']['patch_embed.proj.bias']
        msg, msg1 = model_without_ddp.load_state_dict(model_checkpoint, strict=False)
        print('Missing keys: ', msg)
        print('Unexpected keys: ', msg1)
        print("Resume checkpoint %s" % args.resume)
        if (
            "optimizer" in checkpoint
            and "epoch" in checkpoint
            and not (hasattr(args, "eval") and args.eval)
            and not not_load_optim
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"] + 1
            if "scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["scaler"])
            print("With optim & sched!")
        
        if load_dino:
            assert dino_loss is not None and teacher_model is not None
            print("Loading DINO checkpoint with DINO loss and teacher model")
            dino_loss.load_state_dict(checkpoint["dino_loss"])
            teacher_model.load_state_dict(checkpoint["teacher_model"])

# added by zucks
def load_model_retfound(args, model_without_ddp):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        print(model_without_ddp.state_dict()['pos_embed_spatial'].shape)
        read_in_q_k_v(checkpoint['model'], num_hidden_layers=24, hidden_size=1024)
        read_in_q_k_v(checkpoint['model'], num_hidden_layers=8, hidden_size=512, prefix='decoder_')
        interpolate_pos_embed_2Dto3D(model_without_ddp, checkpoint['model'])
        convert_patchembed_2Dto3D(checkpoint['model'])
        msg, msg2 = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print("Resume checkpoint %s" % args.resume)
        print("Missing keys: ", msg)
        print("Unexpected keys: ", msg2)
        # exit()

# added by zucks 0626 help load retfound and imagenet for 3D
def load_model_retfound_flash_attn(args, model_without_ddp, convert_pos_embed=True, high_res_model=True, encoder_only=False, preload_model=None):
    if args.resume or preload_model is not None:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        elif preload_model is not None:
            checkpoint = {'model': preload_model.state_dict()}
            print('preload model: ', preload_model.state_dict().keys())
        else:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)

        if convert_pos_embed:
            # convert_spatial_pos_embed(model_without_ddp, checkpoint['model'], high_res_model=high_res_model)
            interpolate_pos_embed_2Dto3D(model_without_ddp, checkpoint['model'], high_res_patch_embed=high_res_model)
        convert_patchembed_2Dto3D(checkpoint['model'])
        if high_res_model:
            checkpoint['model']['high_res_patch_embed.proj.weight'] = checkpoint['model']['patch_embed.proj.weight']
            checkpoint['model']['high_res_patch_embed.proj.bias'] = checkpoint['model']['patch_embed.proj.bias']

        msg, msg2 = model_without_ddp.load_state_dict_to_backbone_retfound(checkpoint['model'], strict=False, encoder_only=encoder_only)
        # msg, msg1 = model_without_ddp.load_state_dict_to_backbone(checkpoint["model"])
        print("Resume checkpoint %s" % args.resume)
        print("Missing keys: ", msg)
        print("Unexpected keys: ", msg2)


def interpolate_pos_embed_2Dto3D(model, checkpoint_model, high_res_patch_embed=False):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        print("pos_embed_checkpoint shape: ", pos_embed_checkpoint.shape)
        embedding_size = pos_embed_checkpoint.shape[-1]
        cls_token, pos_embed_spatial = torch.split(pos_embed_checkpoint, [1, 196], dim=1) # This only works for retfound and imagenet_224_vitlarge
        print("cls_token shape: ", cls_token.shape, "pos_embed_spatial shape: ", pos_embed_spatial.shape)
        num_patches = model.patch_embed.num_patches // (model.pred_t_dim // model.t_pred_patch_size)
        if high_res_patch_embed:
            num_patches = num_patches * 4
        num_extra_tokens = model.pos_embed_spatial.shape[-2] - num_patches
        print(f"model.patch_embed.num_patches: {model.patch_embed.num_patches}, model.pred_t_dim: {model.pred_t_dim}, model.t_pred_patch_size: {model.t_pred_patch_size}")
        print(f"num_patches: {num_patches}, num_extra_tokens: {num_extra_tokens}")
        print(f"num_extra_tokens: {num_extra_tokens}", model.pos_embed_spatial.shape)
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_spatial.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)

        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size, orig_size, new_size, new_size)
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_spatial[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)

            checkpoint_model["pos_embed_spatial"] = new_pos_embed
            checkpoint_model["pos_embed_class"] = cls_token
            checkpoint_model.pop("pos_embed")

    if "decoder_pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["decoder_pos_embed"]

        embedding_size = pos_embed_checkpoint.shape[-1]
        cls_token, pos_embed_spatial = torch.split(pos_embed_checkpoint, [1, 196], dim=1)

        num_patches = model.patch_embed.num_patches // (model.pred_t_dim // model.t_pred_patch_size)
        if high_res_patch_embed:
            num_patches = num_patches * 4
        num_extra_tokens = model.decoder_pos_embed_spatial.shape[-2] - num_patches

        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_spatial.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)


        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size, orig_size, new_size, new_size)
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_spatial[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)

            checkpoint_model["decoder_pos_embed_spatial"] = new_pos_embed
            checkpoint_model["decoder_pos_embed_class"] = cls_token
            checkpoint_model.pop("decoder_pos_embed")


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (GB).
    """
    if torch.cuda.is_available():
        mem_usage_bytes = torch.cuda.max_memory_allocated()
    else:
        mem_usage_bytes = 0
    return mem_usage_bytes / 1024**3


def cpu_mem_usage():
    """
    Compute the system memory (RAM) usage for the current device (GB).
    Returns:
        usage (float): used memory (GB).
        total (float): total memory (GB).
    """
    vram = psutil.virtual_memory()
    usage = (vram.total - vram.available) / 1024**3
    total = vram.total / 1024**3

    return usage, total


def all_gather(tensors):
    """
    All gathers the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all gather across all processes in
        all machines.
    """

    gather_list = []
    output_tensor = []
    world_size = dist.get_world_size()
    for tensor in tensors:
        tensor_placeholder = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_placeholder, tensor, async_op=False)
        gather_list.append(tensor_placeholder)
    for gathered_tensor in gather_list:
        output_tensor.append(torch.cat(gathered_tensor, dim=0))
    return output_tensor


def add_weight_decay(model, weight_decay=1e-5, skip_list=(), bias_wd=False):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (
            (not bias_wd)
            and len(param.shape) == 1
            or name.endswith(".bias")
            or name in skip_list
        ):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def inflate(model_2d, model_3d):
    state_dict_inflated = OrderedDict()
    for k, v2d in model_2d.items():
        if "patch_embed.proj.weight" in k:
            v3d = model_3d[k]
            v3d = v2d.unsqueeze(2).repeat(1, 1, v3d.shape[2], 1, 1) / v3d.shape[2]
            state_dict_inflated[k] = v3d.clone()
        elif "pos_embed" in k:
            pos_embed_cls, pos_embed_spatial = torch.split(v2d, [1, 196], dim=1)
            state_dict_inflated["pos_embed_cls"] = pos_embed_cls.clone()
            state_dict_inflated["pos_embed"] = pos_embed_spatial.clone()
        else:
            state_dict_inflated[k] = v2d.clone()
    return state_dict_inflated


def convert_checkpoint(model_2d):
    state_dict_inflated = OrderedDict()
    for k, v2d in model_2d.items():
        if "head.projection.weight" in k:
            state_dict_inflated["head.weight"] = v2d.clone()
        elif "head.projection.bias" in k:
            state_dict_inflated["head.bias"] = v2d.clone()
        else:
            state_dict_inflated[k] = v2d.clone()
    return state_dict_inflated


def untransform_image(image):
    return torch.clip((image.float() * IMG_STD + IMG_MEAN) * 255, 0, 255).int()

def untransform_image_no_normalization(image):
    return torch.clip((image.float() ) * 255, 0, 255).int()


def process_and_adjust_mask(mask, pos_idx=16, fill_gap=2, up_down_clear=3, fill_bottom_center=False, fill_holes=False):
    adjusted_mask = np.copy(mask)

    col_median = np.zeros(mask.shape[1])
    for j in range(mask.shape[1]):
        col = adjusted_mask[:, j]

        boundary_dist = int(pos_idx // 8) 
    
        # if find longest sequence not in [0, up_down_clear] and [-up_down_clear:], adjust the top and bottom of the mask
        if j >= boundary_dist and j <= pos_idx - boundary_dist:
            # print('j:', j, 'boundary_dist:', boundary_dist, type(boundary_dist))            adjusted_mask[(pos_idx-up_down_clear):, j] = 1
            longest_start, longest_end = find_longest_zero_sequence(col)
            if longest_start > up_down_clear:
                adjusted_mask[:up_down_clear, j] = 1

        idx_list_pos_is_0 = np.where(col == 0)[0]
        if len(idx_list_pos_is_0) == 0:
            continue
        median = round(np.median(idx_list_pos_is_0))

        col_median[j] = median
        adjusted_mask[median, j] = 0
        for k in range(fill_gap):

            if median - 1 - k >= 0:
                adjusted_mask[median - k - 1, j] = 0
            if median + 1 + k < mask.shape[0]:
                adjusted_mask[median + k + 1, j] = 0

    
    if fill_bottom_center:
        set_bottom_flag = False
        up_mask = adjusted_mask[:, boundary_dist:pos_idx - boundary_dist]

        col_bottom = np.zeros(up_mask.shape[1])
        for bottom in range(pos_idx - up_down_clear, pos_idx//2 , -1):
            row = up_mask[bottom]
            for j in range(len(row)):

                if row[j] == 0 and col_bottom[j] == 0:

                    col_bottom[j] = bottom

            if sum(row) / len(row) < 0.2:
                set_bottom_flag = True
                adjusted_mask[bottom] = 0
                for j in range(boundary_dist, pos_idx - boundary_dist):
                    adjusted_mask[bottom-fill_gap-1:bottom, j] = 0
            if set_bottom_flag:
                break

        if set_bottom_flag == False and np.sum(col_bottom) > 0:
            max_col_bottom = int(np.max(col_bottom))
            for j in range(len(col_bottom)):
                if col_bottom[j] == max_col_bottom:
                    col_bottom[j] -= 1
            max_col_bottom_new = int(np.max(col_bottom))

            sum_max = 0
            for j in range(len(col_bottom)):
                if col_bottom[j] == max_col_bottom_new:
                    sum_max += 1
            if sum_max / len(col_bottom) > 0.8:

                adjusted_mask[max_col_bottom] = 0
                for j in range(boundary_dist, pos_idx - boundary_dist):
                    adjusted_mask[max_col_bottom-fill_gap-1:max_col_bottom, j] = 0

            
    return adjusted_mask, col_median

def find_longest_zero_sequence(nums):
    longest_start = 0
    longest_length = 0
    current_start = None
    current_length = 0

    for i, num in enumerate(nums):
        if num == 0:
            if current_start is None:
                current_start = i
            current_length += 1
        else:
            if current_length > longest_length:
                longest_start = current_start
                longest_length = current_length
            current_start = None
            current_length = 0
    
    # Check at the end of the loop in case the longest sequence ends at the last element
    if current_length > longest_length:
        longest_start = current_start
        longest_length = current_length

    if longest_length == 0:
        return (0, 0)  # or return (-1, -1) if no zero sequence exists

    return (longest_start, longest_start + longest_length - 1)



def fill_patch_mask_to_ratio(mask, to_mask_number=None):
    num_masked = np.sum(mask) 
    if to_mask_number is None:
        to_mask_number = mask.shape[0] * mask.shape[1] // 2
    if num_masked <= to_mask_number:
        return mask
        # raise ValueError('The number of masked patches is less than the target number')
    else:
        num_to_unmask = num_masked - to_mask_number 
    top_idx = np.zeros(mask.shape[1], dtype=np.int32)
    bottom_idx = np.zeros(mask.shape[1], dtype=np.int32)
    bg_dist = mask.shape[1] // 8
    filled_mask = np.copy(mask)
    for j in range(mask.shape[1]):

        top_idx[j], bottom_idx[j] = find_longest_zero_sequence(mask[:, j])

    number_filled = 0
    fill_idx_start_idx = np.copy(bottom_idx) + 1

    index_iter_type = np.zeros(mask.shape[1]) # 0: bottom, 1: top
    while number_filled < num_to_unmask:
        has_unmask_option = False
        for j in range(bg_dist, mask.shape[0] - bg_dist):
            select_idx = fill_idx_start_idx[j]

            if select_idx >= mask.shape[0]:
                select_idx = top_idx[j] - 1
                index_iter_type[j] = 1
                if select_idx < 0:
                    index_iter_type[j] = -1
            
            if filled_mask[select_idx, j] == 0:
                if index_iter_type[j] == -1:
                    continue 
                elif index_iter_type[j] == 1:
                    fill_idx_start_idx[j] -= 1
                    if fill_idx_start_idx[j] < 0:
                        index_iter_type[j] = -1
                elif index_iter_type[j] == 0:
                    fill_idx_start_idx[j] += 1
                    if fill_idx_start_idx[j] >= mask.shape[0]:
                        index_iter_type[j] = 1
                        fill_idx_start_idx[j] = top_idx[j] - 1
                        if fill_idx_start_idx[j] < 0:
                            index_iter_type[j] = -1
                    
            if filled_mask[select_idx, j] == 1 and index_iter_type[j] != -1:

                filled_mask[select_idx, j] = 0
                number_filled += 1
                has_unmask_option = True
                if index_iter_type[j] == 0:
                    fill_idx_start_idx[j] += 1
                elif index_iter_type[j] == 1:
                    fill_idx_start_idx[j] -= 1
                if fill_idx_start_idx[j] >= mask.shape[0]:
                    fill_idx_start_idx[j] = top_idx[j] - 1
                    index_iter_type[j] = 1

                if fill_idx_start_idx[j] < 0:
                    index_iter_type[j] = -1
                    
                if number_filled >= num_to_unmask:
                    break

    return filled_mask

        

        


def show_image(image, title=''):
    # image is [H, W]
    
    if len(image.shape) == 3:
        image = image[0]
    assert len(image.shape) == 2
    plt.imshow(image, cmap='gray')
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def get_mask(x, img_type='3D', high_res=False, high_res_input_size = (20, 32, 32), input_size=(20, 16, 16), p_emb_mask_ratio=0.6, patch_size=16, pre_masks=None, return_tensor=True):
   
    x_norm = F.normalize(x.float(), p=2, dim=2)   
    cosine_similarity = torch.matmul(x_norm, x_norm.transpose(1, 2))
    
    actual_hwt = cosine_similarity.shape[1]

    if img_type == '3D':
        if high_res:
            pos_hw = high_res_input_size[1] * high_res_input_size[2]
            temporal = actual_hwt // pos_hw 
            high_res_input_size = (temporal, high_res_input_size[1], high_res_input_size[2])
            h = int(high_res_input_size[1])
        else:
            pos_hw = input_size[1] * input_size[2]
            temporal = actual_hwt // pos_hw
            input_size = (temporal, input_size[1], input_size[2])
            h = int(input_size[1])      

        diag_cosine_similarity_list = [torch.stack([cosine_similarity[n, i*pos_hw:(i+1)*pos_hw, i*pos_hw:(i+1)*pos_hw] for i in range(temporal)]) for n in range(cosine_similarity.shape[0])]

        num_frames = temporal
        hw = pos_hw
    
    fill_gap = 5 if high_res else 2
    up_down_clear = 6 if high_res else 3
    fill_bottom_center = True if not high_res else False 
    batched_filled_mask_list = []
    s_size = int(pos_hw ** 0.5)
    top_k = int(pos_hw * p_emb_mask_ratio)
    for b in range(cosine_similarity.shape[0]):
        diag_cosine_similarity = diag_cosine_similarity_list[b]
        unmask_ratio = np.zeros(num_frames)
        adjusted_mask_list = np.zeros((num_frames, s_size, s_size))
        for n in range(num_frames):
            
            mask_cos = diag_cosine_similarity[n]
            sum_cosine_similarity = mask_cos.sum(dim=1) / pos_hw

            # get the top p_emb_mask_ratio index from sum_cosine_similarity
            
            _, indices = torch.topk(sum_cosine_similarity, top_k)

            # mask = torch.zeros(pos_hw, pos_hw)
            patched_imgs = torch.zeros(s_size, s_size)
            j_indices = indices % s_size
            i_indices = indices // s_size
            patched_imgs[i_indices, j_indices] = 1

            adjusted_mask, col_median = process_and_adjust_mask(patched_imgs, pos_idx=h, fill_gap=fill_gap, up_down_clear=up_down_clear, fill_bottom_center=fill_bottom_center)
            ratio_unmasked = 1 - np.sum(adjusted_mask) / (pos_hw)
            unmask_ratio[n] = ratio_unmasked
            adjusted_mask_list[n] = adjusted_mask 
            # adjusted_mask_list.append(adjusted_mask)

        max_unmask_ratio = np.max(unmask_ratio)
        max_to_mask_number = int(pos_hw * max_unmask_ratio)
        anchor_num_mask = pos_hw // 2
        actual_to_mask_number = max(max_to_mask_number, anchor_num_mask)
        # print('max_unmask_ratio:', max_unmask_ratio, 'max_to_mask_number:', max_to_mask_number, 'anchor_num_mask:', anchor_num_mask, 'actual_to_mask_number:', actual_to_mask_number)
        filled_mask_list = torch.zeros(num_frames, s_size, s_size)
        for n in range(num_frames):
            adjusted_mask = adjusted_mask_list[n]
            filled_mask = fill_patch_mask_to_ratio(adjusted_mask, to_mask_number=actual_to_mask_number)
            # filled_mask_list.append(torch.tensor(filled_mask) if return_tensor else filled_mask)
            filled_mask_list[n] = torch.tensor(filled_mask)
        # filled_mask_list = torch.stack(filled_mask_list, dim=0) if return_tensor else filled_mask_list
        batched_filled_mask_list.append(filled_mask_list)
    return batched_filled_mask_list

def get_patch_embed_images(vars_: dict, model, save_dir: str, img_type='2D', p_emb_mask_ratio=0.6, patch_size=16, use_pre_mask=False):
    frame_n = len(vars_['img_names'])

    samples = vars_['samples'].detach().cpu()
    x = vars_['patch_embed'].detach().cpu()
    if use_pre_mask:
        pre_masks = vars_['pre_mask'].detach().cpu()
    h = samples.shape[-1] / patch_size
    w = samples.shape[-2] / patch_size
    hw = h * w
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        input_size = model.module.input_size
        if hasattr(model.module, 'high_res_input_size'):
            high_res_input_size = model.module.high_res_input_size

        else:
            high_res_input_size = 0

    else:
        input_size = model.input_size
        if hasattr(model, 'high_res_input_size'):
            high_res_input_size = model.high_res_input_size

        else:
            high_res_input_size = 0

    if w == input_size[1]:
        high_res = False
    elif w == high_res_input_size[1]:
        high_res = True
    else:
        raise ValueError('Invalid input size')

    # get cosine similarity
    x_norm = F.normalize(x.float(), p=2, dim=2)  # Normalize over the feature dimension

    # Compute pairwise cosine similarity:

    cosine_similarity = torch.matmul(x_norm, x_norm.transpose(1, 2))
    print('cosine_similarity:', cosine_similarity.shape)
    if img_type == '3D':
        if high_res:
            pos_hw = high_res_input_size[1] * high_res_input_size[2]
            temporal = high_res_input_size[0]
        else:
            pos_hw = input_size[1] * input_size[2]
            temporal = input_size[0]
        imgs = samples[0, 0].detach().cpu() 
        diag_cosine_similarity = torch.stack([cosine_similarity[0, i*pos_hw:(i+1)*pos_hw, i*pos_hw:(i+1)*pos_hw] for i in range(temporal)])
        print('diag_cosine_similarity:', diag_cosine_similarity.shape)
        cosine_similarity = torch.stack([diag_cosine_similarity[i//3, :, :] for i in range(0, temporal*3)])
        print('cosine_similarity:', cosine_similarity.shape)
        img_name_list = [save_dir + '/' + vars_['img_names'][0] + '/' + f'frame_{z}_mask.png' for z in range(temporal*3)]
        os.makedirs(os.path.join(save_dir, vars_['img_names'][0]), exist_ok=True)
    elif img_type == '2D':
        img_name_list = []
        for f_idx in range(frame_n):
            os.makedirs(save_dir + '/visible_2d/' + '/'.join(vars_['img_names'][f_idx].split('/')[:-1]), exist_ok=True)
            img_name_list.append(save_dir + '/visible_2d/' + '.'.join(vars_['img_names'][f_idx].split('.')[:-1]) + f'_mask_{samples.shape[-1]}.png')
        imgs = samples[:, 0].detach().cpu()
        if high_res:
            pos_hw = high_res_input_size[1] * high_res_input_size[2]
        else:
            pos_hw = input_size[1] * input_size[2]

    for n in range(len(img_name_list)):
        img_name = img_name_list[n]

        img = imgs[n]
        mask_cos = cosine_similarity[n]
        mask = torch.zeros(pos_hw, pos_hw)
        sum_cosine_similarity = cosine_similarity[n].sum(dim=1) / pos_hw

        # get the top 50% index from sum_cosine_similarity
        top_k = int(pos_hw * p_emb_mask_ratio)
        _, indices = torch.topk(sum_cosine_similarity, top_k)

        s_size = int(pos_hw ** 0.5)
        patched_imgs = torch.zeros(s_size, s_size)
        num_patched_imgs = torch.zeros(s_size, s_size)
        num_mask = torch.zeros(pos_hw, pos_hw)

        # set the mask to 1 for the top 50% index with patch size 16
        for indx in indices:
            i_th_patch = indx // s_size 
            j_th_patch = indx % s_size

            mask[i_th_patch*s_size:(i_th_patch+1)*s_size, j_th_patch*s_size:(j_th_patch+1)*s_size] = 1
            patched_imgs[i_th_patch, j_th_patch] = 1
        for i in range(pos_hw):
            i_th_patch = i // s_size
            j_th_patch = i % s_size
            num_patched_imgs[i_th_patch, j_th_patch] = sum_cosine_similarity[i]
            num_mask[i_th_patch*s_size:(i_th_patch+1)*s_size, j_th_patch*s_size:(j_th_patch+1)*s_size] = sum_cosine_similarity[i]
        num_patched_imgs = num_patched_imgs.detach().cpu().numpy()
        num_mask = num_mask.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()

        fill_gap = 5 if high_res else 2
        up_down_clear = 6 if high_res else 3
        fill_bottom_center = True if not high_res else False
        if use_pre_mask:
            pre_mask = pre_masks[n]

            interpolated_pre_mask = torch.nn.functional.interpolate(pre_mask.float(), size=(s_size, s_size), mode='nearest')
            pre_mask = torch.round(interpolated_pre_mask).bool()

            patched_imgs[pre_mask[0,0]] = 1

        adjusted_mask, col_median = process_and_adjust_mask(patched_imgs, pos_idx=int(h), fill_gap=fill_gap, up_down_clear=up_down_clear, fill_bottom_center=fill_bottom_center)
        ratio_unmasked = 1 - np.sum(adjusted_mask) / (adjusted_mask.shape[0] * adjusted_mask.shape[1])
        filled_mask = fill_patch_mask_to_ratio(adjusted_mask, to_mask_number=None)
        


        # make the plt figure larger
        plt.figure()
        plt.rcParams['figure.figsize'] = [36, 4]
        num_imgs = 6
        plt.subplot(1, num_imgs, 1)
        show_image(img, "original")

        plt.subplot(1, num_imgs, 2)
        show_image(mask_cos, "masked cos similarity")

        plt.subplot(1, num_imgs, 3)
        show_image(num_mask, "num_mask")

        plt.subplot(1, num_imgs, 4)
        show_image(mask, f"mask w/ ratio {p_emb_mask_ratio}")
        
        plt.subplot(1, num_imgs, 5)
        show_image(adjusted_mask, f"adjusted mask w/ ratio {ratio_unmasked:.2f}")


        plt.subplot(1, num_imgs, 6)
        filled_mask_ratio = 1 - np.sum(filled_mask) / (filled_mask.shape[0] * filled_mask.shape[1])
        show_image(filled_mask, f"filled mask w/ ratio {filled_mask_ratio:.2f}")
        plt.savefig(img_name)

        plt.close('all')



def get_visible_images_2d(vars_: dict, model, save_dir: str):
    
    frame_n = len(vars_['img_names'])
    frames = vars_['reconstruct_imgs'].detach().cpu() 
    masks = vars_['mask'].detach().cpu()
    samples = vars_['samples'].detach().cpu()
    hw = masks.shape[-1]
    h = int(hw ** 0.5)
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        input_size = model.module.input_size
        if hasattr(model.module, 'high_res_input_size'):
            high_res_input_size = model.module.high_res_input_size

        else:
            high_res_input_size = 0

        # high_res_input_size = model.module.high_res_input_size
    else:
        input_size = model.input_size
        if hasattr(model, 'high_res_input_size'):
            high_res_input_size = model.high_res_input_size

        else:
            high_res_input_size = 0
        # high_res_input_size = model.high_res_input_size
    if h == input_size[1]:
        high_res = False
    elif h == high_res_input_size[1]:
        high_res = True
    else:
        raise ValueError('Invalid input size')
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        if high_res:
            frames = model.module.unpatchify(frames, high_res=high_res)
            masks = masks.unsqueeze(-1).repeat(1, 1, vars_['reconstruct_imgs'].shape[-1])  # (N, H*W, p*p*3)

            masks = model.module.unpatchify(masks, high_res=high_res)  # 1 is removing, 0 is keeping
        else:
            frames = model.module.unpatchify(frames)
            masks = masks.unsqueeze(-1).repeat(1, 1, vars_['reconstruct_imgs'].shape[-1])
            masks = model.module.unpatchify(masks)

    else:
        if high_res:
            frames = model.unpatchify(frames, high_res=high_res)
            masks = masks.unsqueeze(-1).repeat(1, 1, vars_['reconstruct_imgs'].shape[-1])  # (N, H*W, p*p*3)
            masks = model.unpatchify(masks, high_res) 
        else:
            frames = model.unpatchify(frames)
            masks = masks.unsqueeze(-1).repeat(1, 1, vars_['reconstruct_imgs'].shape[-1])
            masks = model.unpatchify(masks)

    for i in range(frame_n):
        frame_i = frames[i].squeeze()
        frame_i = untransform_image_no_normalization(frame_i)
        mask = masks[i].squeeze()
        img_name_i = vars_['img_names'][i]

        x = samples[i].detach().cpu().squeeze() # shape: [Z, H, W]
        x = untransform_image_no_normalization(x)
        
        # masked image
        im_masked = x * (1 - mask)
        # MAE reconstruction pasted with visible patches
        im_paste = x * (1 - mask) + frame_i * mask
        
        # save the Z slices of the volume


        plt.figure()

        # make the plt figure larger
        plt.rcParams['figure.figsize'] = [24, 4]

        plt.subplot(1, 4, 1)
        show_image(x, "original")

        plt.subplot(1, 4, 2)
        show_image(im_masked, "masked")

        plt.subplot(1, 4, 3)
        show_image(frame_i, "reconstruction")

        plt.subplot(1, 4, 4)
        show_image(im_paste, "reconstruction_paste")
        
        os.makedirs(save_dir + '/visible_2d/' + '/'.join(vars_['img_names'][i].split('/')[:-1]), exist_ok=True)
        plt.savefig(save_dir + '/visible_2d/' + vars_['img_names'][i])

        plt.close('all')

def get_visible_images(vars_: dict, model, save_dir: str, pad_to_pred_t_dim=True, offset: int = 0, suffix: str ='', high_res: bool = False):
    # high_res_input_size = (20, 32, 32)
    # input_size = (20, 16, 16)
    s_size = 16 if not high_res else 32
    print('model patchembed:', model.module.patch_embed.input_size)
    input_size = model.module.patch_embed.input_size

    vol_n = len(vars_['img_names'])
    volumes = vars_['reconstruct_imgs'].detach().cpu()
    actual_t_dim = int(volumes.shape[1] / (input_size[1] * input_size[2]) * 3)
    print('actual_t_dim:', actual_t_dim)
    masks = vars_['mask'].detach().cpu()
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        volumes = model.module.unpatchify(volumes, high_res=high_res, actual_t_dim=actual_t_dim)
        masks = masks.unsqueeze(-1).repeat(1, 1, vars_['reconstruct_imgs'].shape[-1])  # (N, H*W, p*p*3)
        masks = model.module.unpatchify(masks, high_res=high_res, actual_t_dim=actual_t_dim)  # 1 is removing, 0 is keeping
        print('actual_t_dim:', actual_t_dim, 'volumes:', volumes.shape)
        pred_t_dim = model.module.pred_t_dim if pad_to_pred_t_dim else actual_t_dim
    else:
        volumes = model.unpatchify(volumes, high_res=high_res, actual_t_dim=actual_t_dim)
        masks = masks.unsqueeze(-1).repeat(1, 1, vars_['reconstruct_imgs'].shape[-1])  # (N, H*W, p*p*3)
        masks = model.unpatchify(masks, high_res=high_res, actual_t_dim=actual_t_dim)
        pred_t_dim = model.pred_t_dim if pad_to_pred_t_dim else actual_t_dim
        
    samples = torch.index_select(
            vars_['samples'],
            2,
            torch.linspace(
                0,
                vars_['samples'].shape[2] - 1,
                pred_t_dim,
            )
            .long()
            .to(vars_['samples'].device),
        )
    
    for i in range(vol_n):
        vol_i = volumes[i].squeeze()
        vol_i = untransform_image(vol_i)
        # vol_i = untransform_image_no_normalization(vol_i)
        mask = masks[i].squeeze()

        x = samples[i].detach().cpu().squeeze() # shape: [Z, H, W]
        x = untransform_image(x)
        # x = untransform_image_no_normalization(x)
        
        # masked image
        im_masked = x * (1 - mask)
        # MAE reconstruction pasted with visible patches
        im_paste = x * (1 - mask) + vol_i * mask
        
        # save the Z slices of the volume

        for z in range(vol_i.shape[0]):

            plt.figure()
            # make the plt figure larger
            plt.rcParams['figure.figsize'] = [24, 4]

            plt.subplot(1, 4, 1)
            show_image(x[z], "original")

            plt.subplot(1, 4, 2)
            show_image(im_masked[z], "masked")

            plt.subplot(1, 4, 3)
            show_image(vol_i[z], "reconstruction")

            plt.subplot(1, 4, 4)
            show_image(im_paste[z], "reconstruction_paste")
            
            os.makedirs(os.path.join(save_dir, vars_['img_names'][i]), exist_ok=True)
            actual_idx = z + offset
            plt.savefig(os.path.join(save_dir, vars_['img_names'][i], f'frame_{actual_idx}{suffix}.png'))
            plt.close('all')



# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, num_hidden_layers, hidden_size, prefix=""):
    for i in range(num_hidden_layers):

        # read in weights + bias of input projection layer (in timm, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"{prefix}blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"{prefix}blocks.{i}.attn.qkv.bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"{prefix}blocks.{i}.attn.q.weight"] = in_proj_weight[
            : hidden_size, :
        ]
        state_dict[f"{prefix}blocks.{i}.attn.q.bias"] = in_proj_bias[: hidden_size]
        state_dict[f"{prefix}blocks.{i}.attn.k.weight"] = in_proj_weight[
            hidden_size : hidden_size * 2, :
        ]
        state_dict[f"{prefix}blocks.{i}.attn.k.bias"] = in_proj_bias[
            hidden_size : hidden_size * 2
        ]
        state_dict[f"{prefix}blocks.{i}.attn.v.weight"] = in_proj_weight[
            -hidden_size :, :
        ]
        state_dict[f"{prefix}blocks.{i}.attn.v.bias"] = in_proj_bias[-hidden_size :]

def convert_patchembed_2Dto3D(state_dict):
    in_proj_weight = state_dict.pop("patch_embed.proj.weight")
    new_in_project_weight = in_proj_weight.unsqueeze(1)
    state_dict["patch_embed.proj.weight"] = new_in_project_weight





def convert_spatial_pos_embed(model, checkpoint_model, high_res_model=False):
    interpolate_flag = False
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        pos_embed_name = 'pos_embed'
        interpolate_flag = True
    elif 'pos_embed_spatial' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed_spatial']
        pos_embed_name = 'pos_embed_spatial'
        interpolate_flag = True
    if 'pos_embed' in model.state_dict():
        model_pos_embed = model.state_dict()['pos_embed']
        model_pos_embed_name = 'pos_embed'
    elif 'pos_embed_spatial' in model.state_dict():
        model_pos_embed = model.state_dict()['pos_embed_spatial']
        model_pos_embed_name = 'pos_embed_spatial'

    if interpolate_flag:
        embedding_size = pos_embed_checkpoint.shape[-1]
        if model_pos_embed_name == 'pos_embed':
            if high_res_model:
                num_patches = model.high_res_patch_embed.num_patches
            else:
                num_patches = model.patch_embed.num_patches
            num_extra_tokens = model_pos_embed.shape[-2] - num_patches
        elif model_pos_embed_name == 'pos_embed_spatial':
            if high_res_model:
                num_patches = model.high_res_patch_embed.num_patches // (model.high_res_patch_embed.frames // model.high_res_patch_embed.t_patch_size)
            else:
                num_patches = model.patch_embed.num_patches // (model.patch_embed.frames // model.patch_embed.t_patch_size)
            num_extra_tokens = model_pos_embed.shape[-2] - num_patches
        print(pos_embed_checkpoint.shape, num_patches, num_extra_tokens)
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)


        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(f"Position interpolate {pos_embed_name}" + " from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model.pop(pos_embed_name)
            checkpoint_model[model_pos_embed_name] = new_pos_embed

    decoder_interpolate_flag = False
    if 'decoder_pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['decoder_pos_embed']
        pos_embed_name = 'decoder_pos_embed'
        decoder_interpolate_flag = True
    elif 'decoder_pos_embed_spatial' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['decoder_pos_embed_spatial']
        pos_embed_name = 'decoder_pos_embed_spatial'
        decoder_interpolate_flag = True
    if 'decoder_pos_embed' in model.state_dict():
        model_pos_embed = model.state_dict()['decoder_pos_embed']
        model_pos_embed_name = 'decoder_pos_embed'
    elif 'decoder_pos_embed_spatial' in model.state_dict():
        model_pos_embed = model.state_dict()['decoder_pos_embed_spatial']
        model_pos_embed_name = 'decoder_pos_embed'

    if decoder_interpolate_flag:
        embedding_size = pos_embed_checkpoint.shape[-1]
        if model_pos_embed_name == 'decoder_pos_embed':
            if high_res_model:
                num_patches = model.high_res_patch_embed.num_patches
            else:
                num_patches = model.patch_embed.num_patches
            num_extra_tokens = model_pos_embed.shape[-2] - num_patches
        elif model_pos_embed_name == 'decoder_pos_embed_spatial':
            if high_res_model:
                num_patches = model.high_res_patch_embed.num_patches // (model.high_res_patch_embed.frames // model.high_res_patch_embed.t_patch_size)
            else:
                num_patches = model.patch_embed.num_patches // (model.patch_embed.frames // model.patch_embed.t_patch_size)
            num_extra_tokens = model_pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        print(new_size, orig_size)

        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(f"Decoder: Position interpolate {pos_embed_name}" + " from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model[pos_embed_name] = new_pos_embed



def find_and_convert_large_white_region(image, img_type='tensor'):
    if img_type == 'tensor':
        max_val = torch.max(image)
        min_val = torch.min(image)

        min_max_gap = max_val - min_val
        thresh = max_val - (255 - 240) / 255 * min_max_gap
        if len(image.shape) == 3:
            gray_image = image[0].clone()
        else:
            gray_image = image.clone()
        white_mask = gray_image > thresh 
        top_row = torch.argmax(torch.any(white_mask, dim=1).int())
        bottom_row = len(white_mask) - torch.argmax(torch.any(white_mask.flip(0), dim=1).int())

        connected_region = torch.zeros_like(gray_image, dtype=torch.bool)
        for row in range(top_row, bottom_row):
            first_white_pixel = np.argmax(white_mask[row].numpy())

            center_area = row > len(white_mask) * 0.2 and row < len(white_mask) - len(white_mask) * 0.2
            if first_white_pixel > 0.01 * len(white_mask[row]):
                pass
            else:
                for col in range(first_white_pixel, len(white_mask[row])):
                    if not white_mask[row, col]:
                        if col > 0.05 * len(white_mask[row]):
                            connected_region[row, col:col+5] = True
                        break
                    connected_region[row, col] = True
            last_white_pixel = len(white_mask[row]) - torch.argmax(white_mask[row].flip(0).int()) - 1
            if last_white_pixel < len(white_mask[row]) - 0.01 * len(white_mask[row]):
                pass
            else:        
                for col in range(last_white_pixel, -1, -1):
                    if not white_mask[row, col]:
                        if col < len(white_mask[row]) - 0.05 * len(white_mask[row]):
                            connected_region[row, col:col-5] = True
                        break
                    connected_region[row, col] = True
        gray_image[connected_region] = 0
        if len(image.shape) == 3:
            return_img = torch.repeat_interleave(gray_image.unsqueeze(0), image.shape[0], dim=0)
            connected_region = torch.repeat_interleave(connected_region.unsqueeze(0), image.shape[0], dim=0)
        else:
            return_img = gray_image
            
        return return_img, connected_region

    elif img_type == 'pil':
        # Convert image to grayscale
        gray_image = np.array(image.convert('L'))

        # Threshold the image to find white regions
        white_mask = gray_image > 240


        # Find top and bottom rows that are all white pixels
        top_row = np.argmax(np.any(white_mask, axis=1))

        bottom_row = len(white_mask) - np.argmax(np.any(white_mask[::-1], axis=1))

        # Initialize connected region
        connected_region = np.zeros_like(gray_image, dtype=bool)

        # Scan from top_row to bottom_row
        for row in range(top_row, bottom_row):
            # Find the first white pixel in the row
            first_white_pixel = np.argmax(white_mask[row])
            center_area = row > len(white_mask) * 0.2 and row < len(white_mask) - len(white_mask) * 0.2
            if first_white_pixel > 0.01 * len(white_mask[row]):
                pass
            else:

                # Expand the connected region horizontally until the first non-white pixel
                for col in range(first_white_pixel, len(white_mask[row])):
                    if not white_mask[row, col]:
                        if col > 0.05 * len(white_mask[row]):
                            connected_region[row, col:col+5] = True
                        break
                    connected_region[row, col] = True

            # Find the last white pixel in the row
            last_white_pixel = len(white_mask[row]) - np.argmax(white_mask[row][::-1]) - 1
            if last_white_pixel < len(white_mask[row]) - 0.01 * len(white_mask[row]):
                pass
            else:        
                # Expand the connected region horizontally until the first non-white pixel
                for col in range(last_white_pixel, -1, -1):
                    if not white_mask[row, col]:
                        if col < len(white_mask[row]) - 0.05 * len(white_mask[row]):
                            connected_region[row, col:col-5] = True
                        break
                    connected_region[row, col] = True

        # Convert the connected region to black
        gray_image[connected_region] = 0

        return Image.fromarray(gray_image)