#!/bin/bash
# Copyright (c) Zixuan Liu et al, OCTCubeM group
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Lightning joint 3D + 2D pre-training of OCTCube on alleye-macula + Kermany, with
# self-paced learning, initialised from RETFound, on 2 H100s. Hyperparameters follow
# run_chunks_pretraining_vitl_oph_joint_flash_attn.sh, except the depth (48 B-scans,
# centre-cropped from 49) and a 5-epoch warmup instead of 1. Effective batch sizes
# match the original: 3D 2 x 2 GPUs = 4 (original 1 x 4), 2D 128 x 2 = 256 (64 x 4).
#
# Profiled on one H100 (outputs/profiling/20261028-101557): ~59 GB peak at the
# lowest 2D mask ratio, ~0.5 s/step, data loading < 1% of the step. One epoch is
# 6006 steps per GPU, ~55 min with validation -> ~45 h for 50 epochs.
#
# Submit from anywhere (--chdir sets the working directory to Pre-training/):  sbatch Pre-training/scripts/run_pretrain_lightning_alleye_joint.sh
# --ntasks-per-node and --devices must equal the number of GPUs in --gres: Lightning runs one process per GPU.
# To resume, add --ckpt_path <run dir>/lightning_ckpt/last.ckpt and --output_dir <run dir>.
# --------------------------------------------------------

#SBATCH --job-name=octcube_pretrain_joint
#SBATCH --time=3-00:00:00
#SBATCH --output=/storage/homefs/ms22q288/logs/ubelix/%j-%x.out
#SBATCH --chdir=/storage/homefs/ms22q288/projects/OCTCubeM/Pre-training

#SBATCH --account=invest
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_caim
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=90G


DATA_DIR=../data/alleye-macula_n30000
KERMANY_DIR=../data/ucsd_kermany_2018
RETFOUND_CKPT=../ckpt/RETFound_mae_natureOCT.pth

NGPUS=2
# per GPU; lr = blr * BSZ * ACCUMSTEPS * NGPUS / 256 = 1.6e-3 * 4 / 256 = 2.5e-5
BSZ=2
BSZ_2D=128
INPUTSIZE=256
NUMFRAMES=48
ACCUMSTEPS=1
EPOCHS=50
BLR=1.6e-3
RATIO=0.9
# three loaders per GPU (3D, alleye B-scans, Kermany); 5 workers each keep the
# GPU fed several times over, and leave one CPU of 16 for the training process
WORKERS=5

# each run gets its own directory, outputs/pretraining/$SCHEDULER_<job start time>
SCHEDULER="bsz-$BSZ-bsz2d-$BSZ_2D-gpus-$NGPUS-inputsize-$INPUTSIZE-frames-$NUMFRAMES-aacumsteps$ACCUMSTEPS-ep-$EPOCHS-lr-$BLR-2d512-spl-flash-attn-lightning-retfound"

mkdir -p logs
# the environment is created once with `uv sync`; every srun task uses it as is
source ../.venv/bin/activate

# add --wandb_offline if compute nodes have no internet; upload later with `wandb sync`
srun python main_pretrain_lightning.py \
        --data_dir $DATA_DIR \
        --run_name $SCHEDULER \
        --batch_size $BSZ \
        --accum_iter $ACCUMSTEPS \
        --epochs $EPOCHS \
        --blr $BLR \
        --mask_ratio $RATIO \
        --weight_decay 0.05 \
        --num_workers $WORKERS \
        --num_workers_2d $WORKERS \
        --num_frames $NUMFRAMES \
        --t_patch_size 3 \
        --pred_t_dim $NUMFRAMES \
        --input_size $INPUTSIZE \
        --warmup_epochs 5 \
        --checkpoint_period 10 \
        --resume_type retfound_2_flash_attn \
        --resume $RETFOUND_CKPT \
        --model flash_attn_mae_vit_large_patch16 \
        --joint_2d \
        --use_kermany \
        --kermany_data_dir $KERMANY_DIR \
        --self_paced \
        --batch_size_2d $BSZ_2D \
        --high_res_input_size 512 \
        --mask_ratio_2d_min 0.75 \
        --mask_ratio_2d_max 0.85 \
        --K_min 0.15 \
        --K_max 0.3 \
        --devices $NGPUS \
        --num_nodes $SLURM_NNODES \
        --wandb_project retimod
