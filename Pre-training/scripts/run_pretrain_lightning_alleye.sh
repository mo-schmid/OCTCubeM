#!/bin/bash
# Copyright (c) Zixuan Liu et al, OCTCubeM group
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Lightning pre-training of OCTCube on alleye-macula (3D only), initialised from
# RETFound. Hyperparameters follow run_chunks_pretraining_vitl_oph_joint_flash_attn.sh,
# except the depth: 48 B-scans (centre-cropped from 49, never resized).
#
# Submit from Pre-training/:  sbatch scripts/run_pretrain_lightning_alleye.sh
# --ntasks-per-node must equal --gpus-per-node: Lightning runs one process per GPU.
# --------------------------------------------------------

#SBATCH --job-name=octcube-pretrain
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

DATA_DIR=../data/alleye-macula_n30000
RETFOUND_CKPT=../ckpt/RETFound_mae_natureOCT.pth

BSZ=1
INPUTSIZE=256
NUMFRAMES=48
ACCUMSTEPS=1
EPOCHS=50
BLR=1.6e-3
RATIO=0.9

# each run gets its own directory, outputs/pretraining/$SCHEDULER_<job start time>
SCHEDULER="bsz-$BSZ-inputsize-$INPUTSIZE-frames-$NUMFRAMES-aacumsteps$ACCUMSTEPS-ep-$EPOCHS-lr-$BLR-flash-attn-lightning-retfound"

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
        --num_workers $SLURM_CPUS_PER_TASK \
        --num_frames $NUMFRAMES \
        --t_patch_size 3 \
        --pred_t_dim $NUMFRAMES \
        --input_size $INPUTSIZE \
        --warmup_epochs 1 \
        --resume_type retfound_2_flash_attn \
        --resume $RETFOUND_CKPT \
        --model flash_attn_mae_vit_large_patch16 \
        --devices $SLURM_NTASKS_PER_NODE \
        --num_nodes $SLURM_NNODES \
        --wandb_project retimod
