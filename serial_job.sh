#!/bin/bash
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4     # <- or one of: cpu gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bcel-delta-gpu
#SBATCH --job-name=lora_training
#SBATCH --time=04:00:00      # hh:mm:ss for the job
##SBATCH --constraint="scratch"
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,per_task:1
##SBATCH --gpu-bind=none     # <- or closest
##SBATCH --mail-user=zhangzekai2001@gmail.com
##SBATCH --mail-type="BEGIN,END" See sbatch or srun man pages for more email options

module purge
module reset
echo "job is starting on `hostname`"
module load anaconda3_gpu
module list  # job documentation and metadata


echo "job is starting on `hostname`"

source activate /u/zzhang14/.conda/envs/pytorch


## ViT exps
# srun python train_cifar10.py --log-dir logs_new --exp-name ff_layer6_deep --ff-layer 6 --deep --net vit_base_factorized --preempt --r 40 
# srun python train_cifar10.py --log-dir logs_from --exp-name ff_layer3 --ff-layer 3 --net vit_base_factorized --preempt --r 40 
# srun python train_cifar10.py --log-dir logs_from --exp-name attn_layer1_deep --attn-layer 1 --deep --net vit_base_factorized --preempt --r 40 
# srun python train_cifar10.py --log-dir logs_from --exp-name baseline --net vit_base --preempt

# nohup python train_cifar10.py --log-dir tst --exp-name 114514 --ff-layer 1 --deep --net vit_base_factorized --preempt --r 40 &

## mlpmixer exps
# srun python train_cifar10.py --log-dir logs_mlpmixer --exp-name baseline --net mlpmixer --preempt --n_epochs 500 --save-freq 100 --lr 1e-3
# python train_cifar10.py --log-dir logs_mlpmixer --exp-name tst222 --net mlpmixer_factorized --patch-layer 1 --channel-layer 1 --n_epochs 500 --save-freq 100 --lr 1e-3 &
# srun python train_cifar10.py --log-dir logs_CIFAR10_mlpmixer_from --exp-name channel_layer5 --net mlpmixer_factorized --channel-layer 5 --preempt --n_epochs 500 --save-freq 300 --lr 1e-3




# srun python train_gradprojection.py --log-dir logs_projgrad --exp-name r128_freq10_warmup10 --net vit_base --r 128 --project-freq 10 --warmup-epochs 10 --preempt
srun python train_check_rank.py --log-dir logs_checkrank --exp-name ff --net vit_base --warmup-epochs 20
