#!/bin/bash
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4    # <- or one of: cpu gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
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

# nohup python train_cifar10.py --log-dir tst --exp-name 114514 --deep --net vit_tiny --r 8 &

## mlpmixer exps
# srun python train_cifar10.py --log-dir logs_mlpmixer --exp-name baseline --net mlpmixer --preempt --n_epochs 500 --save-freq 100 --lr 1e-3
# python train_cifar10.py --log-dir logs_mlpmixer --exp-name tst222 --net mlpmixer_factorized --patch-layer 1 --channel-layer 1 --n_epochs 500 --save-freq 100 --lr 1e-3 &
# srun python train_cifar10.py --log-dir logs_CIFAR10_mlpmixer_from --exp-name channel_layer5 --net mlpmixer_factorized --channel-layer 5 --preempt --n_epochs 500 --save-freq 300 --lr 1e-3




# srun python train_gradprojection.py --log-dir logs_projgrad --exp-name r128_freq10_warmup10 --net vit_base --r 128 --project-freq 10 --warmup-epochs 10 --preempt
# srun python train_check_rank.py --log-dir logs_checkrank --exp-name ff --net vit_base --warmup-epochs 20

# srun python train_cifar10.py --log-dir logs_CIFAR10_simpleViT --exp-name baseline --net simplevit --preempt --n_epochs 100
# srun python train_gradprojection.py --log-dir logs_CIFAR10_simpleViT --exp-name GaLore_r16_freq20 --net simplevit --r 16 --project-freq 20 --warmup-epochs 0 --preempt --n_epochs 100 --lr 1e-3
# srun python train_cifar10.py --log-dir logs_CIFAR10_simpleViT --exp-name Factorized_r8 --r 8 --net simplevit_factorized --preempt --n_epochs 200 --lr 1e-3

# srun python train_relora.py --log-dir logs_CIFAR10_simpleViT --exp-name Relora_r16_freq5 --net simplevit --r 16 --merge-freq 5 --n_epochs 100 --preempt --lr 1e-3




# exps on ViTtiny
# srun python train_cifar10.py --log-dir logs_CIFAR10_ViTtiny --exp-name baseline --net vit_tiny --preempt --n_epochs 200 --lr 1e-3
# srun python train_cifar10.py --log-dir logs_CIFAR10_ViTtiny --exp-name Factorized_r8 --net vit_tiny_factorized --n_epochs 200 --lr 5e-3 --r 8
# srun python train_relora.py --log-dir logs_CIFAR10_ViTtiny_testinit --exp-name Relora_r8_freq5 --net vit_tiny --r 8 --merge-freq 5 --n_epochs 200 --preempt --lr 1e-3
# srun python train_relora.py --log-dir logs_CIFAR10_ViTtiny --exp-name Relora_r8_freq5 --net vit_tiny --r 8 --merge-freq 5 --n_epochs 200 --lr 5e-3
# srun python train_gradprojection.py --log-dir logs_CIFAR10_ViTtiny --exp-name GaLore_r8_freq20_run2 --net vit_tiny --r 8 --project-freq 20 --warmup-epochs 0 --n_epochs 200 --lr 1e-3



# srun python train_gradcorrection.py --log-dir logs_CIFAR10_ViTtiny_correction_new --exp-name Relora_r8_freq5_correct0.001 --net vit_tiny --r 8 --merge-freq 5 --n_epochs 200 --lr 5e-3 --correct-coef 0.001
# srun python train_gradcorrection_scheduling.py --log-dir logs_CIFAR10_ViTtiny_correction_scheduling_new --exp-name Relora_cosine_r8_freq5_correct0.1_run3 --net vit_tiny --r 8 --merge-freq 5 --n_epochs 200 --lr 5e-3 --correct-coef-A 0.1 --correct-coef-B 0.1 --correct-scheduling cosine
# srun python train_CIFAR10_testing.py --log-dir tst_better_relora --exp-name Relora_cosine_r8_freq5_correct0.1 --net vit_tiny --r 8 --merge-freq 10 --n_epochs 200 --lr 1e-3


# python train_gradfactorization.py --log-dir tst_factorization --exp-name GaLore_randproject_r8_freq20 --net vit_tiny --r 32 --project-freq 20 --warmup-epochs 0 --n_epochs 200 --lr 1e-3




# exps of changing r
# srun python train_relora_changingr.py --log-dir logs_CIFAR10_ViTtiny_changer --exp-name Relora_r32_freq5 --net vit_tiny --r 32 --merge-freq 5 --n_epochs 200 --lr 5e-3
srun python train_relora.py --log-dir logs_CIFAR10_ViTtiny_changer --exp-name Relora400epoch_warmstart_r8_freq10 --net vit_tiny --r 8 --merge-freq 10 --n_epochs 400 --lr 5e-3

# srun python train_plotrank.py --log-dir logs_compute_rank --exp-name compute_rank --net vit_tiny --compute-freq 5 --n_epochs 200 --lr 1e-3

# srun python train_gradcorrection_scheduling.py --log-dir logs_CIFAR10_ViTtiny_correction_scheduling_new --exp-name Relora_resetoptim_cosine_r8_freq5_correct0.1 --net vit_tiny --r 8 --merge-freq 5 --n_epochs 200 --lr 5e-3 --correct-coef-A 0.1 --correct-coef-B 0.1 --correct-scheduling cosine