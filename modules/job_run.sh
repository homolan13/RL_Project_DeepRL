#!/bin/bash

#SBATCH --mail-user=yanis.schaerer@students.unibe.ch
#SBATCH --mail-type=FAIL,END

#SBATCH --job-name="Simglucose DDPG"

#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx2080ti:2

# singularity exec --nv docker://pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime pip freeze | xargs pip uninstall -y
singularity exec --nv docker://pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime pip install -U -e simglucose_local # Gym will also be installed

singularity exec --nv docker://pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime python ubelix_train_yanis.py
