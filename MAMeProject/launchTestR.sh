#!/bin/bash

#sBATCH --job-name="ResNet"

#SBATCH --qos=training

#SBATCH --workdir=.

#SBATCH --output=ResNet_%j.out

#SBATCH --error=ResNet_%j.err

#SBATCH --cpus-per-task=40

#SBATCH --gres gpu:1

#SBATCH --time=10:00:00

module purge; module load gcc/8.3.0 ffmpeg/4.2.1 cuda/10.2 cudnn/7.6.4 nccl/2.4.8 tensorrt/6.0.1 openmpi/4.0.1 atlas/3.10.3 scalapack/2.0.2 fftw/3.3.8 szip/2.1.1 opencv/4.1.1 python/3.7.4_ML

python trainerR.py
