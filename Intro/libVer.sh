#!/bin/bash
#SBATCH --job-name=”test_job”
#SBATCH --qos=debug
#SBATCH --workdir=.
#SBATCH --output=test_job_%j.out
#SBATCH --error=test_job_%j.err
#SBATCH --cpus-per-task=40
#SBATCH --gres gpu:1
#SBATCH --time=00:02:00
module purge; module load numpy/1.22.3 pandas/1.4.1 opencv-python/4.5.5.64 keras/

python versionsOfLib.py