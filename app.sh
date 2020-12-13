#!/bin/bash
#
#SBATCH --job-name=spine-segmentation
#SBATCH --time=1:00:00
#SBATCH --mail-type=END
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
python main.py ~/cmpe257/spine-segmentation/data