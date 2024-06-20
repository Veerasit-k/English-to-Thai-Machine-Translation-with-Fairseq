#!/bin/bash
#SBATCH -p compute
#SBATCH -N 1 -c 128
#SBATCH --ntasks-per-node=1
#SBATCH -t 3:00:00
#SBATCH -A ck900706
#SBATCH -J test

module load Mamba/23.11.0-0
conda activate pytorch-2.2.2
python3 prep_data.py