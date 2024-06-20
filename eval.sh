#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -t 24:00:00
#SBATCH -A ck900706
#SBATCH -J test

module load Mamba/23.11.0-0
conda activate pytorch-2.2.2

echo "Generating translation"
fairseq-generate data-bin/translation \
    --path checkpoints/translation/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe \
    | grep -P "D-[0-9]+" > test_log.txt
