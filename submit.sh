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

export data_name = 'hackathon'
# Learn BPE codes
echo "Learning BPE codes for English"
subword-nmt learn-bpe -s 500000 < ${data_name}_train.en > bpe_en.codes
echo "Learning BPE codes for Thai"
subword-nmt learn-bpe -s 500000 < ${data_name}_train.th > bpe_th.codes

# Apply BPE codes
echo "Splitting data"
subword-nmt apply-bpe -c bpe_en.codes < ${data_name}_train.en > train.bpe.en
subword-nmt apply-bpe -c bpe_th.codes < ${data_name}_train.th > train.bpe.th

subword-nmt apply-bpe -c bpe_en.codes < ${data_name}_valid.en > valid.bpe.en
subword-nmt apply-bpe -c bpe_th.codes < ${data_name}_valid.th > valid.bpe.th

subword-nmt apply-bpe -c bpe_en.codes < ${data_name}_test.en > test.bpe.en
subword-nmt apply-bpe -c bpe_th.codes < ${data_name}_test.th > test.bpe.th

echo "Preprocessing data"
fairseq-preprocess --source-lang en --target-lang th \
    --trainpref train.bpe --validpref valid.bpe --testpref test.bpe \
    --destdir data-bin/translation \
    --workers 20

echo "Training model"
fairseq-train data-bin/translation \
    --arch transformer_wmt_en_de \
    --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --seed 42 \
    --lr 0.0005 --stop-min-lr 1e-09 --max-epoch 10 \
    --weight-decay 0.00005 --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 --max-tokens 8192 --update-freq 2 \
    --log-interval 10 --save-interval-updates 1000 \
    --keep-interval-updates 5 --distributed-world-size 1 \
    --skip-invalid-size-inputs-valid-test


echo "Generating translation"
fairseq-generate data-bin/translation \
    --path checkpoints/translation/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe \
    | grep -P "D-[0-9]+" > train_log.txt
