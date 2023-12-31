#!/bin/bash

model=SocialLGN
decay=1e-4
lr=0.001
layer=3
seed=2020
topks="[10,20]"
recdim=64
bpr_batch=2048

# important parameter
dataset=ciao
epochs=1
k_rec=10
n=3

rm -r "./data/preprocessed/${dataset}_data/"
mkdir "./data/preprocessed/${dataset}_data/"

for timestamp_idx in $(seq 0 $(($n - 1)))
do
    python main.py --epochs $epochs --model=$model --dataset=$dataset --decay=$decay --lr=$lr --layer=$layer --seed=$seed --topks=$topks --recdim=$recdim --bpr_batch=$bpr_batch --timestamp_idx=$timestamp_idx
    python recommend.py --rec_topk $k_rec --timestamp_idx=$timestamp_idx --dataset=$dataset
done
