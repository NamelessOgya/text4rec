#!/bin/bash

# This script tests the training process for the BERTModel (original bert4rec).
# It uses the same data sampling as emb_b4r.sh.

echo "=== Test training for BERTModel using Amazon dataset ==="

read -p "This will delete ./Data/amazon and ./Data/preprocessed. Continue? (y/n): " -n 1 answer
echo
if [[ $answer =~ [Yy] ]]; then
    echo "Deleting preprocessed data..."
    rm -rf ./Data/amazon
    rm -rf ./Data/preprocessed
else
    echo "data deletion skip: use original data."
fi


echo "Starting training..."
poetry run python main.py \
    --mode train \
    --dataset_code amazon \
    --model_code bert \
    --dataloader_code bert \
    --trainer_code bert \
    --experiment_description "bert_test_from_script" \
    --device cuda \
    --num_gpu 1 \
    --min_rating 0 \
    --min_uc 5 \
    --min_sc 0 \
    --split leave_one_out \
    --train_batch_size 128 \
    --val_batch_size 128 \
    --test_batch_size 128 \
    --train_negative_sampler_code random \
    --train_negative_sample_size 0 \
    --train_negative_sampling_seed 0 \
    --test_negative_sampler_code random \
    --test_negative_sample_size 100 \
    --test_negative_sampling_seed 98765 \
    --optimizer Adam \
    --lr 0.00001 \
    --enable_lr_schedule \
    --decay_step 25 \
    --gamma 1.0 \
    --num_epochs 1200 \
    --metric_ks 1 5 10 20 50 100 \
    --best_metric "NDCG@10" \
    --model_init_seed 0 \
    --bert_dropout 0.1 \
    --bert_hidden_units 256 \
    --bert_mask_prob 0.15 \
    --bert_max_len 100 \
    --bert_num_blocks 2 \
    --bert_num_heads 4
