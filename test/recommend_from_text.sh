#!/bin/bash

# This script runs the recommendation engine with a pre-trained model.
# It uses a specific experiment configuration and starts an interactive session
# for providing item sequences to get recommendations.

# --- Configuration ---
# These values are based on the experiment's config.json
DATASET_CODE="amazon"
EXPERIMENT_DIR="experiments"
EXPERIMENT_DESCRIPTION="test_2025-08-24_22"
EPOCH_NUM=200 # Example epoch, can be changed to any saved epoch number

# BERT Model Parameters from the experiment's config
BERT_MAX_LEN=100
BERT_HIDDEN_UNITS=256
BERT_NUM_BLOCKS=2
BERT_NUM_HEADS=4
BERT_DROPOUT=0.1
MODEL_INIT_SEED=0

# --- Run Recommendation ---
# The script will start and wait for user input.
# Example input for the prompt: 1 2 3
echo "Starting recommendation engine..."
echo "Experiment: $EXPERIMENT_DESCRIPTION"
echo "Dataset: $DATASET_CODE"
echo "Model Epoch: $EPOCH_NUM"
echo ""

poetry run python recommend.py \
    --dataset_code $DATASET_CODE \
    --experiment_dir $EXPERIMENT_DIR \
    --experiment_description $EXPERIMENT_DESCRIPTION \
    --epoch_num $EPOCH_NUM \
    --bert_max_len $BERT_MAX_LEN \
    --bert_hidden_units $BERT_HIDDEN_UNITS \
    --bert_num_blocks $BERT_NUM_BLOCKS \
    --bert_num_heads $BERT_NUM_HEADS \
    --bert_dropout $BERT_DROPOUT \
    --model_init_seed $MODEL_INIT_SEED \
    --device "cpu" # Use "cuda" if a GPU is available
