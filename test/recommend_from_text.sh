#!/bin/bash

# This script runs the text-based recommendation engine with a trained BERTEmbeddingModel.

# --- Configuration ---
DATASET_CODE="amazon"
EXPERIMENT_DIR="experiments"
# Use the experiment that was trained with the bertemb model
EXPERIMENT_DESCRIPTION="bertemb_final_try_2025-08-24_2" 
EPOCH_NUM=200 # The epoch number of the saved model

# --- Model Parameters (must match the training configuration) ---
BERT_MAX_LEN=100
BERT_HIDDEN_UNITS=1024 # This must be 1024 for the E5-based model
BERT_NUM_BLOCKS=2
BERT_NUM_HEADS=4
BERT_DROPOUT=0.1
MODEL_INIT_SEED=0

# --- Path to pre-generated item embeddings ---
ITEM_EMBEDDING_PATH="Data/preprocessed/amazon_min_rating4-min_uc5-min_sc0-splitleave_one_out/item_embeddings.npy"

# --- Run Recommendation ---
# The script will read text from sample.txt and recommend items.
echo "Starting text-based recommendation engine..."
echo "Experiment: $EXPERIMENT_DESCRIPTION"
echo "Dataset: $DATASET_CODE"
echo "Model Epoch: $EPOCH_NUM"
echo ""

poetry run python recommend.py \
    --dataset_code $DATASET_CODE \
    --model_code "bert_embedding" \
    --dataloader_code "bert_embedding" \
    --experiment_dir $EXPERIMENT_DIR \
    --experiment_description $EXPERIMENT_DESCRIPTION \
    --epoch_num $EPOCH_NUM \
    --item_embedding_path $ITEM_EMBEDDING_PATH \
    --bert_max_len $BERT_MAX_LEN \
    --bert_hidden_units $BERT_HIDDEN_UNITS \
    --bert_num_blocks $BERT_NUM_BLOCKS \
    --bert_num_heads $BERT_NUM_HEADS \
    --bert_dropout $BERT_DROPOUT \
    --model_init_seed $MODEL_INIT_SEED \
    --device "cpu" < sample.txt
