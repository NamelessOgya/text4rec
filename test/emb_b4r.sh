#!/bin/bash

# This script tests the training process for the BERTEmbeddingModel (bertemb).

echo "=== Test training for BERTEmbeddingModel using Amazon dataset ==="

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
poetry run python main.py


