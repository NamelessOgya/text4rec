#!/bin/bash

# original caseのb4rがちゃんと動くかを試す。
echo "=== test original training using amazon dataset ==="

read -p "delete ./Data/amazon, ./Data/preprocessed (y/n): " -n 1 answer
echo    # 改行を入れる
if [[ $answer =~ [Yy] ]]; then
    echo "deleting..."
    rm -rf ./Data/amazon
    # rm -rf ./Data/preprocessed
    poetry run python main.py --template train_bert \
    --item_embedding_path text4rec/Data/preprocessed/amazon_min_rating4-min_uc5-min_sc0-splitleave_one_out/item_embeddings.npy \
    --bert_hidden_units 1024


else
    echo "中止しました。"
fi
