#!/bin/bash

# original caseのb4rがちゃんと動くかを試す。
echo "=== test original training using amazon dataset ==="

read -p "delete ./Data/amazon, ./Data/preprocessed (y/n): " -n 1 answer
echo    # 改行を入れる
if [[ $answer =~ [Yy] ]]; then
    echo "deleting..."
    rm -rf ./Data/amazon
    rm -rf ./Data/preprocessed
    poetry run python main.py --template train_bert --dataset_code amazon --generate_item_embeddings \
    --bert_hidden_units 1024


else
    echo "中止しました。"
fi
