#!/bin/bash
echo "=== test original training using amazon dataset ==="

read -p "dekete ./Data/amazon, ./Data/preprocessed (y/n): " -n 1 answer
echo    # 改行を入れる
if [[ $answer =~ [Yy] ]]; then
    echo "deleting..."
    rm -rf ./Data/amazon
    rm -rf ./Data/preprocessed
    poetry run python main.py --template train_bert

else
    echo "中止しました。"
fi
