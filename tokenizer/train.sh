#!/usr/bin/env bash
./build/train_tokenizer \
    --data-dir /home/carl/datasets/code_10gb.txt \
    --output-dir ./tok_out \
    --vocab-size 32000 \
    --min-frequency 5
