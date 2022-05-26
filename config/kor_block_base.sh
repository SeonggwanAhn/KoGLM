#! /bin/bash

script_path=$(realpath $BASH_SOURCE)
script_dir=$(dirname $script_path)

config_json="$script_dir/config_block_base.json"
gpt_options=" \
       --block-lm \
       --bert-prob 1.0 \
       --experiment-name blocklm-blank \
       --model-parallel-size 1 \
       --num-layers 12 \
       --hidden-size 768 \
       --num-attention-heads 12 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --save /data/sgahn/ckpt \
       --train-iters 150000 \
       --resume-dataloader \
       --train-data nikl_daily nikl_messenger nikl_newspaper nikl_newspaper_2020 nikl_newspaper_2021 nikl_spoken nikl_written nikl_om \
       --tokenizer-type BertWordPieceTokenizer \
       --tokenizer-model-type monologg/koelectra-base-v3-discriminator \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --lr-decay-iters 120000 \
       --lr-decay-ratio 0.05 \
       --warmup .05 \
       --checkpoint-activations \
       --batch-size 64 \
       --fp16

"
