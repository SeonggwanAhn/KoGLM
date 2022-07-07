#! /bin/bash

script_path=$(realpath $BASH_SOURCE)
script_dir=$(dirname $script_path)

config_json="$script_dir/config_kor_block_base.json"
gpt_options=" \
       --block-lm \
       --bert-prob 1.0 \
       --experiment-name blocklm-blank \
       --model-parallel-size 2 \
       --num-layers 12 \
       --hidden-size 768 \
       --num-attention-heads 12 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --save /home/tempuser/ckpt \
       --train-iters 1500000 \
       --resume-dataloader \
       --train-data nikl_newspaper online_review nikl_kparlty nikl_daily nikl_newspaper_2020 nikl_messenger nikl_newspaper_2021 nikl_spoken nikl_written nikl_om \
       --tokenizer-type BertWordPieceTokenizer \
       --tokenizer-model-type monologg/koelectra-base-v3-discriminator \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --lr-decay-ratio 0.05 \
       --warmup .05 \
       --checkpoint-activations \
       --adam-beta2 0.98 \
       --adam-eps 1e-6 \
       --weight-decay 0.1 \
	   --lr 0.0004 \
       --deepspeed-activation-checkpointing \
       --fp16 \
       --eval-interval 10000 \
	   --save-interval 100000 \
"
gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"
