# bash scripts/finetune_seq2seq_distributed.sh config_tasks/koglm_blocklm_base.sh config_tasks/task_aihub.sh (news/opinion)

EXPERIMENT_NAME=${MODEL_TYPE}-aihub_summary
TASK_NAME=aihub_summary
DATA_PATH="/data/sgahn/seq2seq/aihub-summary/$3"

LR_SINGLE=1e-5
EPOCH_SINGLE=15
BATCH_SINGLE=8

TRAIN_ARGS="--lr-decay-style linear \
            --warmup 0.06 \
            --weight-decay 1.0e-1 \
            --label-smoothing 0.1"

COMMON_ARGS="--save-interval 10000 \
             --log-interval 50 \
             --eval-interval 1000 \
             --eval-iters 100 \
             --eval-epoch 2"

TASK_ARGS="--src-seq-length 608 \
           --tgt-seq-length 160 \
           --min-tgt-length 55 \
           --length-penalty 0.7 \
           --no-repeat-ngram-size 3 \
           --num-beams 5 \
           --select-topk \
           --eval-batch-size 1"
