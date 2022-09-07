EXPERIMENT_NAME=${MODEL_TYPE}-nsmc-pattern$3
TASK_NAME=NSMC
DATA_PATH="${DATA_ROOT}/nsmc"
MAX_SEQ_LEN=256
# lr 3e-5, epoch 5
LR_SINGLE=9e-6
EPOCH_SINGLE=5

TRAIN_ARGS="--lr-decay-style linear \
            --warmup 0.1 \
            --weight-decay 1.0e-1 \
            --pattern-id $3"

COMMON_ARGS="--save-interval 10000 \
             --log-interval 50 \
             --eval-interval 10000000 \
             --eval-iters 100"

PATTERN_IDS=(0 1 2 3)
PROMPT_IDS=(1 2 3)
# batch 16
BATCH_SIZE=128
