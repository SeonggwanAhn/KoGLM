EXPERIMENT_NAME=${MODEL_TYPE}-kornli-pattern$3
TASK_NAME=kornli
DATA_PATH="${DATA_ROOT}/klue-nli-v1.1"
MAX_SEQ_LEN=256
# best hparmas
# lr: 2e-5 & batch 16
LR_SINGLE=2e-5
EPOCH_SINGLE=20

TRAIN_ARGS="--lr-decay-style linear \
            --warmup 0.1 \
            --weight-decay 1.0e-1 \
            --pattern-id $3"

COMMON_ARGS="--save-interval 10000 \
             --log-interval 50 \
             --eval-interval 10000 \
             --eval-iters 100"

PATTERN_IDS=(0 1 2 3)
PROMPT_IDS=(1 2 3)

BATCH_SIZE=16
