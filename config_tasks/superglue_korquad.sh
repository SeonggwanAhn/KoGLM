EXPERIMENT_NAME=${MODEL_TYPE}-korquad-superglue
TASK_NAME=korquad
DATA_PATH="/data/sgahn/seq2seq/korquad/"

LR_SINGLE=1e-5
EPOCH_SINGLE=10
MAX_SEQ_LEN=464

TRAIN_ARGS="--lr-decay-style linear \
            --warmup 0.1 \
			--loss-func mix \
            --weight-decay 1.0e-1 \
			--length-penalty 1"
            # --label-smoothing 0.1"

COMMON_ARGS="--save-interval 10000 \
             --log-interval 50 \
             --eval-interval 10000 \
             --eval-iters 100 \
			 --eval-epoch 1"

PATTERN_IDS=(0)

BATCH_SIZE=8
