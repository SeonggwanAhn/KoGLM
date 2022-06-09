EXPERIMENT_NAME=${MODEL_TYPE}-korquad
TASK_NAME=korquad
DATA_PATH="/data3/sgahn/seq2seq/korquad/"

TRAIN_ARGS="--epochs 10 \
            --batch-size 16 \
            --lr 3e-5 \
            --lr-decay-style linear \
            --warmup 0.06 \
            --weight-decay 1.0e-1 \
            --label-smoothing 0.1"

COMMON_ARGS="--save-interval 10000 \
             --log-interval 50 \
             --eval-interval 1000 \
             --eval-iters 100 \
             --eval-epoch 10"

TASK_ARGS="--src-seq-length 464 \
           --tgt-seq-length 48 \
           --min-tgt-length 0 \
           --length-penalty 0.7 \
           --num-beams 5 \
           --select-topk \
           --eval-batch-size 4"
