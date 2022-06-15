DATA_ROOT=/data/sgahn/seq2seq
CHECKPOINT_PATH=/data/sgahn/ckpt
SAVE_PATH=/data/sgahn/ckpt
DATESTR=$(date +"%m-%d-%H-%M")

source $1    # Model
source $2    # Task

N_GPU=1
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
DISTRIBUTED_ARGS="--nproc_per_node ${N_GPU} --nnodes 1 --node_rank 0 --master_addr localhost --master_port $MASTER_PORT"
TOKENIZERS_PARALLELISM=false

EXPERIMENT_NAME=${EXPERIMENT_NAME}-${DATESTR}
python -m torch.distributed.launch $DISTRIBUTED_ARGS finetune_glm.py \
       --finetune \
       --experiment-name ${EXPERIMENT_NAME} \
       --task ${TASK_NAME} \
       --data-dir ${DATA_PATH} \
       --save ${CHECKPOINT_PATH} \
       --checkpoint-activations \
       --batch-size ${BATCH_SINGLE} \
       --epochs ${EPOCH_SINGLE} \
       --lr ${LR_SINGLE} \
       $MODEL_ARGS \
       $TRAIN_ARGS \
       $COMMON_ARGS \
       $TASK_ARGS \
       2>&1 | tee /data/sgahn/logs/log-${EXPERIMENT_NAME}.txt

