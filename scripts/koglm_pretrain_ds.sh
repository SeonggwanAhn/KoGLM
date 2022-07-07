#! /bin/bash

source $1
N_GPU=2
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
DISTRIBUTED_ARGS="--nproc_per_node ${N_GPU} --nnodes 1 --node_rank 0 --master_addr localhost --master_port $MASTER_PORT"
NUM_WORKERS=1
NUM_GPUS_PER_WORKER=2
HOST_FILE_PATH=/home/tempuser/deepspeed/hostfile
OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"

run_cmd="${OPTIONS_NCCL} deepspeed --master_port ${MASTER_PORT} --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE_PATH} pretrain_glm.py ${gpt_options}"

echo ${run_cmd}
eval ${run_cmd}

set +x
