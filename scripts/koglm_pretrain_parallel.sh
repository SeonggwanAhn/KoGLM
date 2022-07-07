#! /bin/bash

source $1
N_GPU=2
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
DISTRIBUTED_ARGS="--nproc_per_node ${N_GPU} --nnodes 1 --node_rank 0 --master_addr localhost --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_glm.py ${gpt_options}
