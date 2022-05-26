#! /bin/bash

source $1
# rm -rf /data/sgahn/*.lazy/
python pretrain_glm.py ${gpt_options}
