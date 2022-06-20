# KoGLM
__KoGLM(Korean General Language Model)__ is a Korean version of GLM(General Language Model).

This code is based on [GLM](https://github.com/THUDM/GLM)

## Setup
1. A conda environment is used

    ```$ conda env create -f koglm_environment.yml``` you need to change ```prefix```, ```name``` according to your environment.

2. Install apex

    Additionaly, you need to install apex.

     ```git clone https://github.com/NVIDIA/apex.git && cd apex && python setup.py install --cpp_ext --cuda_ext```


## Pretrain
```
$ bash scripts/koglm_pretrain.sh config/kor_block_base.sh
```

## Finetune
On NSMC task
```
$ bash scripts/finetune_superglue.sh config_tasks/koglm_blocklm_base.sh config_tasks/task_nsmc_pattern.sh (pattern-id)
```


### I will make the README more detail soon...


GLM is a General Language Model pretrained with an autoregressive blank-filling objective and can be finetuned on 
various natural language understanding and generation tasks. 

Please refer to our paper for a detailed description of GLM:

[GLM: General Language Model Pretraining with Autoregressive Blank Infilling](https://arxiv.org/abs/2103.10360)


