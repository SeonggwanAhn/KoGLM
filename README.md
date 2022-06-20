# KoGLM
__KoGLM(Korean General Language Model)__ is a Korean pretrained Language model of GLM(General Language Model).

This code is based on [GLM](https://github.com/THUDM/GLM)

## Finetune

## Pretrain
### Environment
1. Create environment

    (1) Using Anaconda

      ```$ conda env create -f koglm_environment.yml```, you need to change ```prefix```, ```name``` according to your environment.

    (2) Other cases
      - Details are in the requirements.txt

2. Install apex

    Additionaly, you need to install apex.

     ```git clone https://github.com/NVIDIA/apex.git && cd apex && python setup.py install --cpp_ext --cuda_ext```

```
bash scripts/koglm_pretrain.sh config/kor_block_base.sh
```

### I will make the README more detail soon...


GLM is a General Language Model pretrained with an autoregressive blank-filling objective and can be finetuned on 
various natural language understanding and generation tasks. 

Please refer to our paper for a detailed description of GLM:

[GLM: General Language Model Pretraining with Autoregressive Blank Infilling](https://arxiv.org/abs/2103.10360)


