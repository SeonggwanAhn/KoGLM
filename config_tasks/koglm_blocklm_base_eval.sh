MODEL_TYPE="kor-blank-base"
MODEL_ARGS="--block-lm \
            --num-layers 12 \
            --hidden-size 768 \
            --num-attention-heads 12 \
            --max-position-embeddings 512 \
            --tokenizer-model-type monologg/koelectra-base-v3-discriminator \
            --tokenizer-type BertWordPieceTokenizer \
            --load ${CHECKPOINT_PATH}/kor-blank-base-nsmc-pattern4-05-24-01-07/"
