#!/bin/sh

EXP="exp_tgt_src"
gpus=0

# Source Task (checkpoint)
SOURCE_TASK='[source task]'
SOURCE_CKPT='best_vanilla_prompt/open_book/[source task]/best_model.pkl'

# Target Task 
TARGET_TASK="[target task]"
CONFIG="config/[target task]PromptT5Base.config"
WANDB_RUN="[Wandb Run 이름]"

echo "==========================="
echo Source task: ${SOURCE_TASK}
echo Checkpoint: ${SOURCE_CKPT}
echo Target task: ${TARGET_TASK}
echo Config: ${CONFIG}
echo "==========================="


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config $CONFIG \
    --gpu $gpus \
    --checkpoint $SOURCE_CKPT \
    --source_task $SOURCE_TASK \
    --exp $EXP \
