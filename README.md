# Is Prompt Transfer Always Effective? An Empirical Study of Prompt Transfer for Question Answering
Our work is accepted to NAACL 2024🎇

This repository include the code for our paper "Is Prompt Transfer Always Effective? An Empirical Study of Prompt Transfer for Question Answering (NAACL short, 2024)."

# Overview
We study the effectiveness of prompt transfer in Question Answering(QA) tasks. We characterize the QA task based on features and empirically investigate the impact of various factors on the performance of PoT using 16 QA datasets. We analyze the impact of initialization during prompt transfer and find that the train dataset size of source and target tasks have the influence significantly. Furthermore, we propose a novel approach for measuring catastrophic forgetting and investigate how it occurs in terms of the amount of evidence. Our findings can help deeply understand transfer learning in prompt tuning.

## Description of Codes

### Setups
Pull an image from a registry:
```
docker pull ssoyaavv/pot
```

### How to Run
```
bash scripts/run_train.sh
```
#### Example: VanillaPT 
```
# !/bin/sh
# mkdir OPTForCausalLM
gpus=0

DATASET="[target task]"
BACKBONE="T5Base"
CONFIG="config/[target task]PromptOPT.config"
WANDB_PROJ="[Wandb project name]"
WANDB_RUN="[Wandb Run name]"

echo "==========================="
echo Backbone model: ${BACKBONE}
echo Config: ${CONFIG}
echo Dataset: ${DATASET}
echo wandb: ${WANDB_PROJ}/${WANDB_RUN}
echo "==========================="

CUDA_VISIBLE_DEVICES=$gpus python3 train.py
    --config $CONFIG \
    --gpu $gpus \
    --wandb_proj $WANDB_PROJ \
    --wandb_run $WANDB_RUN  \
```
#### Example: PoT 
```
# !/bin/sh
# mkdir OPTForCausalLM
gpus=0

# Source Task (checkpoint)
SOURCE_TASK='[source task]'
SOURCE_CKPT='best_vanilla_prompt/open_book/[source task]/best_model.pkl'

# Target Task 
TARGET_TASK="[target task]"
CONFIG="config/[target task]PromptT5Base.config"
WANDB_RUN="[Wandb Run name]"

echo "==========================="
echo Source task: ${SOURCE_TASK}
echo Checkpoint: ${SOURCE_CKPT}
echo Target task: ${TARGET_TASK}
echo Config: ${CONFIG}
echo "==========================="

CUDA_VISIBLE_DEVICES=$gpus python3 train.py  
    --config $CONFIG \
    --gpu $gpus \
    --checkpoint $SOURCE_CKPT \
    --source_task $SOURCE_TASK \
    --exp $EXP \

```

### Evaluate
```
bash scripts/test.sh
```
#### Example
```
gpus=0

# Source task checkpoint
MODEL_PATH='best_vanilla_prompt/open_book/[source task]/best_model.pkl'

# Target task config
CONFIG="config/[target task]PromptOPT.config"

echo "==========================="

echo Model Path : ${MODEL_PATH}
echo Config: ${CONFIG}
echo "==========================="

CUDA_VISIBLE_DEVICES=$gpus python3 test.py
    --config $CONFIG \
    --gpu $gpus \
    --replacing_prompt $MODEL_PATH \
```

# Citations
