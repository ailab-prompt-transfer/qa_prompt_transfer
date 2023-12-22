# hyu-prompt transfer 
* 기간: 230726 ~ Present
* Origin code: https://github.com/thunlp/Prompt-Transferability
      * 자세한 사항은 위의 코드에서 확인 바람

## Description of Codes
* `scripts`: 수행 명령어 파일

## How to Run (Will be Updated!)
### Hyper-parameter Setting
* [수행(예정) 실험에 대한 Hyper-parameter setting](https://github.com/ailab-prompt-transfer/Basic_Model/blob/main/Hyperparamter.md)

### Train
```
bash scripts/train.sh
```
#### Trained prompt path
* **Original PT(실험 1-0)** : model/`{target task}`T5Small/exp_date
* **Transferring initialization PT(실험 1-2)** : model/exp1_2/`{target task}`T5Small/`{source task}`/exp_date

### Evaluate
#### best_model.pkl 테스트하는법
1. `pkl model`이 저장된 폴더에서 best_model.pkl 경로 확인
- validation dataset에서 BEST F1 score를 가진 model을 best_model.pkl로 저장함 
```python
model/squadPromptT5Small/230828_exp3-1-4_with_context/best_model.pkl
```

2. `run_evaluate.sh` 파일에서 test 수행할 CONFIG / MODEL_PATH 수정
```python
# 예시) squad dataset 
gpus=0
CONFIG="config/squadPromptT5Small.config"       
MODEL_PATH='model/squadPromptT5Small/230828_exp3-1-4_with_context/best_model.pkl'

echo "==========================="

echo Model Path : ${MODEL_PATH}
echo Config: ${CONFIG}
echo "==========================="

CUDA_VISIBLE_DEVICES=$gpus python3 test.py --config $CONFIG --gpu $gpus --replacing_prompt $MODEL_PATH
```
  
3. `run_evaluate.sh` 수행
```python
bash scripts/run_evaluate.sh
```
<br>

### epoch_?.pkl 모델 중 best checkpoint 기록 얻는법
- 모든 epoch에서 저장한 pkl model에 대한 EM/F1 score 얻기
- 상세 설명은 `pick_best_epoch_QA.py`에 있음
```python
CUDA_VISIBLE_DEVICES=0 python pick_best_epoch_QA.py --gpu=0 --config=[config 세팅] --ckpt_path=[내가 Test하고자하는 모델 폴더 경로] --output_file=[파일 출력 이름]
```

## Results (Will be Updated!)
# qa_prompt_transfer
