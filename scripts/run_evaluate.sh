
gpus=0
CONFIG="config/[dataset model].config"       
MODEL_PATH='model/[dataset model]/[model 폴더]/best_model.pkl'

echo "==========================="

echo Model Path : ${MODEL_PATH}
echo Config: ${CONFIG}
echo "==========================="

CUDA_VISIBLE_DEVICES=$gpus python3 test.py --config $CONFIG --gpu $gpus --replacing_prompt $MODEL_PATH