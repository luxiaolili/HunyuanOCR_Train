#!/bin/bash


set -e  

# ==================== 配置参数 ====================
MODEL_PATH="tencent/HunyuanOCR"
TRAIN_FILE="train_hunyuanocr.json"
TEST_FILE="test_hunyuanocr.json"
OUTPUT_DIR="HunYuanOCR-SFT"
NUM_GPUS_PER_MODEL=8  
QUESTION_FILE="question.txt"  


EPOCHS=2
BATCH_SIZE=4
GRAD_ACCUM_STEPS=8
LEARNING_RATE=1e-5
MAX_LENGTH=None

# ==================== 环境设置 ====================
echo "设置环境变量..."


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512



accelerate launch \
    --config_file default_config.yaml \
    --num_processes=$NUM_GPUS_PER_MODEL \
    --num_machines=1 \
    --mixed_precision=bf16 \
    train.py \
    --model_name_or_path "$MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --test_file "$TEST_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --max_length "$MAX_LENGTH"

echo "训练完成!"