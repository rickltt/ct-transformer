#!/bin/bash
export CUDA_VISIBLE_DEVICES=3 

OUTPUT_DIR='./output/finetune_output_disflu'
MODEL_PATH='./output/pretrain_output/model.pt'
python finetune.py \
    --data_dir ./dataset/finetune_csc \
    --do_train \
    --do_eval \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --evaluate_after_epoch \
    --per_gpu_train_batch_size 128 \
    --per_gpu_eval_batch_size 128 \
    --dropout_prob 0.1 \
    --max_seq_length 256 \
    --learning_rate 3e-5 \
    --weight_decay 5e-5 \
    --num_train_epochs 10 \
    --seed 42