#!/bin/bash

if [ ! -d "../results" ]; then
    mkdir ../results
fi

if [ ! -d "../results/air-quality" ]; then
    mkdir ../results/air-quality
fi
seq_len=336
model_name=DLinear

pred_lens=(192 336 720)
batch_size=16
learning_rate=0.1

for pred_len in "${pred_lens[@]}"
do
  echo "Running with pred_len=$pred_len, batch_size=$batch_size, learning_rate=$learning_rate"
  python -u run_longExp.py \
    --is_training 1 \
    --root_path /home/pawel/Desktop/ML/ml-project/datasets/ \
    --data_path air-quality.csv \
    --model_id air-quality_${seq_len}_${pred_len}_bs${batch_size}_lr${learning_rate} \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 13 \
    --des 'Exp' \
    --itr 1 --batch_size $batch_size --learning_rate $learning_rate >../results/air-quality/${model_name}_air-quality_${seq_len}_${pred_len}_bs${batch_size}_lr${learning_rate}.log
done
