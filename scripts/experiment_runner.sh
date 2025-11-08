#!/bin/bash
set -euo pipefail

if [ $# -ne 4 ]; then
  echo "Use: $0 <model_name> <dataset_name> <enc_in> <freq>"
  exit 1
fi

model_name="$1"
dataset_name="$2"
enc_in="$3" # Naive model doesn't care about this arg.
freq="$4" # options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]

if [ "$model_name" == "DLinear" ]; then
  script_name="run_longExp.py"
elif [ "$model_name" == "Naive" ]; then
  script_name="run_stat.py"
else
  echo "Unknown model: $model_name"
  echo "Available: DLinear, Naive"
  exit 1
fi

root_path="/content/datasets"
results_dir="../results/${dataset_name}"
mkdir -p "$results_dir"

# Hiperparameters
seq_len=336
pred_lens=(96 192 336 720)
batch_size=16
learning_rate=0.01

for pred_len in "${pred_lens[@]}"; do
  echo "======================================="
  echo "Model: $model_name"
  echo "Script: $script_name"
  echo "Dataset: $dataset_name"
  echo "Enc_in: $enc_in"
  echo "Pred_len: $pred_len"
  echo "Batch size: $batch_size  LR: $learning_rate"
  echo "Log: ${results_dir}/${model_name}_${dataset_name}_${seq_len}_${pred_len}_bs${batch_size}_lr${learning_rate}.log"
  echo "======================================="

  cmd=(python -u "$script_name"
       --is_training 1
       --root_path "$root_path"
       --data_path "${dataset_name}.csv"
       --model_id "${dataset_name}_${seq_len}_${pred_len}_bs${batch_size}_lr${learning_rate}"
       --model "$model_name"
       --data custom
       --features M
       --seq_len "$seq_len"
       --pred_len "$pred_len"
       --des 'Exp'
       --itr 1
       --batch_size "$batch_size"
       --freq "$freq"
  )

  if [ "$script_name" = "run_longExp.py" ]; then
    cmd+=( --enc_in "$enc_in" )
    cmd+=( --learning_rate "$learning_rate" )
  fi

  log_file="${results_dir}/${model_name}_${dataset_name}_${seq_len}_${pred_len}_bs${batch_size}_lr${learning_rate}.log"
  "${cmd[@]}" > "$log_file" 2>&1

  echo "Ended pred_len=$pred_len; log saved: $log_file"
done

