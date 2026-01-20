#!/usr/bin/env bash
set -o pipefail

export CUDA_VISIBLE_DEVICES=0

model_name=PatchTST
root_path=/content/drive/MyDrive/um_proj/datasets

# === RESULTS FILE ===
results_file=results_patchtst.csv
if [ ! -f "$results_file" ]; then
  echo "dataset_name,data_path,freq,feat_count,seq_len,label_len,pred_len,mse,mae" > ./"$results_file"
fi

# === DATASET CONFIGURATION ===
# Format: "dataset_name:data_path:target:freq:feat_count"
declare -a datasets=(
  "household_power:household_power_consumption_hourly_clean.csv:Voltage:h:7"
  "QPS:QPS_clean.csv:y9:t:10"
  "sales:sales_clean.csv:R06:h:8"
  "air_pollution:air-pollution.csv:so2:h:8"
  "ms_stock:microsoft-stock.csv:Close:d:6"
  "wpg:wind-power-generation.csv:Power:h:9"
  # Add more datasets here as needed
)

declare -a pred_lens=(96 192 336 720)

# === IMPORTANT NOTE! ===
# When `features == M` then target can be any feature from the dataset but it must exist!
# Otherwise it won't work...

# === MAIN LOOP ===
for dataset_config in "${datasets[@]}"; do
  IFS=':' read -r dataset_name data_path target freq feat_count <<< "$dataset_config"
  
  echo "Dataset: $dataset_name (data: $data_path, freq: $freq)"
  
  for pred_len in "${pred_lens[@]}"; do
    echo "Running with pred_len=$pred_len (seq_len=96, label_len=48)"
    run_log="/tmp/patchtst_${dataset_name}_${pred_len}.log"
    
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path "$root_path" \
      --data_path "$data_path" \
      --model_id "${dataset_name}_96_${pred_len}" \
      --model "$model_name" \
      --data "custom" \
      --target "$target" \
      --features M \
      --freq "$freq" \
      --seq_len 96 \
      --label_len 48 \
      --pred_len "$pred_len" \
      --e_layers 1 \
      --d_layers 1 \
      --factor 3 \
      --enc_in "$feat_count" \
      --dec_in "$feat_count" \
      --c_out "$feat_count" \
      --des 'Exp' \
      --n_heads 2 \
      --itr 1 \
      --train_epoch 5 \
      2>&1 | tee "$run_log"
    
    # Extract last metrics line containing mse and mae
    metrics_line=$(grep -E 'mse:[0-9eE.+-]+, mae:[0-9eE.+-]+' "$run_log" | tail -n 1)
    if [ -n "$metrics_line" ]; then
      mse=$(echo "$metrics_line" | awk -F 'mse:' '{print $2}' | awk -F ', mae:' '{print $1}')
      mae=$(echo "$metrics_line" | awk -F 'mae:' '{print $2}' | awk -F ', dtw:' '{print $1}')
    else
      mse="NA"; mae="NA"
    fi

    # Append to results CSV
    echo "$dataset_name,$data_path,$freq,$feat_count,96,48,$pred_len,$mse,$mae" >> "$results_file"

    if [ $? -eq 0 ]; then
      echo "Completed: $dataset_name (pred_len=$pred_len) | mse=$mse mae=$mae"
    else
      echo "Failed: $dataset_name (pred_len=$pred_len)"
    fi
  done
done

echo "All experiments completed!"