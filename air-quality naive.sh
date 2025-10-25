
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=Naive

python -u run_stat.py \
  --is_training 1 \
  --root_path /home/pawel/Desktop/ML/ml-project/datasets/ \
  --data_path air-quality.csv \
  --model_id air-quality_$seq_len'_'96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --des 'Exp' \
  --itr 1 --batch_size 16   >logs/LongForecasting/$model_name'_'air-quality_$seq_len'_'96.log

python -u run_stat.py \
  --is_training 1 \
  --root_path /home/pawel/Desktop/ML/ml-project/datasets/ \
  --data_path air-quality.csv \
  --model_id air-quality_$seq_len'_'192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --des 'Exp' \
  --itr 1 --batch_size 16   >logs/LongForecasting/$model_name'_'air-quality_$seq_len'_'192.log

python -u run_stat.py \
  --is_training 1 \
  --root_path /home/pawel/Desktop/ML/ml-project/datasets/ \
  --data_path air-quality.csv \
  --model_id air-quality_$seq_len'_'336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --des 'Exp' \
  --itr 1 --batch_size 16  >logs/LongForecasting/$model_name'_'air-quality_$seq_len'_'336.log

python -u run_stat.py \
  --is_training 1 \
  --root_path /home/pawel/Desktop/ML/ml-project/datasets/ \
  --data_path air-quality.csv \
  --model_id air-quality_$seq_len'_'720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --des 'Exp' \
  --itr 1 --batch_size 16  >logs/LongForecasting/$model_name'_'air-quality_$seq_len'_'720.log
