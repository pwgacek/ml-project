if [ ! -d "../results" ]; then
    mkdir ../results
fi

if [ ! -d "../results/wind-power-generation" ]; then
    mkdir ../results/wind-power-generation
fi
seq_len=336
model_name=DLinear

python -u run_longExp.py \
  --is_training 1 \
  --root_path /home/pawel/Desktop/ML/ml-project/datasets/ \
  --data_path wind-power-generation.csv \
  --model_id wind-power-generation_$seq_len'_'96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 9 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.01  >logs/LongForecasting/$model_name'_'wind-power-generation_$seq_len'_'96.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path /home/pawel/Desktop/ML/ml-project/datasets/ \
  --data_path wind-power-generation.csv \
  --model_id wind-power-generation_$seq_len'_'192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 9 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.01  >logs/LongForecasting/$model_name'_'wind-power-generation_$seq_len'_'192.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path /home/pawel/Desktop/ML/ml-project/datasets/ \
  --data_path wind-power-generation.csv \
  --model_id wind-power-generation_$seq_len'_'336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 9 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.01 >logs/LongForecasting/$model_name'_'wind-power-generation_$seq_len'_'336.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path /home/pawel/Desktop/ML/ml-project/datasets/ \
  --data_path wind-power-generation.csv \
  --model_id wind-power-generation_$seq_len'_'720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 9 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.01  >logs/LongForecasting/$model_name'_'wind-power-generation_$seq_len'_'720.log
