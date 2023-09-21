python ssr2.py \
  --gpu_id=7 \
  --data_dir ./multirc \
  --dataset multi_rc \
  --gradient_accumulation_steps=1 \
  --model_name=SSR2 \
  --lr=2e-05 \
  --max_len=512 \
  --alpha_rationle=0.2 \
  --types=train \
  --epochs=30 \
  --batch_size=4 \
  --class_num=2 \
  --save_path=./output/ \
  --alpha=0.1 \
  --beta=0.01 \
  --seed=1 \
  --is_da=no \