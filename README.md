Official implementation of "Boosting Selective Rationalization with Shortcuts Discovery".

## Dataset

We upload the processed ERASER dataset in ./dataset.



##  How to run SSR?

To train ${SSR}_{unif}$ on ERASER dataset (e.g., Movie):

```python
python ssr1.py \
  --gpu_id=2 \
  --data_dir ./movie \
  --dataset movie_reviews \
  --gradient_accumulation_steps=1 \
  --model_name=SSR1 \
  --lr=2e-05 \
  --max_len=512 \
  --alpha_rationle=0.2 \
  --types=train \
  --epochs=30 \
  --batch_size=4 \
  --class_num=2 \
  --save_path=./output/ \
  --alpha=0.01 \
  --beta=0.01 \
  --seed=3 \
  --is_da=no \ #  control which data augmentation method is performed
```



To train ${SSR}_{virt}$ on ERASER dataset (e.g., MultiRC):

```python
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
```



