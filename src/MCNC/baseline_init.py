import os

os.system('python3 run_baseline.py --seed 2022')
os.system('python3 run_baseline.py --seed 4044')
os.system('python3 run_baseline.py --seed 8088')

os.system('python3 run_baseline.py --encoder_name roberta-large --per_device_train_batch_size 8 --seed 2022')
os.system('python3 run_baseline.py --encoder_name roberta-large --per_device_train_batch_size 8 --seed 4044')
os.system('python3 run_baseline.py --encoder_name roberta-large --per_device_train_batch_size 8 --seed 8088')