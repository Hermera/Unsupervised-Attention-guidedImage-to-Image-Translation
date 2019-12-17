#!/bin/bash
python3 main.py --to_train=2 --log_dir=./output/AGGAN/exp_1 \
	--config_filename=./configs/exp_1.json --checkpoint_dir=./output/AGGAN/exp_1/switch_30_thres_0.1 \
	--checkpoint_name=AGGAN_59
