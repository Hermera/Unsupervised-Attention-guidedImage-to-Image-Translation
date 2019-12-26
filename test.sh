#!/bin/bash
python3 main.py --to_train=0 --log_dir=./output/AGGAN/exp_01 \
	--config_filename=./configs/exp_01_test.json --checkpoint_dir=./output/AGGAN/exp_01/switch_30_thres_0.1 \
	--checkpoint_name=AGGAN_99

