#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export OPT_NUM_THREADS=20

python3 main.py --rsc_path ./rsc \
                --llm 'dolly' \
                --params '12b' \
                --dataset 'cirr' \
                --mode 'train'