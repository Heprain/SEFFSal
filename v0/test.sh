#!/bin/bash
PREFIX=./snapshots/seff_ep101/
MODEL_NAME=model_100
MODEL_PATH=$PREFIX$MODEL_NAME.pth

CUDA_VISIBLE_DEVICES=0 python3 test.py --pretrained $MODEL_PATH \
                                       --savedir ./maps/$MODEL_NAME/ \
                                       --depth 1
