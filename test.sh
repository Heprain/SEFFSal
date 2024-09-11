#!/bin/bash
PREFIX=./snap-d/rgbd-m_ep101/
MODEL_NAME=model_96
MODEL_PATH=$PREFIX$MODEL_NAME.pth

CUDA_VISIBLE_DEVICES=0 python3 test.py --pretrained $MODEL_PATH \
                                       --savedir ./maps/test \
                                       --net_size m
                                       #    --depth 1