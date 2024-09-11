#!/bin/bash
SAVE_PREFIX=./snapshots/seff  

SAVE_PATH=$SAVE_PREFIX  

CUDA_VISIBLE_DEVICES=0 python3 train.py --max_epochs 101 \
                                         --num_workers 2 \
                                         --batch_size 10 \
                                         --savedir $SAVE_PATH \
                                         --depth 1 \
                                         --lr_mode poly \
                                         --lr 1e-4 \
                                         --inWidth 352 \
                                         --inHeight 352 \
                                         --net_size m
                                        #  --resume /home/dl/1Tdriver/zqy/FasterSal/snapshots/v1_ep101/checkpoint.pth.tar
                                            
