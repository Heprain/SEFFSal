#!/bin/bash
SAVE_PREFIX=./snap
SAVE_PATH=$SAVE_PREFIX  

CUDA_VISIBLE_DEVICES=0 python train.py --max_epochs 101\
                                         --num_workers 2 \
                                         --batch_size 10 \
                                         --savedir $SAVE_PATH \
                                         --depth 1 \
                                         --lr_mode poly \
                                         --lr 1e-4 \
                                         --inWidth 352 \
                                         --inHeight 352 \
                                         --net_size m \
                                         --rgbt 1 
                                        # --resume /home/hello/Extra/SEFFSal/snap/rgbd-s_ep101/checkpoint.pth.tar
                                        