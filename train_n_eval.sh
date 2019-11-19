#!/bin/bash

############ Set Configurations######################################################
gpu_no=1
pipeline_config_path='configs/rbox_cnn_resnet101.config'
dst_path='../ckpt/rbox_cnn_resnet101/'
log_every_n_steps=100
save_interval_secs=3600
#####################################################################################


# train
CUDA_VISIBLE_DEVICES=$gpu_no python3 train.py \
    --logtostderr \
    --pipeline_config_path=$pipeline_config_path \
    --train_dir=$dst_path \
    --save_interval_secs=$save_interval_secs \
    --log_every_n_steps=$log_every_n_steps


# evaluation
CUDA_VISIBLE_DEVICES=$gpu_no python3 eval.py \
    --logtostderr \
    --pipeline_config_path=$pipeline_config_path \
    --checkpoint_dir=$dst_path \
    --eval_dir=$dst_path \
    --run_mode=all

