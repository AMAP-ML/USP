#!/bin/bash
model=DiT-XL/2-VAE-simple
data_path=/PATH/TO/YOUR/imagenet-1k/train
finetune=/PATH/TO/YOUR/PRETRAINED_CHECKPOINT
exp_name="DiT-XL/2-VAE-simple"

torchrun --master_addr ${MASTER_ADDR} --master-port ${MASTER_PORT} \
    --nnodes ${WORLD_SIZE} --node_rank ${RANK} --nproc-per-node=${GPUS} train.py \
    --model ${model} \
    --data-path ${data_path} \
    --finetune ${finetune} \
    --exp-name ${exp_name} \


