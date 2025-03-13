#!/bin/bash
model="SiT-XL/2-VAE-simple" # "SiT-XL/2-VAE-simple", "SiT-B/2-VAE-simple"
finetune="/PATH/TO/YOUR/PRETRAINED_CHECKPOINT"
exp_name="SiT-XL/2-VAE-simple"
data_path="/PATH/TO/YOUR/imagenet-1k/train"
path_type="Linear"
prediction="velocity"
global_batch_size="256"


torchrun --master_addr ${MASTER_ADDR} --master-port ${MASTER_PORT} \
  --nnodes ${WORLD_SIZE} --node_rank ${RANK} --nproc-per-node=${GPUS} train.py \
  --model ${model} \
  --data-path ${data_path} \
  --finetune ${finetune} \
  --exp-name ${exp_name} \
  --path-type ${path_type} \
  --prediction ${prediction} \
  --global-batch-size $global_batch_size
