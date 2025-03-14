#!/bin/bash
model="DiT-XL/2-VAE-simple"
ckpt="/PATH/TO/YOUR/FINETUNE_CHECKPOINT"
sample_dir="./samples"
cfg="1.0"

torchrun --master_addr ${MASTER_ADDR} --master-port ${MASTER_PORT} \
    --nnodes ${WORLD_SIZE} --node_rank ${RANK} --nproc-per-node=${GPUS} sample_ddp.py \
    --model ${model} \
    --num-fid-samples 50000 \
    --ckpt ${ckpt} \
    --sample-dir ${sample_dir} \
    --per-proc-batch-size 128 \
    --cfg-scale ${cfg}