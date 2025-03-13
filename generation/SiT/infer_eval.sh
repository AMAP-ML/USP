# # #!/usr/bin/env bash
# # # 设置conda
# # # if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
# # #     . "/opt/conda/etc/profile.d/conda.sh"
# # # else
# # #     export PATH="/opt/conda/bin:$PATH"
# # # fi

if [ -f "/mnt/workspace/lirenda/miniconda3/etc/profile.d/conda.sh" ]; then
    . "/mnt/workspace/lirenda/miniconda3/etc/profile.d/conda.sh"
else
    export PATH="/mnt/workspace/lirenda/miniconda3/bin:$PATH"
fi

sudo mount -o size=20480M -o nr_inodes=1000000 -o noatime,nodiratime -o remount /dev/shm # 增加shm，否则容易出现OOM
# exit

export HF_ENDPOINT=https://hf-mirror.com
# export NCCL_SOCKET_IFNAME=eth # Disable automatically detected interface
# export NCCL_IB_DISABLE=1

## 定义eval_ckpts列表
# eval_ckpts=(50000 100000 200000 300000 400000)  
# eval_ckpts=(400000 300000 200000 100000 50000)
# eval_ckpts=(200000 100000)
# eval_ckpts=(50000)
# eval_ckpts=(1000000 400000)
# eval_ckpts=(800000)
# eval_ckpts=(700000)
# eval_ckpts=(150000)
eval_ckpts=(1000000)
# eval_ckpts=(800000)






# 权重文件所在的目
# checkpoint_dir=/mnt/workspace/lirenda/Code/SiT/SiT/results/022-SiT-XL-2-VAE-simple-Linear-velocity-None-SiT-XL-2-VAE-simple-PL-1600e/checkpoints
checkpoint_dir=/mnt/workspace/lirenda/Code/SiT/SiT/results/035-SiT-XL-2-VAE-simple-Linear-velocity-None-SiT-XL-2-VAE-simple-PL-1600e/checkpoints

model=SiT-XL/2-VAE-simple
save_name=$(basename $(dirname $checkpoint_dir))
mode=SDE
cfg=1.425
echo cfg=${cfg}
if [ "$(echo "$cfg == 1.0" | bc)" -eq 1 ]; then
    sample_dir_root=/mnt/workspace/lirenda/Code/SiT_sample/samples/${save_name}
else   
    sample_dir_root=/mnt/workspace/lirenda/Code/SiT_sample/samples_wCFG/${save_name}
fi

# cd /mnt/workspace/lirenda/Code/DiT_MAE/DiT

# 遍历eval_ckpts列表
echo model:{$model}
for step in "${eval_ckpts[@]}"; do
    ########################################### infer ###########################################
    # conda activate /mnt/workspace/envs/torch22_l20
    conda activate /mnt/workspace/lirenda/miniconda3/envs/torch22_ppue_my_sit

    weight_file="${checkpoint_dir}/$(printf "%07d" $step).pt"

    # 检查权重文件是否存在
    if [ -f "$weight_file" ]; then
        echo "Sample权重文件: $weight_file"
        sample_dir=${sample_dir_root}/${step}
        sample_npz=$(find "$sample_dir" -maxdepth 1 -type f -name "*cfg-${cfg}*.npz" | head -n 1) # 取第一个满足条件的npz文件
        # if ls $sample_dir/*.npz 1> /dev/null 2>&1; then
        if [ -n "$sample_npz" ]; then
            echo "存在 .npz 文件"
        else
            echo "不存在 .npz 文件"
            PORT=$MASTER_PORT NODE_RANK=$RANK NNODES=$WORLD_SIZE  sh sample_h20.sh ${mode} ${model} ${weight_file} ${sample_dir} ${cfg}
        fi

    else
        echo "权重文件不存在: $weight_file"
    fi

    chmod 777 ${sample_dir}
    
    # conda activate /mnt/workspace/lirenda/miniconda3/envs/torch22_ppue_my
    PORT=$MASTER_PORT NODE_RANK=$RANK NNODES=$WORLD_SIZE  sh sample_h20_torch.sh ${sample_dir} ${cfg}
done

# exit

# model=DiT-B/4-VAE
# ckpt=/mnt/workspace/lirenda/Code/DiT/results/005-DiT-B-4-VAE-vae_b_pos/checkpoints/0400000.pt
# sample_dir=/mnt/workspace/lirenda/Code/DiT_sample/samples/000-DiT-B-4-VAE_40w
# PORT=$MASTER_PORT NODE_RANK=$RANK NNODES=$WORLD_SIZE  sh sample_h20.sh ${model} ${ckpt} ${sample_dir}

