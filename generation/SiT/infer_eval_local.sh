# # #!/usr/bin/env bash
# # # 设置conda
# # # if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
# # #     . "/opt/conda/etc/profile.d/conda.sh"
# # # else
# # #     export PATH="/opt/conda/bin:$PATH"
# # # fi

# update at 3.7, only use sample_type=fix

if [ -f "/mnt/workspace/lirenda/miniconda3/etc/profile.d/conda.sh" ]; then
    . "/mnt/workspace/lirenda/miniconda3/etc/profile.d/conda.sh"
else
    export PATH="/mnt/workspace/lirenda/miniconda3/bin:$PATH"
fi

sample_type=default # debault or fix or torch
if [ -n "$1" ]; then
    sample_type=$1
fi


export HF_ENDPOINT=https://hf-mirror.com
# export NCCL_BLOCKING_WAIT=1  # NCCL 允许阻塞等待，确保操作完成
# export TORCH_NCCL_BLOCKING_WAIT=7200000  # 将超时时间增加到 10 分钟

## 定义eval_ckpts列表
eval_ckpts=(800000)
checkpoint_dir=/mnt/workspace/lirenda/Code/SiT/SiT/results/035-SiT-XL-2-VAE-simple-Linear-velocity-None-SiT-XL-2-VAE-simple-PL-1600e/checkpoints
model=SiT-XL/2-VAE-simple
mode=SDE
cfg=1.5
save_name=$(basename $(dirname $checkpoint_dir))

# sample_dir_root=/mnt/workspace/lirenda/Code/DiT_sample/samples/${model}
if [[ ${sample_type} = "default" || ${sample_type} = "torch" ]]; then
    if [[ "${model,,}" == *"sit"* ]]; then
        sample_dir_root=/mnt/workspace/lirenda/Code/SiT_sample/samples/${save_name}
        # sample_dir_root=/mnt/workspace/lirenda/Code/SiT_sample/samples_wCFG/${save_name}

    else
        sample_dir_root=/mnt/workspace/lirenda/Code/DiT_sample/samples/${save_name}
    fi
elif [ ${sample_type} = "fix" ]; then
    sample_dir_root=/mnt/workspace/lirenda/Code/SiT_sample/sample_for_check/${save_name}
fi



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
        if find "$sample_dir" -maxdepth 1 -name "*.npz" | grep -q .; then
            echo "存在 .npz 文件"
        else
            echo "不存在 .npz 文件"
            if [ ${sample_type} = "default" ]; then
                subdir=$(find "$sample_dir" -maxdepth 1 -type d \( -name "DiT*" -o -name "SiT*" \) | head -n 1)
                if [ -n "$subdir" ]; then
                    echo "Found folder: $subdir"
                    # 拼接路径
                    full_path="$sample_dir/$(basename "$subdir")"
                    echo "Full path: $full_path"
                    python /mnt/workspace/lirenda/Code/DiT_MAE/DiT/save_npz.py "${full_path}" 50000
                else
                    echo "No matching subdirectory found."
                    exit
                fi
                
                if [[ "${model,,}" == *"dit"* ]]; then # sit走这里会报错
                    PORT=$MASTER_PORT NODE_RANK=$RANK NNODES=$WORLD_SIZE  sh sample_h20_local.sh ${model} ${weight_file} ${sample_dir}
                fi
            elif [ ${sample_type} = "fix" ]; then
                PORT=$MASTER_PORT NODE_RANK=$RANK NNODES=$WORLD_SIZE  sh sample_h20_local_fixClass.sh ${mode} ${model} ${weight_file} ${sample_dir} ${cfg}
                continue
            fi
        fi

    else
        echo "权重文件不存在: $weight_file"
    fi

    if [ ${sample_type} = "fix" ]; then
        exit
    fi

    # ########################################### eval ###########################################
    # if [ -f "/mnt/workspace/lirenda/miniconda3/etc/profile.d/conda.sh" ]; then
    #     . "/mnt/workspace/lirenda/miniconda3/etc/profile.d/conda.sh"
    # else
    #     export PATH="/mnt/workspace/lirenda/miniconda3/bin:$PATH"
    # fi
    
    # if [ ${sample_type} = "torch" ]; then
    #     conda activate /mnt/workspace/lirenda/miniconda3/envs/torch22_ppue_my
    #     PORT=$MASTER_PORT NODE_RANK=$RANK NNODES=$WORLD_SIZE  sh sample_h20_local_torch.sh ${sample_dir}
    # else
    #     # conda activate /mnt/workspace/lirenda/miniconda3/envs/gd
    #     conda activate /mnt/workspace/lirenda/miniconda3/envs/torch22_ppue_my_fid
    #     npz_files=($sample_dir/*.npz)
    #     if [ ${#npz_files[@]} -eq 1 ]; then
    #         # 如果只有一个 .npz 文件，输出文件路径
    #         echo "找到的 .npz 文件路径: ${npz_files[0]}"
    #         python /mnt/workspace/lirenda/Code/DiT_MAE/DiT/guided_diffusion/evaluations/evaluator.py \
    #         /mnt/workspace/lirenda/Code/DiT_MAE/DiT/pretrained_model/VIRTUAL_imagenet256_labeled.npz \
    #         ${npz_files[0]}

    #     else
    #         if [ ${#npz_files[@]} -eq 0 ]; then
    #             echo "没有找到 .npz 文件"

    #         else
    #             echo "找到多个 .npz 文件，请检查: ${npz_files[@]}"
    #         fi
    #     fi
    # fi
done

