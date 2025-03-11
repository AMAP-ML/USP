For Base(Large, XLarge) with 800 epochs with 64 GPUS, we can use 1600 epoch setting just by replacing 800 with 1600.
```
cd pretrain
model_name=vit_xlarge_patch_vae
python -m torch.distributed.launch --nproc_per_node=$gpu_per_pod --master_addr=$MASTER_ADDR  --master_port=$MASTER_PORT \
--nnodes=$WORLD_SIZE --node_rank=$RANK  --use_env main_pretrain.py --batch_size 64 --model $model_name \
--norm_pix_loss --mask_ratio 0.75 --epochs 800 --warmup_epochs 40 --blr 1.5e-4 --weight_decay 0.05 --data_path $imagenet_dir \
--input_size 224 --vae_version d8 