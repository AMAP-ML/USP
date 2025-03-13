# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

# for sample & eval
import math
from tqdm import tqdm
# from guided_diffusion.evaluations.evaluator import *
# import tensorflow.compat.v1 as tf

import torch_fidelity
from distributed import is_master, init_distributed_device, broadcast_object




#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt", mode='a')]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

@torch.no_grad()
def sample(model, vae, ckpt_path, latent_size, step, rank, args, device):
    import time
    diffusion = create_diffusion(str(args.num_sampling_steps))
    
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

     # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(ckpt_path).replace(".pt", "")
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    # sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    
    exp_name = ckpt_path.split('/')[-3]
    sample_folder_dir = f"{args.sample_dir}/{exp_name}/{step}/{folder_name}"
    sample_root_path = os.path.dirname(sample_folder_dir)
    result_save_path = os.path.join(sample_root_path, 'eval_result.txt')
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
        
        os.chmod(sample_root_path, 0o777)
        print(f"Permissions for {sample_root_path} changed to 777.")
        
    dist.barrier()

    if not len(os.listdir(sample_folder_dir)) >= 50000:
        

        # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
        n = args.per_proc_batch_size
        global_batch_size = n * dist.get_world_size()
        # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
        total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
        if rank == 0:
            print(f"Total number of images that will be sampled: {total_samples}")
        assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
        samples_needed_this_gpu = int(total_samples // dist.get_world_size())
        assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
        iterations = int(samples_needed_this_gpu // n)
        pbar = range(iterations)
        pbar = tqdm(pbar) if rank == 0 else pbar
        total = 0
        for _ in pbar:
            # Sample inputs:
            z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
            y = torch.randint(0, args.num_classes, (n,), device=device)

            # Setup classifier-free guidance:
            if using_cfg:
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([1000] * n, device=device)
                y = torch.cat([y, y_null], 0)
                model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
                sample_fn = model.forward_with_cfg
            else:
                model_kwargs = dict(y=y)
                sample_fn = model.forward

            # Sample images:
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
            )
            if using_cfg:
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

            samples = vae.decode(samples / 0.18215).sample
            samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            # Save samples to disk as individual .png files
            for i, sample in enumerate(samples):
                index = i * dist.get_world_size() + rank + total
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
            total += global_batch_size
    else:
        print(f'Results exists! Save .NPZ now')

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    
    if rank == 0:
        if args.image_size == 256:
            input2 = None
            fid_statistics_file = 'fid_stats/adm_in256_stats.npz'
        else:
            raise NotImplementedError
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=sample_folder_dir,
            input2=input2,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        print("CFG: {:.4f}, FID: {:.4f}, Inception Score: {:.4f}".format(args.cfg_scale, fid, inception_score))
        # # remove temporal saving folder
        # shutil.rmtree(save_folder)
        with open(result_save_path, 'w') as file:
            file.write(f"Inception Score: {inception_score}\n")
            file.write(f"FID: {fid}\n")
            file.write(f"CFG: {args.cfg_scale}\n")
        print(f'save result ready')

    dist.barrier()
    time.sleep(10)


def eval(sample_path, args):
    config = tf.ConfigProto(
        allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
    )
    config.gpu_options.allow_growth = True
    evaluator = Evaluator(tf.Session(config=config))

    print("warming up TensorFlow...")
    # This will cause TF to print a bunch of verbose stuff now rather
    # than after the next print(), to help prevent confusion.
    evaluator.warmup()

    print("computing reference batch activations...")
    ref_acts = evaluator.read_activations(args.ref_batch)
    print("computing/reading reference batch statistics...")
    ref_stats, ref_stats_spatial = evaluator.read_statistics(args.ref_batch, ref_acts)

    print("computing sample batch activations...")
    sample_acts = evaluator.read_activations(sample_path)
    print("computing/reading sample batch statistics...")
    sample_stats, sample_stats_spatial = evaluator.read_statistics(sample_path, sample_acts)

    print("Computing evaluations...")
    Inception_Score = evaluator.compute_inception_score(sample_acts[0])
    print("Inception Score:", Inception_Score)

    FID = sample_stats.frechet_distance(ref_stats)
    print("FID:", FID)

    sFID = sample_stats_spatial.frechet_distance(ref_stats_spatial)
    print("sFID:", sFID)

    # prec, recall = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])
    # print("Precision:", prec)
    # print("Recall:", recall)
    # txt_path = f'{os.path.dirname(args.sample_batch)}/eval_results.txt'
    # with open(txt_path, "w") as file:
    #     file.write(f"Inception Score: {Inception_Score}\n")
    #     file.write(f"FID: {FID}\n")
    #     file.write(f"sFID: {sFID}\n")
    #     file.write(f"Precision: {prec}\n")
    #     file.write(f"Recall: {recall}\n")

    txt_path = f'{os.path.dirname(sample_path)}/eval_results.txt'
    with open(txt_path, "w") as file:
        file.write(f"Inception Score: {Inception_Score}\n")
        file.write(f"FID: {FID}\n")
        file.write(f"sFID: {sFID}\n")

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")

    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank

    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0: 
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    # import pdb; pdb.set_trace()
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}" if not args.resume else os.path.dirname(args.resume).split('/checkpoints')[0]  # Create an experiment folder

    if args.exp_name and not args.resume:
        experiment_dir = f"{experiment_dir}-{args.exp_name}"

    checkpoint_dir = f"{experiment_dir}/checkpoints" if not args.resume else os.path.dirname(args.resume)  # Stores saved model checkpoints
    dist.barrier()

    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)
    

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )

    assert not (args.finetune and args.resume)

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()

        # deal with patch_embed
        ckp_patch_weight = checkpoint_model.pop('patch_embed.proj.weight')
        ckp_patch_bias= checkpoint_model.pop('patch_embed.proj.bias')

        if ckp_patch_weight.shape[1] == model.in_channels: # 都是vae的4通道
            
            # args.load_patch_conv_method == 'none'
            if not ckp_patch_weight == None:
                checkpoint_model['patch_embed.proj.weight'] = ckp_patch_weight
            if not ckp_patch_bias == None:
                checkpoint_model['patch_embed.proj.bias'] = ckp_patch_bias
    
            if 'patch_embed.proj.weight' in checkpoint_model:
                checkpoint_model['x_embedder.proj.weight'] = checkpoint_model.pop('patch_embed.proj.weight')
            if 'patch_embed.proj.bias' in checkpoint_model:
                checkpoint_model['x_embedder.proj.bias'] = checkpoint_model.pop('patch_embed.proj.bias')
            

        # deal with pos_embed
        pos_embed_name = 'pos_embed'
        pos_embed_checkpoint = checkpoint_model[pos_embed_name]

        num_extra_tokens = 1 # cls token
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # new_size = int(model.x_embedder.num_patches ** 0.5)
        new_size = int(model.num_patches ** 0.5)

        embedding_size = pos_embed_checkpoint.shape[-1]
        if orig_size != new_size:
            print("[%s] Position interpolate from %dx%d to %dx%d" % (pos_embed_name, orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)

            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = pos_tokens
            checkpoint_model[pos_embed_name] = new_pos_embed
        else:
            checkpoint_model[pos_embed_name] = pos_embed_checkpoint[:, num_extra_tokens:]
        
        if args.use_pretrain_layer >= 0: # < 0, unuse
            keys_to_delete = []
            for key in checkpoint_model.keys():
                if key.startswith("blocks.") and int(key.split(".")[1]) >= args.use_pretrain_layer:
                    keys_to_delete.append(key)

            for key in keys_to_delete:
                del checkpoint_model[key]

        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
    
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        print("Load resume checkpoint from: %s" % args.resume)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        msg = model.load_state_dict(checkpoint_model, strict=True)
        print(msg)

        ema.load_state_dict(checkpoint["ema"], strict=True)
        print("Succesfully load EMA model from: %s" % args.resume)
        # opt.load_state_dict(state_dict["opt"])


    
    # Note that parameter initialization is done within the DiT constructor

    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank % torch.cuda.device_count()])
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"/mnt/workspace/common/models/sd-vae-ft-ema").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    if args.resume:
        opt.load_state_dict(checkpoint["opt"])
        print("Succesfully load opt from: %s" % args.resume)

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        # rank=rank,
        rank = rank % torch.cuda.device_count(),
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    if args.evaluate:
        torch.cuda.empty_cache()
        train_steps = int(os.path.basename(args.resume).split('.')[0])
        # evaluate(model_without_ddp, vae, ema, args, batch_size=args.eval_bsz, cfg=args.cfg, use_ema=True, device=device)
        sample(ema, vae, args.resume, latent_size, train_steps, rank, args, device)
        torch.cuda.empty_cache()
        return

    # Variables for monitoring/logging purposes:
    train_steps = 0 if not args.resume else int(args.resume.split('/')[-1].split('.')[0])
    log_steps = 0 
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):

        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            print(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if (train_steps % args.ckpt_every == 0 or train_steps in [50000]) and train_steps > 0:
            # if (train_steps % args.ckpt_every == 0 or train_steps in [50000]) and train_steps > 0:
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

                # sample & evaluationgit
                if (train_steps % args.eval_every == 0 or train_steps in [50000]) and train_steps > 0:
                    sample(ema, vae, checkpoint_path, latent_size, train_steps, rank, args, device)
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="/mnt/workspace/lirenda/Code/DiT/results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50000)
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument("--exp-name", type=str, default='')
    parser.add_argument("--use-pretrain-layer", type=int, default=-1)
    parser.add_argument("--load-patch-conv-method", type=str, default='interpolate') # 'interpolate' or 'copy'
    parser.add_argument("--horovod", default=False, action="store_true", help="Use horovod for distributed training.")
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument(
        "--ddp-static-graph",
        default=False,
        action='store_true',
        help="Enable static graph optimization for DDP in PyTorch >= 1.11.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc)."
    )
    
    # evaluation
    parser.add_argument("--per-proc-batch-size", type=int, default=128)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--cfg-scale",  type=float, default=1.0)
    parser.add_argument("--sample-dir", type=str, default="./samples")
    parser.add_argument("--ref-batch", default="./pretrained_model/VIRTUAL_imagenet256_labeled.npz", help="path to reference batch npz file")
    parser.add_argument("--eval-every", type=int, default=100000)

    args = parser.parse_args()
    main(args)
