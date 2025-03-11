# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import numpy
import torch
import os
import util.misc as misc
import util.lr_sched as lr_sched
from vae import DiagonalGaussianDistribution
import cv2
import numpy as np
from PIL import Image


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None, vae=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        if 'vae' in args.model:
            x = samples
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=not args.use_fp32):
                    if args.use_cached:
                        moments = x
                        posterior = DiagonalGaussianDistribution(moments)
                        x = posterior.sample()
                        x = x.mul_(args.scale)

                    else:
                        x = vae.encode(x)
                        if hasattr(x, 'latent_dist'):
                            x = x.latent_dist.mean
                        else:
                            x = x.mean
                        x = x.mul_(args.scale)  # Map input images to latent space + normalize latents:
            samples = x
        if args.add_noise:
            gauss_noise = torch.randn(x.shape, device=x.device)
            t = torch.rand((x.shape[0], 1, 1, 1), device=x.device) # [0, 1) -> [0.75, 1.0)
            t = t * 0.5 + 0.5
            alpha = torch.sqrt(t)
            beta = torch.sqrt(1 - t)
            target = samples
            samples = alpha * samples + beta * gauss_noise
        else:
            target = None

        with torch.cuda.amp.autocast(enabled=not args.use_fp32):
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio, target=target)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def generate_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None, vae=None):
    model.train(False)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    total = 0
    world_size = misc.get_world_size()
    local_rank = args.gpu
    global_batch_size = args.batch_size * world_size

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        origin_samples = samples
        if 'vae' in args.model:
            origin_images = (samples * 127.5 + 127.5).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        else:
            origin_images = samples * torch.tensor([0.229, 0.224, 0.225], device=device).reshape(1, 3, 1, 1)
            origin_images = origin_images + torch.tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
            origin_images = (origin_images * 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        if 'vae' in args.model:
            x = samples
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=not args.use_fp32):
                    if args.use_cached:
                        moments = x
                        posterior = DiagonalGaussianDistribution(moments)
                        x = posterior.sample()
                        x = x.mul_(args.scale)

                    else:
                        x = vae.encode(x)
                        if hasattr(x, 'latent_dist'):
                            x = x.latent_dist.mean
                        else:
                            x = x.mean
                        x = x.mul_(args.scale)  # Map input images to latent space + normalize latents:
            samples = x
        if args.add_noise:
            gauss_noise = torch.randn(x.shape, device=x.device)
            t = torch.rand((x.shape[0], 1, 1, 1), device=x.device) # [0, 1) -> [0.75, 1.0)
            t = t * 0.5 + 0.5
            alpha = torch.sqrt(t)
            beta = torch.sqrt(1 - t)
            target = samples
            samples = alpha * samples + beta * gauss_noise
        else:
            target = None

        with torch.cuda.amp.autocast(enabled=not args.use_fp32):
            loss, pred, mask = model(samples, mask_ratio=args.mask_ratio, target=target)
            latent = model.module.unpatchify(pred)
            if 'vae' in args.model:
                images = vae.decode(latent / args.scale)
                if args.vae_version == 'd8':
                    images = images.sample
                images = torch.clamp(127.5 * images + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            else:
                B, C, H, W = origin_samples.shape
                patch_origin = model.module.patchify(origin_samples)
                pred = torch.where((mask > 0).unsqueeze(-1).expand(B, patch_origin.shape[-2], patch_origin.shape[-1]), pred, patch_origin)
                latent = model.module.unpatchify(pred)
                images = latent * torch.tensor([0.229, 0.224, 0.225], device=device).reshape(1, 3, 1, 1) + torch.tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
                # r = latent[:, 0] * 0.229 + 0.485
                # g = latent[:, 1] * 0.224 + 0.456
                # b = latent[:, 2] * 0.225 + 0.404
                # images = torch.concat([r.unsqueeze(1), g.unsqueeze(1), b.unsqueeze(1)], dim=1)
                images = torch.clamp(images * 255, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # torch.distributed.barrier()
        # Save samples to disk as individual .png files
        for i, sample in enumerate(images):
            index = i * world_size + local_rank + total
            gen_img = sample.astype(np.uint8)[:, :, ::-1]
            origin_img = origin_images[i]
            # Image.fromarray(gen_img).save(os.path.join(args.save_folder, '{}.png'.format(str(index).zfill(5))))
            Image.fromarray(origin_img).save(os.path.join(args.save_folder, '{}_origin.png'.format(str(index).zfill(5))))
            cv2.imwrite(os.path.join(args.save_folder, '{}.png'.format(str(index).zfill(5))), gen_img)
            # cv2.imwrite(os.path.join(args.save_folder, '{}_origin.png'.format(str(index).zfill(5))), origin_img)
            with open(os.path.join(args.save_folder, '{}.mask'.format(str(index).zfill(5))), 'wb') as f:
                numpy.save(f, mask[i].to("cpu").numpy())
        total += global_batch_size
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
