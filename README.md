<h1 align="center">
[ICCV2025] USP: Unified Self-Supervised Pretraining for Image Generation and Understanding
</h1>

[![arXiv](http://img.shields.io/badge/cs.CV-arXiv%3A2503.06132-B31B1B.svg)](https://arxiv.org/abs/2503.06132)

This is official implementation of USP.

![arch](method.png)

Converge much faster just with weight initialization from pretrain. 
![converge](XL_converge.png)


If you find USP useful in your research or applications, please consider giving a star ⭐ and citing using the following BibTeX:
```
@misc{chu2025uspunifiedselfsupervisedpretraining,
      title={USP: Unified Self-Supervised Pretraining for Image Generation and Understanding}, 
      author={Xiangxiang Chu and Renda Li and Yong Wang},
      year={2025},
      eprint={2503.06132},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.06132}, 
}

```
### Catalog
- [x] 【4.21】Upload image generation finetuning weights 
- [x] Pre-training code
- [x] (ImageNet SFT and linear probe finetuning code)

## Finetuning Weights  
Uploaded image generation finetuning weights in [Hugging Face](https://huggingface.co/GD-ML/USP-Image_Generation/tree/main)

All weights were pretrained for 1600 epochs and then finetuned for 400 K steps. 

Using the above weights and following the inference and evaluation procedures outlined in [GENERATION.md](./generation/GENERATION.md), we obtained the following evaluation results:

| Model Name | Pretrain       | Finetuning     | FID    | IS    | sFID   |
|------------|----------------|----------------|--------|-------|--------|
| DiT_B-2    | 1600 epochs    | 400 K steps    | 27.22  | 50.47  | 7.60   |
| DiT_L-2    | 1600 epochs    | 400 K steps    | 15.05  | 80.11  | 6.41   |
| DiT_XL-2   | 1600 epochs    | 400 K steps    | 9.64  | 112.93  | 6.30   |
| SiT_B-2    | 1600 epochs    | 400 K steps    | 22.10  | 61.59  | 5.88   |
| SiT_XL-2   | 1600 epochs    | 400 K steps    |  7.35  | 128.50  | 5.00   |

Our method is somewhat orthogonal to other DINO based acceleration methods. 

| Model          | Params | Steps      | FID (↓)       | IS (↑)        |
|----------------|--------|------------|---------------|---------------|
| SiT-XL/2       | 130M   | 400K       | 16.97         | 77.50         |
| **USP**       | 130M   | 400K       | **7.38**      | **127.96**    |
| REPA           | 130M   | 400K       | 7.9           | 122.6         |
| **USP + REPA** | 130M   | 400K       | **6.26**      | **139.84**    |
| VAVAE          | 130M   | 64 Epochs  | 5.18/2.15†    | 132.4/245.1†  |
| **USP + VAVAE**| 130M   | 64 Epochs  | **4.2/1.81†** | **144/261.0†**|

*Table: Results Combined with External-Model-Based Methods. †: w/ CFG=10.0.*

## Introduction
Recent studies have highlighted the interplay between diffusion models and representation learning. Intermediate representations from diffusion models can be leveraged for downstream visual tasks, while self-supervised vision models can enhance the convergence and generation quality of diffusion models. However, transferring pretrained weights from vision models to diffusion models is challenging due to input mismatches and the use of latent spaces. To address these challenges, we propose Unified Self-supervised Pretraining (USP), a framework that initializes diffusion models via masked latent modeling in a Variational Autoencoder (VAE) latent space. USP achieves comparable performance in understanding tasks while significantly improving the convergence speed and generation quality of diffusion models.

[//]: # (## Updates)

[//]: # ()
[//]: # (Our code is released.)
## Pretraining
Please refer to  [PRETRAIN.md](./pretrain/PRETRAIN.md)
## Downstream Task
### Generation
Please refer to  [GENERATION.md](./generation/GENERATION.md)

[//]: # (### Image Generation Under the DiT Framework)

[//]: # (### Image Generation Under the SiT Framework)

[//]: # (### Image Understanding)

## Acknowledgement

Our  code are based on  [MAE](https://github.com/facebookresearch/mae), [DiT](https://github.com/facebookresearch/DiT), [SiT](https://github.com/willisma/SiT) and  [VisionLLaMA](https://github.com/Meituan-AutoML/VisionLLaMA). Thanks for their great work.


