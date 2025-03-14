# For DiTs

```bash
cd generation/DiT
```

**Training**
```bash
sh train_dit.sh
```
**Note:**
- ${model}: "DiT-XL/2-VAE-simple", "DiT-L/2-VAE-simple", "DiT-B/2-VAE-simple".  
- During training, sampling occurs every `${eval-every}` steps, and the results are saved as a **NPZ file** for evaluation. You can also use the **script below** to sample any saved fine-tuned weights.
- The default ${global-batch-size} is 256.

**Infer**
```bash
sh sample_dit.sh
```
- Sampling using the weights saved during the **fine-tuning process** and saving them as an **NPZ file**, which can be used for evaluating metrics.

**Eval**
```bash
sh eval_dit.sh
```
**Note:**
- Same as [DiT](https://github.com/facebookresearch/DiT), we use [ADM's TensorFlow evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) to calculate FID, Inception Score and other metrics.


# For SiTs

```
cd generation/SiT
```

**Training**
```bash
sh train_sit.sh
```
**Note:**
- ${model}: "SiT-XL/2-VAE-simple", "SiT-B/2-VAE-simple".  
- During training, sampling occurs every `${eval-every}` steps, and the results are saved as a **NPZ file** for evaluation. You can also use the **script below** to sample any saved fine-tuned weights.
- The default ${global-batch-size} is 256.

**Infer**
```bash
sh sample_sit.sh
```
- Sampling using the weights saved during the **fine-tuning process** and saving them as an **NPZ file**, which can be used for evaluating metrics.

**Eval**
- Same as the evaluation in the DiT.