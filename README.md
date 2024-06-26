# Diffusion_transformer_from_scratch

## Introduction

Diffusion Transformers trained on MNIST dataset

用transformer-backbone来替换unet-backbone，用于实现stable diffusion扩散模型

## Preliminary

- **扩散模型的训练与推理过程**

![diffusion process](./assets/diffusion.png)

## Architecture

![diffusion transformer architecture](./assets/dit_architecture.png)

## key

```python
    def modulate(x, shift, scale):
        r"""
        Perform dit block shift and scale
        Args:
            x:      torch.tensor, [b, L, c]
            shift:  torch.tensor, [b, c]
            scale:  torch.tensor, [b, c]
        Return:
            torch.tensor, [b, L, c]
        """
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

```

## Loss

![loss](./assets/loss.png)

## Inference

![loss](./assets/inference-result.png)

## Todo

## Acknowledgements

- [Scalable Diffusion Models with Transformers (DiT)](https://github.com/facebookresearch/DiT)
- [pytorch-diffusion-transformer](https://github.com/owenliang/mnist-dits)