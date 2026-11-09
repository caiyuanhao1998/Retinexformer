---
title: Retinexformer
emoji: 💡
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.50.0
python_version: 3.10.13
app_file: space_app.py
pinned: false
short_description: Low-light image enhancement with Retinexformer
models:
  - CaiYuanhao/Retinexformer
tags:
  - image-to-image
  - low-light-image-enhancement
  - retinexformer
---

# Retinexformer

Online demo for **Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement (ICCV 2023)**.

Upload a low-light photo or choose an example, select a model and click **Enhance image**. Adjust **Output brightness** after inference and download the result as a PNG.

All nine official Retinexformer checkpoints are downloaded automatically from the model repository. GPU inference uses ZeroGPU when available; CPU hosting is also supported. Images are resized to the selected maximum edge length to fit the hosted resources.

[Paper](https://arxiv.org/abs/2303.06705) · [GitHub](https://github.com/caiyuanhao1998/Retinexformer) · [Pretrained models](https://huggingface.co/CaiYuanhao/Retinexformer)

![Retinexformer demo](figure/inference_demo.png)

The five example images come from the LOL-v1 test set; see [source and attribution](demo/examples/README.md).
