&nbsp;

<div align="center">
<p align="center"> <img src="figure/logo.png" width="200px"> </p>


[![arXiv](https://img.shields.io/badge/arxiv-paper-179bd3)](https://arxiv.org/abs/2303.06705)
[![NTIRE](https://img.shields.io/badge/NTIRE_2024-leaderboard-179bd3)](https://codalab.lisn.upsaclay.fr/competitions/17640#results)
[![zhihu](https://img.shields.io/badge/zhihu-知乎-179bd3)](https://zhuanlan.zhihu.com/p/657927878)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/retinexformer-one-stage-retinex-based/low-light-image-enhancement-on-lol-v2-1)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lol-v2-1?p=retinexformer-one-stage-retinex-based)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/retinexformer-one-stage-retinex-based/low-light-image-enhancement-on-mit-adobe-1)](https://paperswithcode.com/sota/low-light-image-enhancement-on-mit-adobe-1?p=retinexformer-one-stage-retinex-based)



[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/retinexformer-one-stage-retinex-based/low-light-image-enhancement-on-sdsd-indoor)](https://paperswithcode.com/sota/low-light-image-enhancement-on-sdsd-indoor?p=retinexformer-one-stage-retinex-based)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/retinexformer-one-stage-retinex-based/low-light-image-enhancement-on-sdsd-outdoor)](https://paperswithcode.com/sota/low-light-image-enhancement-on-sdsd-outdoor?p=retinexformer-one-stage-retinex-based)


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/retinexformer-one-stage-retinex-based/low-light-image-enhancement-on-smid)](https://paperswithcode.com/sota/low-light-image-enhancement-on-smid?p=retinexformer-one-stage-retinex-based)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/retinexformer-one-stage-retinex-based/low-light-image-enhancement-on-sid)](https://paperswithcode.com/sota/low-light-image-enhancement-on-sid?p=retinexformer-one-stage-retinex-based)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/retinexformer-one-stage-retinex-based/low-light-image-enhancement-on-lol-v2)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lol-v2?p=retinexformer-one-stage-retinex-based)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/retinexformer-one-stage-retinex-based/low-light-image-enhancement-on-lol)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lol?p=retinexformer-one-stage-retinex-based)


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/retinexformer-one-stage-retinex-based/low-light-image-enhancement-on-dicm)](https://paperswithcode.com/sota/low-light-image-enhancement-on-dicm?p=retinexformer-one-stage-retinex-based)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/retinexformer-one-stage-retinex-based/low-light-image-enhancement-on-lime)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lime?p=retinexformer-one-stage-retinex-based)


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/retinexformer-one-stage-retinex-based/low-light-image-enhancement-on-mef)](https://paperswithcode.com/sota/low-light-image-enhancement-on-mef?p=retinexformer-one-stage-retinex-based)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/retinexformer-one-stage-retinex-based/low-light-image-enhancement-on-npe)](https://paperswithcode.com/sota/low-light-image-enhancement-on-npe?p=retinexformer-one-stage-retinex-based)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/retinexformer-one-stage-retinex-based/low-light-image-enhancement-on-vv)](https://paperswithcode.com/sota/low-light-image-enhancement-on-vv?p=retinexformer-one-stage-retinex-based)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/retinexformer-one-stage-retinex-based/image-enhancement-on-mit-adobe-5k)](https://paperswithcode.com/sota/image-enhancement-on-mit-adobe-5k?p=retinexformer-one-stage-retinex-based)


</div>


&nbsp;

### Introduction
This is a baseline and toolbox for wide-range low-light image enhancement. This repo **supports over 15 benchmarks** and extremely high-resolution (up to 4000x6000) low-light enhancement. Our method Retinexformer **won the second place** in the [NTIRE 2024 Challenge on Low Light Enhancement](https://codalab.lisn.upsaclay.fr/competitions/17640).  If you find this repo useful, please give it a star ⭐ and consider citing our paper in your research. Thank you.

### News
- **2024.03.22 :** We release `distributed data parallel (DDP)` and `mix-precision` training strategies to help you train larger models. We release `self-ensemble` testing strategy to help you derive better results. In addition, we also release an adaptive `split-and-test` testing strategy for high-resolution up to 4000x6000 low-light image enhancement. Feel free to use them. 🚀
- **2024.03.21 :** Our methods [Retinexformer](https://github.com/caiyuanhao1998/Retinexformer) and [MST++](https://github.com/caiyuanhao1998/MST-plus-plus) (NTIRE 2022 Spectral Reconstruction Challenge Winner) ranked top-2 in the [NTIRE 2024 Challenge on Low Light Enhancement](https://codalab.lisn.upsaclay.fr/competitions/17640). Code, pre-trained weights, training logs, and enhancement results have been released in this repo. Feel free to use them! 🚀
- **2024.02.15 :** [NTIRE 2024 Challenge on Low Light Enhancement](https://codalab.lisn.upsaclay.fr/competitions/17640) begins. Welcome to use our [Retinexformer](https://github.com/caiyuanhao1998/Retinexformer) or [MST++](https://github.com/caiyuanhao1998/MST-plus-plus) (NTIRE 2022 Spectral Reconstruction Challenge Winner) to participate in this challenge! :trophy:
- **2023.11.03 :** The test setting of KinD, LLFlow, and recent diffusion models and the corresponding results on LOL are provided. Please note that we do not suggest this test setting because it uses the mean of the ground truth to obtain better results. But, if you want to follow KinD, LLFlow, and recent diffusion-based works for fair comparison, it is your choice to use this test setting. Please refer to the `Testing` part for details.
- **2023.11.02 :** Retinexformer is added to the [Awesome-Transformer-Attention](https://github.com/cmhungsteve/Awesome-Transformer-Attention/blob/main/README_2.md#image-restoration) collection. 💫
- **2023.10.20 :** Params and FLOPS evaluating function is provided. Feel free to check and use it.
- **2023.10.12 :** Retinexformer is added to the [ICCV-2023-paper](https://github.com/DmitryRyumin/ICCV-2023-Papers#low-level-and-physics-based-vision) collection. 🚀
- **2023.10.10 :** Retinexformer is added to the [low-level-vision-paper-record](https://github.com/lcybuzz/Low-Level-Vision-Paper-Record) collection. ⭐
- **2023.10.06 :** Retinexformer is added to the [awesome-low-light-image-enhancement](https://github.com/dawnlh/awesome-low-light-image-enhancement) collection. :tada:
- **2023.09.20 :** Some results on ExDark nighttime object detection are released.
- **2023.09.20 :** Code, models, results, and training logs have been released. Feel free to use them. ⭐
- **2023.07.14 :** Our paper has been accepted by ICCV 2023. Code and Models will be released. :rocket:

### Results
- Results on LOL-v1, LOL-v2-real, LOL-v2-synthetic, SID, SMID, SDSD-in, SDSD-out, and MIT Adobe FiveK datasets can be downloaded from [Baidu Disk](https://pan.baidu.com/s/1DC6A-I9S7yJ-pmMVTLAHaw?pwd=cyh2) (code: `cyh2`) or [Google Drive](https://drive.google.com/drive/folders/1UCpHh3MkV4bxzWgiiULnb3BOPWS_8crP?usp=drive_link)

- Results on LOL-v1, LOL-v2-real, and LOL-v2-synthetic datasets with the same test setting as KinD, LLFlow, and recent diffusion models can be downloaded from [Baidu Disk](https://pan.baidu.com/s/1Kbq9pASf1O_0Y9QMc88obQ?pwd=cyh2) (code: `cyh2`) or [Google Drive](https://drive.google.com/drive/folders/1_ugNFblIYOCIam4cVJiXVX1aBGEYhn1o?usp=drive_link).

- Results on the NTIRE 2024 low-light enhancement dataset can be downloaded from [Baidu Disk](https://pan.baidu.com/s/1DC6A-I9S7yJ-pmMVTLAHaw?pwd=cyh2) (code: `cyh2`) or [Google Drive](https://drive.google.com/drive/folders/1bzICalpU1RprfnepLsMyCYT569UWTTlV?usp=sharing)

- Results on LIME, NPE, MEF, DICM, and VV datasets can be downloaded from [Baidu Disk](https://pan.baidu.com/s/1cqBwmuXk83h6u1NZJVbfkg?pwd=cyh2) (code: `cyh2`) or [Google Drive](https://drive.google.com/drive/folders/1rWa_WRX5bqlW2HnBNMUGFKWrou7gIQpO?usp=drive_link)

- Results on ExDark nighttime object detection can be downloaded from [Baidu Disk](https://pan.baidu.com/s/1ZvoPzYQePRc80-o7rrJuRQ?pwd=cyh2) (code: `cyh2`) or [Google Drive](https://drive.google.com/drive/folders/1nZQnwKkGvswv--JunzgBLTXVRlOdcxb6?usp=sharing). Please use [this repo](https://github.com/cuiziteng/Illumination-Adaptive-Transformer/tree/main/IAT_high/IAT_mmdetection) to run experiments on the ExDark dataset


<details close>
<summary><b>Performance on LOL-v1, LOL-v2-real, LOL-v2-synthetic, SID, SMID, SDSD-in, and SDSD-out:</b></summary>

![results1](/figure/seven_results.png)


</details>

<details close>
<summary><b>Performance on LOL with the same test setting as KinD, LLFlow, and diffusion models:</b></summary>

|  Metric  |   LOL-v1    |  LOL-v2-real  |  LOL-v2-synthetic  |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| PSNR | 27.18 | 27.71 | 29.04 |
| SSIM | 0.850 | 0.856 | 0.939 |

Please note that we do not suggest this test setting because it uses the mean of the ground truth to obtain better results. But, if you want to follow KinD, LLFlow, and recent diffusion-based works, it is your choice to use this test setting. Please refer to the `Testing` part for details.

</details>


</details>

<details close>
<summary><b>Performance on NTIRE 2024 test-challenge:</b></summary>

|  Method  |   Retinexformer    |  MST++  |  Ensemble  |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| PSNR | 24.61 | 24.59 | 25.30 |
| SSIM | 0.85 | 0.85 | 0.85 |

Feel free to check the [Codalab leaderboard](https://codalab.lisn.upsaclay.fr/competitions/17640#results). Our method ranks second.

![results_ntire](/figure/ntire_2024.png)

</details>


<details close>
<summary><b>Performance on MIT Adobe FiveK:</b></summary>

![results2](/figure/fivek_results.png)


</details>


<details close>
<summary><b>Performance on LIME, NPE, MEF, DICM, and VV:</b></summary>

![results3](/figure/visual_compare_no_gt.png)


</details>


<details close>
<summary><b>Performance on ExDark Nighttime object detection:</b></summary>

![results4](/figure/exdark_results.png)


</details>


### Gallery
|                          *NTIRE - dev - 2000x3000*                           |                          *NTIRE - challenge - 4000x6000*                           |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| [<img src="/figure/ntire_dev.png" height="250px"/>](https://imgsli.com/MjQ5Mzk5) | [<img src="/figure/ntire_challenge.png" height="250px"/>](https://imgsli.com/MjQ5NDAy) |

&nbsp;


## 1. Create Environment

We suggest you use pytorch 1.11 to re-implement the results in our ICCV 2023 paper and pytorch 2 to re-implement the results in NTIRE 2024 Challenge because pytorch 2 can save more memory in mix-precision training.

### 1.1 Install the environment with Pytorch 1.11

- Make Conda Environment
```
conda create -n Retinexformer python=3.7
conda activate Retinexformer
```

- Install Dependencies
```
conda install pytorch=1.11 torchvision cudatoolkit=11.3 -c pytorch

pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard

pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips
```

- Install BasicSR
```
python setup.py develop --no_cuda_ext
```

### 1.2 Install the environment with Pytorch 2

- Make Conda Environment
```
conda create -n torch2 python=3.9 -y
conda activate torch2
```

- Install Dependencies
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard

pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips thop timm
```

- Install BasicSR
```
python setup.py develop --no_cuda_ext
```

&nbsp;

## 2. Prepare Dataset
Download the following datasets:

LOL-v1 [Baidu Disk](https://pan.baidu.com/s/1ZAC9TWR-YeuLIkWs3L7z4g?pwd=cyh2) (code: `cyh2`), [Google Drive](https://drive.google.com/file/d/1L-kqSQyrmMueBh_ziWoPFhfsAh50h20H/view?usp=sharing)

LOL-v2 [Baidu Disk](https://pan.baidu.com/s/1X4HykuVL_1WyB3LWJJhBQg?pwd=cyh2) (code: `cyh2`), [Google Drive](https://drive.google.com/file/d/1Ou9EljYZW8o5dbDCf9R34FS8Pd8kEp2U/view?usp=sharing)

SID [Baidu Disk](https://pan.baidu.com/share/init?surl=HRr-5LJO0V0CWqtoctQp9w) (code: `gplv`), [Google Drive](https://drive.google.com/drive/folders/1eQ-5Z303sbASEvsgCBSDbhijzLTWQJtR?usp=share_link&pli=1)

SMID [Baidu Disk](https://pan.baidu.com/share/init?surl=Qol_4GsIjGDR8UT9IRZbBQ) (code: `btux`), [Google Drive](https://drive.google.com/drive/folders/1OV4XgVhipsRqjbp8SYr-4Rpk3mPwvdvG)

SDSD-indoor [Baidu Disk](https://pan.baidu.com/s/1rfRzshGNcL0MX5soRNuwTA?errmsg=Auth+Login+Params+Not+Corret&errno=2&ssnerror=0#list/path=%2F) (code: `jo1v`), [Google Drive](https://drive.google.com/drive/folders/14TF0f9YQwZEntry06M93AMd70WH00Mg6)

SDSD-outdoor [Baidu Disk](https://pan.baidu.com/share/init?surl=JzDQnFov-u6aBPPgjSzSxQ) (code: `uibk`), [Google Drive](https://drive.google.com/drive/folders/14TF0f9YQwZEntry06M93AMd70WH00Mg6)

MIT-Adobe FiveK [Baidu Disk](https://pan.baidu.com/s/1ajax7N9JmttTwY84-8URxA?pwd=cyh2) (code:`cyh2`), [Google Drive](https://drive.google.com/file/d/11HEUmchFXyepI4v3dhjnDnmhW_DgwfRR/view?usp=sharing), [Official](https://data.csail.mit.edu/graphics/fivek/)

NTIRE 2024 [Baidu Disk](https://pan.baidu.com/s/1Tl-LUhwsPh6XFA2SqR5c8Q?pwd=cyh2) (code:`cyh2`), Google Drive links for [training input](https://drive.google.com/file/d/1Js9yHmV0xAWhT5oJKzfx6oOr_7k5hcNg/view), [training GT](https://drive.google.com/file/d/1PUJgJiEyrIj5TgwcQlFvVGuIe3_PXMLY/view), and [mini-val set](https://drive.google.com/drive/folders/1M-WVWToH1HhMtmQlYrb8qlNCQgi0kG3y?usp=sharing).

**Note:** 

(1) Please use [bandizip](https://www.bandisoft.com/bandizip/) to jointly unzip the `.zip` and `.z01` files of SMID, SDSD-indoor, and SDSD-outdoor 

(2) Please process the raw images of the MIT Adobe FiveK dataset following [the sRGB output mode](https://github.com/nothinglo/Deep-Photo-Enhancer/issues/38) or directly download and use the sRGB image pairs processed by us in the [Baidu Disk](https://pan.baidu.com/s/1ajax7N9JmttTwY84-8URxA?pwd=cyh2) (code:`cyh2`) and [Google Drive](https://drive.google.com/file/d/11HEUmchFXyepI4v3dhjnDnmhW_DgwfRR/view?usp=sharing)

(3) Please download the `text_list.txt` from [Google Drive](https://drive.google.com/file/d/199qrfizUeZfgq3qVjrM74mZ_nlacgwiP/view?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/1GQfaQLI6tvB0IrTMPOM_9Q?pwd=ggbh) (code: `ggbh`) and then put it into the folder `data/SMID/SMID_Long_np/`

<details close>
<summary><b> Then organize these datasets as follows: </b></summary>

```
    |--data   
    |    |--LOLv1
    |    |    |--Train
    |    |    |    |--input
    |    |    |    |    |--100.png
    |    |    |    |    |--101.png
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |    |--100.png
    |    |    |    |    |--101.png
    |    |    |    |     ...
    |    |    |--Test
    |    |    |    |--input
    |    |    |    |    |--111.png
    |    |    |    |    |--146.png
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |    |--111.png
    |    |    |    |    |--146.png
    |    |    |    |     ...
    |    |--LOLv2
    |    |    |--Real_captured
    |    |    |    |--Train
    |    |    |    |    |--Low
    |    |    |    |    |    |--00001.png
    |    |    |    |    |    |--00002.png
    |    |    |    |    |     ...
    |    |    |    |    |--Normal
    |    |    |    |    |    |--00001.png
    |    |    |    |    |    |--00002.png
    |    |    |    |    |     ...
    |    |    |    |--Test
    |    |    |    |    |--Low
    |    |    |    |    |    |--00690.png
    |    |    |    |    |    |--00691.png
    |    |    |    |    |     ...
    |    |    |    |    |--Normal
    |    |    |    |    |    |--00690.png
    |    |    |    |    |    |--00691.png
    |    |    |    |    |     ...
    |    |    |--Synthetic
    |    |    |    |--Train
    |    |    |    |    |--Low
    |    |    |    |    |   |--r000da54ft.png
    |    |    |    |    |   |--r02e1abe2t.png
    |    |    |    |    |    ...
    |    |    |    |    |--Normal
    |    |    |    |    |   |--r000da54ft.png
    |    |    |    |    |   |--r02e1abe2t.png
    |    |    |    |    |    ...
    |    |    |    |--Test
    |    |    |    |    |--Low
    |    |    |    |    |   |--r00816405t.png
    |    |    |    |    |   |--r02189767t.png
    |    |    |    |    |    ...
    |    |    |    |    |--Normal
    |    |    |    |    |   |--r00816405t.png
    |    |    |    |    |   |--r02189767t.png
    |    |    |    |    |    ...
    |    |--SDSD
    |    |    |--indoor_static_np
    |    |    |    |--input
    |    |    |    |    |--pair1
    |    |    |    |    |   |--0001.npy
    |    |    |    |    |   |--0002.npy
    |    |    |    |    |    ...
    |    |    |    |    |--pair2
    |    |    |    |    |   |--0001.npy
    |    |    |    |    |   |--0002.npy
    |    |    |    |    |    ...
    |    |    |    |     ...
    |    |    |    |--GT
    |    |    |    |    |--pair1
    |    |    |    |    |   |--0001.npy
    |    |    |    |    |   |--0002.npy
    |    |    |    |    |    ...
    |    |    |    |    |--pair2
    |    |    |    |    |   |--0001.npy
    |    |    |    |    |   |--0002.npy
    |    |    |    |    |    ...
    |    |    |    |     ...
    |    |    |--outdoor_static_np
    |    |    |    |--input
    |    |    |    |    |--MVI_0898
    |    |    |    |    |   |--0001.npy
    |    |    |    |    |   |--0002.npy
    |    |    |    |    |    ...
    |    |    |    |    |--MVI_0918
    |    |    |    |    |   |--0001.npy
    |    |    |    |    |   |--0002.npy
    |    |    |    |    |    ...
    |    |    |    |     ...
    |    |    |    |--GT
    |    |    |    |    |--MVI_0898
    |    |    |    |    |   |--0001.npy
    |    |    |    |    |   |--0002.npy
    |    |    |    |    |    ...
    |    |    |    |    |--MVI_0918
    |    |    |    |    |   |--0001.npy
    |    |    |    |    |   |--0002.npy
    |    |    |    |    |    ...
    |    |    |    |     ...
    |    |--SID
    |    |    |--short_sid2
    |    |    |    |--00001
    |    |    |    |    |--00001_00_0.04s.npy
    |    |    |    |    |--00001_00_0.1s.npy
    |    |    |    |    |--00001_01_0.04s.npy
    |    |    |    |    |--00001_01_0.1s.npy
    |    |    |    |     ...
    |    |    |    |--00002
    |    |    |    |    |--00002_00_0.04s.npy
    |    |    |    |    |--00002_00_0.1s.npy
    |    |    |    |    |--00002_01_0.04s.npy
    |    |    |    |    |--00002_01_0.1s.npy
    |    |    |    |     ...
    |    |    |     ...
    |    |    |--long_sid2
    |    |    |    |--00001
    |    |    |    |    |--00001_00_0.04s.npy
    |    |    |    |    |--00001_00_0.1s.npy
    |    |    |    |    |--00001_01_0.04s.npy
    |    |    |    |    |--00001_01_0.1s.npy
    |    |    |    |     ...
    |    |    |    |--00002
    |    |    |    |    |--00002_00_0.04s.npy
    |    |    |    |    |--00002_00_0.1s.npy
    |    |    |    |    |--00002_01_0.04s.npy
    |    |    |    |    |--00002_01_0.1s.npy
    |    |    |    |     ...
    |    |    |     ...
    |    |--SMID
    |    |    |--SMID_LQ_np
    |    |    |    |--0001
    |    |    |    |    |--0001.npy
    |    |    |    |    |--0002.npy
    |    |    |    |     ...
    |    |    |    |--0002
    |    |    |    |    |--0001.npy
    |    |    |    |    |--0002.npy
    |    |    |    |     ...
    |    |    |     ...
    |    |    |--SMID_Long_np
    |    |    |    |--text_list.txt
    |    |    |    |--0001
    |    |    |    |    |--0001.npy
    |    |    |    |    |--0002.npy
    |    |    |    |     ...
    |    |    |    |--0002
    |    |    |    |    |--0001.npy
    |    |    |    |    |--0002.npy
    |    |    |    |     ...
    |    |    |     ...
    |    |--FiveK
    |    |    |--train
    |    |    |    |--input
    |    |    |    |    |--a0099-kme_264.jpg
    |    |    |    |    |--a0101-kme_610.jpg
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |    |--a0099-kme_264.jpg
    |    |    |    |    |--a0101-kme_610.jpg
    |    |    |    |     ...
    |    |    |--test
    |    |    |    |--input
    |    |    |    |    |--a4574-DSC_0038.jpg
    |    |    |    |    |--a4576-DSC_0217.jpg
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |    |--a4574-DSC_0038.jpg
    |    |    |    |    |--a4576-DSC_0217.jpg
    |    |    |    |     ...
    |    |--NTIRE
    |    |    |--train
    |    |    |    |--input
    |    |    |    |    |--1.png
    |    |    |    |    |--3.png
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |    |--1.png
    |    |    |    |    |--3.png
    |    |    |    |     ...
    |    |    |--minival
    |    |    |    |--input
    |    |    |    |    |--1.png
    |    |    |    |    |--31.png
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |    |--1.png
    |    |    |    |    |--31.png
    |    |    |    |     ...

```

</details>

We also provide download links for LIME, NPE, MEF, DICM, and VV datasets that have no ground truth:

[Baidu Disk](https://pan.baidu.com/s/1oHg03tOfWWLp4q1R6rlzww?pwd=cyh2) (code: `cyh2`)
 or [Google Drive](https://drive.google.com/drive/folders/1RR50EJYGIHaUYwq4NtK7dx8faMSvX8Xp?usp=drive_link)


&nbsp;                    


## 3. Testing

Download our models from [Baidu Disk](https://pan.baidu.com/s/13zNqyKuxvLBiQunIxG_VhQ?pwd=cyh2) (code: `cyh2`) or [Google Drive](https://drive.google.com/drive/folders/1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV?usp=drive_link). Put them in folder `pretrained_weights`

```shell
# activate the environment
conda activate Retinexformer

# LOL-v1
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v1.yml --weights pretrained_weights/LOL_v1.pth --dataset LOL_v1

# LOL-v2-real
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v2_real.yml --weights pretrained_weights/LOL_v2_real.pth --dataset LOL_v2_real

# LOL-v2-synthetic
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v2_synthetic.yml --weights pretrained_weights/LOL_v2_synthetic.pth --dataset LOL_v2_synthetic

# SID
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_SID.yml --weights pretrained_weights/SID.pth --dataset SID

# SMID
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_SMID.yml --weights pretrained_weights/SMID.pth --dataset SMID

# SDSD-indoor
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_SDSD_indoor.yml --weights pretrained_weights/SDSD_indoor.pth --dataset SDSD_indoor

# SDSD-outdoor
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_SDSD_outdoor.yml --weights pretrained_weights/SDSD_outdoor.pth --dataset SDSD_outdoor

# FiveK
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_FiveK.yml --weights pretrained_weights/FiveK.pth --dataset FiveK

# NTIRE
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_NTIRE.yml --weights pretrained_weights/NTIRE.pth --dataset NTIRE --self_ensemble

# MST_Plus_Plus trained with 4 GPUs on NTIRE 
python3 Enhancement/test_from_dataset.py --opt Options/MST_Plus_Plus_NTIRE_4x1800.yml --weights pretrained_weights/MST_Plus_Plus_4x1800.pth --dataset NTIRE --self_ensemble

# MST_Plus_Plus trained with 8 GPUs on NTIRE 
python3 Enhancement/test_from_dataset.py --opt Options/MST_Plus_Plus_NTIRE_8x1150.yml --weights pretrained_weights/MST_Plus_Plus_8x1150.pth --dataset NTIRE --self_ensemble

```

- #### Self-ensemble testing strategy
We add the self-ensemble strategy in the testing code to derive better results. Just add a `--self_ensemble` action at the end of the above test command to use it.


- #### The same test setting as LLFlow, KinD, and recent diffusion models
We provide the same test setting as LLFlow, KinD, and recent diffusion models. Please note that we do not suggest this test setting because it uses the mean of ground truth to enhance the output of the model. But if you want to follow this test setting, just add a `--GT_mean` action at the end of the above test command as

```shell
# LOL-v1
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v1.yml --weights pretrained_weights/LOL_v1.pth --dataset LOL_v1 --GT_mean

# LOL-v2-real
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v2_real.yml --weights pretrained_weights/LOL_v2_real.pth --dataset LOL_v2_real --GT_mean

# LOL-v2-synthetic
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v2_synthetic.yml --weights pretrained_weights/LOL_v2_synthetic.pth --dataset LOL_v2_synthetic --GT_mean
```


- #### Evaluating the Params and FLOPS of models
We have provided a function `my_summary()` in `Enhancement/utils.py`, please use this function to evaluate the parameters and computational complexity of the models, especially the Transformers as

```shell
from utils import my_summary
my_summary(RetinexFormer(), 256, 256, 3, 1)
```


&nbsp;


## 4. Training

Feel free to check our training logs from [Baidu Disk](https://pan.baidu.com/s/16NtLba_ANe3Vzji-eZ1xAA?pwd=cyh2) (code: `cyh2`) or [Google Drive](https://drive.google.com/drive/folders/1HU_wEn_95Hakxi_ze-pS6Htikmml5MTA?usp=sharing)

We suggest you use the environment with pytorch 2 to train our model on the NTIRE 2024 dataset and the environment with pytorch 1.11 to train our model on other datasets.

```shell
# activate the enviroment
conda activate Retinexformer

# LOL-v1
python3 basicsr/train.py --opt Options/RetinexFormer_LOL_v1.yml

# LOL-v2-real
python3 basicsr/train.py --opt Options/RetinexFormer_LOL_v2_real.yml

# LOL-v2-synthetic
python3 basicsr/train.py --opt Options/RetinexFormer_LOL_v2_synthetic.yml

# SID
python3 basicsr/train.py --opt Options/RetinexFormer_SID.yml

# SMID
python3 basicsr/train.py --opt Options/RetinexFormer_SMID.yml

# SDSD-indoor
python3 basicsr/train.py --opt Options/RetinexFormer_SDSD_indoor.yml

# SDSD-outdoor
python3 basicsr/train.py --opt Options/RetinexFormer_SDSD_outdoor.yml

# FiveK
python3 basicsr/train.py --opt Options/RetinexFormer_FiveK.yml
```

Train  our Retinexformer and MST++ with the distributed data parallel (DDP) strategy of pytorch on the NTIRE 2024 Low-Light Enhancement dataset. Please note that we use the mix-precision strategy in the training process, which is controlled by the bool hyperparameter `use_amp`  in the config file.

```shell
# activate the enviroment
conda activate torch2

# Train Retinexformer with 8 GPUs on NTIRE
bash train_multigpu.sh Options/RetinexFormer_NTIRE_8x2000.yml 0,1,2,3,4,5,6,7 4321

# Train MST++ with 4 GPUs on NTIRE
bash train_multigpu.sh Options/RetinexFormer_NTIRE_4x1800.yml 0,1,2,3,4,5,6,7 4329

# Train MST++ with 8 GPUs on NTIRE
bash train_multigpu.sh Options/MST_Plus_Plus_NTIRE_8x1150.yml 0,1,2,3,4,5,6,7 4343

```



&nbsp;


## 5. Citation

```shell
@InProceedings{Cai_2023_ICCV,
    author    = {Cai, Yuanhao and Bian, Hao and Lin, Jing and Wang, Haoqian and Timofte, Radu and Zhang, Yulun},
    title     = {Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {12504-12513}
}

@inproceedings{retinexformer,
  title={Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement},
  author={Yuanhao Cai and Hao Bian and Jing Lin and Haoqian Wang and Radu Timofte and Yulun Zhang},
  booktitle={ICCV},
  year={2023}
}


# MST++
@inproceedings{mst,
  title={Mask-guided Spectral-wise Transformer for Efficient Hyperspectral Image Reconstruction},
  author={Yuanhao Cai and Jing Lin and Xiaowan Hu and Haoqian Wang and Xin Yuan and Yulun Zhang and Radu Timofte and Luc Van Gool},
  booktitle={CVPR},
  year={2022}
}
```

