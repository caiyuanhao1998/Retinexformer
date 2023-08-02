[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/retinexformer-one-stage-retinex-based/low-light-image-enhancement-on-sid)](https://paperswithcode.com/sota/low-light-image-enhancement-on-sid?p=retinexformer-one-stage-retinex-based)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/retinexformer-one-stage-retinex-based/low-light-image-enhancement-on-sid)](https://paperswithcode.com/sota/low-light-image-enhancement-on-sid?p=retinexformer-one-stage-retinex-based)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/retinexformer-one-stage-retinex-based/low-light-image-enhancement-on-lol)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lol?p=retinexformer-one-stage-retinex-based)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/retinexformer-one-stage-retinex-based/photo-retouching-on-mit-adobe-5k)](https://paperswithcode.com/sota/photo-retouching-on-mit-adobe-5k?p=retinexformer-one-stage-retinex-based)
# Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement (ICCV 2023)


![pipeline](/figure/pipeline.png)

#### News
- **2023.07.14 :** Our paper has been accepted by ICCV 2023, code and models will released. :rocket:

#### Results
- Results on LOL-v1, LOL-v2-real, LOL-v2-synthetic, SID, SMID, SDSD-in, SDSD-out, and MIT Adobe FiveK datasets can be downloaded from [Baidu Disk](https://pan.baidu.com/s/1DC6A-I9S7yJ-pmMVTLAHaw?pwd=cyh2) (code: `cyh2`) or Google Drive
- Results on LIME, NPE, MEF, DICM, and VV datasets can be downloaded from [Baidu Disk](https://pan.baidu.com/s/1cqBwmuXk83h6u1NZJVbfkg?pwd=cyh2) (code: `cyh2`) of Google Drive

<details close>
<summary><b>Performance on LOL-v1, LOL-v2-real, LOL-v2-synthetic, SID, SMID, SDSD-in, and SDSD-out:</b></summary>

* ![results1](/figure/seven_results.png)

</details>


<details close>
<summary><b>Performance on MIT Adobe FiveK:</b></summary>

* ![results2](/figure/fivek_results.png)

</details>


<details close>
<summary><b>Performance on LIME, NPE, MEF, DICM, and VV:</b></summary>

* ![results1](/figure/visual_compare_no_gt.png)

</details>




## 1. Create Envirement

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

## 2. Prepare Dataset
Download the following datasets:
LOL-v1 [Baidu Disk (code: ssd0)](https://pan.baidu.com/share/init?surl=spt0kYU3OqsQSND-be4UaA)

LOL-v2 [Baidu Disk (code: cyh2)](https://pan.baidu.com/s/1X4HykuVL_1WyB3LWJJhBQg?pwd=cyh2), [Google Drive](https://drive.google.com/file/d/1Ou9EljYZW8o5dbDCf9R34FS8Pd8kEp2U/view?usp=sharing)

SID [Baidu Disk (code: gplv)](https://pan.baidu.com/share/init?surl=HRr-5LJO0V0CWqtoctQp9w), [Google Drive](https://drive.google.com/drive/folders/1eQ-5Z303sbASEvsgCBSDbhijzLTWQJtR?usp=share_link&pli=1)

SMID [Baidu Disk (code: btux)](https://pan.baidu.com/share/init?surl=Qol_4GsIjGDR8UT9IRZbBQ), [Google Drive](https://drive.google.com/drive/folders/1OV4XgVhipsRqjbp8SYr-4Rpk3mPwvdvG)

SDSD-indoor [Baidu Disk (code: jo1v)](https://pan.baidu.com/s/1rfRzshGNcL0MX5soRNuwTA?errmsg=Auth+Login+Params+Not+Corret&errno=2&ssnerror=0#list/path=%2F), [Google Drive](https://drive.google.com/drive/folders/14TF0f9YQwZEntry06M93AMd70WH00Mg6)

SDSD-outdoor [Baidu Disk (code: uibk)](https://pan.baidu.com/share/init?surl=JzDQnFov-u6aBPPgjSzSxQ), [Google Drive](https://drive.google.com/drive/folders/14TF0f9YQwZEntry06M93AMd70WH00Mg6)

MIT-Adobe FiveK [Baidu Disk (processed, code:cyh2)](https://pan.baidu.com/s/1ajax7N9JmttTwY84-8URxA?pwd=cyh2), [Google Drive](https://drive.google.com/file/d/11HEUmchFXyepI4v3dhjnDnmhW_DgwfRR/view?usp=sharing), [Official](https://data.csail.mit.edu/graphics/fivek/)

Please process the MIT Adobe FiveK dataset following [the sRGB setting](nothinglo/Deep-Photo-Enhancer#38)

Then orgnize these datasets as follows:

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

```

Besides, we also provide the download links for the five datasets LIME, NPE, MEF, DICM, and VV that have no ground truth:

[Baidu Disk](https://pan.baidu.com/s/1oHg03tOfWWLp4q1R6rlzww?pwd=cyh2) (code: `cyh2`)

[Google Drive]()

                    


## 3. Testing

Download our trained models from [Baidu Disk (code: cyh2)](https://pan.baidu.com/s/13zNqyKuxvLBiQunIxG_VhQ?pwd=cyh2) or [Google Drive](https://drive.google.com/drive/folders/1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV?usp=drive_link). Put them in the folder `pretrained_weights`

```shell
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
```

## 4. Training

Feel free to check our training logs from [Baidu Disk (code: cyh2)](https://pan.baidu.com/s/16NtLba_ANe3Vzji-eZ1xAA?pwd=cyh2) or [Google Drive](https://drive.google.com/drive/folders/1HU_wEn_95Hakxi_ze-pS6Htikmml5MTA?usp=sharing)

```shell
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




## 5. Citation

```shell
@inproceedings{retinexformer,
  title={Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement},
  author={Yuanhao Cai and Hao Bian and Jing Lin and Haoqian Wang and Radu Timofte and Yulun Zhang},
  booktitle={ICCV},
  year={2023}
}
```

**Acknowledgment:** Our code is based on the [BasicSR](https://github.com/xinntao/BasicSR) toolbox. 

