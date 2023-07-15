#### This is the demo code of our paper "Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement" in submission to ICCV 2023.


This repo can reproduce the main results in Tab. (1) of our main paper.
All the source code and pre-trained models will be released to the public for further research.

#### Installation

1. Make conda environment
```
conda create -n Retinexformer python=3.7
conda activate Retinexformer
```

2. Install dependencies
```
conda install pytorch=1.11 torchvision cudatoolkit=10.2 -c pytorch
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm
pip install einops gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips
```

3. Install basicsr
```
python setup.py develop --no_cuda_ext
```

#### Prepare the dataset
Download LOL-v1, LOL-v2, SDSD, SMID, SID datasets and orgnize them as follows:

```
    |--data   
        |--LOLv1  
	        |--Test
                |--input
                    |--111.png
                    ...
                |--target
                    |--111.png
                    ...
        |--LOLv2
            |--Real_captured
                |--Test
                    |--Low
                        |--00690.png
                        ...
                    |--Normal
                        |--00690.png
                        ...
            |--Synthetic
                |--Test
                    |--Low
                        |--00690.png
                        ...
                    |--Normal
                        |--00690.png
                        ...
        |--SDSD
            |--indoor_static_np
                |--input
                    |--pair1
                    |--pair2
                    ...
                |--output
                    |--pair1
                    |--pair2
                    ...
            |--outdoor_static_np
                |--input
                    |--MVI_0898
                    |--MVI_0918
                    ...
                |--output
                    |--MVI_0898
                    |--MVI_0918
                    ...
        |--sid
            |--short_sid2
                |--00001
                |--00002
                ...
            |--long_sid2
                |--00001
                |--00002
                ...
        |--smid
            |--SMID_LQ_np
                |--0001
                |--0002
                ...
            |--SMID_Long_np
                |--0001
                |--0002
                ...
```

                    


#### Evaluate the models
```
# evaluate on the LOL_v1 dataset
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v1.yml --weights pretrained_weights/LOL_v1.pth --dataset LOL_v1

# evaluate on the LOL_v2_real dataset
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v2_real.yml --weights pretrained_weights/LOL_v2_real.pth --dataset LOL_v2_real

# evaluate on the LOL_v2_synthetic dataset
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v2_synthetic.yml --weights pretrained_weights/LOL_v2_synthetic.pth --dataset LOL_v2_synthetic

# evaluate on the SID dataset
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_SID.yml --weights pretrained_weights/SID.pth --dataset SID

# evaluate on the SMID dataset
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_SMID.yml --weights pretrained_weights/SMID.pth --dataset SMID

# evaluate on the SDSD_indoor dataset
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_SDSD_indoor.yml --weights pretrained_weights/SDSD_indoor.pth --dataset SDSD_indoor

# evaluate on the SDSD_outdoor dataset
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_SDSD_outdoor.yml --weights pretrained_weights/SDSD_outdoor.pth --dataset SDSD_outdoor
```



**Acknowledgment:** Our code is based on the [BasicSR](https://github.com/xinntao/BasicSR) toolbox. 

