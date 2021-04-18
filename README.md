##  A Two-branch Neural Network for Non-homogeneous Dehazing via Ensemble Learning
### Dependencies and Installation

* python3.7
* PyTorch >= 1.0
* NVIDIA GPU+CUDA
* numpy
* matplotlib
* tensorboardX(optional)

### Pretrained Weights & Dataset

- Download [ImageNet pretrained weights](https://drive.google.com/file/d/1aZQyF16pziCxKlo7BvHHkrMwb8-RurO_/view?usp=sharing) and [our model weights](https://drive.google.com/file/d/1M2n6g7S5_sqPmTIAuI-IC30fhUmQr199/view?usp=sharing).

- Download the [NH-Haze 2020](https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/) and [NH-Haze 2021](https://drive.google.com/drive/folders/1jBoP1d8eSCHcPgxcWQ42RKIA2Fxo_Thw?usp=sharing) dataset.

  


#### Test

 ```shell
python test.py --data_dir data_21 --model_save_dir results
 ```



## Qualitative Results

Results on NTIRE 2021 NonHomogeneous Dehazing Challenge testing images:

<div style="text-align: center">
<img alt="" src="/Image/test_results.png" style="display: inline-block;" />
</div>

## Citation

If you use any part of this code, please kindly cite

```
@article{
}
```



