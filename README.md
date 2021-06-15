##  A Two-branch Neural Network for Non-homogeneous Dehazing via Ensemble Learning

This is the official PyTorch implementation of Two-branch Dehazing.  See more details in  [[report]](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Ancuti_NTIRE_2021_NonHomogeneous_Dehazing_Challenge_Report_CVPRW_2021_paper.pdf) , [[paper]](https://arxiv.org/pdf/2104.08902.pdf), [[certificates]]( )

### Dependencies and Installation

* python3.7
* PyTorch >= 1.0
* NVIDIA GPU+CUDA
* numpy
* matplotlib
* tensorboardX(optional)

### Pretrained Weights & Dataset

- Download [ImageNet pretrained weights](https://drive.google.com/file/d/1aZQyF16pziCxKlo7BvHHkrMwb8-RurO_/view?usp=sharing) and [our model weights](https://drive.google.com/file/d/1M2n6g7S5_sqPmTIAuI-IC30fhUmQr199/view?usp=sharing).
- Download our [dataset](https://drive.google.com/drive/folders/1eeBA2V_l9-evSJ0XWhRAww6ftweq8hU_?usp=sharing)


  
#### Train
```shell
python train.py --data_dir data -train_batch_size 8 --model_save_dir train_result
```

#### Test
 ```shell
python test.py --model_save_dir results
 ```



## Qualitative Results

Results on NTIRE 2021 NonHomogeneous Dehazing Challenge testing images:

<div style="text-align: center">
<img alt="" src="/images/test_results.png" style="display: inline-block;" />
</div>

## Citation

If you use any part of this code, please kindly cite

```
@article{yu2021two,
  title={A Two-branch Neural Network for Non-homogeneous Dehazing via Ensemble Learning},
  author={Yu, Yankun and Liu, Huan and Fu, Minghan and Chen, Jun and Wang, Xiyao and Wang, Keyan},
  journal={arXiv preprint arXiv:2104.08902},
  year={2021}
}
```



