##  A Two-branch Neural Network for Non-homogeneous Dehazing via Ensemble Learning

This is the official PyTorch implementation of Two-branch Dehazing.  See more details in  [[report]](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Ancuti_NTIRE_2021_NonHomogeneous_Dehazing_Challenge_Report_CVPRW_2021_paper.pdf) , [[paper]](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Yu_A_Two-Branch_Neural_Network_for_Non-Homogeneous_Dehazing_via_Ensemble_Learning_CVPRW_2021_paper.pdf), [[certificates]](https://data.vision.ee.ethz.ch/cvl/ntire21/NTIRE2021awards_certificates.pdf)

Our method wins runner-up award in NTIRE 2021 Non-homogeneous Dehazing Challenge.
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

Training set is the one that we used during the competition.  Please be noted that we conduct gamma correction on NH-HAZE-2020. Test set is composed of random samples from the training set. The test set is used to provide effective training accuracy. If you want to obtain val and test accuracy on NH-HAZE-2021, please step towards the official [competition server.](https://competitions.codalab.org/competitions/28032#learn_the_details-overview)

  
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
@InProceedings{Yu_2021_CVPR,
    author    = {Yu, Yankun and Liu, Huan and Fu, Minghan and Chen, Jun and Wang, Xiyao and Wang, Keyan},
    title     = {A Two-Branch Neural Network for Non-Homogeneous Dehazing via Ensemble Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2021},
    pages     = {193-202}
}
```



