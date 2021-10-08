# Cross-MPI: Cross-scale Stereo for Image Super-Resolution using Multiplane Images

This is the released code for the paper:

Cross-MPI: Cross-scale Stereo for Image Super-Resolution using Multiplane Images\
Yuemei Zhou, Gaochang Wu, Ying Fu, Kun Li, Yebin Liu\
CVPR 2021 

[Project Page](http://www.liuyebin.com/crossMPI/crossMPI.html)

**Please note that**, this code is built based on [Stereo Magnification](https://github.com/google/stereo-magnification) (SIGGRAPH' 18)


## Training the Cross-MPI model

The main training script is the `train.py`. We use the training dataloader from 
Stereo Magnification, so the input flags are in these two places: 1) `train.py` and 2) `crossmpi/loader.py`.

The input flag `which_model_predict` specifies different module combinations for 
realizing Cross-MPI, and our final model is `guided_nnup_denseASPP_lowres_image_psv_lowresL1`
 which is composed of ResASPP feature extraction, plane-aware attention, 
multi-scale guided upsampling and internal supervision loss. (Please see our paper 
for more details.)

Note that when training the VGG loss, you will need to download the pre-trained VGG model
[`imagenet-vgg-verydeep-19.mat`](http://www.vlfeat.org/matconvnet/pretrained/#downloading-the-pre-trained-models) and 
place it in `./models/`. The path to this file can be set by the `vgg_model_file` flag
in `train.py`.

We train the proposed network in three steps. Specifically, we first pretrain initial
alpha maps with internal supervision loss, second the multiscale guided upsampling module 
is added to warp the reference image, and finally, train the whole network together
with all losses. As examples in `./train_scripts/` where these three train steps are 
`train_lowres.sh`, `train_lowres_pretrained.sh` and `train_pretrained.sh` separately.

## Testing the Cross-MPI model

The main testing script for testing the models is `test.py`, and the script for testing multi-frame 
optical zoom cross-scale data pairs with fixed two camera views is `test_giga.py`.

Example command lines for `test.py` are in `./scripts/test_testset.sh` and for `test_giga.py` is in 
`./scripts/real_stereo_pairs_test/giga_four_people.sh`


## Quantitative evaluation

`evaluate.py` contains sample code for evaluating the super-resolution performance
based on the SSIM and PSNR metrics. 

Example command lines for `evaluate.py` are in `./scripts/evaluate.sh`.

## Test on a cross-scale stereo image pair

To test a pair of cross-scale stereo images, please first get the calibration results 
(intrinsics and extrinsics of these two views) as long as the depth ranges, then use the 
`crossmpi_from_images.py`.

Example command lines for `evaluate.py` are in `./scripts/real_stereo_pairs_test/giga_setup.sh`.


## Our pre-trained models

Please download our [pre-trained cross-mpi model](https://drive.google.com/file/d/16Gobd1moYZIAeyQm9HxFhJ6C3EllPebK/view?usp=sharing).


## Datasets

### RealEstate10K dataset
We train our cross-mpi model mainly on [RealEstate10K dataset](https://google.github.io/realestate10k/)

### Our optical zoom-in dataset
Please download our [optical zoom-in dataset](https://drive.google.com/file/d/1KD2rXVq8f5TRsC2jC9u4SxWZWxk1vOH5/view?usp=sharing), 
with data structure as below:
```
[Data Folder]
+[train]
    +[cameras]
        -sequence_0.txt  (????????????????.txt; contains image folder name + {fx fy cx cy k1 k2 3x4{RT}}; consistent with RealEstate10K)
        -sequence_1.txt
        ...
        -sequence_i.txt
    +[images]
        +[sequence_0_folder]
            -000_global.png  (LR_upsampled image of view 000)
            -000_local.png   (HR image of view 000)
            ...
            -00i_global.png
            -00i_local.png
        +[sequence_1_folder]
        ...
        +[sequence_i_folder]
+[test]
    +[cameras]
    +[images]

```




If you find our work **Cross-MPI** useful, please cite:
```
@inproceedings{zhou2021cross,
  title={Cross-MPI: Cross-scale Stereo for Image Super-Resolution using Multiplane Images},
  author={Zhou, Yuemei and Wu, Gaochang and Fu, Ying and Li, Kun and Liu, Yebin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14842--14851},
  year={2021}
}
```
