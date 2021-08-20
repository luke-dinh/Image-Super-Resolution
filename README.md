# Image-Super-Resolution

Image Super-Resollution using Deep Learning, implemented in PyTorch, with an API for easily usage.

## 1. Orriginal model

* SRGAN Architecture

<p align = "center">
  <img src = "https://user-images.githubusercontent.com/51883796/82537161-c4970f80-9b73-11ea-9a95-330f8525f8ba.PNG">
</p>


## 2. Proposed method:

The whole framework: A GAN-based model (mSRGAN) is used to train the model. After training, the Generator block (G block) is used for API to create new higher resolution image from the lower one. 

We modified the G block architecture from the previous model for better training and lowwer weights. The architecture of the block is shown as below (for training and usage): 

<p align = "center">
  <img src = "https://user-images.githubusercontent.com/51883796/130203219-1528644a-2397-405b-a3a6-82844fc01e41.png">
</p>

