A GAN Exercise
============== 

This repo provides some sample code for training GAN networks. 

### Requirements
python >= 3.6  
pytorch  
torchvision  
pillow  
numpy  


### Generators and Discriminators
There are two types of generators and discriminators, the MLP and the deep convolutional (DC) networks in [./models/G_D.py](./models/G_D.py).
We use separated classes for discriminators of standard GAN and Wasserstein AN.  

We can define various combinations of standard GAN/Wasserstein GAN with MLP/DC generators or discriminations in [args.py](args.py).

### Dataset Preparation
We train the models with MNIST and LSUN-edroom dataset.  

MNIST images are small, with image size as 28\*28, one channel. Models can be quickly trained even with MLP for both generator and discriminator.  
We have included the MNIST dataset in the `dataset` folder  

We reshape the LSUN-bedroom images to 64\*64, three channels. In this case, training with MLP as both generator and discriminator cannot converge.  
To train LSUN from scratch, we refer to [fyu/lsun](https://github.com/fyu/lsun.git) to prepare. We have modified the [data.py](https://github.com/fyu/lsun/blob/master/data.py)
 to extract the 64\*64 `.jpg` images. The modified files is provided under the `dataset` folder. 
 
 1. cd ./dataset/lsun/
 2. python3 download.py -c bedroom  
 3. unzip the lsun_bedroom_train_lmdb.zip.  
    e.g.:  unzip lsun_bedroom_train_lmdb.zip
 4. python3 data.py export \<image db path\> --out_dir \<output directory\>  
    e.g.:  
    mkdir ./bedroom_train_64  
    python3 data.py export ./bedroom_train_lmdb/ --out_dir ./bedroom_train_64/
### Train models from scratch

Define all the args and run [train.py](./train.py). Sample args for standard GAN and Wasserstein GAN are in [args.py](./args.py)  
The model checkpoints and generated images will be saved under `./output/<model type>/<generator type>_<discriminator type>_<datasetname>/`

### Generate fake images use pre-trained models

Specify the saved generator state_dict path in `args` and run the `test.py` accordingly.  
The generated images are saved under `./output/<model type>/<generator type>_<discriminator type>_<datasetname>/test_imgs/`


### Comparison
We compare the generated images of different combination of generators and discriminators  
We trained 20000 iterations on MNIST and 200000 iterations on LSUN dataset.  
All the images are resized to 64\*64. The batch size is 64.  

| mnist  | gan | wgan|  
| -------|:-------------------------------------------------------------------------------:|-----------------------------------------------------------------------------: |
| dc_dc  | ![gan](./output/test_imgs/dc_dc_mnist/gan_test_generated_imgs_iter_20000.jpg)   |![wgan](./output/test_imgs/dc_dc_mnist/wgan_test_generated_imgs_iter_20000.jpg)|
| mlp_dc | ![gan](./output/test_imgs/mlp_dc_mnist/gan_test_generated_imgs_iter_20000.jpg)  |![wgan](./output/test_imgs/mlp_dc_mnist/wgan_test_generated_imgs_iter_20000.jpg)|
| dc_mlp | ![gan](./output/test_imgs/dc_mlp_mnist/gan_test_generated_imgs_iter_20000.jpg)  |![wgan](./output/test_imgs/dc_mlp_mnist/wgan_test_generated_imgs_iter_20000.jpg)|
| mlp_mlp| ![gan](./output/test_imgs/mlp_mlp_mnist/gan_test_generated_imgs_iter_20000.jpg) |![wgan](./output/test_imgs/mlp_mlp_mnist/wgan_test_generated_imgs_iter_20000.jpg)|


| lsun  | gan | wgan|  
| -------|:-------------------------------------------------------------------------------:|-----------------------------------------------------------------------------: |
| dc_dc  | ![gan](./output/test_imgs/dc_dc_lsun/gan_test_generated_imgs_iter_100000.jpg)   |![wgan](./output/test_imgs/dc_dc_lsun/wgan_test_generated_imgs_iter_200000.jpg)|
| mlp_dc | ![gan](./output/test_imgs/mlp_dc_lsun/gan_test_generated_imgs_iter_200000.jpg)  |![wgan](./output/test_imgs/mlp_dc_lsun/wgan_test_generated_imgs_iter_200000.jpg)|
| dc_mlp | ![gan](./output/test_imgs/dc_mlp_lsun/gan_test_generated_imgs_iter_200000.jpg)  |![wgan](./output/test_imgs/dc_mlp_lsun/wgan_test_generated_imgs_iter_200000.jpg)|
| mlp_mlp| ![gan](./output/test_imgs/mlp_mlp_lsun/gan_test_generated_imgs_iter_200000.jpg) |![wgan](./output/test_imgs/mlp_mlp_lsun/wgan_test_generated_imgs_iter_5000.jpg)|

Standard GAN_dc_dc will collapse after some iterations. For example, the standard GAN dc_dc_lsun model does not converge after 150000 iterations.  
The training of WGAN mlp_mlp_lsun failed, with a learning rate of 5e-5 or even a smaller one 5e-6.

### Acknowledgement
https://github.com/fyu/lsun  
https://github.com/eriklindernoren/PyTorch-GAN  
https://github.com/martinarjovsky/WassersteinGAN
