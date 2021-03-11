import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class GeneratorMLP(nn.Module):
    '''
    Input: a noise vector with shape (batch_size, tent_dim)
    Model:The MLP generator withfour blocks consisting of one linear layer and one leakyReLU layer.
    Output:The output is a flattened vector with the same size of channel*img_size*img_size
    '''
    def __init__(self, latent_dim, img_shape):
        super(GeneratorMLP, self).__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape

        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, 128),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),  # for lsun dataset, the output vector is 3*64*64, for mnist, the size is 1* 28*28
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        # for lsun dataset, the output vector is 3*64*64, for mnist, the size is 1* 28*28
        # we then reshape the output to the image_shape, which is 3,64,64 for lsun and 1,28,28 for mnist, to construct the image
        img = img.view(img.size(0), *self.img_shape)
        return img


class GeneratorDC(nn.Module):
    '''
    The generator with transpose2d convolutional layers.
    Input: a noise vector with shape (latent_dim, 1,1).
    Model: The model consists of four blocks, each with a ConvTranspose2d->BatchNorm->ReLU layers.
    Output: An constructed image, for lsun, image shape is (3,64,64), for mnist, it is (1,28,28)

    '''
    def __init__(self, latent_dim, img_shape):
        super(GeneratorDC, self).__init__()
        self.nz = latent_dim # the dim of the noise for image generation100
        self.ngf = 64 #  the hidden_dim of the generator default is 64
        self.nc = img_shape[0]  # the number of channels of image

        '''
        nn.ConvTranspose2d: Applies a 2D transposed convolution operator over an input image.
        #nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, \
         padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
        #bias=True: adds a learnable bias to the output.
        '''
        self.main = nn.Sequential(
            # 100->512, stride=1
            nn.ConvTranspose2d(self.nz, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(inplace=True),
            # 512->256, stride=2, padding=1
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(inplace=True),
            # 256->128, stride=2, padding=1
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(inplace=True),
            # 128->64, stride=2, padding=1
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(inplace=True),
            ##If you need extra layers, uncomment this block and copy it the number you want.
            ##128->64, kernel_size=1, stride=1, padding=1
            # nn.ConvTranspose2d(64, 64, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        # we need the noise with shape (batch_size, latent_dim, 1, 1),
        # one can consider the input as an image with only one pixel and 100 channels
        if z.dim() == 2:
            z = z.unsqueeze(-1).unsqueeze(-1)
        out = self.main(z)   # (batch_size, channel, img_size, img_size)
        return out


class DiscriminatorMLPstandard(nn.Module):
    '''
    DiscriminatorMLP using standard GAN
    Input: an image
    Model: three linear layers with LeakyReLU, note that the output score is normalized with Sigmoid function
    Output: a tensor with shape (batch_size, 1)
    '''
    def __init__(self, img_shape):
        super(DiscriminatorMLPstandard, self).__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


class DiscriminatorMLPWasserstein(nn.Module):
    '''
    DiscriminatorMLP using Wasserstein 
    The difference of this discriminator from the standard one is the ablation of the last sigmoid activation. 
    The loss function is also different, refer the train() function in wgan.py
    '''
    def __init__(self, img_shape):
        super(DiscriminatorMLPWasserstein, self).__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            # nn.Sigmoid()  Wasserstein removes the sigmoid
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


class DiscriminatorDCstandard(nn.Module):
    '''
    DiscriminatorDC using standard GAN
    Input: an image
    Model: a classification model with several blocks of Conv2d->BatchNorm->LeakyReLU, Note that the output is normalized
            with Sigmoid activation
    Output: a tensor with shape (batch_size, 1) indicating real or fake
    '''
    def __init__(self, img_shape):
        super(DiscriminatorDCstandard, self).__init__()
        self.nc = img_shape[0]
        self.ndf = 64
        '''
        nn.Conv2d: Applies a 2D convolution operator over an input image.
        #nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
         bias=True, padding_mode='zeros')
        '''
        self.main = nn.Sequential(
            # 3->64, stride=2, padding=1
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            ##If you need extra layers, uncomment this block and copy it the number you want.
            ##64->64, kernel_size=1, stride=1, padding=1
            # nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(64),
            # nn.LeakyReLU(0.2, inplace=True),
            # 64->128, stride=2, padding=1
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 128->256, stride=2, padding=1
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 256->512, stride=2, padding=1
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 512->1, stride=1, padding=0
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )


    def forward(self, img):
        output = self.main(img)

        output = output.squeeze(-1).squeeze(-1)  # (batch_size, 1)  the two-class predicted score
        # print('discriminator', output.size())
        # print(output)
        return output


class DiscriminatorDCWasserstein(nn.Module):
    '''
    DiscriminatorDC using standard GAN
    Input: an image
    Model: a classification model with several blocks of Conv2d->BatchNorm->LeakyReLU, Note that there is no Sigmoid layer
    Output: a tensor with shape (batch_size, 1) indicating real or fake
    '''

    def __init__(self, img_shape):
        super(DiscriminatorDCWasserstein, self).__init__()
        self.nc = img_shape[0]
        self.ndf = 64
        '''
        nn.Conv2d: Applies a 2D convolution operator over an input image.
        #nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,\
         bias=True, padding_mode='zeros')
        '''
        self.main = nn.Sequential(
            # 3->64, stride=2, padding=1
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            ##If you need extra layers, uncomment this block and copy it the number you want.
            ##64->64, kernel_size=1, stride=1, padding=1
            # nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(64),
            # nn.LeakyReLU(0.2, inplace=True),
            # 64->128, stride=2, padding=1
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 128->256, stride=2, padding=1
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 256->512, stride=2, padding=1
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 512->1, stride=1, padding=0
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False)
        )


    def forward(self, img):
        output = self.main(img)

        output = output.squeeze()  # (batch_size,)  the two-class predicted score 1 for real 0 for fake
        # print('discriminator', output.size())
        return output
