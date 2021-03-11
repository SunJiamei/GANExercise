from args import gan_args, wgan_args
from models import gan, wgan
import torch
from preparedataset import Lsundaset
import os
from torchvision import transforms
from torchvision import datasets

def main(args):
    '''

    :param args: the args parser of training parameters
    :return:
    '''

    model_type = args.model_type
    print(model_type, args.generator_type, args.discriminator_type)
    if model_type == 'gan':
        gan.test( args, args.datasetname)
    elif model_type == 'wgan':
        wgan.test(args,args.datasetname)
    else:
        raise NotImplementedError("the model type is currently not available. please choose model type as gan/dcgan/wgan")
if __name__ == '__main__':
    # parser = gan_args()
    parser = wgan_args()
    args = parser.parse_args()
    main(args)