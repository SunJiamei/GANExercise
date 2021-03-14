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
    :param datasetname: either mnist or lsun
    :return:
    '''
    if args.datasetname == 'lsun':
        dataset = Lsundaset(args.data_file_folder)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    elif args.datasetname == 'mnist':
        dataloader = torch.utils.data.DataLoader(datasets.MNIST('../dataset/', train=True, download=True,
                                                                         transform=transforms.Compose([
                                                                             transforms.Resize(
                                                                                 (args.img_size, args.img_size)),
                                                                           transforms.ToTensor(),
                                                                           transforms.Normalize(
                                                                             (0.5,), (0.5,))
                                                                         ])),
                                                batch_size=args.batch_size, shuffle=True)
    else:
        raise NotImplementedError("the dataset is not available for now, please add the dataset loader and adjust the training hyperparameters accordingly")
    model_type = args.model_type
    print(model_type, args.generator_type, args.discriminator_type)
    if model_type == 'gan':
        gan.train(dataloader, args)
    elif model_type == 'wgan':
        wgan.train(dataloader, args)
    else:
        raise NotImplementedError("the model type is currently not available. please choose model type as gan/dcgan/wgan")
if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    parser = gan_args()
    args = parser.parse_args()
    main(args)