from args import gan_args, wgan_args
from models import gan, wgan
import torch
import numpy as np
import os
from torch.autograd import Variable
from torchvision.utils import save_image
from models.G_D import GeneratorDC, GeneratorMLP

def main(args):
    '''

    :param args: the args parser of training parameters
    :return:
    '''

    model_type = args.model_type
    print(model_type, args.generator_type, args.discriminator_type)
    if model_type == 'gan':
        gan.test( args)
    elif model_type == 'wgan':
        wgan.test(args)
    else:
        raise NotImplementedError("the model type is currently not available. please choose model type as gan/dcgan/wgan")


def test_all_models(args):
    '''
    This function test various combinations of generators and discriminators using the same noise
    Before you run this function, you should have prepared the all the models under ./output
    '''

    # generate a noise variable
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    z = Variable(Tensor(np.random.normal(0, 1, (25, args.latent_dim))))
    model_root_dir = './output/'
    if args.datasetname == 'mnist':
        models = {'dc_dc_gan': os.path.join(model_root_dir, 'gan', '0.0002_dc_dc_mnist', '20000_gan_generator_checkpoint.pt'),
                  'mlp_dc_gan': os.path.join(model_root_dir, 'gan','0.0002_mlp_dc_mnist', '20000_gan_generator_checkpoint.pt'),
                  'dc_mlp_gan': os.path.join(model_root_dir, 'gan','0.0002_dc_mlp_mnist', '20000_gan_generator_checkpoint.pt'),
                  'mlp_mlp_gan': os.path.join(model_root_dir,'gan', '0.0002_mlp_mlp_mnist', '20000_gan_generator_checkpoint.pt'),
                  'dc_dc_wgan': os.path.join(model_root_dir, 'wgan','5e-05_dc_dc_mnist', '20000_wgan_generator_checkpoint.pt'),
                  'mlp_dc_wgan': os.path.join(model_root_dir,'wgan', '5e-05_mlp_dc_mnist', '20000_wgan_generator_checkpoint.pt'),
                  'dc_mlp_wgan': os.path.join(model_root_dir, 'wgan','5e-05_dc_mlp_mnist', '20000_wgan_generator_checkpoint.pt'),
                  'mlp_mlp_wgan': os.path.join(model_root_dir,'wgan', '5e-05_mlp_mlp_mnist', '20000_wgan_generator_checkpoint.pt')
                  }
    elif args.datasetname == 'lsun':
        models = {'dc_dc_gan':os.path.join(model_root_dir, 'gan','0.0002_dc_dc_lsun', '100000_gan_generator_checkpoint.pt'),
                  'mlp_dc_gan':os.path.join(model_root_dir, 'gan','0.0002_mlp_dc_lsun', '200000_gan_generator_checkpoint.pt'),
                  'dc_mlp_gan':os.path.join(model_root_dir,'gan', '0.0002_dc_mlp_lsun', '200000_gan_generator_checkpoint.pt'),
                  'mlp_mlp_gan': os.path.join(model_root_dir,'gan', '0.0002_mlp_mlp_lsun', '200000_gan_generator_checkpoint.pt'),
                  'dc_dc_wgan':os.path.join(model_root_dir, 'wgan','5e-05_dc_dc_lsun', '200000_wgan_generator_checkpoint.pt'),
                  'mlp_dc_wgan':os.path.join(model_root_dir, 'wgan','5e-05_mlp_dc_lsun', '100000_wgan_generator_checkpoint.pt'),
                  'dc_mlp_wgan':os.path.join(model_root_dir,'wgan', '5e-05_dc_mlp_lsun', '200000_wgan_generator_checkpoint.pt'),
                  # 'mlp_mlp': os.path.join(model_root_dir, '5e-06_dc_mlp_lsun', '200000_gan_generator_checkpoint.pt'),
                  }
    else:
        raise NotImplementedError("Dataset does not exist, args.datasetname in [lsun, mnist]")

    for k in models:
        model_path = models[k]
        print(model_path)
        generator_type = k.split('_')[0]
        discriminator_type = k.split('_')[1]
        model_type = k.split('_')[-1]
        image_shape = (args.channels, args.img_size, args.img_size)
        if generator_type == 'mlp':
            generator = GeneratorMLP(args.latent_dim, image_shape)
        elif generator_type == 'dc':
            generator = GeneratorDC(args.latent_dim, image_shape)
        else:
            raise NotImplementedError("the generator_type should be mlp or dc")
        if torch.cuda.is_available():
            generator.cuda()
        if not os.path.isfile(model_path):
            raise NotImplementedError("generator file does not exist")
        num_iter = model_path.split('/')[-1].split('_')[0]

        print('loading state dict')
        state_dict = torch.load(model_path)
        # print(state_dict)
        generator.load_state_dict(state_dict)
        generator.eval()
        print("model loaded")
        gen_imgs = generator(z)
        output_dir = "./output/" + 'test_imgs/' + generator_type + '_' + discriminator_type + '_' + args.datasetname
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        save_image(gen_imgs.data, os.path.join(output_dir, f"{model_type}_test_generated_imgs_iter_{num_iter}.jpg"), nrow=5,
                   normalize=True)


if __name__ == '__main__':
    # parser = gan_args()
    parser = wgan_args()
    args = parser.parse_args()
    args.datasetname = 'lsun'
    args.channels = 3
    test_all_models(args)