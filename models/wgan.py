import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
from models.G_D import GeneratorDC, GeneratorMLP, DiscriminatorDCWasserstein, DiscriminatorMLPWasserstein, weights_init_normal

def train(dataloader, args):
    # whether to use cuda
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    one = Tensor([1])
    mone = one * -1
    # output dir to save generated images
    output_dir = "./output/wgan/" + str(args.lr)+'_'+args.generator_type+'_'+args.discriminator_type+'_'+args.datasetname
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # image shape
    image_shape = (args.channels, args.img_size, args.img_size)

    # model
    if args.generator_type == 'mlp':
        generator = GeneratorMLP(args.latent_dim, image_shape)
    elif args.generator_type == 'dc':
        generator = GeneratorDC(args.latent_dim, image_shape)
    else:
        raise NotImplementedError("the generator_type should be mlp or dc")
    if args.discriminator_type == 'mlp':
        discriminator = DiscriminatorMLPWasserstein(image_shape)
    elif args.discriminator_type == 'dc':
        discriminator = DiscriminatorDCWasserstein(image_shape)
    else:
        raise NotImplementedError("the generator_type should be mlp or dc")


    # we can resume the previous training
    if args.resume_train:
        print(f"Training from checkpoint {args.resume_generator} and {args.resume_discriminator}")
        if os.path.isfile(args.resume_generator):
            generator_state = torch.load(args.resume_generator)
            generator.load_state_dict(generator_state)
        else:
            raise NotImplementedError("Generator checkpoint does not exist")
        if os.path.isfile(args.resume_discriminator):
            discriminator_state = torch.load(args.resume_discriminator)
            discriminator.load_state_dict(discriminator_state)
        else:
            raise NotImplementedError("Discriminator checkpoint does not exist")
        finished_batches = int(args.ressume_generator.split('/')[-1].split('_')[0])  # get the batches number from the saved model

    else:
        print("Training from scratch")
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)
        finished_batches = 0


    if cuda:
        generator.cuda()
        discriminator.cuda()
    # optimizer, note that wgan use RMSprop
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=args.lr)

    #start training
    batches_done = 0
    gen_iterations = 0
    for epoch in range(args.n_epochs):
        for i, imgs in enumerate(dataloader):
            if isinstance(imgs,list):
                imgs = imgs[0]
            # Configure input
            real_imgs = Variable(imgs.type(Tensor))
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))))

            # ---------------------
            #  Train Discriminator
            # ---------------------
            for p in discriminator.parameters(): # reset requires_grad
                p.requires_grad = True # they are set to False below in generator update
            for p in generator.parameters(): # reset requires_grad
                p.requires_grad = False # they are set to True below in generator update
            # we train the discriminator more than the generator
            for j in range(args.n_critic):
                # Clip weights of discriminator
                for p in discriminator.parameters():
                    p.data.clamp_(-args.clip_value, args.clip_value)
                # optimizer_D.zero_grad()
                discriminator.zero_grad()

                # real loss
                errD_real = discriminator(real_imgs)
                errD_real.backward(one)

                # Generate a batch of images
                fake_imgs = Variable(generator(z).data)

                errD_fake = discriminator(fake_imgs)
                errD_fake.backward(mone)

                errD = errD_real - errD_fake
                optimizer_D.step()

            # Train the generator every n_critic iterations
            # -----------------
            #  Train Generator
            # -----------------
            for p in discriminator.parameters(): # reset requires_grad
                p.requires_grad = False # they are set to False in generator update
            for p in generator.parameters(): # reset requires_grad
                p.requires_grad = True # they are set to True in generator update
            generator.zero_grad()
            # Generate a batch of images
            gen_imgs = generator(z)
            # Adversarial loss
            errG = discriminator(gen_imgs)
            errG.backward(one)
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (
                epoch, args.n_epochs, i, len(dataloader), errD.item(), errG.item())
            )

            # save checkpoint and generated images
            batches_done = epoch * len(dataloader) + i + finished_batches
            if batches_done % 5000 == 0:
                save_image(gen_imgs.data[:25],os.path.join(output_dir,f"{batches_done}_lossd_{errD.item()}_loss_g_{errG.item()}.jpg"), nrow=5, normalize=True)
            if batches_done % 50000 == 0:
                torch.save(generator.state_dict(),
                           os.path.join(output_dir, f"{batches_done}_wgan_generator_checkpoint.pt"))
                torch.save(discriminator.state_dict(),
                           os.path.join(output_dir, f"{batches_done}_wgan_discriminator_checkpoint.pt"))
            gen_iterations+=1

def test(args):
    args.resume_generator_path = './output/wgan/dc_dc_lsun/10000_wgan_generator_checkpoint.pt'
    output_dir = "./output/wgan/" + args.generator_type+'_'+args.discriminator_type+'_'+args.datasetname + '/test_imgs/'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    image_shape = (args.channels, args.img_size, args.img_size)
    if args.generator_type == 'mlp':
        generator = GeneratorMLP(args.latent_dim, image_shape)
    elif args.generator_type == 'dc':
        generator = GeneratorDC(args.latent_dim, image_shape)
    else:
        raise NotImplementedError("the generator_type should be mlp or dc")
    if torch.cuda.is_available():
        generator.cuda()
    if not os.path.isfile(args.resume_generator):
        raise NotImplementedError("generator file does not exist")
    model_path = args.resume_generator_path
    num_iter = model_path.split('/')[-1].split('_')[0]
    generator.load_state_dict(torch.load(model_path))
    generator.eval()

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    z = Variable(Tensor(np.random.normal(0, 1, (25, args.latent_dim))))
    gen_imgs = generator(z)
    save_image(gen_imgs.data, os.path.join(output_dir, f"test_generated_imgs_iter_{num_iter}.jpg"), nrow=5, normalize=True)


def test_all_models(args):
    '''
    This function test various combinations of generators and discriminators using the same noise
    Before you run this function, you should have prepared the all the models under ./output
    '''

    # generate a noise variable
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    z = Variable(Tensor(np.random.normal(0, 1, (25, args.latent_dim))))
    model_root_dir = '../output/wgan/'
    if args.datasetname == 'mnist':
        models = {'dc_dc': os.path.join(model_root_dir, '5e-05_dc_dc_mnist', '20000_wgan_generator_checkpoint.pt'),
                  'mlp_dc': os.path.join(model_root_dir, '5e-05_mlp_dc_mnist', '20000_wgan_generator_checkpoint.pt'),
                  'dc_mlp': os.path.join(model_root_dir, '5e-05_dc_mlp_mnist', '20000_wgan_generator_checkpoint.pt'),
                  'mlp_mlp': os.path.join(model_root_dir, '5e-05_mlp_mlp_mnist', '20000_wgan_generator_checkpoint.pt')
                  }
    elif args.datasetname == 'lsun':
        models = {'dc_dc':os.path.join(model_root_dir, '5e-05_dc_dc_lsun', '200000_wgan_generator_checkpoint.pt'),
                  'mlp_dc':os.path.join(model_root_dir, '5e-05_mlp_dc_lsun', '100000_wgan_generator_checkpoint.pt'),
                  'dc_mlp':os.path.join(model_root_dir, '5e-05_dc_mlp_lsun', '200000_wgan_generator_checkpoint.pt'),
                  # 'mlp_mlp': os.path.join(model_root_dir, '5e-06_dc_mlp_lsun', '200000_gan_generator_checkpoint.pt'),
                  }
    else:
        raise NotImplementedError("Dataset does not exist, args.datasetname in [lsun, mnist]")

    for k in models:
        model_path = models[k]
        print(model_path)
        generator_type = k.split('_')[0]
        discriminator_type = k.split('_')[1]

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
        output_dir = "../output/wgan/" + 'test_imgs/' + generator_type + '_' + discriminator_type + '_' + args.datasetname
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        save_image(gen_imgs.data, os.path.join(output_dir, f"test_generated_imgs_iter_{num_iter}.jpg"), nrow=5,
                   normalize=True)


if __name__ == '__main__':
    from args import wgan_args
    parser = wgan_args()
    args = parser.parse_args()
    args.datasetname = 'mnist'
    args.channels = 1
    test_all_models(args)
