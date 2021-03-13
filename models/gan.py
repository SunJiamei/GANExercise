import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

from models.G_D import GeneratorDC, GeneratorMLP, DiscriminatorDCstandard, DiscriminatorMLPstandard, weights_init_normal


def train(dataloader, args, datasetname):
    output_dir = "./output/gan/" + str(args.lr)+'_'+args.generator_type+'_'+args.discriminator_type+'_'+datasetname
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    # Loss function
    adversarial_loss = torch.nn.BCELoss()
    # data_params
    image_shape = (args.channels, args.img_size, args.img_size)


    # model
    if args.generator_type == 'mlp':
        generator = GeneratorMLP(args.latent_dim, image_shape)
    elif args.generator_type == 'dc':
        generator = GeneratorDC(args.latent_dim, image_shape)
    else:
        raise NotImplementedError("the generator_type should be mlp or dc")
    if args.discriminator_type == 'mlp':
        discriminator = DiscriminatorMLPstandard(image_shape)
    elif args.discriminator_type == 'dc':
        discriminator = DiscriminatorDCstandard(image_shape)
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
    else:
        print("Training from scratch")
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)


    # optimizer
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))


    # whether to use cuda
    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


    # start train
    batches_done = 0
    generator.train()
    discriminator.train()
    for epoch in range(args.n_epochs):
        for i, imgs in enumerate(dataloader):
            if isinstance(imgs,list):
                imgs = imgs[0]
            # Adversarial ground truths
            valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))


            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))))

            # Generate a batch of images
            fake_images = generator(z)
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(fake_images.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()
            # Loss measures generator's ability to fool the discriminator
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()


            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, args.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.sample_interval == 0:
                save_image(gen_imgs.data[:25], os.path.join(output_dir,f"{batches_done}_lossd_{d_loss}_loss_g_{g_loss}.jpg"), nrow=5, normalize=True)
            if batches_done % args.sample_interval == 0:
                torch.save(generator.state_dict(),
                           os.path.join(output_dir, f"{batches_done}_gan_generator_checkpoint.pt"))
                torch.save(discriminator.state_dict(),
                           os.path.join(output_dir, f"{batches_done}_gan_discriminator_checkpoint.pt"))



def test(args, datasetname):
    # args.resume_generator = './output/gan/dc_dc_lsun/10000_gan_generator_checkpoint.pt'
    output_dir = "./output/gan/" + args.generator_type+'_'+args.discriminator_type+'_'+datasetname + '/test_imgs/'
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
    model_path = args.resume_generator
    num_iter = model_path.split('/')[-1].split('_')[0]
    print(num_iter)
    print('loading state dict')
    state_dict = torch.load(model_path)
    # print(state_dict)
    generator.load_state_dict(state_dict)
    generator.eval()
    print("model loaded")
    print("test model")
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    z = Variable(Tensor(np.random.normal(0, 1, (25, args.latent_dim))))
    gen_imgs = generator(z)
    save_image(gen_imgs.data, os.path.join(output_dir, f"test_generated_imgs_iter_{num_iter}.jpg"), nrow=5, normalize=True)
