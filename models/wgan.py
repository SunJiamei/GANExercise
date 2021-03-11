import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
from models.G_D import GeneratorDC, GeneratorMLP, DiscriminatorDCWasserstein, DiscriminatorMLPWasserstein, weights_init_normal

def train(dataloader, args, datasetname):
    # whether to use cuda
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # output dir to save generated images
    output_dir = "./output/wgan/" + args.generator_type+'_'+args.discriminator_type+'_'+datasetname
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
    else:
        print("Training from scratch")
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

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
            if isinstance(imgs,tuple):
                imgs = imgs[0]
            # Configure input
            real_imgs = Variable(imgs.type(Tensor))
            if gen_iterations < 25 or gen_iterations % 500 == 0:  # a warm up at the first 25 iterations train discriminator more
                Diters = 100
            else:
                Diters = args.n_critic
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))))

            # ---------------------
            #  Train Discriminator
            # ---------------------
            for p in discriminator.parameters(): # reset requires_grad
                p.requires_grad = True # they are set to False below in netG update
            # we train the discriminator more than the generator
            for j in range(Diters):
                optimizer_D.zero_grad()

                noise = z.normal_(0, 1)
                noisev = Variable(noise, volatile=True)  # totally freeze generator

                # Generate a batch of images
                fake_imgs = generator(noisev).detach()
                # Adversarial loss
                loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

                loss_D.backward()
                optimizer_D.step()

                # Clip weights of discriminator
                for p in discriminator.parameters():
                    p.data.clamp_(-args.clip_value, args.clip_value)

            # Train the generator every n_critic iterations
            # -----------------
            #  Train Generator
            # -----------------
            for p in discriminator.parameters(): # reset requires_grad
                p.requires_grad = False # they are set to False in generator update

            optimizer_G.zero_grad()
            noise = z.normal_(0, 1)
            noisev = Variable(noise)
            # Generate a batch of images
            gen_imgs = generator(noisev)
            # Adversarial loss
            loss_G = -torch.mean(discriminator(gen_imgs))

            loss_G.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (
                epoch, args.n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
            )

            # save checkpoint and generated images
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.sample_interval == 0:
                save_image(gen_imgs.data[:25],os.path.join(output_dir,f"{batches_done}_lossd_{loss_D}_loss_g_{loss_G}.jpg"), nrow=5, normalize=True)
            if batches_done % 10000 == 0:
                torch.save(generator.state_dict(),
                           os.path.join(output_dir, f"{batches_done}_wgan_generator_checkpoint.pt"))
                torch.save(discriminator.state_dict(),
                           os.path.join(output_dir, f"{batches_done}_wgan_discriminator_checkpoint.pt"))
            gen_iterations+=1

def test(args, datasetname):
    args.resume_generator_path = './output/wgan/dc_dc_lsun/10000_wgan_generator_checkpoint.pt'
    output_dir = "./output/wgan/" + args.generator_type+'_'+args.discriminator_type+'_'+datasetname + '/test_imgs/'
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