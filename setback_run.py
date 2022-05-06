import sys
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

import torch.nn.functional as F

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.autograd as autograd
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from torchvision.utils import save_image

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

class Classifier(nn.Module):
    def __init__(self, n_classes):
        super(Classifier, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same'),
        )
        self.fully_connected = nn.Sequential(
            nn.Linear(128*8*8, 128*8), nn.ReLU(),
            nn.Linear(128*8, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_classes),
        )
        
        self.classify =  nn.Softmax(dim=1)
        
        
    def forward(self, img):
        out = self.conv_blocks(img)
        out = torch.flatten(out, start_dim=1)
        out = self.fully_connected(out)
        out = self.classify(out)
        return out

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
            print("but I don't want to take up memory on my GPU....")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    def get_pred(x):
        if resize:
            x = up(x)
        x = model1(x)
        return x.data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 8))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        
        # First, transform the input into a 8x8 256-channels feature map
        self.init_size = width // 16  # one 16th the image size 
        self._latent_dim = latent_dim
        self.l1 = nn.Sequential(
            nn.Linear(self._latent_dim, 64), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256 * self.init_size ** 2), nn.LeakyReLU(0.2, inplace=True), # 4X4
        )
        
        self.conv_blocks = nn.Sequential(
            #input 256 X init_size X init_size
            
            ############# 4x4 #########################################
            
            nn.BatchNorm2d(256), 
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), #8X8
            nn.LeakyReLU(0.2, inplace=True),
            
            ############# 8X8 #########################################
            
            nn.BatchNorm2d(128), 
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), #16X16
            nn.LeakyReLU(0.2, inplace=True),
            
            ############# 16X16 #########################################
            
            nn.BatchNorm2d(64), 
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), #32X32
            nn.LeakyReLU(0.2, inplace=True),
            
            ############# 32X32 #########################################
            
            nn.BatchNorm2d(32), 
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), #64X64
            nn.LeakyReLU(0.2, inplace=True),
            
            ############# 64X64 #########################################
            
            # Produce a 32x32xRGB-channel feature map
            nn.Conv2d(16, channels, kernel_size=3, padding='same'),
            nn.Tanh(),
        )
        
    def forward(self, z):
        
        # expand the sampled z to 8x8
        out = self.l1(z)
        
        out = torch.reshape(out, (out.shape[0], 256, self.init_size, self.init_size))
        
        # use the view function to reshape the layer output
        #  old way for earlier Torch versions: out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):

    def __init__(self, latent_dim):
        super(Discriminator, self).__init__()
        
        self.init_size = width // 16 # one 16th the image size 
        self._latent_dim = latent_dim

        self.model = nn.Sequential(
            
            ############# 64X64 #########################################
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding='same'), #64X64
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(16, 0.8),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1), #32X32
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(32, 0.8),
            
            ############# 32X32 #########################################
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1), #16X16
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64, 0.8),
            
            ############# 16X16 #########################################
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1), #8x8
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128, 0.8),
            
            ############# 8X8 #########################################
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1), #4X4
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(256, 0.8),
            
            ############# 4x4 #########################################
            
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(256, 0.8),
        )
        
        # laten_dim_layer layer
        self.laten_dim_layer = nn.Sequential(nn.Linear(256 * self.init_size ** 2, 128), nn.LeakyReLU(0.2, inplace=True),
                                             nn.Linear(128, 64), nn.LeakyReLU(0.2, inplace=True),
                                             nn.Linear(64, self._latent_dim), nn.LeakyReLU(0.2, inplace=True),
                                            )
        # Classification layer
        self.classification_layer = nn.Sequential(nn.Linear(self._latent_dim, 1), 
                                                  nn.Sigmoid())
    def forward(self, img):
        
        out = self.model(img)
        # use the view function to flatten the layer output
        #    old way for earlier Torch versions: out = out.view(out.shape[0], -1)
        out = torch.flatten(out, start_dim=1) # don't flatten over batch size
        
        # move to the latent dim
        out = self.laten_dim_layer(out)
        
        #validate the latent dim
        validity = self.classification_layer(out)
        return validity

def train_setback(generator, gan_optimizer,
          discriminator, discriminator_optimizer,
          iterations, dataloader, 
          setback_frequency=None,
          setback_percentage=None,
          begin_step=None):
    
    setback_amount = int(setback_percentage*setback_frequency)
    
    setback_name = f"setback_{setback_frequency}_{setback_percentage}"
    
    total_steps = 0
    if begin_step is not None:
        total_steps = begin_step

    for step in range(iterations):
        total_steps = total_steps+1
            
        # Apply setback 
        if setback_frequency is not None and setback_percentage is not None:
            # Check if we need to save off the weights
            if (total_steps+setback_amount) % setback_frequency == 0:
                torch.save({'state_dict': discriminator.state_dict()}, 'models/gan_models/setback_dis_temp.pth')
            # Check if we need to load in the weights
            if total_steps % setback_frequency == 0:
                checkpoint = torch.load('models/gan_models/setback_dis_temp.pth')
                discriminator.load_state_dict(checkpoint['state_dict'])

        for i, (imgs, classes) in enumerate(dataloader):
            #===================================
            # GENERATOR OPTIMIZE AND GET LABELS

            # Zero out any previous calculated gradients
            gan_optimizer.zero_grad()

            # Sample random points in the latent space
            random_latent_vectors = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))
            # Decode them to fake images, through the generator
            generated_images = generator(random_latent_vectors)

            # Assemble labels that say "all real images"
            # misleading target, c=1
            misleading_targets = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)

            # Get MSE Loss function
            # want generator output to generate images that are "close" to all "ones" 
            g_loss = adversarial_loss(discriminator(generated_images), misleading_targets)

            # now back propagate to get derivatives
            g_loss.backward()

            # use gan optimizer to only update the parameters of the generator
            # this was setup above to only use the params of generator
            gan_optimizer.step()

            #===================================
            # DISCRIMINATOR OPTIMIZE AND GET LABELS

            # Zero out any previous calculated gradients
            discriminator_optimizer.zero_grad()

            # Combine real images with some generator images
            real_images = Variable(imgs.type(Tensor))
            combined_images = torch.cat([real_images, generated_images.detach()])
            # in the above line, we "detach" the generated images from the generator
            # this is to ensure that no needless gradients are calculated 
            # those parameters wouldn't be updated (becasue we already defined the optimized parameters)
            # but they would be calculated here, which wastes time.

            # Assemble labels discriminating real from fake images
            # real label, a=1 and fake label, b=0
            labels = torch.cat((
                Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False),
                Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
            ))
            # Add random noise to the labels - important trick!
            labels += 0.05 * torch.rand(labels.shape).to(device)

            # Setup Discriminator loss
            # this takes the average of MSE(real images labeled as real) + MSE(fake images labeled as fake)
            mid_point = combined_images.shape[0] // 2
            d_loss = (
                adversarial_loss(discriminator(combined_images[:mid_point]), labels[:mid_point]) + \
                adversarial_loss(discriminator(combined_images[mid_point:]), labels[mid_point:])
                ) / 2

            # get gradients according to loss above
            d_loss.backward()
            # optimize the discriminator parameters to better classify images
            discriminator_optimizer.step()

            # Now Clip weights of discriminator (manually)
            for p in discriminator.parameters():
                p.data.clamp_(-clip_value, clip_value)

            #===================================
        
        # Occasionally save / plot
        if step % 10 == 0:
            # Print metrics
            print('Loss at step %s: D(z_c)=%s, D(G(z_mis))=%s' % (total_steps, d_loss.item(),g_loss.item()))
            # save images in a list for display later
        
        if step % 25 == 0:
            
            with torch.no_grad():
                fake_output = generator(fixed_random_latent_vectors).detach().cpu()
            img_list.append(torchvision.utils.make_grid(fake_output[0:25], padding=2, normalize=True, nrow=5))
            
            IS_scores.append(inception_score(fake_output.to(device))[0])
            np.save(f'models/gan_models/{setback_name}_is_scores_celeb.npy', IS_scores)

            # in addition, save off a checkpoint of the current models and images
            ims = np.array([np.transpose(np.hstack((i,real_image_numpy)), (2,1,0)) for i in img_list])
            np.save(f'models/gan_models/{setback_name}_images_celeb.npy',ims)

            # save the state of the models (will need to recreate upon reloading)
            torch.save({'state_dict': generator.state_dict()}, f'models/gan_models/{setback_name}_gen_celeb.pth')
            torch.save({'state_dict': discriminator.state_dict()}, f'models/gan_models/{setback_name}_dis_celeb.pth')
    return (ims)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02) # filters are zero mean, small STDev
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02) # batch norm is unit mean, small STDev
        m.bias.data.fill_(0) # like normal, biases start at zero

if __name__ == '__main__':
    
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        # work on a single GPU or CPU
        cudnn.benchmark=True
        #generator.cuda()
        #discriminator.cuda()
        #adversarial_loss.cuda()
        Tensor = torch.cuda.FloatTensor
        print(device)
    
    
    checkpoint = torch.load( 'models/gan_models/face_classifier.pth')
    model1 = Classifier(8).to(device)
    model1.load_state_dict(checkpoint['state_dict'])
    args = sys.argv

    if len(args) != 4:
        exit()

    setback_frequency = int(args[1])
    setback_percentage = float(args[2])
    EPOCHS = int(args[3])

    print(f"Running EPOCHS:{EPOCHS} setback:{setback_frequency} percentag{setback_percentage}")

    ###Load the data###
    workers = 32

    batch_size = 128

    image_size = 64

    nc = 3

    nz = 100
    dataroot = "./data/celeba_smaller/"
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]))
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=1)
    
    dataiter = iter(dataloader)
    real_image_examples, classes = dataiter.next()
    latent_dim = 100
    height = real_image_examples[0].shape[1]
    width = real_image_examples[0].shape[2]
    channels = real_image_examples[0].shape[0]
    
    ###Create the Gen/disc###
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator(latent_dim).to(device)

    # LSGAN says they use ADAM, but follow up papers say RMSProp is lsightly better
    #lr = 0.0002
    #betas = (0.5, 0.999)

    # To stabilize training, we use learning rate decay
    # and gradient clipping (by value) in the optimizer.
    clip_value = 1.0 # This value will use in the future training process since 
                     # PyTorch didn't has the feature to set clipvalue for 
                     # RMSprop optimizer.
        
    # set discriminator learning higher than generator
    discriminator_optimizer = torch.optim.RMSprop(discriminator.parameters(),
                                                  lr=0.0002, weight_decay=1e-8)

    gan_optimizer = torch.optim.RMSprop(generator.parameters(),
                                        lr=0.0001, weight_decay=1e-8)

    # used to be: adversarial_loss = torch.nn.BCELoss() # binary cross entropy 
    adversarial_loss = torch.nn.MSELoss() # mean squared error loss 

    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    
    iterations = 21

    log_every = 50

    # Sample random points in the latent space
    plot_num_examples = 200
    fixed_random_latent_vectors = torch.randn(plot_num_examples, latent_dim, device=device)
    img_list = []
    IS_scores = []
    total_steps = 0

    real_image_numpy = np.transpose(
        torchvision.utils.make_grid(real_image_examples[:25,:,:,:],
                                    padding=2, normalize=False, nrow=5),(0,1,2))

    ims = train_setback(generator=generator, gan_optimizer=gan_optimizer,
                             discriminator=discriminator, discriminator_optimizer=discriminator_optimizer,
                             iterations=iterations, dataloader=dataloader,
                             setback_frequency=setback_frequency,setback_percentage=setback_percentage
                             )

    exit()