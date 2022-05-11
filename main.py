#!/usr/bin/env python
# coding: utf-8

# In[1]:
"""
main.py will take the parameters below and run a training session.
models will be saved to models/gan_models

Main program takes the following parameters
(1) # of epochs 
(2) # of latent dim (100)
(3) # phi: percentage weight for feature matching
(4) setback freuency (needs both 4 and 5)
(5) setback percentage (needs both 4 and 5)
"""

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
#from IPython.display import HTML
from torchvision.utils import save_image

from PIL import Image
from matplotlib import image
from matplotlib import pyplot
from matplotlib import image
from matplotlib import pyplot

from python_files.Generator import Generator
from python_files.Discriminator import Discriminator
from python_files.train import setup_train, train

import sys

# Seed control, for better reproducibility 
# NOTE: this does not gurantee results are always the same
seed = 22
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

#Global parameters
""""
Main program takes the following parameters
(1) # of epochs 
(2) # of latent dim (100)
(3) # phi: percentage weight for feature matching
(4) setback freuency (needs both 4 and 5)
(5) setback percentage (needs both 4 and 5)
"""

print(sys.argv)

if len(sys.argv) not in (4, 6):
    exit()

iterations = int(sys.argv[1])
latent_dim = int(sys.argv[2])
phi = float(sys.argv[3])

setback_frequency = None
setback_percentage = None
if len(sys.argv) == 6:
    setback_frequency = int(sys.argv[4])
    setback_percentage = float(sys.argv[5])
    
dataroot = "./data/celeba_smaller/"

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    # work on a single GPU or CPU
    cudnn.benchmark=True
    #generator.cuda()
    #discriminator.cuda()
    #adversarial_loss.cuda()
    Tensor = torch.cuda.FloatTensor
    print(device)
else:
    device = torch.device("cpu")
    cudnn.benchmark=False
    Tensor = torch.FloatTensor


# In[2]:


def norm_grid(im):
    # first half should be normalized and second half also, separately
    im = im.astype(np.float)
    rows,cols,chan = im.shape
    cols_over2 = int(cols/2)
    tmp = im[:,:cols_over2,:]
    im[:,:cols_over2,:] = (tmp-tmp.min())/(tmp.max()-tmp.min())
    tmp = im[:,cols_over2:,:]
    im[:,cols_over2:,:] = (tmp-tmp.min())/(tmp.max()-tmp.min())
    return im


# In[3]:



workers = 32

batch_size = 128

image_size = 64

nc = 3

nz = 100
print(dataroot)
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


# In[4]:


def imshow(img):
    # custom show in order to display
    # torch tensors as numpy
    npimg = img.numpy() / 2 + 0.5 # from tensor to numpy
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()



height = real_image_examples[0].shape[1]
width = real_image_examples[0].shape[2]
channels = real_image_examples[0].shape[0]


# In[7]:




# In[8]:
gen = Generator(latent_dim, width, channels).to(device)


# In[9]:


random_latent_vectors = Variable(Tensor(np.random.normal(0, 1, (32, latent_dim))))
generated_images = gen(random_latent_vectors)
imshow(generated_images[1].detach().cpu())


# In[10]:


disc = Discriminator(latent_dim, width, channels).to(device)
disc(generated_images)


# In[11]:




# In[12]:


setup_train(latent_dim, device)


# In[13]:


# custom weights initialization called on netG and netD
# this function from PyTorch's officail DCGAN example:
# https://github.com/pytorch/examples/blob/master/dcgan/main.py#L112
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02) # filters are zero mean, small STDev
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02) # batch norm is unit mean, small STDev
        m.bias.data.fill_(0) # like normal, biases start at zero


# In[14]:


generator = Generator(latent_dim, width, channels).to(device)
discriminator = Discriminator(latent_dim, width, channels).to(device)

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


# In[15]:


real_image_numpy = np.transpose(
    torchvision.utils.make_grid(real_image_examples[:25,:,:,:],
                                padding=2, normalize=False, nrow=5),(0,1,2))


train(generator=generator, gan_optimizer=gan_optimizer,
      discriminator=discriminator, discriminator_optimizer=discriminator_optimizer, 
      iterations=iterations, 
      dataloader=dataloader, 
      latent_dim=latent_dim,
      real_image_numpy=real_image_numpy,
      setback_frequency=setback_frequency,
      setback_percentage=setback_percentage,
      phi=phi
)


# In[ ]:




