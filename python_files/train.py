"""
This file contains the train function that is called by main.py
setup_train must be called before train
Classifer, is used for IS and has been pre trained in models/gan_models/face_classifier.pth.
Any model that can classify the dataset can be used here.
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
from IPython.display import HTML
from torchvision.utils import save_image

from PIL import Image
from matplotlib import image
from matplotlib import pyplot
from matplotlib import image
from matplotlib import pyplot
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

def setup_train(latent_dim, device_in):
    global fixed_random_latent_vectors
    global plot_num_examples
    global device
    global fixed_random_latent_vectors
    global img_list
    global IS_scores 
    global total_steps 
    global plot_num_examples 
    global device
    fixed_random_latent_vectors = torch.randn(plot_num_examples, latent_dim, device=device)
    img_list= []
    IS_scores = []
    total_steps = 0
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

def train(generator, gan_optimizer,
          discriminator, discriminator_optimizer,
          iterations, dataloader, 
          latent_dim,
          real_image_numpy,
          setback_frequency=None,
          setback_percentage=None,
          begin_step=None,
          log_every=50,
          save_off_every=10, 
          phi=1):
    
    setback_name = "ls"
    if setback_frequency is not None and setback_percentage is not None:
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

            real_images = Variable(imgs.type(Tensor)).to(device)
            # Sample random points in the latent space
            random_latent_vectors = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim)))).to(device)
            # Decode them to fake images, through the generator
            
            generated_images = generator(random_latent_vectors)

            # Assemble labels that say "all real images"
            # misleading target, c=1
            misleading_targets = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False).to(device)

            # Get MSE Loss function
            # want generator output to generate images that are "close" to all "ones" 
            g_loss = adversarial_loss(discriminator(generated_images), misleading_targets) * (phi)
            
            #fake_features = torch.mean(discriminator.get_feature_layer_1(generated_images), dim=0)
            #real_features = torch.mean(discriminator.get_feature_layer_1(real_images), dim=0)
            #g_loss += torch.mean(torch.square(fake_features-real_features)) / 4
            
            if phi != 1: 
                fake_features = torch.mean(discriminator.get_feature_layer_2(generated_images), dim=0)
                real_features = torch.mean(discriminator.get_feature_layer_2(real_images), dim=0)
                g_loss += torch.mean(torch.square(fake_features-real_features)) * (1 - phi)
            
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
        if step % log_every == 0:
            # Print metrics
            print('Loss at step %s: D(z_c)=%s, D(G(z_mis))=%s' % (total_steps, d_loss.item(),g_loss.item()))
            # save images in a list for display later
        
        if step % save_off_every == 0:
            
            with torch.no_grad():
                fake_output = generator(fixed_random_latent_vectors).detach().cpu()
            img_list.append(torchvision.utils.make_grid(fake_output[0:25], padding=2, normalize=True, nrow=5))
            
            IS_scores.append(
                [inception_score(fake_output.to(device))[0],
                g_loss.item(),
                d_loss.item()]
            )
            np.save(f'models/gan_models/{setback_name}_is_scores_celeb.npy', IS_scores)

            # in addition, save off a checkpoint of the current models and images
            ims = np.array([np.transpose(np.hstack((i,real_image_numpy)), (2,1,0)) for i in img_list])
            np.save(f'models/gan_models/{setback_name}_images_celeb.npy',ims)

            # save the state of the models (will need to recreate upon reloading)
            torch.save({'state_dict': generator.state_dict()}, f'models/gan_models/{setback_name}_gen_celeb.pth')
            torch.save({'state_dict': discriminator.state_dict()}, f'models/gan_models/{setback_name}_dis_celeb.pth')
    return (ims)


    
plot_num_examples = 200
img_list= []
IS_scores = []
total_steps = 0
device = 'cuda:0'
fixed_random_latent_vectors = torch.randn(1, 1000, device=device)
device = torch.device("cuda:0")
cudnn.benchmark=False
Tensor = torch.cuda.FloatTensor
adversarial_loss = torch.nn.MSELoss() 
clip_value = 1.0

checkpoint = torch.load( 'models/gan_models/face_classifier.pth')
model1 = Classifier(8).to(device)
model1.load_state_dict(checkpoint['state_dict'])
    