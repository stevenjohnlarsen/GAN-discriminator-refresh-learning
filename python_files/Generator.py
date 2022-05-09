"""
Generator:
a nn.Module clas that it the generator for a GAN
This one taked notated for 64x64 sized images.
"""

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, latent_dim, width, channels):
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
    