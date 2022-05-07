import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

class Discriminator(nn.Module):

    def __init__(self, latent_dim, width, channels):
        super(Discriminator, self).__init__()
        
        self.init_size = width // 16 # one 16th the image size 
        self._latent_dim = latent_dim

        self.model = nn.Sequential(
            
            ############# 64X64 #########################################
            nn.Conv2d(in_channels=channels, out_channels=16, kernel_size=3, stride=1, padding='same'), #64X64
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
    def get_feature_layer_1(self, img):
        
        # use the view function to flatten the layer output
        #    old way for earlier Torch versions: out = out.view(out.shape[0], -1)
        out = self.model(img)
        return out
    def get_feature_layer_2(self, img):
        
        # move to the latent dim
        out = self.get_feature_layer_1(img)
        out = torch.flatten(out, start_dim=1)
        out = self.laten_dim_layer(out)
        return out 
    def forward(self, img):
        
        out = self.get_feature_layer_2(img)
        validity = self.classification_layer(out)
        return validity
