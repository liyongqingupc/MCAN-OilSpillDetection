import torch
import torch.nn as nn


# Conv(3*3)-BatchNorm-LeakyReLU
class ConvBlock(nn.Sequential):   # nn.Sequential: father class
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        self.add_module('norm',nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))

def weights_init(m):  # m
    classname = m.__class__.__name__  # get class name
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)  # (0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)  # (1.0, 0.02)
        m.bias.data.fill_(0)
   
class WDiscriminator(nn.Module):  # W
    def __init__(self, opt):
        super(WDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()  # is GPu or not
        N = int(opt.nfc)  # N: out_channel  #opt.nfc: num of ker
        self.head = ConvBlock(opt.nc_im + 3, N, opt.ker_size, opt.padd_size, 1) # lyq add +3 0911
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1))) # pow: 2^(i+1)
            block = ConvBlock(max(2*N, opt.min_nfc),max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1) # max(N,opt.min_nfc)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Conv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)
        # WGAN: Discriminator do not have activation
        # The num of ker is decided by max() function, including opt.nfc and opt.min_nfc. # lyq 0315


    def forward(self,x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x

class GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im + 3, N, opt.ker_size, opt.padd_size, 1) #lyq +3#GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Sequential(
            nn.Conv2d(max(N,opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride =1, padding=opt.padd_size),
            nn.Tanh()  #  Tanh: activation
        )
    def forward(self,x,y): # y is prev, used for residual learning
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x+y
