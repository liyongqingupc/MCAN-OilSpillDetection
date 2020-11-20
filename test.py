from __future__ import print_function
from MCAN.training import *
from config import get_arguments
from MCAN.functions import *
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from skimage import io as img
from MCAN.imresize import imresize

# opt
parser = get_arguments()
parser.add_argument('--input_dir', help='input image dir', default='Input/TestSet')
parser.add_argument('--input_name', help='input image name',  default="test0.jpg")   # input test image name
parser.add_argument('--mode', help='task to be done', default='test')
parser.add_argument('--trained_model', help='folder name of trained model', default='model')

opt = parser.parse_args()
opt = functions.post_config(opt)

# input the folder name of trained model
trained_model = opt.trained_model
########################################
# Gs
pthfile1 = r'TrainedModels/%s/Gs.pth' % trained_model
Gs = torch.load(pthfile1)

# NoiseAmp
pthfile3 = r'TrainedModels/%s/NoiseAmp.pth' % trained_model
NoiseAmp = torch.load(pthfile3)

# real2s
pthfile4 = r'TrainedModels/%s/real2s.pth' % trained_model
real2s = torch.load(pthfile4)

for test_num in range(1,31,1):  # the index of test images
    opt.input_name = 'test' + str(test_num) + '.jpg'   # the name of test image
    x = img.imread('%s/%s' % (opt.input_dir, opt.input_name))  # img.imread
    x = np2torch(x, opt)
    x = x[:, 0:3, :, :]
    a = x.shape
    b = int(a[3])
    real1_ = x[:, 0:3, :, 0:int(b / 2)]  # left image
    real1_ = (real1_ - real1_.min()) / (real1_.max() - real1_.min())  # normalize()

    real2_ = x[:, 0:3, :, int(b / 2):b]  # right image
    #real2_ = (real2_ - real2_.min()) / (real2_.max() - real2_.min())  # normalize()

    functions.adjust_scales2image(real1_, opt)
    real1 = imresize(real1_, opt.scale1, opt)
    real1s = []
    real1s = functions.creat_reals_pyramid(real1, real1s, opt)

    in_s = None
    scale_v = 1
    scale_h = 1
    n = 0  # floor num
    gen_start_scale = 0  # gen_start_scale
    num_samples = 1  # num_samples: num of generated image for each scale


    if in_s is None:
        in_s = torch.full(real2s[0].shape, 0, device=opt.device)  # creat 0 matrix
    images_cur = []  # images_cur

    for G, Z_opt, noise_amp in zip(Gs, real1s, NoiseAmp):
        pad1 = ((opt.ker_size - 1) * opt.num_layer) / 2  # pad1  (2*5)/2=5
        m = nn.ZeroPad2d(int(pad1))  # padding

        images_prev = images_cur  # set images_prev
        images_cur = []  # set images_cur as [],

        for i in range(0, num_samples, 1):  # num_samples =1 : run 1 loop
            if n == 0:  # n == 0 is the coarsest scale
                opt.nzx = real1s[n].shape[2]
                opt.nzy = real1s[n].shape[3]
                z_curr = real1s[n].expand(1, 3, opt.nzx, opt.nzy)
                z_curr = m(z_curr)
            else:
                opt.nzx = real1s[n].shape[2]
                opt.nzy = real1s[n].shape[3]
                z_curr = real1s[n].expand(1, opt.nc_z, opt.nzx, opt.nzy)
                z_curr = m(z_curr)

            if images_prev == []:
                I_prev = m(in_s)  # now, in_s is 0 matrix
            else:
                I_prev = images_prev[i]
                I_prev = imresize(I_prev, 1/opt.scale_factor, opt)
                if opt.mode == "test":
                    I_prev = I_prev[:, :, 0:round(scale_v * real2s[n].shape[2]), 0:round(scale_h * real2s[n].shape[3])]
                    I_prev = m(I_prev)  # m
                    I_prev = I_prev[:, :, 0:z_curr.shape[2], 0:z_curr.shape[3]]
                    I_prev = functions.upsampling(I_prev, z_curr.shape[2], z_curr.shape[3])  # I_prev upsampling
                else:
                    I_prev = m(I_prev)

            if n < gen_start_scale:
                z_curr = Z_opt

            z_in = noise_amp * (z_curr) + I_prev  # input
            I_curr = G(z_in.detach(), I_prev)  # output
            ################## output --> 1 ##################
            I_curr = (I_curr - I_curr.min())/(I_curr.max() - I_curr.min())  # normalize()
            I_cu = functions.convert_image_np(I_curr.detach())

            if n == len(real2s) - 1:  # reach at finest scale
                threshold = 0.9  # Set the threshold of output
                I_cu[np.where(I_cu <= threshold)] = 0
                I_cu[np.where(I_cu > threshold)] = 1

                if opt.mode == 'test':
                    dir2save = '%s/TestResult_%s' % (opt.out, trained_model)
                try:
                    os.makedirs(dir2save)
                except OSError:
                    pass
                # save the generated image
                plt.imsave('%s/%s' % (dir2save, opt.input_name), I_cu, vmin=0, vmax=1)
            images_cur.append(I_curr)  # images_cur has been cleared above

        n += 1  #  n + 1, enter next scale

