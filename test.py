from __future__ import print_function
from MCAN.training import *
from config import get_arguments
from MCAN.functions import *
import matplotlib.pyplot as plt
import numpy as np
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

for test_num in range(5, 31, 1):  # the index of test images
    opt.input_name = 'test' + str(test_num) + '.jpg'   # the name of test image
    x = img.imread('%s/%s' % (opt.input_dir, opt.input_name))  # img.imread
    x = np2torch(x, opt)

    x = x[:, 0:3, :, :]
    a = x.shape
    b = int(a[3])
    real1_ = x[:, 0:3, :, 0:int(b / 2)]  # extract left image

    ###### normalize() -> [0,1] ######
    real1_ = (real1_ - real1_.min()) / (real1_.max() - real1_.min())

    functions.adjust_scales2image(real1_, opt)
    real1 = imresize(real1_, 0, opt)
    real1s = []
    real1s = functions.creat_reals_pyramid(real1, real1s, opt)

    images_cur = None
    n = 0  # scale num
    
    images_cur = torch.full(real1s[0].shape, 0, device = opt.device)  # images_cur

    for G, real_in in zip(Gs, real1s):  # use each trained G
        images_prev = images_cur  # set images_prev

        #print('images_prev:', images_prev.shape)
        #print('real_in:', real_in.shape)
        g_in = torch.cat((real_in, images_prev),1).detach()  # input of G
        I_curr = G(g_in, images_prev)  # output of G

        images_cur = []  # set as []
        images_cur = I_curr
        images_cur = imresize(images_cur, -1, opt)  # upsample

        ############################### SET opt.scale_num #######################################
        if n == opt.scale_num:  # reach the finest scale

            for threshold in range(70, 72, 2):   #
                ################## output normalize()--> 1 ###################
                I_curr = (I_curr - I_curr.min()) / (I_curr.max() - I_curr.min())

                I_cu = functions.convert_image_np(I_curr.detach())  

                I_cu = (I_cu[:, :, 0] + I_cu[:, :, 1] + I_cu[:, :, 2]) / 3  # 3D --> 2D lyq210118

                if opt.mode == 'test':
                    dir2save = '%s/TestResult_%s' % (opt.out, trained_model)
                try:
                    os.makedirs(dir2save)
                except OSError:
                    pass
                ####################### POW ###########################
                I_save = pow(I_cu, 1)  #
                #print('I_save.shape:', I_save.shape, 'I_save',I_save)  # 256*256
                #print('I_cu.shape:', I_cu.shape, 'I_cu', I_cu)
                I_save[np.where(I_save <= float(threshold/100))] = 0
                I_save[np.where(I_save > float(threshold/100))] = 255

                # save the generated image
                opt.save_name = 'test' + str(test_num) + '_' + str(float(threshold/100)) + '.jpg'  # the name of test image
                plt.imsave('%s/%s' % (dir2save, opt.save_name), I_save, cmap='gray', vmin=0, vmax=1)


        n += 1  #  n + 1, enter next scale

