import MCAN.functions as functions
import MCAN.models as models
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import math
import matplotlib.pyplot as plt
from MCAN.imresize import imresize
import numpy as np

def train(opt,Gs,Ds,real1s,real2s,train_range,train_index,train_i,train_loopnum):
    in_s = 0
    scale_num = 0

    opt.input_name = 'train' + str(train_range[train_index]) + '.jpg'  # the name of training image
    real1_, real2_ = functions.read_image(opt)   #read_image

    real1 = imresize(real1_, 0, opt)  #opt.scale1->opt.scale_num->0
    real2 = imresize(real2_, 0, opt)  #opt.scale1->opt.scale_num->0
    #print('real2:', real2.shape)  #[1, 3, 256, 256]
    real1s = functions.creat_reals_pyramid(real1, real1s, opt)
    real2s = functions.creat_reals_pyramid(real2, real2s, opt)

    while scale_num < opt.scale_num + 1:  # opt.stop_scale->opt.scale_num
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)  # adjust the num of kernel
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.out_ = functions.generate_dir2save(opt)
        opt.outf = '%s/%d' % (opt.out_, scale_num)  # output path, used in functions.save_networks(netG,netD,opt)
        try:
            os.makedirs(opt.outf)
        except OSError:
            pass

        if train_i == 0:  # if it is the first image
            D_curr, G_curr = init_models(opt)  # init_models()
        else:
            D_curr, G_curr = init_models(opt)
            G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out_, scale_num)))  # load the previous model
            D_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_, scale_num)))

        # train_single_scale
        in_s, G_curr, D_curr = train_single_scale(
                               D_curr, G_curr, real1s, real2s, Gs, in_s, opt, train_index, train_i, train_loopnum, scale_num)  # reals -- real1s,real2s  # lyq add D_cu

        G_curr = functions.reset_grads(G_curr, False)  # reset_grads
        G_curr.eval()                                  # eval()
        D_curr = functions.reset_grads(D_curr, False)
        D_curr.eval()

        Gs.append(G_curr)  # Gs
        Ds.append(D_curr)  # Ds

        scale_num += 1  # enter next scale

        del D_curr, G_curr  # delete current model, creat new model at next scale

    # saved in TrainedModels
    torch.save(Gs, '%s/Gs.pth' % (opt.out_))
    torch.save(Ds, '%s/Ds.pth' % (opt.out_))   #
    torch.save(real2s, '%s/real2s.pth' % (opt.out_))  #
    torch.save(real1s, '%s/real1s.pth' % (opt.out_))  #

    return

####################################
###    train_single_scale       ###
####################################
def train_single_scale(netD,netG,real1s,real2s,Gs,in_s,opt,train_index,train_i, train_loopnum, scale_num):

    #print('train_index:', train_index)
    real2 = real2s[len(Gs)]  #get the current image from reals, len(Gs) is index
    real1 = real1s[len(Gs)]

    opt.nzx = real2.shape[2]  #real --real2
    opt.nzy = real2.shape[3]  #real --real2
    #print('opt.nzx:', opt.nzx, 'opt.nzy:', opt.nzy)  # lyq # 64,64

    alpha = opt.alpha  # alpha

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, opt.beta2)) # netD.parameters()
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, opt.beta2))  #lyq add beta2
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD,milestones=[1600],gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,milestones=[1600],gamma=opt.gamma) #gamma=0.1

    errD2plot = []
    errG2plot = []
    D_real2plot = []
    D_fake2plot = []

    if scale_num == opt.scale_num:  #lyq 0317
        niter_num = 1  # finest scale train 1 time
    else:
        niter_num = opt.niter # other scale train opt.niter time

    for epoch in range(niter_num):
        ############################
        # (1) Update D network: minimize -D(x) + D(G(z))
        ###########################
        for j in range(opt.Dsteps): # The num of training D
            # train with real
            netD.zero_grad()  # zero_grad:
            output = netD(torch.cat((real1, real2), 1)).to(opt.device)  # input image pairs
            errD_real = -output.mean()  #-a
            errD_real.backward(retain_graph=True)  # retain_graph=True
            D_x = -errD_real.item()

            if (Gs == []): # lyq add 0315 
                prev = torch.full([1,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)  # full:generete 0 matrix
            else:
                #print('real1_prev:',real1_prev.shape, 'in_s:',in_s.shape)
                prev = in_s
            print('real1:',real1.shape, 'prev:',prev.shape)  #Dsteps=3 has error
            fake = netG(torch.cat((real1, prev),1).detach(), prev) # lyq add 0315

            output = netD(torch.cat((real1, fake),1).detach())   # lyq add torch.cat 0911 # 
            errD_fake = output.mean()
            errD_fake.backward(retain_graph=True)
            D_G_z = output.mean().item()

            # calc_gradient_penalty
            gradient_penalty = functions.calc_gradient_penalty(netD, real1, real2, fake, opt.lambda_grad, opt.device)
            gradient_penalty.backward()

            ######## errD: fake - real + gp ########
            errD = errD_real + errD_fake + gradient_penalty   ### The errD ###

            if train_i % 20 == 0 and epoch % 10 == 0:
                print('errD:', errD)
                print('gradient_penalty:', gradient_penalty)

            optimizerD.step()   # optimizerD

        errD2plot.append(errD.detach())  # errD: tensor(1.8132, device='cuda:0', grad_fn=<AddBackward0>)

        ############################
        # (2) Update G network:
        ###########################
        for j in range(opt.Gsteps):
            netG.zero_grad()
            output = netD(torch.cat((real1, fake), 1))
            errG = -output.mean()
            errG.backward(retain_graph=True)

            if alpha != 0:
                gen_image = fake  # lyq add 0315
                norm_loss = alpha * l1(gen_image, real2)  #
                norm_loss.backward(retain_graph=True)   # lyq add
                if train_i % 20 == 0 and epoch % 10 == 0:
                    print('errG:', errG)
                    print('norm_loss:', norm_loss)
                norm_loss = norm_loss.detach()   # lyq add

            optimizerG.step()  # optimizerG

        errG2plot.append(errG.detach() + norm_loss)

        D_real2plot.append(D_x)
        D_fake2plot.append(D_G_z)

        if train_i % 20 == 0 and epoch % 10 == 0:
            print('train_i:[%d/%d]-train_index:%d-scale:%d-iteration:[%d/%d]' % (train_i+1,train_loopnum,train_index,len(Gs), epoch+1, opt.niter))

        if train_i % 10 == 0 and epoch % 10 == 0:
            plt.imsave('%s/fake_sample.png' % (opt.outf), functions.convert_image_np(fake.detach()), vmin=0, vmax=1)
            plt.imsave('%s/z_prev.png' % (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)

        schedulerD.step()
        schedulerG.step()

    functions.save_networks(netG, netD, opt)   #save_networks
    in_s = imresize(fake.detach(), -1, opt) # in_s is prepared for the next scale # +.detach()  # lyq 0317

    return in_s, netG, netD    # return z_opt,in_s,netG  #lyq delete z_opt


def l2(x, y):
    return (torch.pow(x - y, 2)).mean()  # lyq add

def l1(x, y):
    return (torch.abs(x - y)).mean()  # lyq add


def init_models(opt):
    #generator initialization:
    netG = models.GeneratorConcatSkip2CleanAdd(opt).to(opt.device)
    netG.apply(models.weights_init)
    #print(netG)   #lyq #

    #discriminator initialization:
    netD = models.WDiscriminator(opt).to(opt.device)
    netD.apply(models.weights_init)
    if opt.netD != '':  #
        netD.load_state_dict(torch.load(opt.netD))
    #print(netD)   #lyq #
    return netD, netG
