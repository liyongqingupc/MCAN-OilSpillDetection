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

def train(opt,Gs,Ds,Zs,real1s,real2s,NoiseAmp,train_range,train_index,train_i,train_loopnum):
    in_s = 0
    scale_num = 0
    nfc_prev = 0

    opt.input_name = 'train' + str(train_range[train_index]) + '.jpg'  # the name of training image
    real1_, real2_ = functions.read_image(opt)   #read_image

    real1 = imresize(real1_, opt.scale1, opt)  #opt.scale1
    real2 = imresize(real2_, opt.scale1, opt)
    real1s = functions.creat_reals_pyramid(real1, real1s, opt)
    real2s = functions.creat_reals_pyramid(real2, real2s, opt)

    while scale_num < opt.stop_scale + 1:  # opt.stop_scale is defined in functions.py
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.out_ = functions.generate_dir2save(opt)
        opt.outf = '%s/%d' % (opt.out_, scale_num)  # output path
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
            if (nfc_prev == opt.nfc):
                G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out_, scale_num - 1)))
                D_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_, scale_num - 1)))

        # train_single_scale
        z_curr, in_s, G_curr, D_curr = train_single_scale(D_curr, G_curr, real1s, real2s, Gs, Zs, in_s, NoiseAmp, opt, train_index, train_i, train_loopnum)  # reals -- real1s,real2s  # lyq add D_cu

        G_curr = functions.reset_grads(G_curr, False)  # reset_grads
        G_curr.eval()
        D_curr = functions.reset_grads(D_curr, False)
        D_curr.eval()

        Gs.append(G_curr)  # Gs
        Ds.append(D_curr)  #
        Zs.append(z_curr)  # Zs
        NoiseAmp.append(opt.noise_amp)  # NoiseAmp: weight of noise

        scale_num += 1  # enter next scale
        nfc_prev = opt.nfc  # nfc_prev
        del D_curr, G_curr  # delete

    # saved in TrainedModels
    torch.save(Gs, '%s/Gs.pth' % (opt.out_))
    torch.save(Ds, '%s/Ds.pth' % (opt.out_))   #
    torch.save(real2s, '%s/real2s.pth' % (opt.out_))  #
    torch.save(real1s, '%s/real1s.pth' % (opt.out_))  #
    torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))
    return


####################################
###    train_single_scale       ###
####################################
def train_single_scale(netD,netG,real1s,real2s,Gs,Zs,in_s,NoiseAmp,opt,train_index,train_i, train_loopnum):

    real2 = real2s[len(Gs)]  #get the current image from reals, len(Gs) is index
    real1 = real1s[len(Gs)]

    opt.nzx = real2.shape[2]  #real --real2
    opt.nzy = real2.shape[3]  #real --real2

    opt.receptive_field = opt.ker_size + ((opt.ker_size-1)*(opt.num_layer-1))*opt.stride   # receptive_field = 11  original

    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)  #pad_noise = 5
    pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)  #pad_image = 5

    m_noise = nn.ZeroPad2d(int(pad_noise))  # m_noise
    m_image = nn.ZeroPad2d(int(pad_image))  # m_image

    alpha = opt.alpha  # alpha

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999)) # netD.parameters()
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD,milestones=[1600],gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,milestones=[1600],gamma=opt.gamma)

    errD2plot = []
    errG2plot = []
    D_real2plot = []
    D_fake2plot = []
    #z_opt2plot = []

    for epoch in range(opt.niter):
        z_opt = m_noise(real1.expand(1, 3, opt.nzx, opt.nzy))
        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
        for j in range(opt.Dsteps):
            # train with real
            netD.zero_grad()  # zero_grad: clear gradient
            output = netD(torch.cat((real1, real2), 1)).to(opt.device)  # input is image pairs
            # D_real_map = output.detach()
            errD_real = -output.mean()  #-a
            errD_real.backward(retain_graph=True)  # retain_graph=True
            D_x = -errD_real.item()

            # train with fake
            if (j==0) & (epoch == 0):  # epoch == 0
                if (Gs == []) & (opt.mode != 'SR_train'):
                    prev = torch.full([1,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)  # full:generete 0 matrix
                    in_s = prev  # in_s is prev
                    prev = m_image(prev)   # m_image: add 0
                    z_prev = torch.full([1,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device) # z_prev
                    z_prev = m_noise(z_prev)   # m_noise
                    opt.noise_amp = 1
                else:
                    prev = draw_concat(Gs,Zs,real2s,NoiseAmp,in_s,'rand',m_noise,m_image,opt) #Zs--real1s[0:len(Gs)-1]
                    prev = m_image(prev)
                    z_prev = draw_concat(Gs,Zs,real2s,NoiseAmp,in_s,'rec',m_noise,m_image,opt) #Zs--real1s[0:len(Gs)-1]
                    criterion = nn.MSELoss()   # MSELoss()
                    RMSE = torch.sqrt(criterion(real2, z_prev))   # RMSE
                    opt.noise_amp = opt.noise_amp_init * RMSE
                    z_prev = m_image(z_prev)
            else:
                prev = draw_concat(Gs,Zs,real2s,NoiseAmp,in_s,'rand',m_noise,m_image,opt) #Zs--real1s[0:len(Gs)-1]
                prev = m_image(prev)

            if (Gs == []):
                real11 = z_opt
            else:
                real11= opt.noise_amp * z_opt + prev  # prev: from underlayer
            fake = netG(real11.detach(), prev)  #input real1  #detach()

            output = netD(torch.cat((real1, fake),1).detach())   # lyq add torch.cat 0911
            errD_fake = output.mean()
            errD_fake.backward(retain_graph=True)
            D_G_z = output.mean().item()

            # calc_gradient_penalty
            gradient_penalty = functions.calc_gradient_penalty(netD, real1, real2, fake, opt.lambda_grad, opt.device)
            gradient_penalty.backward()
            ######## errD:fake-real+gp ########
            errD = errD_real + errD_fake + gradient_penalty
            optimizerD.step()

        errD2plot.append(errD.detach())

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################

        for j in range(opt.Gsteps):
            netG.zero_grad()
            output = netD(torch.cat((real1, fake), 1))
            errG = -output.mean()
            errG.backward(retain_graph=True)
            if alpha != 0:
                #loss = nn.MSELoss()  #
                Z_opt = opt.noise_amp * z_opt + z_prev
                gen_image = netG(Z_opt.detach(), z_prev)   #
                norm_loss = alpha * l1(gen_image,real2)  #
                norm_loss.backward(retain_graph=True)   # lyq add
                norm_loss = norm_loss.detach()   # lyq add
            else:
                Z_opt = z_opt  # original
                norm_loss = 0  # lyq add
            optimizerG.step()

        errG2plot.append(errG.detach() + norm_loss)  # add "+ norm_loss"
        D_real2plot.append(D_x)
        D_fake2plot.append(D_G_z)

        if epoch % 10 == 0 or epoch == (opt.niter-1):
            print('train_i:[%d/%d]-train_index:%d-scale:%d-iteration:[%d/%d]' % (train_i+1,train_loopnum,train_index,len(Gs), epoch+1, opt.niter))

        if epoch % 10 == 0 or epoch == (opt.niter-1):
            plt.imsave('%s/fake_sample.png' % (opt.outf), functions.convert_image_np(fake.detach()), vmin=0, vmax=1)
            plt.imsave('%s/G(z_opt).png' % (opt.outf),  functions.convert_image_np(netG(Z_opt.detach(), z_prev).detach()), vmin=0, vmax=1)
            plt.imsave('%s/z_opt.png' % (opt.outf), functions.convert_image_np(z_opt.detach()), vmin=0, vmax=1)
            plt.imsave('%s/z_prev.png' % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)
            torch.save(z_opt, '%s/z_opt.pth' % (opt.outf))
        schedulerD.step()
        schedulerG.step()

    functions.save_networks(netG,netD,z_opt,opt)   #save_networks
    return z_opt,in_s,netG,netD    # return z_opt,in_s,netG


def l2(x, y):
    return (torch.pow(x - y, 2)).mean()  # lyq add

def l1(x, y):
    return (torch.abs(x - y)).mean()  # lyq add

###### draw_concat ######
def draw_concat(Gs,Zs,reals,NoiseAmp,in_s,mode,m_noise,m_image,opt):
    G_z = in_s
    if len(Gs) > 0:
        if mode == 'rand':
            count = 0
            pad_noise = int(((opt.ker_size-1)*opt.num_layer)/2)
            if opt.mode == 'animation_train':
                pad_noise = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                if count == 0:
                    z = functions.generate_noise([1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                    z = z.expand(1, 3, z.shape[2], z.shape[3])
                else:
                    z = functions.generate_noise([opt.nc_z,Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                z = m_noise(z)
                G_z = G_z[:,:,0:real_curr.shape[2],0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp*z+G_z
                G_z = G(z_in.detach(),G_z)
                G_z = imresize(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                count += 1
        if mode == 'rec':
            count = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp*Z_opt+G_z
                G_z = G(z_in.detach(),G_z)
                G_z = imresize(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                #if count != (len(Gs)-1):
                #    G_z = m_image(G_z)
                count += 1
    return G_z


def init_models(opt):

    #generator initialization:
    netG = models.GeneratorConcatSkip2CleanAdd(opt).to(opt.device)
    netG.apply(models.weights_init)
    #print(netG)   #lyq #

    #discriminator initialization:
    netD = models.WDiscriminator(opt).to(opt.device)
    netD.apply(models.weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    #print(netD)   #lyq #

    return netD, netG
