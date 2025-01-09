from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import glob
import models.multiloss
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
print(torch.cuda.is_available())
from models.stackhourglass import PSMNet
# import wandb_demo
import wandb
from ImageLoader import myImageloader
from utils.preprocess import get_transform
from local_utils_main import load_model, scale_invariant
import time
from models.configurable_stn_projective import ConfigNet
from models.configurable_stn_projective_both_images import ConfigNet as ConfigNetLeftStn
from models.stn import Net
import math
from models import hsm
from PIL import Image
from local_utils_main import disparity2depth
from unet.predict_cont import predict_full_img, get_Unet
from models.MiDaS import MonoDepthNet
from models.submodule import *
import cv2
# from models.multiloss import criterion1_prob, criterion2, criterion3, criterion4, \
#                           evaluate, evaluate_kitti, predict,    \
                        #   WrappedModel
from models.multiloss import *

def get_mask(stereo_out, right_transformed):
    mono_mask = (stereo_out > 0.35) & (stereo_out < 4.5)
    mask = (right_transformed != 0)[:, 0, :, :]
    mask = mask & mono_mask
    nan_mask = ~torch.isnan(right_transformed)
    nan_mask = nan_mask.to(mask.dtype)
    if mask.dtype == torch.float32:
        mask = (mask > 0)  
    mask = mask & nan_mask
    return mask

im_cnt = 0

def get_mono_and_unrect_stereo(stereo_model, mono_net, midas_model, left, small_left, unet, dfd_net, device, small_right):
    global mono_out
    stereo_model.eval()
    with torch.no_grad():
        if mono_net == 'midas':
            mono_out = midas_model.forward(small_left)
            mono_out = torch.squeeze(mono_out, 0)
        elif mono_net == 'phase-mask':
            dfd_mono_out, _ = dfd_net(left, focal_point=1.5)
            mono_out = torch.unsqueeze(dfd_mono_out, 0)
        else:
            mono_out_unet = predict_full_img(unet, left, device=device)
            # mono_out_unet = psi_to_depth(mono_out_unet, focal_point=1.5)
            mono_out = torch.unsqueeze(mono_out_unet, 0)

        _, stereo_unrect = stereo_model(small_left, small_right)
        stereo_unrect = 100 / stereo_unrect
    if mono_net == 'midas':
        mono_out -= torch.min(mono_out)
        mono_out = torch.clamp(1 / mono_out, 0, 3)
    else:
        mono_out = get_small_mono(mono_out, device)
    return stereo_unrect
idx = 1

def train(model,  disp_model, optimizer, optimizer1, weight,left, right,width,height):
    global fig, ax_list, loss_list,idx,pse_rec,right_transformed_np
    model.train()
    # model.eval()
    # stereo_model.eval()
    # disp_model.eval()
    optimizer.zero_grad()
    stereo_out, theta, right_transformed = model(left, right)
    print(theta)

    stereo_out = 100 / stereo_out
    # right_transformed_image = right_transformed[0, 0, :, :].detach().cpu()
    # right_transformed_color = torch.stack([right_transformed_image, right_transformed_image, right_transformed_image], dim=0)
    # right_transformed_np = right_transformed_color.numpy()
    # plt.imshow(right_transformed_np.transpose(1, 2, 0), cmap='jet')  
    # plt.axis('off')  
    right_transformed_image = right_transformed[0, 0, :, :].detach().cpu()
    right_transformed_color = torch.stack([right_transformed_image, right_transformed_image, right_transformed_image], dim=0)
    right_transformed_np = right_transformed_color.numpy()
    plt.imshow(right_transformed_np.transpose(1, 2, 0), cmap=None)
    plt.axis('off')
    # plt.show()  # 显示图像
    folder_path = 'figures/transform_r'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    save_path = os.path.join(folder_path, f'right_transformed_{idx}_batch.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    idx = idx + 1
    mask = get_mask(stereo_out, right_transformed)    
    mask = mask[:, 0, :, :] 
    stereo_out = stereo_out[:, :256, :256]    
    mask = mask[:, :256, :256]
    loss1 = F.l1_loss(stereo_out[mask],pse_rec[mask])    
    # loss2 = F.l1_loss(right, pse_rec)
    # right_transformed_color = right_transformed_color.unsqueeze(0)
    loss_disp = disp_train(disp_model,optimizer1, left, right_transformed_color,width, height)
    loss =weight*loss1+(1-weight)*loss_disp
    loss.requires_grad_(True) 
    loss_list.append(loss)    # show_depth_maps(left, right_transformed, mono_out_for_train, stereo_unrect, stereo_out)
    # freeze_specific_hourglass_modules(model, [8,16])#冻结1、3
    loss.backward()
    # optimizer.zero_grad()
    optimizer.step()    
    # print(param.grad)
    return loss

def show_best_calibration(cp_file, model, small_left, small_right, mono_net, left, stereo_unrect,epoch):
    model.train()
    state_dict = torch.load(cp_file)
    model.load_state_dict(state_dict)
    with torch.no_grad():
        stereo_out, theta, right_transformed = model(small_left, small_right)
    stereo_out = 100 / stereo_out
    if mono_net == 'midas':
        mono_out_for_train = mono_out * (
                    torch.mean(stereo_out[stereo_out < 3.0]) / torch.mean(mono_out[mono_out < 3.0]))
    else:
        mono_out_for_train = mono_out
    # show_depth_maps(left, right_transformed, mono_out_for_train, stereo_unrect, stereo_out, blocking=True)

def disp_train(model,optimizer1, left, right_transformed,width,height):
    from torch.autograd import Variable
    global pred_disp
    max_disp = 400
    alpha = 0.1
    tmpdisp = int(max_disp//64*64)
    if (max_disp /64*64) > tmpdisp:
        model.module.maxdisp = tmpdisp + 64
    else:
        model.module.maxdisp = tmpdisp
    if model.module.maxdisp ==64: model.module.maxdisp=400
    model.module.disp_reg8 =  disparityregression(model.module.maxdisp,16).cuda()
    model.module.disp_reg16 = disparityregression(model.module.maxdisp,16).cuda()
    model.module.disp_reg32 = disparityregression(model.module.maxdisp,32).cuda()
    model.module.disp_reg64 = disparityregression(model.module.maxdisp,64).cuda()

        #DISP:
    imgL = left
    imgR = right_transformed
    folder_path = 'figures/transform_r'
    imgL = imgL.detach().cpu().numpy()
    # imgR = imgR.detach().cpu().numpy()
    imgL = np.reshape(imgL, (3, height, width))
    imgR = np.reshape(imgR, (3, height, width))
    imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
    imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])    
    right_transformed = Variable(torch.FloatTensor(imgR).cuda())
    max_h = int(imgL.shape[2] // 64 * 64)
    max_w = int(imgL.shape[3] // 64 * 64)
    if max_h < imgL.shape[2]: max_h += 64
    if max_w < imgL.shape[3]: max_w += 64
    top_pad = max_h-imgL.shape[2]
    left_pad = max_w-imgL.shape[3]
    imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
    imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
    # imgL_padded = np.lib.pad(imgL_np, ((0,0),(0,0),(top_pad,0),(0,left_pad)), mode='constant', constant_values=0)
    # test
    
    imgL = Variable(torch.FloatTensor(imgL).cuda())
    imgR = Variable(torch.FloatTensor(imgR).cuda())
    optimizer1.zero_grad()
    with torch.no_grad():
        model.eval()
        pred_disp, entropy = model(imgL, imgR)

    # resize to highres
    # pred_disp = cv2.resize(pred_disp/args.testres,(imgsize[1],imgsize[0]),interpolation=cv2.INTER_LINEAR)
    # clip while keep inf
    pred_disp = torch.squeeze(pred_disp).data.cpu().numpy()
    pred_disp = pred_disp[top_pad:,:pred_disp.shape[1]-left_pad]
    # 使用NumPy的logical_or函数检查无效值
    invalid = np.logical_or(pred_disp == np.inf, pred_disp != pred_disp)
    pred_disp[invalid] = np.nan
    loss1 = criterion2(left, right_transformed, pred_disp)  #重构
    # loss4 = criterion6(left, pred_disp)   #拿视差算的极线，肯定是错的，考虑能否和SURF/SIFT组合计算损失
    # loss5 = criterion5(pred_disp, left, right_transformed)  #极线
    
    # loss3 = criterion3(pred_disp, right_transformed)        #平滑
    from save_image_only import save_image_only
    min_val, max_val = np.nanpercentile(pred_disp, [0.5, 99.5])
    pred_disp = np.clip(pred_disp, min_val, max_val)
    pred_disp = (pred_disp - min_val) / (max_val - min_val)
    colored_disp = plt.cm.jet(pred_disp.squeeze())[:, :, :3] 
    # colored_disp = plt.cm.jet(pred_disp.squeeze().cpu().numpy())[:, :, :3]  # 只取RGB通道，去掉alpha通道
    colored_disp = torch.from_numpy(colored_disp).permute(2, 0, 1)     
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colored_disp = colored_disp.float() 
    colored_disp = colored_disp.to(device)
    loss2 = criterion1(left,colored_disp)
    save_image_only(pred_disp, f'%s/disp{idx}.jpg'% (folder_path), cmap='jet', save_cbar=True, save_mask=True)
    # left = left.detach().cpu().numpy()
    loss1 = alpha * loss1 + (1-alpha) * loss2
    loss1 = compute_epipolar_error(left, right_transformed)
    # loss1.requires_grad_(True)     
    # # freeze_specific_hourglass_modules(model, [8,16])#冻结1、3
    # loss1.backward()
    # optimizer1.step() 
    return loss1

def freeze_specific_hourglass_modules(model, modules_to_freeze):
    for module in modules_to_freeze:
        # 通过访问 model.module 来获取原始模型的属性
        for param in getattr(model.modules, f'disp_reg{module}').parameters():
            param.requires_grad = False

def get_data(l_img,r_img, pse_img):
    device = torch.device("cuda:{}".format(0))
    if l_img is not None:
        right_train_filelist = [r_img]
        left_train_filelist = [l_img]
        pse_rec_right_file = [pse_img]
    else:
        # left_train_filelist = glob.glob('/data2/zjq/program/imprecise_rectify/dataset/*/im0.png')
        # right_train_filelist = glob.glob('/data2/zjq/program/imprecise_rectify/dataset/*/im1.png')
        left_train_filelist = ['/data2/zjq/program/imprecise_rectify/dataset/001.jpg']
        right_train_filelist = ['/data2/zjq/program/imprecise_rectify/dataset/002.jpg']
        # left_train_filelist = ['/data2/zjq/program/DepthSensingBeyondLiDARRange-master/DepthSensingBeyondLiDARRange-master/high-res-stereo/data-mbtest/mid-pse2/im0.png']
        # right_train_filelist = ['/data2/zjq/program/DepthSensingBeyondLiDARRange-master/DepthSensingBeyondLiDARRange-master/high-res-stereo/data-mbtest/mid-pse2/im1.png']
        # pse_rec_right_file = ['/data2/zjq/program/CalibrationNet/Sample_Images/pse_rec/im1.png']       
        # pse_rec_left_file = ['/data2/zjq/program/CalibrationNet/Sample_Images/pse_rec/im0.png']
        # left_train_filelist = ['/data2/zjq/program/imprecise_rectify/resize_data/4/4_2024-10-23-162316-997_1729671796830181_YUYV.png']#棋盘格
        # right_train_filelist = ['/data2/zjq/program/imprecise_rectify/resize_data/6/6_2024-10-23-162316-960_1729671796830181_YUYV.png']
        # pse_rec_right_file = ['/data2/zjq/program/imprecise_rectify/ori_resize_data/6/6_2024-10-23-162316-960_1729671796830181_YUYV.png']
        # left_train_filelist = ['/data2/zjq/program/RAFT-Stereo/datasets/01/005/171714.389691.jpg']#办公室走廊，特征点少
        # right_train_filelist = ['/data2/zjq/program/RAFT-Stereo/datasets/01/006/171714.389691.jpg']
        # pse_rec_right_file = ['/data2/zjq/program/RAFT-Stereo/datasets/01/006/171714.389691.jpg'] 
        # left_train_filelist = ['/data2/zjq/program/RAFT-Stereo/datasets/02mid/4_2024-10-16-171819-252_1729070298239406_YUYV.png']#远距离大楼
        # right_train_filelist = ['/data2/zjq/program/RAFT-Stereo/datasets/02mid/6_2024-10-16-171819-898_1729070298772740_YUYV.png']
        # pse_rec_right_file =  ['/data2/zjq/program/RAFT-Stereo/datasets/02mid/6_2024-10-16-171819-898_1729070298772740_YUYV.png']
        # left_train_filelist = ['/data2/zjq/program/imprecise_rectify/short/left.png']
        # right_train_filelist = ['/data2/zjq/program/imprecise_rectify/short/right.png']
        # pse_rec_right_file = ['/data2/zjq/program/imprecise_rectify/short/right.png']
        # left_train_filelist = ['/data2/zjq/program/calib/data/short/undis0/174922775370.png']
        # right_train_filelist = ['/data2/zjq/program/calib/data/short/undis1/174922775370.png']
        # pse_rec_right_file = ['/data2/zjq/program/calib/data/short/undis1/174922775370.png']
        # left_train_filelist = ['/data2/zjq/program/calib/data/short/overlap0/174922775370.png']
        # right_train_filelist = ['/data2/zjq/program/calib/data/short/overlap1/174922775370.png']
        # pse_rec_right_file = ['/data2/zjq/program/calib/data/short/overlap1/174922775370.png']
        pse_rec_right_file = right_train_filelist  # Assuming pse_rec_right_file is the same as right_train_filelist

    patch_size = 256
    train_db = myImageloader(left_img_files=left_train_filelist, right_img_files=right_train_filelist, supervised=False,
                             train_patch_w=patch_size,
                             transform=transforms.Compose(
                                 [transforms.ToTensor()]),
                             label_transform=transforms.Compose([transforms.ToTensor()]), get_filelist=True)
    pse_rec = cv2.imread(pse_rec_right_file[0])
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((patch_size, patch_size)),  # 如果需要调整大小
        transforms.ToTensor()
    ])
    pse_rec_pil = Image.fromarray(pse_rec)
    grayscale_transform = transforms.Grayscale(num_output_channels=1)
    pse_rec_gray = grayscale_transform(pse_rec_pil)
    pse_rec_gray = transforms.ToTensor()(pse_rec_gray)
    pse_rec = pse_rec_gray.squeeze(1)
    pse_rec = transform(pse_rec)
    pse_rec = pse_rec.to(device)  # 增加批次维度并移动到GPU

    train_loader = torch.utils.data.DataLoader(train_db, batch_size=1, shuffle=True, num_workers=0)
    for batch_idx, (left, right, small_left, small_right) in enumerate(train_loader):
        left, right, small_left, small_right = left.to(device), right.to(device), small_left.to(device), small_right.to(device)
    return left, right, pse_rec

def main(l_img=None, r_img=None):
    global loss_list, ax_list, fig, pse_rec
    torch.cuda.empty_cache()
    device = torch.device("cuda:{}".format(0))
    # left_train_filelist = glob.glob('/data3/zjq/document/short_rectify/002/left/000524.png')#('/data2/zjq/program/RAFT-Stereo/datasets/1/*.png')    
    # right_train_filelist = glob.glob('/data3/zjq/document/short_rectify/002/right/000524.png')
    # pse_rec_right_file = glob.glob('/data3/zjq/document/short_rectify/002/right/000524.png')
    
    left_train_filelist = glob.glob('/data2/zjq/program/imprecise_rectify/dataset/*/im0.png')
    right_train_filelist = glob.glob('/data2/zjq/program/imprecise_rectify/dataset/*/im1_rectified.png')
    pse_rec_right_file = glob.glob('/data2/zjq/program/imprecise_rectify/dataset/*/im1.png')
    ind=0
    stereo_model2 = PSMNet(128, device=device, dfd_net=False, dfd_at_end=False, right_head=False)
    stereo_model2 = nn.DataParallel(stereo_model2)
    stereo_model2.to(device)
    # state_dict = torch.load('checkpoints/PSM/pretrained_model_KITTI2015.tar')
    # state_dict = torch.load('checkpoints/Dfd/checkpoint_257.pth.tar')
    state_dict = torch.load('checkpoints/demo_DPT/CP49.pth')
    stereo_model2.load_state_dict(state_dict['state_dict'], strict=False)
    # state_dict = torch.load('checkpoints/demo_cp/CP40.pth')
    # state_dict = torch.load('checkpoints/demo_cp/CP92.pth')

    # stereo_model2.load_state_dict(state_dict['theta_var'], strict=False)
    # stereo_model2.train()
    model = ConfigNet(stereo_model=stereo_model2, stn_mode='projective', ext_disp2depth=False, device=device).to(device)
    model.train()
    iii = 0
    for param in model.parameters():
        # iii+=1
        # if iii <= 50:
        #     continue
        param.requires_grad = True
    lr = 0.03
    num_of_epochs = 50
    optimizer = optim.SGD(model.parameters(), lr=lr)    
    plt.ion()
    fig, ax_list = plt.subplots(3, 2, figsize=(16, 9))
    fig.tight_layout()
    ax_list = ax_list.ravel()
    loss_list = list()
    dir_checkpoint = 'checkpoints/demo_cp'
    best_test_loss = 2.0
    loss_array = np.zeros(num_of_epochs, dtype=float)
    level = 1
    clean = -1
    #loadmodel = '/data2/zjq/program/DepthSensingBeyondLiDARRange-master/DepthSensingBeyondLiDARRange-master/high-res-stereo/final-768px.pth'
    loadmodel = '/data2/zjq/program/DepthSensingBeyondLiDARRange-master/DepthSensingBeyondLiDARRange-master/high-res-stereo/kitti.pth'
    disp_model = hsm(128, clean, level)
    disp_model = nn.DataParallel(disp_model, device_ids=[0])
    disp_model.cuda()
    if loadmodel is not None:
        pretrained_dict = torch.load(loadmodel)
        pretrained_dict['state_dict'] = {k: v for k, v in pretrained_dict['state_dict'].items() if 'disp' not in k}
        disp_model.load_state_dict(pretrained_dict['state_dict'], strict=False)
    else:
        print('run with random init')
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    for param in disp_model.parameters():
        param.requires_grad = True
    # '''
    wandb.init(
        # set the wandb project where this run will be logged
        # project="my-awesome-project",
        project="pse_rec",
        # track hyperparameters and run metadata
        config={
        "learning_rate": lr,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": num_of_epochs * len(left_train_filelist),
        }
    )
    wandb.watch(model, log="all")
    # '''
    '''
    wandb.init(
        project="DISP",
        config={
            "learning_rate": lr,
            "architecture": "CNN",
            "dataset": "CIFAR-100",
            "epochs": num_of_epochs,
        }
    )
    wandb.watch(disp_model, log="all")
    '''
    optimizer1 = optim.SGD(disp_model.parameters(), lr=lr)
    # optimizer1 = optim.Adam(disp_model.parameters(),lr = lr) 
    # optimizer1 = optim.Adam(disp_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    weight = 0.5
    # optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-5)
    for l_img, r_img, p_img in zip(left_train_filelist, right_train_filelist, pse_rec_right_file):     
        loss = 0               
        epoch = 1
        should_finish = False
        while not should_finish:       
            left, right, pse_rec = get_data(l_img, r_img, p_img)
            loss = train(model, disp_model, optimizer, optimizer1, weight, left, right, left.shape[3], left.shape[2])
            # loss_disp = disp_train(disp_model, optimizer1, left, right_transformed_np, left.shape[3], left.shape[2])
            # loss = loss * weight + loss_disp * (1 - weight)
            # loss = loss/len(left_train_filelist)
            if loss < best_test_loss:
                cp_file = os.path.join(dir_checkpoint, 'CP{}.pth'.format(idx))
                torch.save({'state_dict': model.state_dict()}, cp_file)
                print('Checkpoint {} saved !'.format(idx))
                best_test_loss = loss
            print(epoch, loss)           
            wandb.log({"loss": loss})
            should_finish = epoch == num_of_epochs
            epoch += 1        
        ind+=1
    wandb.finish()

if __name__ == '__main__':
    
    # print(torch.cuda.current_device())
    main()
