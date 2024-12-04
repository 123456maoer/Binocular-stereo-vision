from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import glob
import models.multiloss
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from models.dfd import Dfd_net,psi_to_depth
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
# run = wandb_demo.init(
#     # Set the project where this run will be logged
#     project="my-awesome-project",
#     # Track hyperparameters and run metadata
#     config={
#         "learning_rate": 0.01,
#         "epochs": 50,
#     },
# )
def get_dirs(scene_name):
    if scene_name == 'Demo':
        right_train_dir = '/home/yotamg/ForDemo/Right'
        left_train_dir = '/home/yotamg/ForDemo/Left'
    elif scene_name == 'Dynamic':
        right_train_dir = '/home/yotamg/ForDynamic/Right'
        left_train_dir = '/home/yotamg/ForDynamic/Left'
    elif scene_name == 'Outdoor_one':
        # Outdoor one image
        right_train_dir = '/media/yotamg/Yotam/Stereo/Outdoor_one/Right'
        left_train_dir = '/media/yotamg/Yotam/Stereo/Outdoor_one/Left'
    elif scene_name == 'Outdoor_one_rectified':
        # Outdoor one image
        right_train_dir = '/media/yotamg/Yotam/Stereo/Outdoor_one/rectified/Right'
        left_train_dir = '/media/yotamg/Yotam/Stereo/Outdoor_one/rectified/Left'
    elif scene_name == 'Outdoor_one_2':
        # Outdoor one image
        right_train_dir = '/media/yotamg/Yotam/Stereo/Outdoor_one_2/Right'
        left_train_dir = '/media/yotamg/Yotam/Stereo/Outdoor_one_2/Left'
    return right_train_dir, left_train_dir


right_train_dir, left_train_dir = get_dirs('Outdoor_one_rectified')

# right_train_filelist = [os.path.join(right_train_dir, img) for img in os.listdir(right_train_dir) if img.endswith('.png') or img.endswith('.tif') or img.endswith('.bmp')]
# left_train_filelist = [img.replace(right_train_dir, left_train_dir).replace('R', 'L') for img in right_train_filelist]


fig_output_dir = '/home/yotamg/PycharmProjects/affine_transform/figures/calibration_images/'

def get_small_mono(mono_out, device):
    mono_out_normalized = (mono_out - torch.min(mono_out)) / (torch.max(mono_out) / torch.min(mono_out))
    mono_out_small = transforms.ToPILImage()(mono_out_normalized[0].cpu())
    mono_out_small = mono_out_small.resize((256, 256), resample=Image.LANCZOS)
    mono_out_small = transforms.ToTensor()(mono_out_small).to(device)
    mono_out = ((mono_out_small) * (torch.max(mono_out) / torch.min(mono_out)) + torch.min(mono_out))
    return mono_out

# def get_mask(stereo_out, right_transformed):
#     mono_mask = (stereo_out > 0.5) & (stereo_out < 4.5)
#     print("right train shape",right_transformed.shape)
#     mask = (right_transformed != 0)[:, 0, :, :]
#     mask = mask & mono_mask
#     nan_mask = ~torch.isnan(right_transformed)
#     nan_mask = nan_mask.to(mask.dtype)
#     if mask.dtype == torch.float32:
#         mask = (mask > 0)  
#     mask = mask & nan_mask
#     return mask
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

def show_depth_maps(left, right_transformed, mono, stereo_rect, stereo, blocking=True):
    global im_cnt, fig, ax_list, loss_list
    vmin = torch.min(stereo_rect).item()
    vmax = torch.max(stereo_rect).item()
    folder_path = 'figures/seperate'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    if im_cnt == 0:
        ax_list[0].imshow(left[0].permute(1,2,0).detach().cpu())
        plt.setp(ax_list[0].get_xticklabels(), visible=False)
        plt.setp(ax_list[0].get_yticklabels(), visible=False)
        ax_list[0].tick_params(axis='both', which='both', length=0)
        ax_list[0].set_title('Left Image')

        ax_list[1].imshow(right_transformed[0].permute(1,2,0).detach().cpu(), cmap='jet', vmin=vmin, vmax=vmax)
        plt.setp(ax_list[1].get_xticklabels(), visible=False)
        plt.setp(ax_list[1].get_yticklabels(), visible=False)
        ax_list[1].tick_params(axis='both', which='both', length=0)
        ax_list[1].set_title('Right (transformed) Image')

        ax_list[2].imshow(mono[0].detach().cpu(), cmap='jet', vmin=vmin, vmax=vmax)
        plt.setp(ax_list[2].get_xticklabels(), visible=False)
        plt.setp(ax_list[2].get_yticklabels(), visible=False)
        ax_list[2].tick_params(axis='both', which='both', length=0)
        ax_list[2].set_title('Monocular Depth Map')

        ax_list[3].imshow(stereo_rect[0].detach().cpu(),cmap='jet', vmin=vmin, vmax=vmax)
        plt.setp(ax_list[3].get_xticklabels(), visible=False)
        plt.setp(ax_list[3].get_yticklabels(), visible=False)
        ax_list[3].tick_params(axis='both', which='both', length=0)
        ax_list[3].set_title('Stereo Before Calibration')

        ax_list[4].imshow(stereo[0].detach().cpu(),cmap='jet', vmin=vmin, vmax=vmax)
        plt.setp(ax_list[4].get_xticklabels(), visible=False)
        plt.setp(ax_list[4].get_yticklabels(), visible=False)
        ax_list[4].tick_params(axis='both', which='both', length=0)
        ax_list[4].set_title('Stereo After Calibration')
        
        loss_list_cpu = [loss_tensor.detach().cpu().numpy() for loss_tensor in loss_list]
        ax_list[5].plot(loss_list_cpu, 'b')

        # ax_list[5].plot(loss_list, 'b')
        ax_list[5].set_title('Train Loss (Mono vs. Stereo)')
        plt.setp(ax_list[5].get_xticklabels(), visible=False)
        plt.setp(ax_list[5].get_yticklabels(), visible=False)
        ax_list[5].tick_params(axis='both', which='both', length=0)
        for i, ax in enumerate(ax_list):
            fig = plt.figure()
            ax.figure = fig
            ax.set_position([0, 0, 1, 1])
            fig.savefig(f'figures/seperate/output_subfigure_{i}.png', bbox_inches='tight', dpi=300)
            plt.close(fig)  # 关闭临时创建的 Figure 对象
    else:
        fig1 = plt.figure(figsize=(1280/16,720/9))
        ax1 = fig1.add_subplot(111)        
        ax_list[1].imshow(right_transformed[0].permute(1, 2, 0).detach().cpu())
        image_data1 = ax_list[1].images[0].get_array()
        plt.imsave(f'figures/seperate/output_image{im_cnt}.png',image_data1)
        ax1.imshow(image_data1, cmap='jet', vmin=vmin, vmax=vmax)  
        ax1.axis('off')
        plt.savefig(f'figures/seperate/ax_list1{im_cnt}.png',bbox_inches='tight', pad_inches=0)
        plt.close(fig1)

        
        fig4 = plt.figure()
        ax4 = fig4.add_subplot(111)
        image_data4 = ax_list[4].images[0].get_array()
        ax4.imshow(image_data4, cmap='jet', vmin=vmin, vmax=vmax)  # 假设你已经有了vmin和vmax的值
        ax4.set_xticks([])  # 隐藏x轴刻度
        ax4.set_yticks([])  # 隐藏y轴刻度
        # 保存图像
        plt.savefig('figures/seperate/ax_list4'+str(im_cnt)+'.png', bbox_inches='tight')
        # 关闭新创建的图形
        plt.close(fig4)
        ax_list[4].imshow(stereo[0].detach().cpu(), cmap='jet', vmin=vmin, vmax=vmax)
        ax_list[1].imshow(right_transformed[0].permute(1, 2, 0).detach().cpu())
        # loss_list_cpu = [loss_tensor.detach().cpu().numpy() for loss_tensor in loss_list]
        # ax_list[5].plot(loss_list_cpu, 'b')
    if blocking:
        fig.canvas.draw()
        plt.ioff()
        plt.show()
    else:
        fig.canvas.draw()
    plt.close()
    # plt.savefig('figures/tmp/dfd_'+str(im_cnt)+'.png')
    # plt.close()
    # plt.pause(0.0001)
    im_cnt += 1
    # plt.show()

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
            mono_out_unet = psi_to_depth(mono_out_unet, focal_point=1.5)
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

def train(model, stereo_model, disp_model, optimizer, small_left, small_right, left, right):
    global fig, ax_list, loss_list,idx,pse_rec,right_transformed_np
    model.train()
    stereo_model.eval()
    # disp_model.eval()
    optimizer.zero_grad()
    stereo_out, theta, right_transformed = model(left, right)

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
    plt.show()  # 显示图像
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
    loss = F.l1_loss(stereo_out[mask],pse_rec[mask])    
    loss_list.append(loss)    # show_depth_maps(left, right_transformed, mono_out_for_train, stereo_unrect, stereo_out)
    loss.backward()
    optimizer.step()    
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

def disp_train(model,optimizer1, left, right_transformed):
    from torch.autograd import Variable
    global pred_disp
    max_disp = 400
    tmpdisp = int(max_disp//64*64)
    if (max_disp /64*64) > tmpdisp:
        model.module.maxdisp = tmpdisp + 64
    else:
        model.module.maxdisp = tmpdisp
    if model.module.maxdisp ==64: model.module.maxdisp=128
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
    imgL = np.reshape(imgL, (3, 720, 1280))
    imgR = np.reshape(imgR, (3, 720, 1280))
    imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
    imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])    
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

    from save_image_only import save_image_only
    min_val, max_val = np.nanpercentile(pred_disp, [0.5, 99.5])
    pred_disp = np.clip(pred_disp, min_val, max_val)
    pred_disp = (pred_disp - min_val) / (max_val - min_val)
    colored_disp = plt.cm.jet(pred_disp.squeeze())[:, :, :3] 
    # colored_disp = plt.cm.jet(pred_disp.squeeze().cpu().numpy())[:, :, :3]  # 只取RGB通道，去掉alpha通道
    colored_disp = torch.from_numpy(colored_disp).permute(2, 0, 1) 
    save_image_only(pred_disp, f'%s/disp{idx}.jpg'% (folder_path), cmap='jet', save_cbar=True, save_mask=True)
    # left = left.detach().cpu().numpy()
    loss1 = criterion1(left,colored_disp)
    # loss1 = criterion2(left, pred_disp)
    loss1.requires_grad_(True) 
    loss1.backward()
    optimizer1.step()  
    return loss1


def main(l_img=None, r_img=None):
    global loss_list,ax_list,fig,pse_rec
    torch.cuda.empty_cache()
    #device = torch.device('cuda:0')    
    device=torch.device("cuda:{}".format(0))
    print(device)
    midas_model = None
    dfd_net = None
    unet = None
    mono_net = 'phase-mask'
    # mono_net = 'midas'
    #mono_net = 'unet  '
    stereo_model = PSMNet(128, device=device, dfd_net=False, dfd_at_end=False, right_head=False)
    stereo_model = nn.DataParallel(stereo_model)
    stereo_model.to(device)
    # stereo_model.cuda()
    state_dict = torch.load('checkpoints/PSM/pretrained_model_KITTI2015.tar')
    stereo_model.load_state_dict(state_dict['state_dict'], strict=False)
    stereo_model.train()

    stereo_model2 = PSMNet(128, device=device, dfd_net=False, dfd_at_end=False, right_head=False)
    stereo_model2 = nn.DataParallel(stereo_model2)
    # stereo_model2.cuda()
    stereo_model2.to(device)
    state_dict = torch.load('checkpoints/PSM/pretrained_model_KITTI2015.tar')
    stereo_model2.load_state_dict(state_dict['state_dict'], strict=False)
    stereo_model2.train()

    if mono_net == 'phase-mask':
        dfd_net = Dfd_net(mode='segmentation', target_mode='cont', pool=False)
        dfd_net = dfd_net.eval()
        dfd_net = dfd_net.to(device)
        load_model(dfd_net, device, model_path='checkpoints/Dfd/checkpoint_257.pth.tar')

    elif mono_net == 'midas':
        # load network
        midas_model_path = 'checkpoints/Midas/model.pt'
        midas_model = MonoDepthNet(midas_model_path)
        midas_model.to(device)
        midas_model.eval()
    else:
        unet = get_Unet('models/unet/CP100_w_noise.pth', device=device)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    if l_img is not None:
        right_train_filelist = [r_img]
        left_train_filelist = [l_img]
    else:       
        # left_train_filelist = ['/data2/zjq/program/CalibrationNet/Sample_Images/undistort/im0.png']
        # right_train_filelist = ['/data2/zjq/program/CalibrationNet/Sample_Images/undistort/im1.png']
        # left_train_filelist = ['/data2/zjq/program/DepthSensingBeyondLiDARRange-master/DepthSensingBeyondLiDARRange-master/high-res-stereo/data-mbtest/mid-pse2/im0.png']
        # right_train_filelist = ['/data2/zjq/program/DepthSensingBeyondLiDARRange-master/DepthSensingBeyondLiDARRange-master/high-res-stereo/data-mbtest/mid-pse2/im1.png']
        # pse_rec_left_file = ['/data2/zjq/program/CalibrationNet/Sample_Images/pse_rec/im0.png']
        left_train_filelist = ['/data2/zjq/program/imprecise_rectify/resize_data/4/4_2024-10-23-162316-997_1729671796830181_YUYV.png']
        right_train_filelist = ['/data2/zjq/program/imprecise_rectify/resize_data/6/6_2024-10-23-162316-960_1729671796830181_YUYV.png']
        pse_rec_right_file = ['/data2/zjq/program/imprecise_rectify/ori_resize_data/6/6_2024-10-23-162316-960_1729671796830181_YUYV.png']
        # pse_rec_right_file = ['/data2/zjq/program/CalibrationNet/Sample_Images/pse_rec/im1.png']
    patch_size = 256
    train_db = myImageloader(left_img_files=left_train_filelist, right_img_files=right_train_filelist, supervised=False,
                            train_patch_w=patch_size,
                            transform=transforms.Compose(
                                [transforms.ToTensor()]),
                            label_transform=transforms.Compose([transforms.ToTensor()]), get_filelist=True)
    # pse_rec = cv2.imread(pse_rec_file)
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
    # 去除颜色通道这一维，得到 [1, 256, 256] 形状的张量
    pse_rec = pse_rec_gray.squeeze(1)
    pse_rec = transform(pse_rec)
    pse_rec = pse_rec.to(device)  # 增加批次维度并移动到GPU
    
    train_loader = torch.utils.data.DataLoader(train_db, batch_size=1, shuffle=True, num_workers=0)
    # model = Net(stereo_model=stereo_model).to(device)
    model = ConfigNet(stereo_model=stereo_model2, stn_mode='projective', ext_disp2depth=False, device=device).to(device)
    lr = 0.00005
    # if mono_net == 'midas':
    #     # lr = 0.0001
    #     lr = 0.0001
    # else:
    #     lr = 0.0001
    num_of_epochs = 50
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for param in model.stereo_model.parameters():
        param.requires_grad = True
    for batch_idx, (left, right, small_left, small_right) in enumerate(train_loader):
        left, right, small_left, small_right = left.to(device), right.to(device), small_left.to(device), small_right.to(device)

    plt.ion()
    fig, ax_list = plt.subplots(3, 2, figsize=(16,9))
    fig.tight_layout()
    ax_list = ax_list.ravel()
    loss_list = list()  
    # stereo_unrect = get_mono_and_unrect_stereo(stereo_model, mono_net, midas_model, left, small_left, unet, dfd_net, device, small_right)
    # print("small left: ",small_left.shape, small_right.shape)

    dir_checkpoint = 'checkpoints/demo_cp'
    best_test_loss = 2.0
    epoch = 1
    should_finish = False
    start_over = False
    loss_array = np.zeros(num_of_epochs, dtype=float)
    '''
    wandb.init(
        # set the wandb project where this run will be logged
        # project="my-awesome-project",
        project="pse_rec",
        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 50,
        }
    )
    wandb.watch(model, log="all")
'''
    import argparse
    parser = argparse.ArgumentParser(description='HSM')
    parser.add_argument('--datapath', default='./data-mbtest',
                        help='test data path')
    parser.add_argument('--loadmodel', default='final-768px.pth',
                        help='model path')
    parser.add_argument('--outdir', default='mboutput/1',
                        help='output dir')
    parser.add_argument('--clean', type=float, default=-1,
                        help='clean up output using entropy estimation')
    parser.add_argument('--testres', type=float, default=1,
                        help='test time resolution ratio 0-x')
    parser.add_argument('--max_disp', type=float, default=400,
                        help='maximum disparity to search for')
    parser.add_argument('--level', type=int, default=1,
                        help='output level of output, default is level 1 (stage 3),\
                            can also use level 2 (stage 2) or level 3 (stage 1)')
    args = parser.parse_args()
    # left_file = '/data2/zjq/program/DepthSensingBeyondLiDARRange-master/DepthSensingBeyondLiDARRange-master/high-res-stereo/data-mbtest/mid-pse2/im0.png'
    # right_file = '/data2/zjq/program/DepthSensingBeyondLiDARRange-master/DepthSensingBeyondLiDARRange-master/high-res-stereo/data-mbtest/mid-pse2/im1.png'
    left_file = '/data2/zjq/program/imprecise_rectify/resize_data/4/4_2024-10-23-162316-997_1729671796830181_YUYV.png'
    right_file = '/data2/zjq/program/imprecise_rectify/resize_data/6/6_2024-10-23-162316-960_1729671796830181_YUYV.png'
    import skimage.io
    imgL_o = (skimage.io.imread(left_file).astype('float32'))[:,:,:3]
    imgR_o = (skimage.io.imread(right_file).astype('float32'))[:,:,:3]

    args.datapath = './data-mbtest'
    args.loadmodel = '/data2/zjq/program/DepthSensingBeyondLiDARRange-master/DepthSensingBeyondLiDARRange-master/high-res-stereo/final-768px.pth'
    args.outdir = 'mboutput/1'
    args.clean = -1
    args.testres = 1
    args.max_disp=300
    args.level=2
    disp_model = hsm(128,args.clean,level=args.level)
    disp_model = nn.DataParallel(disp_model, device_ids=[0])
    disp_model.cuda()
    if args.loadmodel is not None:
        pretrained_dict = torch.load(args.loadmodel)
        pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items() if 'disp' not in k}
        disp_model.load_state_dict(pretrained_dict['state_dict'],strict=False)
    else:
        print('run with random init')
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    from torch.autograd import Variable
    processed = get_transform()
    # '''
    wandb.init(
        # set the wandb project where this run will be logged
        # project="my-awesome-project",
        project="DISP",
        # track hyperparameters and run metadata
        config={
        "learning_rate": lr,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 50,
        }
    )
    wandb.watch(disp_model, log="all")      
    # '''
  
    # optimizer1 = optim.SGD(disp_model.parameters(),lr = lr)    
    optimizer1 = optim.Adam(disp_model.parameters(),lr = lr) 
    weight = 0.2#右目变换损失的权重
    
    #DISP:
    # multip = 48
    # imgL = np.zeros((1,3,24*multip,32*multip))
    # imgR = np.zeros((1,3,24*multip,32*multip))
    # imgL = Variable(torch.FloatTensor(imgL).cuda())
    # imgR = Variable(torch.FloatTensor(imgR).cuda())    
    # # optimizer1 = optim.Adam(disp_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    # with torch.no_grad():
    #     disp_model.eval()
    #     pred_disp, entropy = disp_model(imgL, imgR)
    # imgL = processed(imgL_o).numpy()
    # imgR = processed(imgR_o).numpy()   
    while not should_finish:
        print(epoch)
        loss = train(model, stereo_model, disp_model, optimizer, small_left, small_right, left, right)
        loss_disp = disp_train(disp_model,optimizer1, left, right_transformed_np)
        # disp_train(disp_model, left, right)
        loss = loss*weight + loss_disp * (1-weight)
        # disp_train(disp_model, imgL, imgR)
        if loss < best_test_loss:
            cp_file = os.path.join(dir_checkpoint, 'CP{}.pth'.format(epoch))
            torch.save(model.state_dict(),cp_file)
            print('Checkpoint {} saved !'.format(epoch))
            best_test_loss = loss
        # wandb.log({"loss": loss})
        wandb.log({"loss":loss_disp})
        epoch += 1
        should_finish = epoch == num_of_epochs
    # show_best_calibration(cp_file, model, small_left, small_right, mono_net, left, epoch)
    plt.plot(loss_array, marker='o')
    plt.title('Line Plot of Zeros Array')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend(['Loss Array'])
    save_path = 'figures/seperate/loss_line_plot.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()      

# [optional] finish the wandb run, necessary in notebooks
    wandb.finish()

if __name__ == '__main__':
    
    # print(torch.cuda.current_device())
    main()