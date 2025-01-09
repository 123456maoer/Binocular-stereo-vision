import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.pytorch_ssim as pytorch_ssim
from PIL import Image
import time
import cv2
# import IPython, cv2

SSIM_WIN = 5


class WrappedModel(nn.Module):
	def __init__(self, module):
		super(WrappedModel, self).__init__()
		self.module = module # that I actually define.
	def forward(self, x):
		return self.module(x)


def gradient_xy(img):
    gx = img[ :, :-1] - img[ :, 1:]
    gy = img[ :-1, :] - img[ 1:, :]
    return gx, gy
def gradient_xy1(img):
    gx = img[:, :, :, :-1] - img[:, :, :, 1:]
    gy = img[:, :, :-1, :] - img[:, :, 1:, :]
    return gx, gy

def warp_disp(x, disp):
    # result + flow(-disp) = x
    # warp back to result
    # x = x.cuda()
    disp = disp.astype(np.float32)  # 确保数据类型兼容，例如使用 float32
    disp = torch.from_numpy(disp)  # 将 NumPy 数组转换为 PyTorch 张量
    disp = disp.cuda()  # 将 PyTorch 张量移动到 GPU 上
    N, _, H, W = x.shape

    x_ = torch.arange(W).view(1, -1).expand(H, -1)
    y_ = torch.arange(H).view(-1, 1).expand(-1, W)
    grid = torch.stack([x_, y_], dim=0).float()
    # if args.cuda:
    #     grid = grid.cuda()
    grid = grid.cuda()
    grid = grid.unsqueeze(0).expand(N, -1, -1, -1)
    grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (W - 1) - 1
    grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (H - 1) - 1
    # disp = 30*torch.ones(N, H, W).cuda()
    grid2 = grid.clone()
    grid2[:, 0, :, :] = grid[:, 0, :, :] + 2*disp/W
    grid2 = grid2.permute(0, 2, 3, 1)
    return F.grid_sample(x, grid2, padding_mode='zeros')


def Loss_prob(y, target, logvar):
    y = y*50
    target = target*50

    thresh = 10
    logvar = logvar.clamp(-50, 50)
    loss = 2**0.5 * F.smooth_l1_loss(y, target, reduction='none').mean(1) * (torch.exp(-logvar) ) + logvar
    loss = loss.clamp(-thresh, thresh).mean()
    return loss
def compute_epipolar_error(left, right_transformed):
    delta=0.5
    imgL_np = left.squeeze().cpu().permute(1, 2, 0).numpy()  # 转换为 [H, W, C]
    imgR_np = right_transformed.squeeze().cpu().permute(1, 2, 0).numpy()  # 转换为 [H, W, C]

    # 将图像数据类型转换为 uint8
    imgL_np = (imgL_np * 255).astype(np.uint8)
    imgR_np = (imgR_np * 255).astype(np.uint8)

    # 转换为灰度图像
    left_cv = cv2.cvtColor(imgL_np, cv2.COLOR_RGB2GRAY)
    right_cv = cv2.cvtColor(imgR_np, cv2.COLOR_RGB2GRAY)
    
    # # 将图像从Tensor转换为NumPy数组
    # left_np = left[0].permute(1, 2, 0).cpu().numpy()
    # right_transformed_np = right_transformed[0].permute(1, 2, 0).cpu().numpy()

    # # 将NumPy数组转换为OpenCV图像
    # left_cv = cv2.cvtColor(left_np, cv2.COLOR_RGB2BGR)
    # right_transformed_cv = cv2.cvtColor(right_transformed_np, cv2.COLOR_RGB2BGR)
    # 检测关键点和描述符
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(left_cv, None)
    kp2, des2 = orb.detectAndCompute(right_cv, None)

    # 匹配描述符
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # 提取匹配点
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # 计算基础矩阵
    F, mask = cv2.findFundamentalMat(pts1.reshape(-1, 2), pts2.reshape(-1, 2), cv2.FM_RANSAC)
    if mask is None:
        raise ValueError("Fundamental matrix could not be computed. Check the input points.")
    # 计算极线误差
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)

    error = 0
    for pt1, pt2, line1, line2 in zip(pts1, pts2, lines1, lines2):
        error += abs(line1[0] * pt1[0] + line1[1] * pt1[1] + line1[2])
        error += abs(line2[0] * pt2[0] + line2[1] * pt2[1] + line2[2])
    A = error/(len(pt1)+len(pt2))
    return A/(A+1)
# loss1
# appearance loss: the difference between reconstructed image and original image
def criterion1_normal(imgC, imgR, imgL, outputR, outputL, maxdisp, args, down_factor=1):
    if down_factor != 1:
        imgC = F.interpolate(imgC, scale_factor=1.0/down_factor, mode='bicubic')
        imgR = F.interpolate(imgR, scale_factor=1.0/down_factor, mode='bicubic')
        imgL = F.interpolate(imgL, scale_factor=1.0/down_factor, mode='bicubic')
        outputR = F.interpolate(outputR.unsqueeze(1), scale_factor=1.0/down_factor, mode='bicubic') / down_factor
        outputL = F.interpolate(outputL.unsqueeze(1), scale_factor=1.0/down_factor, mode='bicubic') / down_factor

        outputR = outputR.squeeze(1)
        outputL = outputL.squeeze(1)

    imgR2C = warp_disp(imgR, -outputR, args)
    imgL2C = warp_disp(imgL, outputL, args)
    imgR2C2 = warp_disp(imgR, -outputL, args)
    imgL2C2 = warp_disp(imgL, outputR, args)

    alpha2 = 0.85
    crop_edge = 200
    if imgC.shape[2] > SSIM_WIN:
        ssim_loss = pytorch_ssim.SSIM(window_size = SSIM_WIN)
    else:
        ssim_loss = pytorch_ssim.SSIM(window_size = imgC.shape[2])

    if crop_edge == 0:
        diff_ssim = (1 - ssim_loss(imgC, imgR2C)) / 2.0 + \
                    (1 - ssim_loss(imgC, imgL2C)) / 2.0 + \
                    (1 - ssim_loss(imgC, imgR2C2)) / 2.0 + \
                    (1 - ssim_loss(imgC, imgL2C2)) / 2.0
        diff_L1 = (F.smooth_l1_loss(imgC, imgR2C, reduction='mean')) + \
                  (F.smooth_l1_loss(imgC, imgL2C, reduction='mean')) + \
                  (F.smooth_l1_loss(imgC, imgR2C2, reduction='mean')) + \
                  (F.smooth_l1_loss(imgC, imgL2C2, reduction='mean'))
    else:
        diff_ssim = (1 - ssim_loss(imgC[:,:,:,crop_edge:], imgR2C[:,:,:,crop_edge:])) / 2.0 + \
                    (1 - ssim_loss(imgC[:,:,:,:-crop_edge], imgL2C[:,:,:,:-crop_edge])) / 2.0 + \
                    (1 - ssim_loss(imgC[:,:,:,crop_edge:], imgR2C2[:,:,:,crop_edge:])) / 2.0 + \
                    (1 - ssim_loss(imgC[:,:,:,:-crop_edge], imgL2C2[:,:,:,:-crop_edge])) / 2.0
        diff_L1 = (F.smooth_l1_loss(imgC[:,:,:,crop_edge:], imgR2C[:,:,:,crop_edge:], reduction='mean')) + \
                  (F.smooth_l1_loss(imgC[:,:,:,:-crop_edge], imgL2C[:,:,:,:-crop_edge], reduction='mean')) + \
                  (F.smooth_l1_loss(imgC[:,:,:,crop_edge:], imgR2C2[:,:,:,crop_edge:], reduction='mean')) + \
                  (F.smooth_l1_loss(imgC[:,:,:,:-crop_edge], imgL2C2[:,:,:,:-crop_edge], reduction='mean'))
    
    loss1 = 1.0/4 * (alpha2 * diff_ssim + (1-alpha2) * diff_L1)
    
    return loss1, imgR2C, imgL2C, imgC, outputR

def criterion1_2frame(imgC, imgR, outputR, maxdisp, args, down_factor=1):
    if down_factor != 1:
        imgC = F.interpolate(imgC, scale_factor=1.0/down_factor, mode='bicubic')
        imgR = F.interpolate(imgR, scale_factor=1.0/down_factor, mode='bicubic')
        outputR = F.interpolate(outputR.unsqueeze(1), scale_factor=1.0/down_factor, mode='bicubic') / down_factor
        
        outputR = outputR.squeeze(1)

    imgR2C = warp_disp(imgR, -outputR, args)

    alpha2 = 0.85
    crop_edge = 0
    if imgC.shape[2] > SSIM_WIN:
        ssim_loss = pytorch_ssim.SSIM(window_size = SSIM_WIN)
    else:
        ssim_loss = pytorch_ssim.SSIM(window_size = imgC.shape[2])

    if crop_edge == 0:
        diff_ssim = (1 - ssim_loss(imgC, imgR2C)) / 2.0
        diff_L1 = (F.smooth_l1_loss(imgC, imgR2C, reduction='mean'))
    else:
        diff_ssim = (1 - ssim_loss(imgC[:,:,:,crop_edge:], imgR2C[:,:,:,crop_edge:])) / 2.0
        diff_L1 = (F.smooth_l1_loss(imgC[:,:,:,crop_edge:], imgR2C[:,:,:,crop_edge:], reduction='mean'))
    
    loss1 = (alpha2 * diff_ssim + (1-alpha2) * diff_L1)
    
    return loss1, imgR2C
import os              
idx = 1
# loss2
# consistency loss the difference between left output and right output
def sav_gridsample(matrix):
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.cpu().numpy()
    elif isinstance(matrix, np.ndarray):
        pass
    else:
        raise TypeError("matrix must be a PyTorch Tensor or a NumPy ndarray")
    nan_mask = np.isnan(matrix)
    matrix[nan_mask] = np.nanmin(matrix)
    output_np = matrix[0].transpose(1, 2, 0) 
    current_file_path = os.path.abspath(__file__)
    # 获取当前文件所在的目录
    current_dir = os.path.dirname(current_file_path)
    # 获取父目录的路径
    parent_dir = os.path.dirname(current_dir)
    # 构造完整的图片保存路径
    image_path = os.path.join(parent_dir, f'figures/transform_r/image{idx}.png')
    Image.fromarray((output_np * 255).astype(np.uint8)).save(image_path)

def criterion1(L, R): #对左图和视差彩色图进行SSIM或smoothL1
    global idx
    alpha1 = 0
    alpha2=0.85
    tau = 10   
    # R = R.type(torch.float32).to('cuda')
    # 假设 R 是一个 NumPy 数组
    # R = torch.from_numpy(R).type(torch.float32).to('cuda')  # 如果你使用的是 PyTorch 1.x
    # # 或者如果你使用的是 PyTorch 0.4 或更高版本，可以使用以下方式：
    # R= torch.tensor(R, dtype=torch.float32, device='cuda')
    ssim_loss = pytorch_ssim.SSIM(window_size = SSIM_WIN)
    diff_ssim = (1 - ssim_loss(L, R)) /2
    
    L = L.squeeze(0)
    diff_L1 = (F.smooth_l1_loss(L, R, reduction='mean'))
    loss1 = (alpha2 * diff_ssim + (1-alpha2) * diff_L1)
    return loss1 
def criterion2(L, R, disp):#对左图进行视差移动，再和右图进行SSIMloss计算
    global idx
    alpha1 = 0
    alpha2=0.85
    tau = 10    # truncation for occluded region
    # L = np.reshape(L, (3, 720, 1280))
    # L = torch.from_numpy(L)
    L1 = warp_disp(L, disp)
    sav_gridsample(L1)
    idx = idx+1
    # R = torch.randn(10, 3)  # 假设这是目标值
    ssim_loss = pytorch_ssim.SSIM(window_size = SSIM_WIN)
    diff_ssim = (1 - ssim_loss(R, L1)) /2
    diff_L1 = (F.smooth_l1_loss(R, L1, reduction='mean',beta=1.5))
    loss1 = (alpha2 * diff_ssim + (1-alpha2) * diff_L1)
    # L1loss = F.smooth_l1_loss(L, R, reduction='none').clamp(min=alpha1, max=tau).mean()

    return loss1

# loss3
# smooth loss: force grident of intensity to be small
def criterion3(disp, R):
    img=warp_disp(R.cuda(), -disp)
    disp = torch.from_numpy(disp).cuda()
    # disp = disp.unsqueeze(1)
    disp_gx, disp_gy = gradient_xy(disp)
    
    disp_gx = torch.nn.functional.pad(disp_gx.unsqueeze(0), (0, 1), "constant", 0).squeeze(0)
    disp_gy = torch.nn.functional.pad(disp_gy.unsqueeze(0), (0, 0, 1, 0), "constant", 0).squeeze(0)
    # disp_gx = F.pad(disp_gx, (0, 1), "constant", 0)
    intensity_gx, intensity_gy = gradient_xy1(img)
    intensity_gx = F.pad(intensity_gx, (0, 1), "constant", 0)
    intensity_gy = F.pad(intensity_gy, (0, 0, 1, 0), "constant", 0)
    weights_x = torch.exp(-10 * torch.abs(intensity_gx).mean(1).unsqueeze(1))
    weights_y = torch.exp(-10 * torch.abs(intensity_gy).mean(1).unsqueeze(1))

    disp_gx = torch.abs(disp_gx)
    gx = disp_gx.clone()
    gx[gx>0.5] = disp_gx[disp_gx>0.5] + 10

    disp_gy = torch.abs(disp_gy)
    gy = disp_gy.clone()
    gy[gy>0.5] = disp_gy[disp_gy>0.5] + 10

    smoothness_x = gx * weights_x
    smoothness_y = gy * weights_y

    return smoothness_x.mean() + smoothness_y.mean()

# loss4
# regularization term: 
def criterion4(disp, maxdisp):
    # r1 = disp.mean()
    # r = torch.exp(-1 / 5.0 * disp) + torch.exp(1 / 5.0 * (disp - 90))
    # r = torch.exp(-1 / 5.0 * disp)
    r = (disp*2/maxdisp - 1).pow(2)
    return r.mean()

def criterion5(disp, imgL, imgR):
    delta=0.5
    imgL_np = imgL.squeeze().cpu().permute(1, 2, 0).numpy()  # 转换为 [H, W, C]
    imgR_np = imgR.squeeze().cpu().permute(1, 2, 0).numpy()  # 转换为 [H, W, C]

    # 将图像数据类型转换为 uint8
    imgL_np = (imgL_np * 255).astype(np.uint8)
    imgR_np = (imgR_np * 255).astype(np.uint8)

    # 转换为灰度图像
    imgL = cv2.cvtColor(imgL_np, cv2.COLOR_RGB2GRAY)
    imgR = cv2.cvtColor(imgR_np, cv2.COLOR_RGB2GRAY)

    # 初始化SIFT检测器
    sift = cv2.SIFT_create()

    # 检测关键点和描述符
    kp1, des1 = sift.detectAndCompute(imgL, None)
    kp2, des2 = sift.detectAndCompute(imgR, None)

    # 创建FLANN匹配器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 匹配描述符
    matches = flann.knnMatch(des1, des2, k=2)

    # 应用比率测试
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good_matches.append(m)
    img_matches = cv2.drawMatches(imgL, kp1, imgR, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # save_path = 'figures/seperate'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # file_name = 'matches.jpg'
    # full_path = os.path.join(save_path, file_name)
    # cv2.imwrite(full_path, img_matches)
    # print(f"图像已保存到 {full_path}")
    points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 设置纵坐标差异的最大允许值
    y_diff_threshold = 50
    x_diff_threshold = 500
    # 筛选纵坐标差异较小的匹配对
    filtered_good_matches = []
    for m, (p1, p2) in zip(good_matches, zip(points1, points2)):
        y_diff = abs(p1[0][1] - p2[0][1])
        x_diff = abs(p1[0][0] - p2[0][0])
        if y_diff <= y_diff_threshold and x_diff <= x_diff_threshold:
            filtered_good_matches.append(m)

    # 绘制筛选后的匹配结果
    img_filtered_matches = cv2.drawMatches(imgL, kp1, imgR, kp2, filtered_good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)    
    save_path = 'figures/seperate'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = 'filtered_good_matches.jpg'
    full_path = os.path.join(save_path, file_name)
    cv2.imwrite(full_path, img_filtered_matches)
    for i, match in enumerate(good_matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt
    
    # 将点转换为正确的格式
    points1 = points1.reshape(-1, 1, 2)
    points2 = points2.reshape(-1, 1, 2)

    # 计算基础矩阵
    F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_LMEDS)

    # 计算极线误差
    epipolar_errors = []
    for i in range(len(points1)):
        # 计算左图像点在右图像中的极线
        line = cv2.computeCorrespondEpilines(points1[i].reshape(1, 1, 2), 1, F)[0]
        a, b, c = line.flatten()  # 极线方程的系数
        x, y = points2[i][0][0], points2[i][0][1] 
        error = abs(a*x + b*y + c) / np.sqrt(a**2 + b**2) 
        epipolar_errors.append(error)
    errors = np.mean(np.abs(epipolar_errors))
    print("平均极线误差:", errors)
    is_small_error = np.abs(errors) < delta
    squared_loss = 0.5 * errors ** 2
    linear_loss = delta * (np.abs(errors) - 0.5 * delta)
    errors = errors/(errors+1)
    return errors
def criterion6(left_image, disp):    #disparity epipolar loss@@@@@@wrong
    height, width = left_image.shape[2],left_image.shape[3]
    points1 = []
    points2 = []
    for v in range(height):
        for u in range(width):
            disparity = disp[v, u]
            if disparity != 0 and disparity != np.inf:
                x_right = u + disparity
                if 0 <= x_right < width:
                    points1.append((u, v))
                    points2.append((x_right, v))
    points1 = np.array(points1)
    points2 = np.array(points2)
    points1 = points1.reshape(-1, 1, 2)
    points2 = points2.reshape(-1, 1, 2)
    F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_LMEDS)
    print(F)
    # epipolar_errors = []
    epipolar_errors = 0
    for i in range(len(points1)):
        # 计算极线
        line = cv2.computeCorrespondEpilines(points1[i].reshape(1, 1, 2), 1, F)[0]
        a, b, c = line.flatten()  # 极线方程的系数
        x, y = points2[i][0][0], points2[i][0][1] 
        error = abs(a*x + b*y + c) / np.sqrt(a**2 + b**2) 
        epipolar_errors+=error
    # mean_error = np.mean(np.abs(epipolar_errors))
    print("平均极线误差:", epipolar_errors)
    


def evaluate(model, imgL, imgC, imgR, gt, args, maxdisp):
    use_cuda = args.cuda
    # use_cuda = False
    height = imgL.shape[1]
    width = imgL.shape[2]
    pad_h = (height // 32 + 1) * 32
    pad_w = (width // 32 + 1) * 32
    imgL = np.reshape(imgL, [1, imgL.shape[0], imgL.shape[1], imgL.shape[2]])
    imgR = np.reshape(imgR, [1, imgR.shape[0], imgR.shape[1], imgR.shape[2]])
    if imgC is not None:
        imgC = np.reshape(imgC, [1, imgC.shape[0], imgC.shape[1], imgC.shape[2]])

    # pad to (M x 32, N x 32)
    top_pad = pad_h - imgL.shape[2]
    left_pad = pad_w - imgL.shape[3]
    imgL = np.lib.pad(imgL, ((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
    imgR = np.lib.pad(imgR, ((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
    if imgC is not None:
        imgC = np.lib.pad(imgC, ((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

    imgL = torch.from_numpy(imgL)
    imgR = torch.from_numpy(imgR)
    if imgC is not None:
        imgC = torch.from_numpy(imgC)

    # model.eval()

    if imgC is not None:
        # multiscopic mode
        imgC_rot = imgC.flip(2).flip(3)
        imgL_rot = imgL.flip(2).flip(3)

        if use_cuda:
            imgL, imgR, imgC, imgC_rot, imgL_rot = \
                    imgL.cuda(), imgR.cuda(), imgC.cuda(), imgC_rot.cuda(), imgL_rot.cuda()
        
        if args.model == 'stackhourglass':
            outputR, outputR_prob, _, _ = model(imgC, imgR, maxdisp)
            if args.cuda and (not use_cuda):
                outputR = outputR.cpu()
                outputR_prob = outputR_prob.cpu()
            outputL_rot, outputL_prob_rot, _, _ = model(imgC_rot, imgL_rot, maxdisp)
            outputL = outputL_rot.flip(1).flip(2)
            outputL_prob = outputL_prob_rot.flip(2).flip(3)
            if args.cuda and (not use_cuda):
                outputL = outputL.cpu()
                outputL_prob = outputL_prob.cpu()
        elif args.model == 'basic':
            outputR = model(imgC, imgR, maxdisp)
            outputL_rot = model(imgC_rot, imgL_rot)
            outputL = outputL_rot.flip(1).flip(2)

        mindisp = torch.min(torch.cat([outputR, outputL]), 0)[0]
        diff = (outputR - outputL).squeeze()
        outputR = outputR.squeeze()
        outputL = outputL.squeeze()
        outputR[diff>3] = mindisp[diff>3]

        disp = outputL
        disp = disp[top_pad:, :-left_pad]
    
    else:
        # stereo mode
        if use_cuda:
            imgL, imgR = imgL.cuda(), imgR.cuda()
        if args.model == 'stackhourglass':
            output, _, _, _ = model(imgL, imgR, maxdisp)
        elif args.model == 'basic':
            output = model(imgL, imgR, maxdisp)
        if args.cuda and (not use_cuda):
            output = output.cpu()
        disp = output.squeeze()[top_pad:, :-left_pad]

    gt = torch.from_numpy(gt).float()
    if(use_cuda): gt = gt.cuda()
    mask = (gt != 0)

    diff = torch.abs(disp[mask] - gt[mask])
    avgerr = torch.mean(diff)
    rms = torch.sqrt( (diff**2).mean() ) 
    bad05 = len(diff[diff>0.5])/float(len(diff))
    bad1 = len(diff[diff>1])/float(len(diff))
    bad2 = len(diff[diff>2])/float(len(diff))
    bad3 = len(diff[diff>3])/float(len(diff))

    return [avgerr.data.item(), rms.data.item(), bad05, bad1, bad2, bad3], disp.cpu().numpy()


def evaluate_kitti(model, imgL, imgR, gt_occ, gt_noc, args, maxd=160):
    height = imgL.shape[1]
    width = imgL.shape[2]
    maxdisp = maxd

    pad_h = (height / 32 + 1) * 32
    pad_w = (width / 32 + 1) * 32
    imgL = np.reshape(imgL, [1, imgL.shape[0], imgL.shape[1], imgL.shape[2]])
    imgR = np.reshape(imgR, [1, imgR.shape[0], imgR.shape[1], imgR.shape[2]])

    # pad to (M x 32, N x 32)
    top_pad = pad_h - imgL.shape[2]
    left_pad = pad_w - imgL.shape[3]
    imgL = np.lib.pad(imgL, ((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
    imgR = np.lib.pad(imgR, ((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

    imgL = torch.from_numpy(imgL)
    imgR = torch.from_numpy(imgR)

    # model.eval()
    
    if args.cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()

    if args.model == 'stackhourglass':
        output, _, _, _ = model(imgL, imgR, maxdisp)
    elif args.model == 'basic':
        output = model(imgL, imgR, maxdisp)

    disp = output.squeeze()[top_pad:, :-left_pad]

    if gt_noc.any() == None:
        return disp.cpu().numpy()

    gt_occ = torch.from_numpy(gt_occ).float()
    gt_noc = torch.from_numpy(gt_noc).float()
    if args.cuda:
        gt_noc = gt_noc.cuda()
        gt_occ = gt_occ.cuda()
    mask_occ = (gt_occ != 0)
    mask_noc = (gt_noc != 0)

    diff_occ = torch.abs(disp[mask_occ] - gt_occ[mask_occ])
    diff_noc = torch.abs(disp[mask_noc] - gt_noc[mask_noc])
    # bad3_occ = len(diff_occ[diff_occ>3])/float(len(diff_occ))
    # bad3_noc = len(diff_noc[diff_noc>3])/float(len(diff_noc))

    bad3_occ = torch.sum((diff_occ>3) & (diff_occ/gt_occ[mask_occ]>0.05)).float() / float(len(diff_occ))
    bad3_noc = torch.sum((diff_noc>3) & (diff_noc/gt_noc[mask_noc]>0.05)).float() / float(len(diff_noc))

    return [bad3_occ, bad3_noc], disp.cpu().numpy()


def predict(model, imgL, imgR, args, maxd):
    height = imgL.shape[1]
    width = imgL.shape[2]

    pad_h = (height / 32 + 1) * 32
    pad_w = (width / 32 + 1) * 32
    imgL = np.reshape(imgL, [1, imgL.shape[0], imgL.shape[1], imgL.shape[2]])
    imgR = np.reshape(imgR, [1, imgR.shape[0], imgR.shape[1], imgR.shape[2]])

    # pad to (M x 32, N x 32)
    top_pad = pad_h - imgL.shape[2]
    left_pad = pad_w - imgL.shape[3]
    imgL = np.lib.pad(imgL, ((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
    imgR = np.lib.pad(imgR, ((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

    imgL = torch.from_numpy(imgL)
    imgR = torch.from_numpy(imgR)

    # model.eval()

    if args.cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()
    
    if args.model == 'stackhourglass':
        output, _, _, _ = model(imgL, imgR, maxd)
        
    elif args.model == 'basic':
        output = model(imgL, imgR, maxd)

    disp = output.squeeze()[top_pad:, :-left_pad]

    return disp.cpu().numpy()
