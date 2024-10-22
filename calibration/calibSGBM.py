import cv2
import numpy as np
import time
import random
import math
import os
import yaml
def matrix_to_string(matrix):
    return "\n".join([" ".join(map(str, row)) for row in matrix])

# -----------------------------------双目相机的基本参数---------------------------------------------------------
#   left_camera_matrix          左相机的内参矩阵
#   right_camera_matrix         右相机的内参矩阵
#
#   left_distortion             左相机的畸变系数    格式(K1,K2,P1,P2,0)
#   right_distortion            右相机的畸变系数
# -------------------------------------------------------------------------------------------------------------
# 左镜头的内参，如焦距
# left_camera_matrix = np.array([[479.55052148941473, 0.0, 348.994582059884],[0.0, 479.5208206701134, 249.50377047147833],[0.,0.,1.]])
# right_camera_matrix = np.array([[483.0792290729504, 0, 330.3141627786868],[0,483.09956102308314, 259.6327628978444],[0.,0.,1.]])

# # 畸变系数,K1、K2、K3为径向畸变,P1、P2为切向畸变
# left_distortion = np.array([[0.08217728251119759, 0.08680868940188831, 0.0009404912371921342, -0.0004196630166092444, -0.3535012081537213]])
# right_distortion = np.array([[0.07318857195438393, 0.08980421622019015, 0.00030315373024372916, 7.041304057446092e-05, -0.34956841142077094]])

# # 旋转矩阵
# R = np.array([[0.9999845563063311, -0.00047368126050488095, 0.00553739784498367],
#               [0.00048748258869173486, 0.9999967778843306, -0.0024913012026770284],
#               [-0.005536199920153326, 0.0024939621128205035, 0.9999815651517902]])
# # 平移矩阵
# T = np.array([-0.06653063457946741,-0.001695318529047398,0.0017436357141747513])

size = (640, 480)

# R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
#                                                                   right_camera_matrix, right_distortion, size, R,
#                                                                   T)
# 读取 YAML 文件
with open('calibration.yaml', 'r') as file:
    data = yaml.safe_load(file)

# 提取相机参数
left_camera_matrix = np.array(data['K1'])
right_camera_matrix = np.array(data['K2'])
left_distortion = np.array(data['D1'])
right_distortion = np.array(data['D2'])
R = np.array(data['R'])
T = np.array(data['T'])
R1 = np.array(data['R1'])
R2 = np.array(data['R2'])
P1 = np.array(data['P1'])
P2 = np.array(data['P2'])
Q = np.array(data['Q'])

# 立体校正
# R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
#     left_camera_matrix, left_distortion,
#     right_camera_matrix, right_distortion,
#     size, R, T
# )
# 校正查找映射表,将原始图像和校正后的图像上的点一一对应起来
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)

# --------------------------鼠标回调函数---------------------------------------------------------
#   event               鼠标事件
#   param               输入参数
# -----------------------------------------------------------------------------------------------
def onmouse_pick_points(event, x, y, flag, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        threeD = param
        print('\n像素坐标 x = %d, y = %d' % (x, y))
        print("世界坐标是：", threeD[y][x][0], threeD[y][x][1], threeD[y][x][2], "m")
        #print("世界坐标xyz 是：", threeD[y][x][0] / 1000.0, threeD[y][x][1] / 1000.0, threeD[y][x][2] / 1000.0, "m")

        distance = math.sqrt(threeD[y][x][0] ** 2 + threeD[y][x][1] ** 2 + threeD[y][x][2] ** 2)
        distance = distance# mm -> m
        print("距离是：", distance, "m")
#选取左右文件夹里的第一张照片来测试
def get_pic(train_dir2, train_dir3):
    pic_file2 = os.listdir(train_dir2)
    pic_data2 = cv2.imread(os.path.join(train_dir2, pic_file2[0]))
    pic_file3 = os.listdir(train_dir3)
    pic_data3 = cv2.imread(os.path.join(train_dir3, pic_file3[0]))
    #print(pic_file2,'+', pic_file3)
    return pic_data2, pic_data3

left_dir2 = r'data/005'
right_dir3 = r'data/006'
dst, dst2 = get_pic(left_dir2, right_dir3)
# 加载视频文件
capture = cv2.VideoCapture("./close_video.avi")
cv2.namedWindow('Deep disp', cv2.WINDOW_AUTOSIZE)

# 读取视频
fps = 0.0
ret, frame = capture.read()
while ret:
# while 1:
    # 开始计时
    t1 = time.time()
    # 切割为左右两张图片
    frame1 = frame[0:480, 0:640]
    frame2 = frame[0:480, 640:1280]
    # 将BGR格式转换成灰度图片，用于畸变矫正
    imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # #转换为灰度图后计算视差图
    # imgL = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    # imgR = cv2.cvtColor(dst2, cv2.COLOR_BGR2GRAY)

    # 重映射，就是把一幅图像中某位置的像素放置到另一个图片指定位置的过程。
    # 依据MATLAB测量数据重建无畸变图片,输入图片要求为灰度图
    # img1_rectified = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
    # img2_rectified = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)
    
    # ------------------------------------SGBM算法----------------------------------------------------------
    #   blockSize                   深度图成块，blocksize越低，其深度图就越零碎，0<blockSize<10
    #   img_channels                BGR图像的颜色通道，img_channels=3，不可更改
    #   numDisparities              SGBM感知的范围，越大生成的精度越好，速度越慢，需要被16整除，如numDisparities
    #                               取16、32、48、64等
    #   mode                        sgbm算法选择模式，以速度由快到慢为：STEREO_SGBM_MODE_SGBM_3WAY、
    #                               STEREO_SGBM_MODE_HH4、STEREO_SGBM_MODE_SGBM、STEREO_SGBM_MODE_HH。精度反之
    # ------------------------------------------------------------------------------------------------------
    blockSize = 9
    img_channels = 3
    stereo = cv2.StereoSGBM_create(minDisparity=0,
                                   numDisparities=112,
                                   blockSize=blockSize,
                                   P1=8 * img_channels * blockSize * blockSize,
                                   P2=32 * img_channels * blockSize * blockSize,
                                   disp12MaxDiff=-1,
                                   #preFilterCap=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,#阈值
                                   speckleRange=32,
                                   mode=cv2.STEREO_SGBM_MODE_HH)
    # 计算视差
    disparity = stereo.compute(imgL, imgR)

    # 归一化函数算法，生成深度图（灰度图）
    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 生成深度图（颜色图）
    dis_color = disparity
    dis_color = cv2.normalize(dis_color, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    dis_color = cv2.applyColorMap(dis_color, 2)

    # 计算三维坐标数据值
    #output = np.zeros_like(disparity)
    #threeD = cv2.reprojectImageTo3D(disparity,output, Q, handleMissingValues=True)#output输出三维坐标图像
    threeD = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
    # 计算出的threeD，需要乘以尺度因子(=focal length*baseline)，才等于现实中的距离
    threeD = threeD * 12
    

    # 鼠标回调事件
    cv2.namedWindow('depth', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("depth", onmouse_pick_points, threeD)

    #完成计时，计算帧率
    fps = (fps + (1. / (time.time() - t1))) / 2
    frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("depth", dis_color)
    cv2.imshow("left", frame1)
    cv2.imshow("Deep disp", disp)
    # cv2.waitKey()
    # exit()
    if cv2.waitKey(100) & 0xff == ord('q'):
        break
        # 是否读取到了帧，读取到了则为True
    ret, frame = capture.read()
    # dst, dst2 = get_pic(left_dir2, right_dir3)

# 释放资源
capture.release()

# 关闭所有窗口
cv2.destroyAllWindows()
