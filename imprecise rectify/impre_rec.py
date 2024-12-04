import cv2
import numpy as np
import glob
import os
import yaml
from PIL import Image

# image_width, image_height = 3840, 2160
#image_width, image_height = 768,432
image_width, image_height = 1280, 720
dim = (image_width, image_height) 
def matrix_to_string(matrix):
    return "\n".join([" ".join(map(str, row)) for row in matrix])
def calibrate_camera(images,w, h, size):
    objp = np.zeros((w * h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    objp=size*objp
    objpoints = [] 
    imgpoints = []  
    size = tuple()
    flag=0
    for fname in images:
        img = cv2.imread(fname)
        flag +=1
        if flag % 5 !=0:
            continue
                                    # Resize the image to fit the screen 
        # scale_percent = 25  # percent of original size
        # width = int(w * scale_percent / 150)
        # height = int(image_height * scale_percent / 150)
        # dim = (width, height)        
        # Resize image
        #img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)  
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   
        size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
        if flag > 200:
            break
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria) 
            imgpoints.append(corners2)
            print(flag, fname, "success\n")
        
            objpoints.append(objp)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, size, None, None)
    with open("E:/calib/cameradata.txt", "a") as file:
        file.write("RET:\n"+str(ret)+"\n")
        file.write("K:\n")
        file.write(matrix_to_string(mtx) + "\n\n")
        file.write("D:\n")
        file.write(matrix_to_string(dist) + "\n\n")
    return mtx, dist
def stereo_calibrate(images1, images2, image_width, image_height, points_per_row, points_per_col, K1, D1, K2, D2, size):
    image_points_l = []
    image_points_r = []
    obj_points = []
    objp = np.zeros((points_per_row * points_per_col, 3), np.float32)
    objp[:, :2] = np.mgrid[0:points_per_row, 0:points_per_col].T.reshape(-1, 2)
    objp = size * objp
    flag = 0
    for img_path1, img_path2 in zip(images1, images2):
        flag += 1
        left_img = cv2.imread(img_path1)
        right_img = cv2.imread(img_path2)
        left_img = cv2.resize(left_img, dim, interpolation=cv2.INTER_AREA)  
        right_img = cv2.resize(right_img, dim, interpolation=cv2.INTER_AREA)
        # cv2.imshow('left', left_img)
        # cv2.imshow('right', right_img)
        # cv2.waitKey(0)
        gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        ret_l, corners_l = cv2.findChessboardCorners(gray_left, (points_per_row, points_per_col), None)
        ret_r, corners_r = cv2.findChessboardCorners(gray_right, (points_per_row, points_per_col), None)
        if ret_l and ret_r:
            corners_l2 = cv2.cornerSubPix(gray_left, corners_l, (11, 11), (-1, -1), 
                                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
            corners_r2 = cv2.cornerSubPix(gray_right, corners_r, (11, 11), (-1, -1), 
                                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
            obj_points.append(objp)
            image_points_l.append(corners_l2)
            image_points_r.append(corners_r2)
            print(flag, "Found corners in", img_path1, "and", img_path2)
            #obj_points.append([(point[0][0] * square_size, point[0][1] * square_size, 0) for point in np.mgrid[0:points_per_col, 0:points_per_row].reshape(2, -1).transpose()])
        # ˫Ŀ�궨
    ret, camera_matrix_l, dist_coeffs_l, camera_matrix_r, dist_coeffs_r, R, T, E, F = cv2.stereoCalibrate(obj_points, image_points_l, image_points_r,  K1, D1, K2, D2, (image_width, image_height), None, cv2.CALIB_FIX_INTRINSIC)
    print(camera_matrix_l, dist_coeffs_l, camera_matrix_r, dist_coeffs_r)
    print("R:\n", R)
    print("T:\n", T)
    # print("E:\n", E)
    # print("F:\n", F)
    with open("cameradata.txt", "a") as file:
        # file.write(ret+"\n")
        # file.write("camera_matrix_l:\n")
        # file.write(matrix_to_string(camera_matrix_l) + "\n\n")
        # file.write("dist_coeffs_l:\n")
        # file.write(matrix_to_string(dist_coeffs_l) + "\n\n")    
        # file.write("camera_matrix_r:\n")
        # file.write(matrix_to_string(camera_matrix_r) + "\n\n")
        # file.write("dist_coeffs_r:\n")
        # file.write(matrix_to_string(dist_coeffs_r) + "\n\n")
        file.write("R:\n")
        file.write(matrix_to_string(R) + "\n\n")
        file.write("T:\n")
        file.write(matrix_to_string(T) + "\n\n")
        file.write("E:\n")
        file.write(matrix_to_string(E) + "\n\n")
        file.write("F:\n")
        file.write(matrix_to_string(F) + "\n")
    return R, T
def undistort_images(image, camera_matrix, dist_coeffs):
    newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (image.shape[1], image.shape[0]), 1, (image.shape[1], image.shape[0]))
    undistorted_img = cv2.undistort(image, camera_matrix, dist_coeffs, None, newcameramatrix)
    return undistorted_img

def save_calibration_parameters(filename, K1, K2, D1, D2, R1, R2, P1, P2, Q, R, T):
    calibration_data = {
        'K1': K1.tolist(),
        'K2': K2.tolist(),
        'D1': D1.tolist(),
        'D2': D2.tolist(),
        'R1': R1.tolist(),        
        'R2': R2.tolist(),
        'R': R.tolist(),
        'T': T.tolist(),
        'P1': P1.tolist(),
        'P2': P2.tolist(),
        'Q': Q.tolist()
    }
    with open(filename, 'w') as file:
        yaml.dump(calibration_data, file)
def stereo_rectify(K1, D1, K2, D2, img_size, R, T):
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        K1, D1, K2, D2, img_size, R, T
    )
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_size, cv2.CV_32FC1)
    print(validPixROI1, validPixROI2)
    save_calibration_parameters('calibmid.yaml', K1, K2, D1, D2, R1, R2, P1, P2, Q, R, T)
    return map1x, map1y, map2x, map2y, P1, P2, Q

def main():
    # image_width, image_height = 1280,720
    image_width, image_height = 3840, 2160
    # w, h = 9, 6    
    w,h = 8,5
    # square_size = 0.1 #meter
    square_size = 0.054  #meter
    images1 = glob.glob('E:/calib/data/new/2024-10-23clearchess/4/4_2024-10-23-162544-534_1729671944396874_YUYV.bmp')
    images2 = glob.glob('E:/calib/data/new/2024-10-23clearchess/6/6_2024-10-23-162544-573_1729671944396874_YUYV.bmp')   
    file_path1 = glob.glob('ori_data/4/*.png')
    file_path2 = glob.glob('ori_data/6/*.png') 
    K1 = np.array([[7229.42402686822, 0.0, 1873.80377996716], 
                   [0.0, 7165.40168964290, 576.123339380286], 
                   [0.0, 0.0, 1.0]])
    D1 = np.array([[-0.642987432438166,6.142612560470050,0.021377426123003,-0.007620805444321,-46.837448090089980]])
    K2 = np.array([[7321.83549959599, 0.0, 1803.55138995725],
                     [0.0,7295.37342601286,710.306725057748],
                     [0.0, 0.0, 1.0]])
    D2 = np.array([[-0.139806438615313,-7.151026228368141,0.016417022030518,-0.014427938801353,48.007696967843330]])

    R = np.array([[0.999978890835931, -0.000869279306927973, -0.00643911764357976],
                    [0.00101354381307824, 0.999747787333197, 0.0224351164765358],  
                    [0.00641799123404545, -0.0224411692178306, 0.999727564545790]])    
    # R=R.T
    T = np.array([-1.568785941155917e+02,9.498546206358995,87.221697879758110])
    dim = (640, 480) 
    sav1 = "undistorted_data/4/"
    sav2 = "undistorted_data/6/"
    # for fname1, fname2 in zip(file_path1, file_path2):
    # #for fname in images1:
    #     img1 = cv2.imread(fname1)
    #     img2 = cv2.imread(fname2)
    #     # img1 = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA) 
    #     # img2 = cv2.resize(img2, dim, interpolation=cv2.INTER_AREA)
    #     undistorted_img1 = undistort_images(img1, K1, D1)
    #     undistorted_img2 = undistort_images(img2, K2, D2) 
    #     filename1 = os.path.join(sav1, os.path.basename(fname1))
    #     filename2 = os.path.join(sav2, os.path.basename(fname2)) 
    #     cv2.imwrite(filename1, undistorted_img1)
    #     cv2.imwrite(filename2, undistorted_img2)
    #     undistorted_img1 = cv2.resize(undistorted_img1, dim, interpolation=cv2.INTER_AREA)
    #     undistorted_img2 = cv2.resize(undistorted_img2, dim, interpolation=cv2.INTER_AREA)  
    #     # cv2.imshow('Undistorted Left', undistorted_img1)
    #     # cv2.imshow('Undistorted Right', undistorted_img2)
    #     # cv2.waitKey(0)
    map1x, map1y, map2x, map2y, P1, P2, Q = stereo_rectify(K1, D1, K2, D2, (image_width, image_height), R, T) 
    # filename = 'data/came.yaml'
    # calibration_data = {
    #     'K1': K1.tolist(),'K2': K2.tolist(),'D1': D1.tolist(),'D2': D2.tolist(),
    #     'R': R.tolist(),'T': T.tolist(),'P1': P1.tolist(),'P2': P2.tolist(),'Q': Q.tolist()}
    # with open(filename, 'w') as file:
    #     yaml.dump(calibration_data, file)   
    file_path3 = glob.glob(sav1+'*.png')
    file_path4 = glob.glob(sav2+'*.png')
    sav3 = "rectify_data/1/"
    sav4 = "rectify_data/2/"
    sav5 = "rectify_data/3/"
    for fname1, fname2 in zip(file_path3, file_path4):        
        dim = (1280,720)
        img1 = cv2.imread(fname1)
        img2 = cv2.imread(fname2)
        img1 = undistort_images(img1, K1, D1)
        img2 = undistort_images(img2, K2, D2)
        rectified_img1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
        rectified_img2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)   
        rectified_img1 = cv2.resize(rectified_img1, dim, interpolation=cv2.INTER_AREA)
        rectified_img2 = cv2.resize(rectified_img2, dim, interpolation=cv2.INTER_AREA)  
        num_lines = 10  # 需要绘制的横线数量
        line_spacing = rectified_img1.shape[0] // (num_lines + 1)  # 计算横线间隔
        for i in range(1, num_lines + 1):
            y = i * line_spacing
            cv2.line(rectified_img1, (0, y), (rectified_img1.shape[1], y), (0, 255, 0), 2)
            cv2.line(rectified_img2, (0, y), (rectified_img2.shape[1], y), (0, 255, 0), 2)
        filename1 = os.path.join(sav3, os.path.basename(fname1))
        filename2 = os.path.join(sav4, os.path.basename(fname2)) 
        print(os.path.basename(fname1), os.path.basename(fname2))
        cv2.imwrite(filename1, rectified_img1)
        cv2.imwrite(filename2, rectified_img2)
        img1 = Image.open(filename1)
        img2 = Image.open(filename2)
        
        path3 = os.path.join(sav5,  os.path.basename(fname1))
        # 确保两个图片的高度相同
        if img1.height == img2.height:
            # 将两个图片并列拼接
            combined_img = Image.new('RGB', (img1.width + img2.width, img1.height))
            combined_img.paste(img1, (0, 0))
            combined_img.paste(img2, (img1.width, 0))
            
            # 保存新的图片
            combined_img.save(path3)  # 保存到第一个文件夹，也可以选择其他位置
        else:
            print("mismatch")
    cv2.destroyAllWindows()    
   
if __name__ == "__main__":
    main()