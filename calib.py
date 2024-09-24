import cv2
import numpy as np
import glob
def matrix_to_string(matrix):
    return "\n".join([" ".join(map(str, row)) for row in matrix])
# 相机标定
def calibrate_camera(images,w, h, size):
    w = 10
    h = 7
    objp = np.zeros((w * h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    objp=size*objp
    objpoints = []  # 在世界坐标系中的三维点
    imgpoints = []  # 在图像平面的二维点
 
    size = tuple()
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # 转灰度
        size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 
        # 执行亚像素级角点检测
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria) 
        objpoints.append(objp)
        imgpoints.append(corners2)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, size, None, None)
    print("ret:", ret)
    print("内参数矩阵:\n", mtx,'\n')
    print("畸变系数:\n", dist,'\n')
    # print("旋转向量(外参数):\n", rvecs,'\n')
    # print("平移向量(外参数):\n", tvecs,'\n')
    with open("cameradata.txt", "a") as file:
        file.write("RET:\n"+str(ret)+"\n")
        file.write("K:\n")
        file.write(matrix_to_string(mtx) + "\n\n")
        file.write("D:\n")
        file.write(matrix_to_string(dist) + "\n\n")
    return mtx, dist
# 双目标定
def stereo_calibrate(images1, images2, image_width, image_height, points_per_row, points_per_col, K1, D1, K2, D2, size):
    image_points_l = []
    image_points_r = []
    obj_points = []
    objp = np.zeros((points_per_row * points_per_col, 3), np.float32)
    objp[:, :2] = np.mgrid[0:points_per_row, 0:points_per_col].T.reshape(-1, 2)
    objp = size * objp
    for img_path1, img_path2 in zip(images1, images2):
        left_img = cv2.imread(img_path1)
        right_img = cv2.imread(img_path2)
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
            #obj_points.append([(point[0][0] * square_size, point[0][1] * square_size, 0) for point in np.mgrid[0:points_per_col, 0:points_per_row].reshape(2, -1).transpose()])
        # 双目标定
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
def stereo_rectify(K1, D1, K2, D2, img_size, R, T):
    # 计算立体校正的旋转矩阵和平移向量
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        K1, D1, K2, D2, img_size, R, T
    )
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_size, cv2.CV_32FC1)
    return map1x, map1y, map2x, map2y, P1, P2, Q

# 主函数
def main():
    image_width, image_height = 640, 480
    w, h = 10, 7
    square_size = 0.3975  #meter
    images1 = glob.glob('D:\calib\images3\*.png') #left
    images2 = glob.glob('D:\calib\images4\*.png')#right
    file_path1 = './images5'
    file_path2 = './images6'
    K1, D1 = calibrate_camera(images1, w, h, square_size)    
    K2, D2 = calibrate_camera(images2, w, h, square_size)
    R, T = stereo_calibrate(images1, images2, image_width, image_height, w, h, K1, D1, K2, D2, square_size)
     # 畸变矫正
    for fname1, fname2 in zip(images1, images2):
    #for fname in images1:
        img1 = cv2.imread(fname1)
        img2 = cv2.imread(fname2)
        undistorted_img1 = undistort_images(img1, K1, D1)
        undistorted_img2 = undistort_images(img2, K2, D2) 
    #     cv2.imshow('Undistorted Left', undistorted_img1)
    #     cv2.imshow('Undistorted Right', undistorted_img2)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 立体校正
    map1x, map1y, map2x, map2y, P1, P2, Q = stereo_rectify(
        K1, D1, K2, D2, (image_width, image_height), R, T
    )
    print('P1:',P1)
    print('P2:',P2)
    print('Q:',Q)
    # 应用畸变矫正映射
    for fname1, fname2 in zip(images1,images2):
        img1 = cv2.imread(fname1)
        img2 = cv2.imread(fname2)
        rectified_img1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
        rectified_img2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
        #draw = (ImageDraw.Draw(fname1))        # 获取图片的宽度和高度
        line_color = (0, 255, 0)
        # 每隔200个像素画一条横线
        for y in range(0, image_height, 20):
            cv2.line(rectified_img1, (0, y), (image_width, y), line_color, 1)
            cv2.line(rectified_img2, (0, y), (image_width, y), line_color, 1)
       
        # target_folder = 'D:\calib\images5'
        # output_file_path = os.path.join(target_folder)
        # img1.save(output_file_path)
        # target_folder = 'D:\calib\images6'
        # output_file_path = os.path.join(target_folder)
        # img2.save(output_file_path) 
        # filename1 = f"{folder_name1}/image_{timestamp}.jpg"
        # cv2.imwrite(filename1, rectified_img1)
        # cv2.imwrite(filename1, rectified_img2)
        # print(f"Image with lines has been saved to {output_file_path}")
        
        # cv2.imshow('Rectified Left', rectified_img1)
        # cv2.imshow('Rectified right', rectified_img2)
        # cv2.waitKey(0)
    cv2.destroyAllWindows()
   
if __name__ == "__main__":
    main()