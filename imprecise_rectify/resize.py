import cv2
import os
#rename.py
# 设置文件夹路径
# folder_path = 'C:/Users/LASER/Downloads/baidudiskdownload/2024-10-23closechess/2024-10-23/6'  # 替换为你的BMP文件所在的文件夹路径
# output_folder = 'C:/Users/LASER/Downloads/baidudiskdownload/2024-10-23closechess/2024-10-23/06'  # 替换为你想保存PNG文件的文件夹路径

# # 确保输出文件夹存在
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# # 遍历文件夹中的所有文件
# for filename in os.listdir(folder_path):
#     if filename.endswith(".bmp"):  # 检查文件扩展名是否为BMP
#         # 构建完整的文件路径
#         file_path = os.path.join(folder_path, filename)
        
#         # 读取BMP图像
#         image = cv2.imread(file_path)
        
#         # 构建输出文件的路径
#         output_file = os.path.join(output_folder, os.path.splitext(filename)[0] + '.png')
        
#         # 将图像保存为PNG格式
#         cv2.imwrite(output_file, image)
#         #print(f'Converted {filename} to PNG format.')

# print("All BMP files have been converted to PNG format.")


# #resize.py
# # 设置文件夹路径
folder_path = '/data2/zjq/program/imprecise_rectify/ori_data/6/'  # 替换为你的BMP文件所在的文件夹路径
output_folder = '/data2/zjq/program/imprecise_rectify/ori_resize_data/6/'  # 替换为你想保存PNG文件的文件夹路径

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
dim = (1280,720)
# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith(".png"):  # 检查文件扩展名是否为PNG
        # 构建完整的文件路径
        file_path = os.path.join(folder_path, filename)
        
        # 读取BMP图像
        image = cv2.imread(file_path)
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        # 构建输出文件的路径
        output_file = os.path.join(output_folder, os.path.splitext(filename)[0] + '.png')
        
        # 将图像保存为PNG格式
        cv2.imwrite(output_file, image)
        print(f'Converted {filename} to PNG format.')

print("All BMP files have been converted to PNG format.")