from PIL import Image
import os

# 设置两个文件夹的路径
folder1 = 'D:/calib/data/005'
folder2 = 'D:/calib/data/006'
folder3 = 'D:/calib/data/003'
# 获取第一个文件夹中的所有文件名
files1 = os.listdir(folder1)

# 遍历文件名
for file in files1:
    if file.endswith('.png'):
        # 构建完整的文件路径
        path1 = os.path.join(folder1, file)
        path2 = os.path.join(folder2, file)
        path3 = os.path.join(folder3, file)
        # 确保第二个文件夹中也有相同的文件
        if os.path.exists(path2):
            # 打开两个图片文件
            img1 = Image.open(path1)
            img2 = Image.open(path2)
            
            # 确保两个图片的高度相同
            if img1.height == img2.height:
                # 将两个图片并列拼接
                combined_img = Image.new('RGB', (img1.width + img2.width, img1.height))
                combined_img.paste(img1, (0, 0))
                combined_img.paste(img2, (img1.width, 0))
                
                # 保存新的图片
                combined_img.save(path3)  # 保存到第一个文件夹，也可以选择其他位置
            else:
                print(f"Height mismatch for {file}")

print("Images have been combined.")