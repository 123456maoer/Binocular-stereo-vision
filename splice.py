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
        path1 = os.path.join(folder1, file)
        path2 = os.path.join(folder2, file)
        path3 = os.path.join(folder3, file)
        if os.path.exists(path2):
            img1 = Image.open(path1)
            img2 = Image.open(path2)
            if img1.height == img2.height:
                combined_img = Image.new('RGB', (img1.width + img2.width, img1.height))
                combined_img.paste(img1, (0, 0))
                combined_img.paste(img2, (img1.width, 0))
                combined_img.save(path3)
            else:
                print(f"Height mismatch for {file}")

print("Images have been combined.")
