import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# 设置图片文件夹路径
folder_path = 'captured_images/sofa'
sav1 = 'images9'
sav2 = 'imagesx'
if not os.path.exists(sav1):
    os.makedirs(sav1)
if not os.path.exists(sav2):
    os.makedirs(sav2)
for filename in os.listdir(folder_path):
    image_path = os.path.join(folder_path, filename)
        
        # 打开图片
    image = Image.open(image_path)
        
        # 获取图片的宽度和高度
    width, height = image.size
        
        # 切割图片
    left_image = image.crop((0, 0, width/2, height))
    right_image = image.crop((width/2, 0, width, height))
        
        # 保存切割后的图片
    left_image.save(os.path.join(sav1, f'{filename}'))
    right_image.save(os.path.join(sav2, f'{filename}'))
print("success")