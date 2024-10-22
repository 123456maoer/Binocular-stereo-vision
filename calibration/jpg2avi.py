import cv2
import glob
import os

# 图片所在文件夹路径
images_path = 'D:/calib/data/003'  # 替换为你的图片文件夹路径
# 所有图片名称
image_files = glob.glob(os.path.join(images_path, '*.png'))

# 按文件名排序，确保顺序正确
image_files.sort()

# 视频输出路径
output_path = 'close_video.avi'

# 获取第一张图片以获取视频的宽度和高度
frame = cv2.imread(image_files[0])
height, width, layers = frame.shape

# 设置视频编解码器和帧率等参数
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 30  # 可以根据需要调整帧率

# 创建 VideoWriter 对象
video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 遍历所有图片并写入视频
for image_file in image_files:
    img = cv2.imread(image_file)
    video.write(img)

# 释放资源
video.release()
cv2.destroyAllWindows()

print(f"Video saved as {output_path/video}")
