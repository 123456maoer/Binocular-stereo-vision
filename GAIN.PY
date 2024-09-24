import cv2
import os
import datetime

# 创建一个文件夹来保存图像
folder_name1 = "captured_images"
if not os.path.exists(folder_name1):
    os.makedirs(folder_name1)

# 打开摄像头
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 4) # 设置缓冲区大小
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
if not cap.isOpened():
    print("无法打开双目摄像头")
    exit()
while True:
    ret1, frame1 = cap.read()
    if not (ret1):
            print("无法从摄像头读取帧")
            break

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:23]  
    filename1 = f"{folder_name1}/image_{timestamp}.jpg"
    cv2.imwrite(filename1, frame1)
    cv2.imshow('frame', frame1)
    key =cv2.waitKey(1)         
    if key== ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
print("图像已保存到：", folder_name1)