import cv2
import dlib
import os
import sys
import random

#输出文件夹
output_dir = './myfaces'
size = 64

#如果文件夹不存在。则建立文件夹
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#改变图片对比度和亮度
def relight(img,light=1,bias=0):
    h = img.shape[0]
    w = img.shape[1]

    for i in range(0,w):
        for j in range(0,h):
            for k in range(3):
                tmp = int(img[j,i,k]*light+bias)
                if tmp > 255:
                    tmp = 255
                elif tmp<0:
                    tmp = 0
                img[j,i,k] = tmp

    return img

#使用dlib库中的frontal_face_detector作为特征提取器
detector = dlib.get_frontal_face_detector()

#打开摄像头
camera = cv2.VideoCapture(0)
success,img = camera.read()
#记录照片采集数量
index = 1
while True:
    if(index<=10000):
        print('Being collected picture %s' % index)
        success,img = camera.read()
        #转化为灰度图像
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        face_image = detector(gray_img,1)

        #看不懂了
        for i, d in enumerate(face_image):
            x1 = d.top() if d.top() > 0 else 0
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0

            face = img[x1:y1, x2:y2]
            # 调整图片的对比度与亮度， 对比度与亮度值都取随机数，这样能增加样本的多样性
            face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
            face = cv2.resize(face, (size, size))
            cv2.imshow('image', face)
            cv2.imwrite(output_dir + '/' + str(index) + '.jpg', face)
            index += 1

        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break

    else:
        print('Finished!')
        break

