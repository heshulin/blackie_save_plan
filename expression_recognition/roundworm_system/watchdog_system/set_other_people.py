import sys
import os
import cv2
import dlib


input_dir  = './input_image'
output_dir = './otherfaces'
size = 64
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

deterctor = dlib.get_frontal_face_detector()

index = 1
for (path,dirnames,filenames) in os.walk(input_dir):
    for filename in filenames:
        if filename.endswith('.jpg'):
            print("Being processed picture %s" % index)
            #构造文件路径
            img_path = path + '/' +filename
            #读取图片
            img = cv2.imread(img_path)
            #转化为灰度图片
            gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            #人脸检测，返回结果
            dets = deterctor(gray_img,1)
            for i, d in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0
                # img[y:y+h,x:x+w]
                face = img[x1:y1, x2:y2]
                # 调整图片的尺寸
                face = cv2.resize(face, (size, size))
                cv2.imshow('image', face)
                # 保存图片
                cv2.imwrite(output_dir + '/' + str(index) + '.jpg', face)
                index += 1

            key = cv2.waitKey(30) & 0xff
            if key == 27:
                sys.exit(0)

