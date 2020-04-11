import random
import numpy as np
import cv2
import os

IMGSIZE = 64


# 创建目录
def createdir(*args):
    for item in args:
        if not os.path.exists(item):
            os.makedirs(item)


def relight(imgsrc, alpha=1, bias=0):
    imgsrc = imgsrc.astype(float)
    imgsrc = imgsrc * alpha + bias
    imgsrc[imgsrc < 0] = 0
    imgsrc[imgsrc > 255] = 255
    imgsrc = imgsrc.astype(np.uint8)
    return imgsrc


def getfacefromcamera(outdir):
    createdir(outdir)
    camera = cv2.VideoCapture(0)
    haar = cv2.CascadeClassifier('D:/haarcascades/haarcascade_frontalface_default.xml')
    n = 1
    while 1:
        # 创建200张64*64图片
        if n <= 200:
            print('It`s processing %s image.' % n)
            success, img = camera.read()
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = haar.detectMultiScale(gray_img, 1.3, 5)
            # f_x, f_y, f_w, f_h分别为获取面部的左上角x, y坐标值，宽高值（原点（0， 0）在图片左上角）
            for f_x, f_y, f_w, f_h in faces:
                # 截取面部图片，先写y方向，再写x方向，别写反了（可以尝试尝试写反获取的图片）
                face = img[f_y:f_y+f_h, f_x:f_x+f_w]
                # 修改图片大小为64*64
                face = cv2.resize(face, (IMGSIZE, IMGSIZE))
                # 随机改变图片的明亮程度，增加图片复杂度
                face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
                # 保存图片
                cv2.imwrite(os.path.join(outdir, str(n)+'.jpg'), face)
                # 在原图img面部上方20处写下你的名字
                cv2.putText(img, 'haha', (f_x, f_y-20), cv2.FONT_HERSHEY_COMPLEX, 1, 255, 2)
                # 画出方框，框选出你的face
                img = cv2.rectangle(img, (f_x, f_y), (f_x + f_w, f_y + f_h), (255, 0, 0), 2)
                n += 1
            cv2.imshow('img', img)
            key = cv2.waitKey(30) & 0xff
            if key == 27:
                break
        else:
            break
    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # 输入你的名字，创建属于你的face文件夹
    name = input('please input your name: ')
    # 执行这段代码前必须在当前目录手动创建一个‘face_images’文件夹，否则下面代码找不到‘face_images'
    getfacefromcamera(os.path.join('face_images', name))
