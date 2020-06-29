import tensorflow as tf
import numpy as np
from dataset import make_anime_dataset
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets, Sequential
import glob
from dataset import make_anime_dataset
from skimage import io, transform
import random
import csv
import time
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

root_img = 'D:\\tensorflow\\My_Work\\Target_Detect\\face_images\\'
img_mean = tf.constant([0.485, 0.456, 0.406])
img_std = tf.constant([0.229, 0.224, 0.225])


# 由卷积层和全连接层组成，中间用Flatten层进行平铺, inputs: [None, 64, 64, 3]
my_layers = [
    layers.Conv2D(32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

    layers.Flatten(),

    layers.Dense(512, activation=tf.nn.relu),
    layers.Dropout(rate=0.5),
    layers.Dense(2, activation=None)
]


# root为我们之前获得图片数据的根目录face_images，filename为我们要加载的csv文件，
# name2label为我们获取的图片类型字典
def load_csv(root, filename, name2label):
    # 如果根目录root下不存在filename文件，那么创建一个filename文件
    if not os.path.exists(os.path.join(root, filename)):
        # 创建一个图片路径的列表images
        images = []
        # 遍历字典里所有的元素，例如我的第一个为'xu'，第二个为‘zheng’
        for name in name2label.keys():
            # 将路径下所有的jpg图片的路径写至images列表中
            images += glob.glob(os.path.join(root, name, '*.jpg'))
            # print('addr:', glob.glob(os.path.join(root, name, '*.jpg')))
        print(len(images), images)
        # 对images进行随机打乱
        random.shuffle(images)
        with open(os.path.join(root, filename), mode='w', newline='') as f:
            writer = csv.writer(f)
            for img in images:
                # 获取路径最底层文件夹的名字
                # os.sep为路径的分隔符，split()函数以给定分隔符进行切片（默认以空格切片），取从右往左数第二个
                # img = '...a/b/c/haha.jpg' =>['...a', 'b', 'c', 'haha.jpg'], -2指的是'c'
                name = img.split(os.sep)[-2]
                # 查找字典对应元素的值
                label = name2label[name]
                # 添加到路径的后面
                writer.writerow([img, label])
            print('written into csv file:', filename)
    # 如果存在filename文件，将其读取至imgs, labels这两个列表中
    imgs, labels = [], []
    with open(os.path.join(root, filename)) as f:
        reader = csv.reader(f)
        for row in reader:
            # 读取路径和对应的label值
            img, label = row
            label = int(label)
            # 将其分别压入列表中，并返回出来
            imgs.append(img)
            labels.append(label)
    return imgs, labels


def load_faceimg(root, mode='train'):
    # 创建图片类型字典，准备调用load_csv()方法
    name2label = {}
    for name in sorted(os.listdir(os.path.join(root))):
        # 跳过root目录下不是文件夹的文件
        if not os.path.isdir(os.path.join(root, name)):
            continue
        # name为根目录下各个文件夹的名字
        # name2label.keys()表示字典name2label里所有的元素，len()表示字典所有元素的个数
        # 一开始字典是没有元素的，所以'xu'的值为0, 之后字典元素有个一个，所以'zheng'的值为1
        name2label[name] = len(name2label.keys())
    # 调用load_csv（）方法，返回值images为储存图片的目录的列表，labels为储存图片种类(0, 1两种)的列表
    images, labels = load_csv(root, 'images.csv', name2label)
    # 我们将前60%取为训练集，后20%取为验证集，最后20%取为测试集，并返回
    if mode == 'train':
        images = images[:int(0.6 * len(images))]
        labels = labels[:int(0.6 * len(labels))]
    elif mode == 'val':
        images = images[int(0.6 * len(images)):int(0.8 * len(images))]
        labels = labels[int(0.6 * len(labels)):int(0.8 * len(labels))]
    else:
        images = images[int(0.8 * len(images)):]
        labels = labels[int(0.8 * len(labels)):]
    return images, labels, name2label


def normalize(x, mean=img_mean, std=img_std):
    # 标准化
    # x: [64, 64, 3]
    # mean: [64, 64, 3], std: [3]
    x = (x - mean) / std
    return x


def denormalize(x, mean=img_mean, std=img_std):
    # 标准化的逆过程
    x = x * std + mean
    return x


# x：图片的路径List, y: 图片种类的数字编码List
def get_tensor(x, y):
    # 创建一个列表ims
    ims = []
    for i in x:
        # 读取路径下的图片
        p = tf.io.read_file(i)
        # 对图片进行解码，RGB，3通道
        p = tf.image.decode_jpeg(p, channels=3)
        # 修改图片大小为64*64
        p = tf.image.resize(p, [64, 64])
        # 将图片压入ims列表中
        ims.append(p)
    # 将List类型转换为tensor类型，并返回
    ims = tf.convert_to_tensor(ims)
    y = tf.convert_to_tensor(y)
    return ims, y


# 预处理函数，x, y均为tensor类型
def preprocess(x, y):
    # 数据增强
    x = tf.image.random_flip_left_right(x)  # 左右镜像
    x = tf.image.random_crop(x, [64, 64, 3])  # 随机裁剪
    # x: [0,255]=>0~1，将其值转换为float32
    x = tf.cast(x, dtype=tf.float32) / 255.
    # 0~1 => D(0, 1)
    x = normalize(x)
    # 将其值转换为int32
    y = tf.cast(y, dtype=tf.int32)
    return x, y


xu = [0]
zheng = [1]
sq = [1]
zheng = tf.convert_to_tensor(zheng, dtype=tf.int32)
xu = tf.convert_to_tensor(xu, dtype=tf.int32)
sq = tf.convert_to_tensor(sq, dtype=tf.int32)
if tf.equal(zheng, sq):
    print('cool!')

# 加载图片，获得图片路径与图片种类编码的列表
images_train, labels_train, name2label = load_faceimg(root_img, mode='train')
images_val, labels_val, _ = load_faceimg(root_img, mode='val')
images_test, labels_test, _ = load_faceimg(root_img, mode='test')

print('images_train:', images_train)

# 从对应路径读取图片，并将列表转换为张量
x_train, y_train = get_tensor(images_train, labels_train)
x_val, y_val = get_tensor(images_val, labels_val)
x_test, y_test = get_tensor(images_test, labels_test)

# 可输出查看它们的shape
print('x_train:', x_train.shape, 'y_train:', y_train.shape)
print('x_val:', x_val.shape, 'y_val:', y_val.shape)
print('x_test:', x_test.shape, 'y_test:', y_test.shape)
# print('images:', len(images_test))
# print('labels:', len(labels_test))
# print('name2label', name2label)

# 切分传入参数的第一个维度，并进行随机打散，预处理和打包处理
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(1000).map(preprocess).batch(10)

db_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
db_val = db_val.map(preprocess).batch(10)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).batch(10)

# 创建一个迭代器，可以查看其shape大小
sample_train = next(iter(db_train))
sample_val = next(iter(db_val))
sample_test = next(iter(db_test))
print('sample_train:', sample_train[0].shape, sample_train[1].shape)
print('sample_val:', sample_val[0].shape, sample_val[1].shape)
print('sample_test:', sample_test[0].shape, sample_test[1].shape)


def main():

    # my_net = Sequential(my_layers)
    #
    # my_net.build(input_shape=[None, 64, 64, 3])
    # my_net.summary()
    #
    # optimizer = optimizers.Adam(lr=1e-3)
    # acc_best = 0
    # patience_num = 10
    # no_improved_num = 0
    # for epoch in range(50):
    #     for step, (x, y) in enumerate(db_train):
    #         with tf.GradientTape() as tape:
    #             out = my_net(x)
    #             # print('out', out.shape)
    #             logits = out
    #             y_onehot = tf.one_hot(y, depth=2)
    #             loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
    #             loss = tf.reduce_mean(loss)
    #         grads = tape.gradient(loss, my_net.trainable_variables)
    #         optimizer.apply_gradients(zip(grads, my_net.trainable_variables))
    #
    #         if step % 5 == 0:
    #             print(epoch, step, 'loss:', float(loss))
    #
    #     total_num = 0
    #     total_correct = 0
    #     for x2, y2 in db_test:
    #         out = my_net(x2)
    #         logits = out
    #         prob = tf.nn.softmax(logits, axis=1)
    #         # tf.argmax() : axis=1 表示返回每一行最大值对应的索引, axis=0 表示返回每一列最大值对应的索引
    #         pred = tf.argmax(prob, axis=1)
    #         # 将pred转化为int32数据类型，便于后面与y2进行比较
    #         pred = tf.cast(pred, dtype=tf.int32)
    #
    #         correct = tf.cast(tf.equal(pred, y2), dtype=tf.int32)
    #         correct = tf.reduce_sum(correct)
    #
    #         total_num += x2.shape[0]
    #         total_correct += int(correct)
    #     acc = total_correct / total_num
    #     if acc > acc_best:
    #         acc_best = acc
    #         no_improved_num = 0
    #         my_net.save('model1.h5')
    #     else:
    #         no_improved_num += 1
    #     print(epoch, 'acc:', acc, 'no_improved_num:', no_improved_num)
    #     if no_improved_num >= patience_num:
    #         break

    my_net = tf.keras.models.load_model('my_net.h5')

    camera = cv2.VideoCapture(0)
    haar = cv2.CascadeClassifier('D:/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    n = 1
    while 1:
        if n <= 20000:
            print('It`s processing %s image.' % n)
            success, img = camera.read()

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = haar.detectMultiScale(gray_img, 1.3, 5)
            for f_x, f_y, f_w, f_h in faces:
                # 获得面部图片
                face = img[f_y:f_y + f_h, f_x:f_x + f_w]
                # 修改图片大小为64*64
                face = cv2.resize(face, (64, 64))
                # 将图片数据类型转换为tensor类型，完成后shape为[64, 64, 3]
                face_tensor = tf.convert_to_tensor(face)
                # 在0维度左侧增加一个维度，即[64, 64, 3]=>[1, 64, 64, 3]
                face_tensor = tf.expand_dims(face_tensor, axis=0)
                # 将tensor类型从uint8转换为float32
                face_tensor = tf.cast(face_tensor, dtype=tf.float32)
                # print('face_tensor', face_tensor)
                # 输入至网络
                logits = my_net(face_tensor)
                # 将每一行进行softmax
                prob = tf.nn.softmax(logits, axis=1)
                print('prob:', prob)
                # 取出prob中每一行最大值对应的索引
                pred = tf.argmax(prob, axis=1)
                pred = tf.cast(pred, dtype=tf.int32)
                print('pred:', pred)
                if tf.equal(pred, zheng):
                    cv2.putText(img, 'zheng', (f_x, f_y - 20), cv2.FONT_HERSHEY_COMPLEX, 1, 255, 2)
                if tf.equal(pred, xu):
                    cv2.putText(img, 'xu', (f_x, f_y - 20), cv2.FONT_HERSHEY_COMPLEX, 1, 255, 2)
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

    # my_net.save('my_net.h5')
    # del my_net
    # new_net = tf.keras.models.load_model('my_net.h5')
    # new_net.compile(optimizer=optimizers.Adam(lr=1e-3),
    #                 loss=tf.losses.CategoricalCrossentropy(from_logits=True),
    #                 metrics=['accuracy'])
    # new_net.fit(x=x_train, y=y_train, epochs=50)
    # loss1, acc1 = new_net.evaluate(x_test, y_test)


if __name__ == '__main__':
    main()
