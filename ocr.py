# coding: utf-8
# 用MNIST数据集训练的一个单层的神经网络模型，准确率不高

import cv2
import numpy as np
import tensorflow as tf

WINNAME = 'ocr'
ix, iy = -1, -1
clean = False  # 按下空格键，True-清除写字板里面的内容，False-对手写数字进行分类


# mouse callback function
def draw_line(event, x, y, flags, param):
    global ix, iy
    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y
    elif (event == cv2.EVENT_MOUSEMOVE) & (flags == cv2.EVENT_FLAG_LBUTTON):
        cv2.line(img, (ix, iy), (x, y), 255, 5, cv2.LINE_AA)
        ix, iy = x, y


# Create a black image, a window and bind the function to window
img = np.zeros((140, 140, 1), np.uint8)
cv2.namedWindow(WINNAME)
cv2.setMouseCallback(WINNAME, draw_line)

# 构建图 #

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 创建一个简单的神经网络，输入层784个神经元，输出层10个神经元
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, W) + b)

prediction = tf.argmax(prediction, 1)

# 初始化变量
init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, 'model/nn_mnist_model/my_net.ckpt')

    while True:
        cv2.imshow(WINNAME, img)
        key = cv2.waitKey(20)

        # 把图片resize成MNIST数据集的标准尺寸14*14
        resized_img = cv2.resize(img, (28, 28), cv2.INTER_CUBIC)
        # cv2.imshow('resized_img', resized_img)
        # key = cv2.waitKey(1)

        if key == 32:
            if clean:
                img = np.zeros((140, 140, 1), np.uint8)
                cv2.imshow(WINNAME, img)
            else:
                print(sess.run(prediction, feed_dict={x: resized_img.reshape([1, 784])}))
            clean = not clean
        elif key == 27:
            break

cv2.destroyAllWindows()
