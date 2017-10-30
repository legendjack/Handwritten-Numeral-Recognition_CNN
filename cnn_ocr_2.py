# coding: utf-8
# 基于CNN模型的手写数字识别，用MNIST数据集训练
# cnn_ocr.py中重复定义了网络结构图（graph）
# 这里直接从cnn_mnist.ckpt.meta文件中加载已经持久化的图，line27
# 需要在训练的时候为Tensorflow指定名称: keep_prob = tf.placeholder(tf.float32, name="keep_prob")

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


# 直接加载持久化的图
saver = tf.train.import_meta_graph("model/cnn_mnist_model/cnn_mnist.ckpt.meta")

# The name 'x_input' refers to an Operation, not a Tensor.
# Tensor names must be of the form "<op_name>:<output_index>".
x = tf.get_default_graph().get_tensor_by_name("x_input:0")

keep_prob = tf.get_default_graph().get_tensor_by_name("keep_prob:0")

result = tf.get_default_graph().get_tensor_by_name("result:0")

# opencv画图 #
img = np.zeros((140, 140, 1), np.uint8)
cv2.namedWindow(WINNAME)
cv2.setMouseCallback(WINNAME, draw_line)

with tf.Session() as sess:
    saver.restore(sess, 'model/cnn_mnist_model/cnn_mnist.ckpt')

    while True:
        cv2.imshow(WINNAME, img)
        key = cv2.waitKey(20)

        if key == 32:
            if clean:
                img = np.zeros((140, 140, 1), np.uint8)
                cv2.imshow(WINNAME, img)
            else:
                # 把图片resize成MNIST数据集的标准尺寸14*14
                resized_img = cv2.resize(img, (28, 28), cv2.INTER_CUBIC)
                print(sess.run(result, feed_dict={x: resized_img.reshape([1, -1]).astype(float), keep_prob: 1.0}))
            clean = not clean
        elif key == 27:
            break

cv2.destroyAllWindows()
