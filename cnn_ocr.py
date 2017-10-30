# coding: utf-8
# 基于CNN模型的手写数字识别，用MNIST数据集训练

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


# 初始化权值
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 生成一个截断的正态分布（高斯分布）
    return tf.Variable(initial)


# 初始化偏置值
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)  # 全部置为0.1
    return tf.Variable(initial)


# 卷积层
def conv2d(x, W):
    # x : input tensor of shape [batch, in_height, in_width, in_channels]，如果是图片，shape就是 [批次大小，高，宽，通道数]
    # W : filter/kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
    # strides : 步长，strides[0] = strides[3] = 1，strides[1]代表x方向上的步长，strides[2]代表y方向上的步长
    # padding : 'SAME'/'VALID'
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 池化层
def max_pool_2x2(x):
    # ksize : [1, x, y, 1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 构图 #

# 定义两个placeholder，x和y分别用于表示输入的手写数字图像和标签，（具体数据）在执行图的时候喂进来
x = tf.placeholder(tf.float32, [None, 784], name="x_input")  # placeholder(dtype, shape=None, name=None)
# y = tf.placeholder(tf.float32, [None, 10])

# 改变x的格式为4D的向量[batch, in_height, in_width, in_channels]，把图片恢复成原来的尺寸28*28
# -1会由实际计算出来的置代替，比如一个batch是100，实际数据量是100*784，那么-1由100*784*1/28/28/1=100代替（通道数为1）
x_image = tf.reshape(x, [-1, 28, 28, 1])  # 对应conv2d函数的参数x，输入的图片数据原来是一维的张量，而conv2d函数要将图片数据恢复成原来的形状

# 初始化第一个卷积层的权值和偏置值
W_conv1 = weight_variable(
    [5, 5, 1, 32])  # 5*5的采样窗口，32个卷积核从1个平面抽取特征（输出32个特征平面）。对应conv2d函数的输入W，函数会将该矩阵reshape成[5*5*1, 32]
b_conv1 = bias_variable([32])  # 每一个卷积核需要一个偏置值

# 把x_image和权值进行卷积，加上偏置值，然后应用于relu激活函数
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 第一个卷积层的结果
h_pool1 = max_pool_2x2(h_conv1)  # 进行max-pooling，得到池化结果

# 初始化第二个卷积层的权值和偏置值
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 28*28的图片经过第一次卷积之后还是28*28，第一次池化之后变成14*14（14*14*32）
# 第二次卷积之后是14*14，第二次池化之后变成7*7（14*14*64）
# 经过上面的操作之后变成64张7*7的平面
# 图片的维度变高了，第一次卷积池化后由1维变成32维（32个卷积核），第二次变成64维

# 初始化第一个全连接层的权值
W_fc1 = weight_variable([7 * 7 * 64, 1024])  # 上一层有7*7*64个神经元，全连接层有1024个神经元
b_fc1 = bias_variable([1024])

# 把池化层2的输出扁平化为1维
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
# 求第一个全连接层的输出
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# keep_prob用来表示神经元的输出概率
keep_prob = tf.placeholder(tf.float32, name="keep_prob")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 初始化第二个全连接层的权值
W_fc2 = weight_variable([1024, 10])  # 上一层有7*7*64个神经元，全连接层有1024个神经元
b_fc2 = bias_variable([10])

# 计算输出
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

result = tf.argmax(prediction, 1, name="result")

# 初始化变量
init = tf.global_variables_initializer()

saver = tf.train.Saver()

# opencv画图 #
img = np.zeros((140, 140, 1), np.uint8)
cv2.namedWindow(WINNAME)
cv2.setMouseCallback(WINNAME, draw_line)

with tf.Session() as sess:
    sess.run(init)
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
