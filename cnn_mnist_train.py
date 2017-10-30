# 将卷积神经网络应用于MNIST识别，结构为两个卷积层（conv+pooling)加两个全连接层
# 注意测试准确率的时候，不能直接将1w个测试数据全部读入（2GB显存不够），分成10个批次，每个批次1000个数据
# 训练21次准确率达到99.16%（训练1次准确率95%，11次准确率就超过了99%，在GTX760上训练一次约13秒）

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST', one_hot=True)

# 每个批次的大小
batch_size = 100
# 一共有多少个批次
n_batch = mnist.train.num_examples // batch_size


# 初始化权值
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)    # 生成一个截断的正态分布（高斯分布）
    return tf.Variable(initial)


# 初始化偏置值
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)             # 全部置为0.1
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


# 定义两个placeholder，x和y分别用于表示输入的手写数字图像和标签，（具体数据）在执行图的时候喂进来
x = tf.placeholder(tf.float32, [None, 784], name="x_input")     # placeholder(dtype, shape=None, name=None)
y = tf.placeholder(tf.float32, [None, 10])

# 改变x的格式为4D的向量[batch, in_height, in_width, in_channels]，把图片恢复成原来的尺寸28*28
# -1会由实际计算出来的置代替，比如一个batch是100，实际数据量是100*784，那么-1由100*784*1/28/28/1=100代替（通道数为1）
x_image = tf.reshape(x, [-1, 28, 28, 1])    # 对应conv2d函数的参数x，输入的图片数据原来是一维的张量，而conv2d函数要将图片数据恢复成原来的形状

# 初始化第一个卷积层的权值和偏置值
W_conv1 = weight_variable([5, 5, 1, 32])    # 5*5的采样窗口，32个卷积核从1个平面抽取特征（输出32个特征平面）。对应conv2d函数的输入W，函数会将该矩阵reshape成[5*5*1, 32]
b_conv1 = bias_variable([32])               # 每一个卷积核需要一个偏置值

# 把x_image和权值进行卷积，加上偏置值，然后应用于relu激活函数
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)    # 第一个卷积层的结果
h_pool1 = max_pool_2x2(h_conv1)                             # 进行max-pooling，得到池化结果

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
W_fc1 = weight_variable([7*7*64, 1024])     # 上一层有7*7*64个神经元，全连接层有1024个神经元
b_fc1 = bias_variable([1024])

# 把池化层2的输出扁平化为1维
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# 求第一个全连接层的输出
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# keep_prob用来表示神经元的输出概率
keep_prob = tf.placeholder(tf.float32, name="keep_prob")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 初始化第二个全连接层的权值
W_fc2 = weight_variable([1024, 10])     # 上一层有7*7*64个神经元，全连接层有1024个神经元
b_fc2 = bias_variable([10])

# 计算输出
predection = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# predection里面存放的是每个分类的概率，这里返回最大概率所在的位置，即最后的分类结果
result = tf.argmax(predection, 1, name="result")

# 交叉熵代价函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predection))
# 使用AdamOptimizer进行优化，学习率是1e-4
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 结果存放在一个布尔列表中
correct_predection = tf.equal(tf.argmax(predection, 1), tf.argmax(y, 1))    # argmax返回一维张量中最大值所在的位置
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_predection, tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})

        # acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        # print("Iter " + str(epoch) + " Test Accuracy " + str(acc))
        acc = list(range(10))
        for i in range(10):
            batch_xs_test, batch_ys_test = mnist.test.next_batch(1000)
            acc[i] = sess.run(accuracy, feed_dict={x: batch_xs_test, y: batch_ys_test, keep_prob: 1.0})
        print("Iter " + str(epoch) + " Test Accuracy " + str(np.mean(acc)))

    saver.save(sess, 'model/cnn_mnist_model/cnn_mnist.ckpt')