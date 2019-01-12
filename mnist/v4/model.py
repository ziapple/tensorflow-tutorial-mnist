# -*- coding: utf-8 -*-
import tensorflow as tf

'''
建立网络训练模型
'''
class Network:
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, mean=0, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # tf.nn.conv2d (input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
    # 参数：
    # **input : ** 输入的要做卷积的图片，要求为一个张量，shape为 [ batch, in_height, in_weight, in_channel ]，其中batch为图片的数量，
    # in_height 为图片高度，in_weight 为图片宽度，in_channel 为图片的通道数，灰度图该值为1，彩色图为3。（也可以用其它值，但是具体含义不是很理解）
    # filter： 卷积核，要求也是一个张量，shape为 [ filter_height, filter_weight, in_channel, out_channels ]，其中 filter_height 为卷积核高度，
    # filter_weight 为卷积核宽度，in_channel 是图像通道数 ，和 input 的 in_channel 要保持一致，out_channel 是卷积核数量。
    # strides： 卷积时在图像每一维的步长，这是一个一维的向量，[ 1, strides, strides, 1]，第一位和最后一位固定必须是1
    # padding： string类型，值为“SAME” 和 “VALID”，表示的是卷积的形式，是否考虑边界。"SAME"是考虑边界，不足的时候用0去填充周围，"VALID"则不考虑
    # use_cudnn_on_gpu： bool类型，是否使用cudnn加速，默认为true
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # tf.nn.max_pool(value, ksize, strides, padding, name=None)
    # 参数是四个，和卷积很类似：
    # 第一个参数value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape
    # 第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
    # 第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
    # 第四个参数padding：和卷积类似，可以取'VALID' 或者'SAME'
    # 返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式
    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def __init__(self):
        self.learning_rate = 1e-4  # 梯度下降速率
        self.global_step = tf.Variable(0, trainable=False, name="global_step") # 全局训练次数
        self.x = tf.placeholder("float", shape=[None, 784])
        self.y_ = tf.placeholder("float", shape=[None, 10])

        # conv1
        W_conv1 = self.weight_variable([5, 5, 1, 32])
        b_conv1 = self.bias_variable([32])
        x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        # pool1,size=28/2=14
        h_pool1 = self.max_pool_2x2(h_conv1)

        # conv2
        W_conv2 = self.weight_variable([5, 5, 32, 64])
        b_conv2 = self.bias_variable([64])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        # pool2,size=14/2=7
        h_pool2 = self.max_pool_2x2(h_conv2)

        # 图片尺寸减小到7x7，我们加入一个有1024个神经元的全连接层，用于处理整个图片。
        W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
        b_fc1 = self.bias_variable([1024])
        # 我们把池化层输出的张量reshape成一些向量，乘上权重矩阵，加上偏置，然后对其使用ReLU。
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # dropout
        self.keep_prob = tf.placeholder("float")
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # softmax
        self.w = self.weight_variable([1024, 10])
        self.b = self.bias_variable([10])
        self.y = tf.nn.softmax(tf.matmul(h_fc1_drop, self.w) + self.b)

        # 训练阶段
        self.loss = -tf.reduce_sum(self.y_ * tf.log(self.y))
        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)
        predict = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(predict, "float"))

        # 创建 summary node
        # w, b 画直方图
        # loss, accuracy画标量图
        tf.summary.histogram('weight', self.w)
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
