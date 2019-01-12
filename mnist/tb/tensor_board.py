# -*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import array
mnist = input_data.read_data_sets('../data_set', one_hot=True)

print(mnist.test.images.shape, mnist.test.labels.shape)
# 建立图像和数字的关系模型y=wx+b
# 1 该网络有两层（输入层不算，包含隐藏层（softmax）和输出层（十个数字的概率）
# 2 输入层有784个节点
# 3 隐藏层是softmax激活函数
# 4 输出层是10个数字的概率例如[0.9,0.1,0,0,0,0,0,0,0,0]表示该图片为1的概率90%

# x模拟每张图片的每个点的像素，None表示图片集合的大小
x = tf.placeholder("float", [None, 784])

# W表示图片的每个像素对应每个数字的权重，共784*10个变量
W = tf.Variable(tf.zeros([784,10]))

# b表示图片数字为i的偏置量
b = tf.Variable(tf.zeros([10]))

# 用softmax归一化函数将y值转化为概率函数，也就是每个图片对应10个数字的概率
y = tf.nn.softmax(tf.matmul(x,W) + b)

# y_表示样本值
y_ = tf.placeholder("float", [None,10])
# 定义loss函数，转换为最小值问题，求样本与目标值交叉墒达到最小的W,b的最优解
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
#自动地使用反向传播算法(backpropagation algorithm)来有效地确定你的变量是如何影响你想要最小化的那个成本值的
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


# tensorboard to graph
writer = tf.summary.FileWriter("logs/", sess.graph)
tf.summary.histogram('weight', W)
merged_summary_op = tf.summary.merge_all()

#训练模型
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  merged_summary = sess.run(merged_summary_op)
  writer.add_summary(merged_summary, i)

#评估模型
#比较两个向量是否相等
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#tf.reduce_mean求平均值，tf.cast将true,false转化成1,0
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

