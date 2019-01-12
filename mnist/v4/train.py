# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from model import Network


'''
训练模型参数
'''
CKPT_DIR = 'ckpt'
class Train:
    def __init__(self):
        self.net = Network()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.data = input_data.read_data_sets('../data_set', one_hot=True)

    def train(self):
        batch_size = 100        # 每次训练批次数量
        train_step = 2000       # 总训练次数
        step = 0                # 当前训练次数
        save_interval = 100     # 保存模型的频率间隔
        keep_prob = 1.0         # dropout 参数
        saver = tf.train.Saver(max_to_keep=5)

        # merge所有的summary node
        merged_summary_op = tf.summary.merge_all()
        # 可视化存储目录为当前文件夹下的 log
        merged_writer = tf.summary.FileWriter("./log", self.sess.graph)

        ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            # 读取网络中的global_step的值，即当前已经训练的次数
            step = self.sess.run(self.net.global_step)
            print('Continue from')
            print('-> Minibatch update : ', step)

        while step < train_step:
            x, y_ = self.data.train.next_batch(batch_size)
            _, loss, merged_summary = self.sess.run([self.net.train, self.net.loss, merged_summary_op],
                                                    feed_dict={self.net.x: x, self.net.y_: y_, self.net.keep_prob: 1.0})
            step = self.sess.run(self.net.global_step)

            if step % save_interval == 0:
                train_accuracy = self.sess.run(self.net.accuracy, feed_dict={self.net.x: x, self.net.y_: y_, self.net.keep_prob: 1.0})
                print "step %d, training accuracy %g" % (step, train_accuracy)
                merged_writer.add_summary(merged_summary, step)
                saver.save(self.sess, CKPT_DIR + '/model', global_step=step)
                print('%s/model-%d saved' % (CKPT_DIR, step))

    def calculate_accuracy(self):
        test_x = self.data.test.images
        test_y_ = self.data.test.labels
        accuracy = self.sess.run(self.net.accuracy, feed_dict={self.net.x: test_x, self.net.y_: test_y_})
        print("准确率: %.2f，共测试了%d张图片 " % (accuracy, len(test_y_)))


if __name__ == "__main__":
    app = Train()
    app.train()
    app.calculate_accuracy()
