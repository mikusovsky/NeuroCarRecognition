import tensorflow as tf
slim = tf.contrib.slim
from PIL import Image
from models.slim.nets.inception_resnet_v2 import *
from models.slim.nets.inception_v1 import *
from models.slim.nets.resnet_v1 import *
from models.slim.nets.resnet_utils import *
from models.slim.nets.alexnet import *
import numpy as np
import Deffinitions
Deffinitions.init()

class Neuronka:
    _sess = None
    _logits = None
    _end_points = None
    _input_tensor = None

    def __init__(self, checkpoint_file, pretreined):
        self.loadNeuro(checkpoint_file=checkpoint_file, pretreined=pretreined)

    def loadNeuro(self, checkpoint_file='D:/Skola/UK/DiplomovaPraca/TensorFlow/resnet/inception_resnet_v2_2016_08_30.ckpt', pretreined=True):
        # Load the model
        self._sess = tf.Session()
        #arg_scope = inception_v1_arg_scope()
        arg_scope = alexnet_v2_arg_scope()

        input_tensor = self.getEmptyTensor()
        # scaled_input_tensor = tf.scalar_mul((1.0 / 255), input_tensor)
        # scaled_input_tensor = tf.sub(scaled_input_tensor, 0.5)
        # scaled_input_tensor = tf.mul(scaled_input_tensor, 2.0)
        with slim.arg_scope(arg_scope):
            #self._logits, self._end_points = inception_resnet_v2(input_tensor, is_training=False)
            #self._logits, self._end_points = inception_v1(input_tensor, is_training=False)
            #self._logits, self._end_points = resnet_v1_50(input_tensor, is_training=False, num_classes=25)
            self._logits, self._end_points = alexnet_v2(input_tensor, num_classes=25, is_training=True)
            for endPoint in self._end_points:
                print(endPoint)


        saver = tf.train.Saver()
        if pretreined:
            new_saver = tf.train.import_meta_graph(Deffinitions.CheckPointPath.directory + 'trained.ckpt.meta')
            new_saver.restore(self._sess, tf.train.latest_checkpoint(Deffinitions.CheckPointPath.directory + './'))
            all_vars = tf.get_collection('vars')
            for v in all_vars:
                v_ = self._sess.run(v)
                print(v_)
            #saver.restore(self._sess, checkpoint_file)

        # self.last_conv = self._end_points['Conv2d_7b_1x1']
        self.last_conv = self._end_points['alexnet_v2/fc8']
        self.flatten = slim.flatten(self.last_conv)
        # TODO pridat regularizaciu ?? slim.dropout(flattne)
        stop_grad = tf.stop_gradient(self.flatten)

        self.logits = slim.fully_connected(stop_grad, num_outputs=25, activation_fn=None)

        self.loss = tf.nn.softmax_cross_entropy_with_logits(self.logits, self.labels)

        self.trainer = tf.train.AdamOptimizer().minimize(self.loss)
        self._sess.run(tf.global_variables_initializer())

        print('neuronka loaded')

    def train(self, img, label):
        feed_dict = {
            self._input_tensor: img,
            self.labels: label
        }
        #_, loss = self._sess.run([self.trainer, self.loss], feed_dict=feed_dict)
        _, loss = self._sess.run([self._optimizer, self.loss], feed_dict=feed_dict)
        return loss

    def get_conv_out(self, feed_dict):
        return self._sess.run(self.flatten, feed_dict=feed_dict)

    def getEmptyTensor(self):
        if self._input_tensor is None:
            #self._input_tensor = tf.placeholder(tf.float32, shape=(None, 225, 225, 3), name='input_image')  # resnet 50
            #self._input_tensor = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='input_image') # inception v1
            #elf._input_tensor = tf.placeholder(tf.float32, shape=(None, 299, 299, 3), name='input_image') # inception resnet v2
            self._input_tensor = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='input_image') # inception resnet v2
            self.labels = tf.placeholder(tf.float32, shape=(None, 25), name='labels')
        return self._input_tensor

    def run(self, feed_dict):
        return self._sess.run([self.logits], feed_dict=feed_dict)