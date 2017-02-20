import tensorflow as tf
slim = tf.contrib.slim
from PIL import Image
from models.slim.nets.inception_resnet_v2 import *
import numpy as np

class Neuronka:
    _sess = None
    _logits = None
    _end_points = None
    _input_tensor = None

    def __init__(self, checkpoint_file):
        self.loadNeuro(checkpoint_file)

    def loadNeuro(self, checkpoint_file='D:/Skola/UK/DiplomovaPraca/TensorFlow/resnet/inception_resnet_v2_2016_08_30.ckpt'):
        # Load the model
        self._sess = tf.Session()
        arg_scope = inception_resnet_v2_arg_scope()

        input_tensor = self.getEmptyTensor()
        scaled_input_tensor = tf.scalar_mul((1.0 / 255), input_tensor)
        scaled_input_tensor = tf.sub(scaled_input_tensor, 0.5)
        scaled_input_tensor = tf.mul(scaled_input_tensor, 2.0)
        with slim.arg_scope(arg_scope):
            self._logits, self._end_points = inception_resnet_v2(input_tensor, is_training=False)
        saver = tf.train.Saver()
        saver.restore(self._sess, checkpoint_file)
        print('neuronka loaded')

    def getEmptyTensor(self):
        if self._input_tensor is None:
            self._input_tensor = tf.placeholder(tf.float32, shape=(None, 299, 299, 3), name='input_image')
        return self._input_tensor

    def run(self, feed_dict):
        return self._sess.run([self._end_points['Predictions'], self._logits], feed_dict=feed_dict)