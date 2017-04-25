import tensorflow as tf
import random
import tensorflow.contrib.layers as layers

slim = tf.contrib.slim
from models.slim.nets.alexnet import *
from models.slim.nets.vgg import *
import numpy as np

from DbLoader import dbLoader
import Utils as utils
import Deffinitions
Deffinitions.init()
train_log_dir = 'D:/Skola/UK/DiplomovaPraca/TensorFlow/myModel/trainingLog'

def xrange(x):

    return iter(range(x))

def getBatch(images, labels, batch_size):
    indices = list(xrange(len(images)))
    random.shuffle(indices)

    img_batch = []
    label_batch = []

    for i in range(0, batch_size):
        img_batch.append(images[indices[i]])
        label_batch.append(labels[indices[i]])

    return np.array(img_batch), np.array(label_batch)

if not tf.gfile.Exists(train_log_dir):
    tf.gfile.MakeDirs(train_log_dir)

shape = (224, 224)
batchSize = 32

with tf.Graph().as_default():
    # Set up the data loading:
    print("loading models")
    allModels = dbLoader.get_all_models(Deffinitions.DBPaths.trainingDBSmall)
    print("loading images")
    images, labels = utils.car_models_to_image_class(carModels=allModels, shape=shape)
    print(images.shape, labels.shape)
    #images, labels = tf.train.batch([images[0], labels[0]], batch_size=32)
    batchImages, batchLabels = getBatch(images, labels, batchSize)
    print("images loaded")
    # Define the model:
    print(batchImages.shape)
    input_tensor = tf.placeholder(
        dtype=tf.float32,
        shape=(batchSize,shape[0],shape[1], 3),
        name='input_image'
    )
    #desired_outputs = tf.placeholder(tf.int32, shape=(None, 25), name='labels')
    desired_outputs = tf.placeholder(
        dtype=tf.int32,
        shape=[32],
        name='desired_outputs'
    )
    #with tf.device('/cpu:0'):
    print(desired_outputs)
    desired_outputs_one_hot = tf.one_hot(desired_outputs, 25, dtype=tf.int32)
    print(desired_outputs_one_hot)
    predictions, end_points = alexnet_v2(inputs=input_tensor, num_classes=25, is_training=True)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=desired_outputs_one_hot)
    loss = tf.reduce_mean(cross_entropy)

    # hyper-parameters
    learning_rate = 1e-3

    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimize = layers.optimize_loss(loss=loss,
                                global_step=global_step,
                                learning_rate=learning_rate,
                                #summaries=layers.optimizers.OPTIMIZER_SUMMARIES,
                                optimizer=tf.train.RMSPropOptimizer
                                )

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    out = sess.run(optimize, feed_dict={input_tensor: batchImages, desired_outputs: batchLabels})
    # Specify the loss function:
    print(batchLabels.shape)
    slim.losses.softmax_cross_entropy(predictions, batchLabels)

    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('losses/total_loss', total_loss)

    # Specify the optimization scheme:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001)

    # create_train_op that ensures that when we evaluate it to get the loss,
    # the update_ops are done and the gradient updates are computed.
    train_tensor = slim.learning.create_train_op(total_loss, optimizer)
    print("start training")
    # Actually runs training.
    slim.learning.train(train_tensor, train_log_dir)
