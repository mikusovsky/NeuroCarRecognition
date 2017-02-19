import tensorflow as tf
slim = tf.contrib.slim
from PIL import Image
from models.slim.nets.inception_resnet_v2 import *
import numpy as np

sess = None
logits = None
end_points = None
input_tensor = None

def loadDb(checkpoint_file='D:/Skola/UK/DiplomovaPraca/TensorFlow/resnet/inception_resnet_v2_2016_08_30.ckpt'):
    global sess, logits, end_points, input_tensor
    #Load the model
    sess = tf.Session()
    arg_scope = inception_resnet_v2_arg_scope()

    input_tensor = tf.placeholder(tf.float32, shape=(None,299,299,3), name='input_image')
    scaled_input_tensor = tf.scalar_mul((1.0/255), input_tensor)
    scaled_input_tensor = tf.sub(scaled_input_tensor, 0.5)
    scaled_input_tensor = tf.mul(scaled_input_tensor, 2.0)

    with slim.arg_scope(arg_scope):
        logits, end_points = inception_resnet_v2(input_tensor, is_training=False)
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_file)

loadDb('D:/Skola/UK/DiplomovaPraca/TensorFlow/resnet/inception_resnet_v2_2016_08_30.ckpt')

sample_images = ['D:/Skola/UK/DiplomovaPraca/Databases/TrainingDb/BMW/Rad_5/Rad_5_1/bmw-rad-5-520-d-m-packet.jpg',
                 'D:/Skola/UK/DiplomovaPraca/Databases/TrainingDb/BMW/Rad_5/Rad_5_1/bmw-rad-5-525-d-e60.jpg',
                 'D:/Skola/UK/DiplomovaPraca/Databases/TrainingDb/Skoda/Superb/Superb2/skoda-superb-19-tdi-pd-elegance (2).jpg']

for image in sample_images:
    im = Image.open(image).resize((299, 299))
    im = np.array(im)
    im = 2 * (im / 255.0) - 1.0
    im = im.reshape(-1, 299, 299, 3)
    predict_values, logit_values = sess.run([end_points['Predictions'], logits], feed_dict={input_tensor: im})
    print('-------------------')
    print(len(predict_values), len(logit_values))
    for value in predict_values:
        print(len(value))
        s = 0.
        for value1 in value:
            print(value1)
            s = s + value1
        print("sum={0}".format(s))
    print('-------------------')
    for value in logit_values:
        print(len(value))
        s = 0.
        for value1 in value:
            print(value1)
            s = s + value1
        print("sum={0}".format(s))
    print('-------------------')
    #print(np.max(predict_values), np.max(logit_values))
    #print(np.argmax(predict_values), np.argmax(logit_values))
