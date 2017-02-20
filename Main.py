import tensorflow as tf
slim = tf.contrib.slim
from PIL import Image
import numpy as np
import Neuronka as neuro
import Classificator as classif
from DbLoader import dbLoader
import Deffinitions
Deffinitions.init()

sess = None
logits = None
end_points = None
input_tensor = None

neuronka = neuro.Neuronka('D:/Skola/UK/DiplomovaPraca/TensorFlow/resnet/inception_resnet_v2_2016_08_30.ckpt')
clasificator = classif.Classificator(neuronka)
allModels = dbLoader.get_all_models(Deffinitions.DBPaths.trainingDB)
print(clasificator._neuronka , 'gg')
clasificator.loadClassifier(allModels, type=1)

'''
while True:
    testPath = input("Set test path: ")
    if (testPath == 'exit'):
        break
    img = Image.open(testPath).resize((299, 299))
    img = np.array(img)
    img = 2 * (img / 255.0) - 1.0
    img = img.reshape(-1, 299, 299, 3)
    predict_values, logit_values = sess.run([end_points['Predictions'], logits], feed_dict={input_tensor: img})
    print(kmeans.predict(logit_values)[0])
'''
'''
sample_images = ['D:/Skola/UK/DiplomovaPraca/PokracovaniePoPredchodcovi/Database/TrainingDb/Audi/A4/A4_1/900038982_1_card.jpg',
                 'D:/Skola/UK/DiplomovaPraca/PokracovaniePoPredchodcovi/Database/TrainingDb/Audi/A4/A4_1/audi-a4-19-tdi (1).jpg',
                 'D:/Skola/UK/DiplomovaPraca/Databases/TrainingDb/Skoda/Superb/Superb2/skoda-superb-19-tdi-pd-elegance (2).jpg']

for image in sample_images:
    im = Image.open(image).resize((299, 299))
    im = np.array(im)
    im = 2 * (im / 255.0) - 1.0
    im = im.reshape(-1, 299, 299, 3)
    predict_values, logit_values = sess.run([end_points['Predictions'], logits], feed_dict={input_tensor: im})
    print(np.max(predict_values), np.max(logit_values))
    print(np.argmax(predict_values), np.argmax(logit_values))
'''