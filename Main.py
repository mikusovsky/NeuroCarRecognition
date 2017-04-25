import tensorflow as tf
slim = tf.contrib.slim
from PIL import Image
import numpy as np
import Neuronka as neuro
import Classificator as classif
from DbLoader import dbLoader
import Deffinitions
import Utils as utils
Deffinitions.init()

import sys

sess = None
logits = None
end_points = None
input_tensor = None

#neuronka = neuro.Neuronka('D:/Skola/UK/DiplomovaPraca/TensorFlow/resnet/inception_resnet_v2_2016_08_30.ckpt')
#neuronka = neuro.Neuronka('D:/Skola/UK/DiplomovaPraca/TensorFlow/resnet/inception_v1.ckpt')
#neuronka = neuro.Neuronka('D:/Skola/UK/DiplomovaPraca/TensorFlow/resnet/resnet_v1_50.ckpt')
neuronka = neuro.Neuronka(checkpoint_file="", pretreined=False)
sess = neuronka._sess
# clasificator = classif.Classificator(neuronka)
allModels = dbLoader.get_all_models(Deffinitions.DBPaths.trainingDB)
listImg = utils.car_models_to_image_class(carModels=allModels, shape=(224, 224))
# print(listImg[0]['label'], listImg[0]['image'], listImg[100]['label'], listImg[100]['image'])
# print(clasificator._neuronka , 'gg')
# clasificator.loadClassifier(allModels, type=1)
# SOFTMAX?
#--listImg = np.array(listImg)

np.random.shuffle(listImg)
print("trainning")
loss = None
for i in range(1000000000):
    if i % 100000 and i != 0:
        print(i, loss)
        np.random.shuffle(listImg)
        saver = tf.train.Saver()
        saver.save(sess, Deffinitions.CheckPointPath.directory + 'trained.ckpt')
    #imgs, labels = get_batch(listImg, 1)
    img, clas = listImg[i % len(listImg)]
    #classes = []
    #for label in labels:
    #    classes.append(class_to_vector(label, 25))
    ##--clas = utils.class_to_vector(label, 25)
    ##--clas = np.reshape(clas, (1, 25))
    #classes = np.array(classes)
    loss = neuronka.train(img, clas)



features = []
labels = []
for pair in listImg:
    img, label = pair
    labels.append(label)
    features.append(neuronka.get_conv_out({neuronka._input_tensor: img}).ravel())

from sklearn.linear_model import LogisticRegression

features = np.array(features)
labels = np.array(labels)

print(features.shape, labels.shape)

model = LogisticRegression()
model.fit(features, labels)

testModels = dbLoader.get_all_models(Deffinitions.DBPaths.testingDB)
testlistImg = utils.car_models_to_image_class(testModels)

features = []
labels = []
i = 1
for pair in testlistImg:
    img, label = pair
    labels.append(label)
    features.append(neuronka.get_conv_out({neuronka._input_tensor: img}).ravel())
    print(i)
    i = i + 1

score = model.score(features, labels)
print(score)
# SOFTMAX?
listImg = np.array(listImg)
np.random.shuffle(listImg)
for i in range(1000):
    #imgs, labels = get_batch(listImg, 1)
    img, label = listImg[i % len(listImg)]
    #classes = []
    #for label in labels:
    #    classes.append(class_to_vector(label, 25))
    clas = utils.class_to_vector(label, 25)
    clas = np.reshape(clas, (1, 25))
    #classes = np.array(classes)
    neuronka.train(img, clas)

def accuracy():
    pass
    #TODO pridat testovacie obrazky a kazdy pustit na neuronke z vystupu zobrat argmax a index argmax povazovat za klasifikovanu triedu obrazka
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