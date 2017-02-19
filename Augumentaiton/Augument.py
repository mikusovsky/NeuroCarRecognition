import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import numpy as np
import six
import six.moves as sm
import os
import gc

import cv2

def doAugumentation():
    path = "D:/Skola/UK/DiplomovaPraca/PokracovaniePoPredchodcovi/DBTensor/TrainingDb/Audi/A4/A4_1/900038982_1_card.jpg"
    #path = "C:/Users/Adam/Desktop/900038982_1_card.jpg"
    print(os.path.isfile(path))
    # Load an color image in grayscale
    img = cv2.imread(path, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (224, 224))

    aimgs = _makeAugumentation([img])
    length = len(aimgs)
    print(length)
    name = 1
    print("{0} of {1}".format(name, length))
    for image in aimgs:
        print("{0} of {1}".format(name, length))
        cv2.imwrite('D:/Skola/UK/DiplomovaPraca/PokracovaniePoPredchodcovi/DBTensor/TrainingNormalisedDb/{0}.png'.format(name), image)
        del image
        name += 1
    del aimgs
    gc.collect()
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def _makeAugumentation(images):
    resImgs = []
    for image in images:
        resImgs.extend(_imgAugumentation(image))
    return resImgs

def _imgAugumentation (img):
    resImgs = []
    fliplr = [iaa.Fliplr(1), 1]  # horizontally flip 50% of all images
    flipud = [iaa.Flipud(1), 1]  # vertically flip 50% of all images
    crop = [iaa.Crop(percent=(0, 0.2)), 4]  # crop images by 0-10% of their height/width
    blur = [iaa.GaussianBlur((0, 3.0)), 4] # blur images with a sigma between 0 and 3.0
    noise = [iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.2), per_channel=0.5), 2]  # add gaussian noise to images
    dropout = [iaa.Dropout((0.0, 0.1), per_channel=0.5), 2]  # randomly remove up to 10% of the pixels
    brightness1 = [iaa.Add((-45, 45), per_channel=0.5), 2]  # change brightness of images (by -10 to 10 of original value)
    brightness2 = [iaa.Multiply((0.25, 1.5), per_channel=0.5), 2]  # change brightness of images (50-150% of original value)
    contrast = [iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), 2]  # improve or worsen the contrast
    elastic = [iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25), 2]
    operations = [elastic, contrast, noise, brightness1, brightness2, blur, dropout, crop]
    resImgs.append(img)
    i = 1;
    for operation in operations:
        print(i)
        imgsForOperation = []
        for image in resImgs:
            imgsForOperation.extend(e_xecuteTransform(operation[0], image, operation[1]))
        resImgs.extend(imgsForOperation)
        i += 1
    return resImgs

def _executeTransform(operation, image, count):
    res = []
    index = 0
    while index < count:
        index += 1
        res.append(operation.augment_image(image))
    return res
