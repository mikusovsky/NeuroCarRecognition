import os
from xml.dom import minidom
from PIL import Image
import numpy as np
import gc
import Utils as utils

class CarModel:

    def __init__(self, maker=None, model=None, generation=None, dateFrom=None, dateTo=None, imagePath=None):
        self._maker = maker
        self._model = model
        self._generation = generation
        self._from = dateFrom
        self._from = dateTo
        self._imagePath = imagePath
        self._imagesPath = []
        if imagePath is not None:
            fileNames = os.listdir(imagePath)
            for fileName in fileNames:
                self._imagesPath.append("{0}/{1}".format(imagePath, fileName));

    def id(self):
        return "{0}_{1}_{2}".format(self._maker, self._model, self._generation)

    def maker(self):
        return self._maker

    def mode(self):
        return self._model

    def generation(self):
        return self._generation

    def dateFrom(self):
        return self._from

    def dateTo(self):
        return self._to

    def imagePath(self):
        return self._imagePath

    def imagesPath(self):
        return self._imagesPath

    def getImages(self, shape=(299, 299)):
        #self.createNumpyDb()
        images = []
        #images = utils.readAllImagesFromPath(self._imagePath + "/", shape)
        #return images

        exceptionCount = 0
        for imgPath in self._imagesPath:
            img = Image.open(imgPath).resize(shape)
            np_img = np.array(img)
            #np_img = 2 * (np_img / 255.0) - 1.0
            try:
                np_img = np_img.reshape(-1, shape[0], shape[1], 3)
                np_img1d = np_img#.flatten()
                images.append(np_img1d)
            except: # for png imgs dont work reshape
                exceptionCount = exceptionCount + 1
                print(imgPath)
            del img
            gc.collect()
        ret = None
        if len(images) != 0:
            ret = np.concatenate(images)
        return ret


    def indexFrom(self, value=None):
        if value is None:
            return self._indexFrom
        self._indexFrom = value

    def indexTo(self, value=None):
        if value is None:
            return self._indexTo
        self._indexTo = value

    def kMeanLabel(self, value):
        if value is None:
            return self._kMeanLabel
        self._kMeanLabel = value

    def createNumpyDb(self):
        exceptionCount = 0
        shape = (224, 224)
        for imgPath in self._imagesPath:
            newPath = imgPath.replace("NeuroTrainingDb", "NeuroTrainingDbNumpy")
            newDir = os.path.dirname(newPath)
            img = Image.open(imgPath).resize(shape)
            np_img = np.array(img)
            np_img = 2 * (np_img / 255.0) - 1.0
            try:
                np_img = np_img.reshape(-1, shape[0], shape[1], 3)
                if not os.path.exists(newDir):
                    os.makedirs(newDir)
                np_img.tofile(newPath, sep="", format="%f")
            except:  # for png imgs dont work reshape
                exceptionCount = exceptionCount + 1
                print(imgPath)
            del img
            gc.collect()
        print(exceptionCount)


def get_all_models(configFile):
    allModels = []
    xmldoc = minidom.parse(configFile)
    makers = xmldoc.getElementsByTagName('maker')
    for maker in makers:
        makerName = maker.getAttribute('name')
        models = maker.getElementsByTagName('model')
        for model in models:
            modelName = model.getAttribute('name')
            generations = model.getElementsByTagName('generation')
            for generation in generations:
                generationName = generation.getAttribute('name')
                dateFrom = generation.getElementsByTagName('from')[0].firstChild.data
                dateTo = generation.getElementsByTagName('to')[0].firstChild.data
                imagePath = generation.getElementsByTagName('path')[0].firstChild.data
                allModels.append(CarModel(maker=makerName,model=modelName,generation=generationName,dateFrom=dateFrom,dateTo=dateTo,imagePath=imagePath))
    return allModels