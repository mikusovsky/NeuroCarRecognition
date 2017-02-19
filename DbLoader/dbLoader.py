import os
from xml.dom import minidom

class CarModel:
    _maker = None
    _model = None
    _generation = None
    _from = None
    _to = None
    _imagePath = None
    _imagesPath = None

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