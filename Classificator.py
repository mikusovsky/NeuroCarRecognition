import gc
from sklearn.cluster import KMeans
import itertools

class Classificator:
    _neuronka = None
    _kmeans = None

    def __init__(self, neuronka):
        print('adding neuronka to classifier', neuronka)
        self._neuronka = neuronka

    def loadClassifier(self, carModels, type=1):
        print('classifing', self._neuronka)
        if type == 1:
            neuronka = self._neuronka
            input_tensor = neuronka.getEmptyTensor()
            neuroOut = []
            countCategories = 25
            i = 0
            for category in carModels:
                cImages = category.getImages()
                first = True
                for image in cImages:
                    if first:
                        category.indexFrom = i
                    category.indexTo = i
                    first = False
                    print(i)
                    i = i + 1
                    predict_values, logit_values = neuronka.run(feed_dict={input_tensor: image})
                    for value in logit_values:
                        neuroOut.append(value)
                    del image
                    gc.collect()
            print('kmeans')
            self._kmeans = KMeans(n_clusters=countCategories, random_state=0).fit(neuroOut)
            labels = self._kmeans.labels_
            i = 0
            for category in carModels:
                maxG = []
                categoryPart = labels[category.indexFrom: category.indexTo + 1]
                for key, igroup in itertools.groupby(categoryPart, lambda x: x // 1):
                    l = list(igroup)
                    if len(maxG) < len(l):
                        maxG = l
                category.kMeanLabel = maxG[0]
                i = i + 1
                print("{0} - {1}".format(i, countCategories))

            for category in carModels:
                print("id({0}) = {1}".format(category.id(), category.kMeanLabel))
