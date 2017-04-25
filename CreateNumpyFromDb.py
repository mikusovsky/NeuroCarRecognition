from DbLoader import dbLoader
import numpy as np
import Utils as utils
import Deffinitions
Deffinitions.init()

allModelsTraining = dbLoader.get_all_models(Deffinitions.DBPaths.trainingDB)
imagesTraining, labelsTraining = utils.car_models_to_image_class(carModels=allModelsTraining, shape=(128, 128))

allModelsTesting = dbLoader.get_all_models(Deffinitions.DBPaths.testingDB)
imagesTesting, labelsTesting = utils.car_models_to_image_class(carModels=allModelsTesting, shape=(128, 128))

print(imagesTraining.shape)
print(labelsTraining.shape)
np.save('D:/Skola/UK/DiplomovaPraca/PokracovaniePoPredchodcovi/Database/NeuroNumpy/trainingDb.np', imagesTraining.astype(np.uint8))
np.save('D:/Skola/UK/DiplomovaPraca/PokracovaniePoPredchodcovi/Database/NeuroNumpy/trainingDbLabels.np', labelsTraining.astype(np.uint8))

print(imagesTesting.shape)
print(labelsTesting.shape)
np.save('D:/Skola/UK/DiplomovaPraca/PokracovaniePoPredchodcovi/Database/NeuroNumpy/testingDb.np', imagesTesting.astype(np.uint8))
np.save('D:/Skola/UK/DiplomovaPraca/PokracovaniePoPredchodcovi/Database/NeuroNumpy/testingDbLabels.np', labelsTesting.astype(np.uint8))