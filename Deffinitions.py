from enum import Enum

def init():
    global CV_LOAD_IMAGE_COLOR # set flag to 1 to give colour image
    CV_LOAD_IMAGE_COLOR = 1
    global CV_LOAD_IMAGE_GRAY # set flag to 0 to give a grayscale one
    CV_LOAD_IMAGE_GRAY = 0

    global CheckPointPath
    class CheckPointPath:
        directory = 'D:/Skola/UK/DiplomovaPraca/TensorFlow/myModel/'


    global DBPaths
    class DBPaths(object):
        trainingDBSmall = "D:/Skola/UK/DiplomovaPraca/PokracovaniePoPredchodcovi/zdrojové kódy/NeuroCarRecognition/configuration/LoadDbSmall.xml"
        trainingDB = "D:/Skola/UK/DiplomovaPraca/PokracovaniePoPredchodcovi/zdrojové kódy/NeuroCarRecognition/configuration/LoadDb.xml"
        testingDB = "D:/Skola/UK/DiplomovaPraca/PokracovaniePoPredchodcovi/zdrojové kódy/NeuroCarRecognition/configuration/LoadDbTesting.xml"
        testingDB360 = "D:/Skola/UK/DiplomovaPraca/PokracovaniePoPredchodcovi/zdrojové kódy/NeuroCarRecognition/configuration/LoadDbTesting360.xml"
        testingDB540 = "D:/Skola/UK/DiplomovaPraca/PokracovaniePoPredchodcovi/zdrojové kódy/NeuroCarRecognition/configuration/LoadDbTesting540.xml"
        testingDB576 = "D:/Skola/UK/DiplomovaPraca/PokracovaniePoPredchodcovi/zdrojové kódy/NeuroCarRecognition/configuration/LoadDbTesting576.xml"
        testingDB720 = "D:/Skola/UK/DiplomovaPraca/PokracovaniePoPredchodcovi/zdrojové kódy/NeuroCarRecognition/configuration/LoadDbTesting720.xml"
        testingDB900 = "D:/Skola/UK/DiplomovaPraca/PokracovaniePoPredchodcovi/zdrojové kódy/NeuroCarRecognition/configuration/LoadDbTesting900.xml"

    global DB_DIRECOTRY
    DB_DIRECOTRY = "D:/Skola/UK/DiplomovaPraca/TensorFlow/resnet/"

    global DBType
    class DBType(Enum):
        V1 = 1
        V2 = 2
        V3 = 3
        V4 = 4
        V1_50 = 5
        V1_101 = 6
        V1_152 = 7
        VGG_16 = 8
        VGG_19 = 9
        V2_1 = 10

    global DBFile
    class DBFile(object):
        V1 = "inception_v1.ckpt"
        V2 = "inception_v2.ckpt"
        V3 = "inception_v3.ckpt"
        V4 = "inception_v4.ckpt"
        V1_50 = "resnet_v1_50.ckpt"
        V1_101 = "resnet_v1_101.ckpt"
        V1_152 = "resnet_v1_152.ckpt"
        VGG_16 = "vgg_16.ckpt"
        VGG_19 = "vgg_19.ckpt"
        V2_1 = "inception_resnet_v2_2016_08_30.ckpt"



