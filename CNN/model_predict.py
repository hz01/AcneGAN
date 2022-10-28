from keras.models import model_from_json
import os
import numpy as np
import cv2
import json
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.transform import resize
from keras.models import load_model

MODEL_NAME = "VGG16"
OPTIMIZER = "ADAM"
json1 = open("Results/"+MODEL_NAME+"/"+OPTIMIZER+"/class_mapping.json", "r")
class_mapping_json = json.load(json1)
CLASS_MAPPING = []
for x in class_mapping_json:
    CLASS_MAPPING.append(class_mapping_json[x])


# def LoadModel(model_name, opt):
#     if(os.path.exists("Results/"+model_name+"/"+opt+"/"+model_name+'.json') and os.path.exists("Results/"+model_name+"/"+opt+"/"+model_name+'.h5')):
#         json_file = open("Results/"+model_name+"/" +
#                          opt+"/"+model_name+'.json', 'r')
#         loaded_model_json = json_file.read()
#         json_file.close()
#         loaded_model = model_from_json(loaded_model_json)
#         loaded_model.load_weights(
#             "Results/"+model_name+"/"+opt+"/"+model_name+'.h5')
#         print("Loaded model from disk")
#         return loaded_model

def LoadModel(model_name, opt):
    if(os.path.exists("Results/"+model_name+"/"+opt+"/"+model_name+'.h5')):
        model = load_model("Results/"+model_name+"/"+opt+"/"+model_name+'.best.h5')
        return model


def classify_image(model, path, class_mapping):
    im = plt.imread(path)
    resized_image = resize(im, (224, 224, 3))
    np_arr = np.array([resized_image])
    predictions = model.predict(np_arr)
    return class_mapping[np.argmax(predictions)]


def ClassifyClassesFolder(model, classes_folder, class_mapping):
    array_true = []
    array_pred = []

    for x in os.listdir(classes_folder):
        for y in tqdm(os.listdir(classes_folder+"/"+x)):
            array_true.append(x)
            array_pred.append(classify_image(
                model, classes_folder+"/"+x+"/"+y, class_mapping))
    return array_true, array_pred


def ConfusionMatrix(y_true, y_pred):
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.savefig("test_matrices/"+MODEL_NAME+"_"+OPTIMIZER+"_confusion_matrix.png")

LOADED_MODEL = LoadModel(MODEL_NAME, OPTIMIZER)
arr_true, arr_pred = ClassifyClassesFolder(LOADED_MODEL, "data1", CLASS_MAPPING)
ConfusionMatrix(arr_true, arr_pred)
