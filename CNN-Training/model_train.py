from ast import Load
import numpy as np
import os
import sys
import time
from tensorflow.keras.applications import *
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Flatten, Dense,BatchNormalization
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.models import model_from_json, load_model
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd
from sklearn.metrics import classification_report
def ConfusionMatrix(y_true, y_pred,model_name,opt):
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.savefig("Results/"+model_name+"/"+opt+"/"+model_name+"_"+opt+"_confusion_matrix.png")


def LoadModel(model_name, opt):
    if(os.path.exists("Results/"+model_name+"/"+opt+"/"+model_name+'.h5')):
        model = load_model("Results/"+model_name+"/"+opt+"/"+model_name+'.best.h5')
        return model



# num_classes = 4
# IMAGES_DIRECTORY = "data/"
# MODEL_NAME="InceptionResNetV2"
# BATCH_SIZE=16
# EPOCHS=10
# VERBOSE=1
# OPTIMIZER="ADAM"
# LR=1e-04
# MOMENTUM=0.9
# IMG_SIZE=224

def TrainModel(IMAGES_DIRECTORY,NUM_CLASSES, MODEL_NAME, BATCH_SIZE, EPOCHS, VERBOSE, OPTIMIZER, LR, MOMENTUM, IMG_SIZE):
    datagen = ImageDataGenerator(validation_split=0.2,rescale=1./255,horizontal_flip=True,brightness_range=[0.6,1.0],rotation_range=45)
    train_generator = datagen.flow_from_directory(IMAGES_DIRECTORY, class_mode='categorical', batch_size=BATCH_SIZE,subset='training',shuffle=True,target_size=(IMG_SIZE,IMG_SIZE ))
    valid_generator = datagen.flow_from_directory(IMAGES_DIRECTORY, class_mode='categorical', batch_size=BATCH_SIZE,subset='validation',shuffle=False,target_size=(IMG_SIZE,IMG_SIZE))
    image_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    if(MODEL_NAME=="Xception"):
        model1 = Xception(weights='imagenet',include_top=False,input_tensor=image_input)
    elif(MODEL_NAME=="ResNet50"):
        model1 = ResNet50(weights='imagenet',include_top=False,input_tensor=image_input)
    elif(MODEL_NAME=="VGG16"):
        model1 = VGG16(weights='imagenet',include_top=False,input_tensor=image_input)
    elif(MODEL_NAME=="VGG19"):
        model1 = VGG19(weights='imagenet',include_top=False,input_tensor=image_input)
    elif(MODEL_NAME=="MobileNet"):
        model1 = MobileNet(weights='imagenet',include_top=False,input_tensor=image_input)
    elif(MODEL_NAME=="MobileNetV2"):
        model1 = MobileNetV2(weights='imagenet',include_top=False,input_tensor=image_input)
    elif(MODEL_NAME=="DenseNet121"):
        model1=DenseNet121(weights='imagenet',include_top=False,input_tensor=image_input)
    elif(MODEL_NAME=="DenseNet169"):
        model1=DenseNet169(weights='imagenet',include_top=False,input_tensor=image_input)
    elif(MODEL_NAME=="DenseNet201"):
        model1=DenseNet201(weights='imagenet',include_top=False,input_tensor=image_input)
    elif(MODEL_NAME=="NASNetLarge"):
        model1=NASNetLarge(weights='imagenet',include_top=False,input_tensor=image_input)
    elif(MODEL_NAME=="NASNetMobile"):
        model1=NASNetMobile(weights='imagenet',include_top=False,input_tensor=image_input)
    elif(MODEL_NAME=="ResNet50V2"):
        model1=ResNet50V2(weights='imagenet',include_top=False,input_tensor=image_input)
    elif(MODEL_NAME=="ResNet101V2"):
        model1=ResNet101V2(weights='imagenet',include_top=False,input_tensor=image_input)
    elif(MODEL_NAME=="ResNet152V2"):
        model1=ResNet152V2(weights='imagenet',include_top=False,input_tensor=image_input)
    elif(MODEL_NAME=="InceptionV3"):
        model1=InceptionV3(weights='imagenet',include_top=False,input_tensor=image_input)
    elif(MODEL_NAME=="InceptionResNetV2"):
        model1=InceptionResNetV2(weights='imagenet',include_top=False,input_tensor=image_input)
    elif(MODEL_NAME=="EfficientNetB0"):
        model1=EfficientNetB0(weights='imagenet',include_top=False,input_tensor=image_input)
    elif(MODEL_NAME=="EfficientNetB1"):
        model1=EfficientNetB1(weights='imagenet',include_top=False,input_tensor=image_input)
    elif(MODEL_NAME=="EfficientNetB2"):
        model1=EfficientNetB2(weights='imagenet',include_top=False,input_tensor=image_input)
    elif(MODEL_NAME=="EfficientNetB3"):
        model1=EfficientNetB3(weights='imagenet',include_top=False,input_tensor=image_input)
    elif(MODEL_NAME=="EfficientNetB4"):
        model1=EfficientNetB4(weights='imagenet',include_top=False,input_tensor=image_input)
    elif(MODEL_NAME=="EfficientNetB5"):
        model1=EfficientNetB5(weights='imagenet',include_top=False,input_tensor=image_input)
    elif(MODEL_NAME=="EfficientNetB6"):
        model1=EfficientNetB6(weights='imagenet',include_top=False,input_tensor=image_input)
    elif(MODEL_NAME=="EfficientNetB7"):
        model1=EfficientNetB7(weights='imagenet',include_top=False,input_tensor=image_input)
    else:
        sys.exit("Model Not Found")

    model1.summary()

    for i, layer in enumerate(model1.layers):
        print(i, layer.name, layer.trainable)


    x = model1.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    #x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x) 
    model = Model(inputs=model1.input, outputs=predictions)
    print(model.summary())


    for layer in model1.layers[:]:
        layer.trainable = False

    print('conv_base is now NOT trainable')

    for layer in model1.layers[:15]:
        layer.trainable = False
    for layer in model1.layers[15:]:
        layer.trainable = True

    print('Last block of the conv_base is now trainable')

    for i, layer in enumerate(model1.layers):
        print(i, layer.name, layer.trainable)

    if(OPTIMIZER=="ADAM"):
        opt = Adam(lr=LR)
    elif(OPTIMIZER=="SGD"):
        opt = SGD(lr=LR, momentum=MOMENTUM)
    else:
        sys.exit("Optimizer not found")

    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    model.summary
    t=time.time()
    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

    filepath="Results/"+MODEL_NAME+"/"+OPTIMIZER+"/"+MODEL_NAME+".best.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=VERBOSE, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    # Fit the model
    hist = model.fit_generator(generator=train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=valid_generator,
                        validation_steps=STEP_SIZE_VALID,
                        epochs=EPOCHS,
                        callbacks=callbacks_list
                        )
    (loss, accuracy) = model.evaluate_generator(generator=valid_generator,steps=STEP_SIZE_VALID)
    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))


    if(not os.path.exists("Results/"+MODEL_NAME)):
        os.mkdir("Results/"+MODEL_NAME)

    if(not os.path.exists("Results/"+MODEL_NAME+"/"+OPTIMIZER)):
        os.mkdir("Results/"+MODEL_NAME+"/"+OPTIMIZER)

    pd.DataFrame(hist.history).to_csv("Results/"+MODEL_NAME+"/"+OPTIMIZER+"/history.csv")

    txt_path="Results/"+MODEL_NAME+"/"+OPTIMIZER+"/class_mapping.json"
    text_file=open(txt_path,"w+")
    wr=""
    dict1={}
    count=0
    for x in train_generator.class_indices.keys():
        dict1[count]=x
        count=count+1
    text_file.write(json.dumps(dict1))
    text_file.close()

    params_file=open("Results/"+MODEL_NAME+"/"+OPTIMIZER+"/params.txt","w+")
    dict2={
    "model_name":MODEL_NAME,
    "batch_size":BATCH_SIZE,
    "epochs":EPOCHS,
    "opt":OPTIMIZER,
    "learning_rate":LR,
    "momentum":MOMENTUM,
    "num_classes":NUM_CLASSES,
    "total_images":train_generator.samples,
    "img_size":IMG_SIZE
    }
    params_file.write(json.dumps(dict2))
    params_file.close()
    # serialize model to JSON
    model_json = model.to_json()
    with open("Results/"+MODEL_NAME+"/"+OPTIMIZER+"/"+MODEL_NAME+".json", "w") as json_file:
        json_file.write(model_json)
        
    # serialize weights to HDF5
    model.save_weights("Results/"+MODEL_NAME+"/"+OPTIMIZER+"/"+MODEL_NAME+".h5")
    print("Saved model to disk")


    # visualizing losses and accuracy
    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['accuracy']
    val_acc=hist.history['val_accuracy']
    xc=range(50)
    epochs = range(len(train_acc))



    plt.plot(epochs, train_acc, label='Training accuracy')
    plt.plot(epochs, val_acc,  label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("Results/"+MODEL_NAME+"/"+OPTIMIZER+"/"+MODEL_NAME+"_Accuracy.jpg")

    plt.figure()
    plt.plot(epochs,train_loss , label='Training loss')
    plt.plot(epochs, val_loss,  label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("Results/"+MODEL_NAME+"/"+OPTIMIZER+"/"+MODEL_NAME+"_Loss.jpg")





    #"Results/"+MODEL_NAME+"/"+OPTIMIZER+"/"+MODEL_NAME+".best.h5"
    mod=LoadModel(MODEL_NAME,OPTIMIZER)
    Y_pred = mod.predict_generator(valid_generator, STEP_SIZE_VALID+1)
    y_pred = np.argmax(Y_pred, axis=1)
    
    

    ConfusionMatrix(valid_generator.classes, y_pred,MODEL_NAME,OPTIMIZER)
    target_names = ['Healthy', 'Mild', 'Moderate',"Severe"]
    print(classification_report(valid_generator.classes, y_pred, target_names=target_names,digits=4))
