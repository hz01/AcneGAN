from ast import Load
import numpy as np
import os
import sys
import time
import tensorflow as tf
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
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def ConfusionMatrix(y_true, y_pred,model_name,opt,RUN_NAME):
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.savefig("Results/"+RUN_NAME+"/"+model_name+"/"+opt+"/"+model_name+"_"+opt+"_confusion_matrix.png")


def LoadModel(model_name, opt,RUN_NAME):
    if(os.path.exists("Results/"+RUN_NAME+"/"+model_name+"/"+opt+"/"+model_name+'.h5')):
        model = load_model("Results/"+RUN_NAME+"/"+model_name+"/"+opt+"/"+model_name+'.best.h5')
        return model


def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.utils.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.utils.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    # cv2.imshow("img",cv2.imread(cam_path))
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


def GradCam1(Image,model,predictions,new_img_path):
    img_size = (224, 224)
    preprocess_input = tf.keras.applications.inception_resnet_v2.preprocess_input
    decode_predictions = tf.keras.applications.inception_resnet_v2.decode_predictions
    last_conv_layer_name = "conv_7b"
    img_path = Image
    img_array = preprocess_input(get_img_array(img_path, size=img_size))
    model = Model(inputs=model.inputs, outputs=predictions)
    model.layers[-1].activation = None
    preds = model.predict(img_array)
    print(preds)
    print("Predicted:", np.argmax(preds[0]))
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    # plt.matshow(heatmap)
    # plt.show()
    new_img_path = new_img_path
    save_and_display_gradcam(img_path, heatmap,cam_path=new_img_path)

def TrainModel(RUN_NAME,IMAGES_DIRECTORY,NUM_CLASSES, MODEL_NAME, BATCH_SIZE, EPOCHS, VERBOSE, OPTIMIZER, LR, MOMENTUM, IMG_SIZE,AUGMENT=False):
    if(AUGMENT==True):
        datagen = ImageDataGenerator(validation_split=0.2,rescale=1./255,horizontal_flip=True,brightness_range=[0.6,1.0],rotation_range=45)
    else:
        datagen = ImageDataGenerator(validation_split=0.2,rescale=1./255)
    
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


    # for i, layer in enumerate(model1.layers):
    #     print(i, layer.name, layer.trainable)


    x = model1.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    #x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x) 
    model = Model(inputs=model1.input, outputs=predictions)
    


    for layer in model1.layers[:]:
        layer.trainable = False

    print('conv_base is now NOT trainable')

    for layer in model1.layers[:15]:
        layer.trainable = False
    for layer in model1.layers[15:]:
        layer.trainable = True

    print('Last block of the conv_base is now trainable')

    # for i, layer in enumerate(model1.layers):
    #     print(i, layer.name, layer.trainable)

    if(OPTIMIZER=="ADAM"):
        opt = Adam(lr=LR)
    elif(OPTIMIZER=="SGD"):
        opt = SGD(lr=LR, momentum=MOMENTUM)
    else:
        sys.exit("Optimizer not found")

    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

    t=time.time()
    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

    filepath="Results/"+RUN_NAME+"/"+MODEL_NAME+"/"+OPTIMIZER+"/"+MODEL_NAME+".best.h5"
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


    if(not os.path.exists("Results/"+RUN_NAME+"/"+MODEL_NAME)):
        os.mkdir("Results/"+RUN_NAME+"/"+MODEL_NAME)

    if(not os.path.exists("Results/"+RUN_NAME+"/"+MODEL_NAME+"/"+OPTIMIZER)):
        os.mkdir("Results/"+RUN_NAME+"/"+MODEL_NAME+"/"+OPTIMIZER)

    pd.DataFrame(hist.history).to_csv("Results/"+RUN_NAME+"/"+MODEL_NAME+"/"+OPTIMIZER+"/history.csv")

    txt_path="Results/"+RUN_NAME+"/"+MODEL_NAME+"/"+OPTIMIZER+"/class_mapping.json"
    text_file=open(txt_path,"w+")
    wr=""
    dict1={}
    count=0
    for x in train_generator.class_indices.keys():
        dict1[count]=x
        count=count+1
    text_file.write(json.dumps(dict1))
    text_file.close()

    params_file=open("Results/"+RUN_NAME+"/"+MODEL_NAME+"/"+OPTIMIZER+"/params.txt","w+")
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
    with open("Results/"+RUN_NAME+"/"+MODEL_NAME+"/"+OPTIMIZER+"/"+MODEL_NAME+".json", "w") as json_file:
        json_file.write(model_json)
        
    # serialize weights to HDF5
    model.save_weights("Results/"+RUN_NAME+"/"+MODEL_NAME+"/"+OPTIMIZER+"/"+MODEL_NAME+".h5")
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
    plt.savefig("Results/"+RUN_NAME+"/"+MODEL_NAME+"/"+OPTIMIZER+"/"+MODEL_NAME+"_Accuracy.jpg")

    plt.figure()
    plt.plot(epochs,train_loss , label='Training loss')
    plt.plot(epochs, val_loss,  label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("Results/"+RUN_NAME+"/"+MODEL_NAME+"/"+OPTIMIZER+"/"+MODEL_NAME+"_Loss.jpg")





    "Results/"+MODEL_NAME+"/"+OPTIMIZER+"/"+MODEL_NAME+".best.h5"
    mod=model #LoadModel(MODEL_NAME,OPTIMIZER,RUN_NAME)
    Y_pred = mod.predict_generator(valid_generator, STEP_SIZE_VALID+1)
    y_pred = np.argmax(Y_pred, axis=1)
    
    

    ConfusionMatrix(valid_generator.classes, y_pred,MODEL_NAME,OPTIMIZER,RUN_NAME)
    target_names = ['Healthy', 'Mild', 'Moderate',"Severe"]
    print(classification_report(valid_generator.classes, y_pred, target_names=target_names,digits=4))


    # GradCam1(r"Real\Healthy\32996.png",model,predictions,"cam_healthy.jpg")
    # GradCam1(r"im\seed820548.png",model,predictions,"cam_severe.jpg")
    # GradCam1(r"im\seed136256-20230403-043005.png",model,predictions,"cam_moderate.jpg")
    # GradCam1(r"im\seed240699.png",model,predictions,"cam_severe.jpg")
    
    # GradCam1(r"Dataset\First Train\levle0_151.jpg",model,predictions,"syn_cam_mild.jpg")
    # GradCam1(r"Dataset\First Train\levle0_270.jpg",model,predictions,"syn_cam_moderate.jpg")
    # GradCam1(r"Dataset\First Train\image126.jpg",model,predictions,"syn_cam_severe.jpg")





