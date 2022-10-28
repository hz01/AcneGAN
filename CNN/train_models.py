from model_train import TrainModel
#TrainModel("data",4,"VGG16", 16, 10,1,"ADAM", 1e-04, 0.9,224)
TrainModel("data",4,"MobileNetV2", 16, 3,1,"ADAM", 1e-04, 0.9,224)
#TrainModel("data",4,"MobileNetV2", 32, 7,1,"SGD", 1e-04, 0.9,224)