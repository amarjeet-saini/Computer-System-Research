from tensorflow.keras.applications.resnet50 import ResNet50
# from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.applications.inception_v3 import InceptionV3
# from tensorflow.keras.applications.xception import Xception
# from tensorflow.keras.applications.densenet import DenseNet121
# from tensorflow.keras.applications.resnet import ResNet101
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2


model1 = ResNet50(weights='imagenet')
#model2 = VGG16(weights='imagenet')
#model3 = InceptionV3(weights='imagenet')
#model4 = Xception(weights='imagenet')
#model5 = DenseNet121(weights='imagenet')
#model6 = ResNet101(weights='imagenet')
#model7 = MobileNetV2(weights='imagenet')

# to check model layer summary
#model1.summary()

# to download pre-trained model as .pb format 

model1.save('saved_model/Resnet50')
#model2.save('saved_model/Vgg16')
#model3.save('saved_model/Inceptionv3')
#model4.save('saved_model/Xception')
#model5.save('saved_model/Densenet')
#model6.save('saved_model/Resnet101')
#model7.save('saved_model/MobileNetV2')

# to download pre-trained model as .h5 format
model1.save('saved_model/resnet50.h5')
#model2.save('saved_model/vgg16.h5')
#model3.save('saved_model/inceptionv3.h5')
#model4.save('saved_model/xception.h5')
#model5.save('saved_model/densenet.h5')
#model6.save('saved_model/resnet101.h5')
#model7.save('saved_model/mobileNetv2.h5')


