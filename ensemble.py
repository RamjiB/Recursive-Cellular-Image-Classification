import glob
import keras
from keras.applications import inception_v3,nasnet,mobilenet,vgg19,resnet50,xception,densenet
import keras.backend as K

from keras.models import Sequential, Model
from keras.layers import Conv2D,MaxPool2D,SeparableConv2D,Dropout,Flatten,Dense,BatchNormalization,GlobalAveragePooling2D
from keras import layers,models
from keras import initializers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,CSVLogger
from keras.optimizers import Adam,RMSprop

total_train_images = 58424
total_valid_images = 14606
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 10

datagen = ImageDataGenerator(rescale=1.0/255)

#resizing all the images to 128 x 128
train_gen_s1 = datagen.flow_from_directory('s1/' , 
                                        target_size = (IMG_SIZE,IMG_SIZE) , 
                                        batch_size = BATCH_SIZE,
                                       class_mode ='categorical',
                                       shuffle = True)
train_gen_s2 = datagen.flow_from_directory('s2/',target_size = (IMG_SIZE,IMG_SIZE),
						batch_size=BATCH_SIZE,class_mode = 'categorical',
						shuffle = True)
valid_gen = datagen.flow_from_directory('valid_data/' , 
                                        target_size =(IMG_SIZE,IMG_SIZE) , 
                                        batch_size = BATCH_SIZE,
                                       class_mode ='categorical',
                                       shuffle = True)
def pretrained_model(model):
    if model == 'densenet':
        base_model = densenet.DenseNet121(include_top=False,weights='imagenet',input_shape = (IMG_SIZE,IMG_SIZE,3))
    elif model == 'inception':
        base_model = inception_v3.InceptionV3(include_top=False,weights='imagenet',input_shape = (IMG_SIZE,IMG_SIZE,3))
    elif model == 'mobilenet':
        base_model = mobilenet.MobileNet(include_top=False,weights='imagenet',input_shape = (IMG_SIZE,IMG_SIZE,3))
    elif model == 'vgg':
        base_model = vgg19.VGG19(include_top=True,weights='imagenet',input_shape = (IMG_SIZE,IMG_SIZE,3))
    elif model == 'resnet':
        base_model = resnet50.ResNet50(include_top=False,weights='imagenet',input_shape = (IMG_SIZE,IMG_SIZE,3))
    elif model == 'xception':
        base_model = xception.Xception(include_top=False,weights='imagenet',input_shape = (IMG_SIZE,IMG_SIZE,3))
        
    for layer in base_model.layers:
        layer.trainable = False
        
    base_model_1 = Model(inputs = base_model_1.input, outputs = base_model_1.get_layer('fc2').output)
    base_model_2 = Model(inputs = base_model_2.input,outputs = base_model_2.get_layer('fc2').output)
    #x = Dense(2048,activation='relu')(x)
    #x = Dropout(0.2)(x)
    predictions_1 = Dense(1108,activation='softmax')(base_model_1.output)
    predictions_2 = Dense(1108,activation='softmax')(base_model_2.output)

    return models.Model(base_model_1.input,predictions_1),models.Model(base_model_2.input,predictions_2)

s1_model,s2_model = pretrained_model('vgg')
s1_model.summary()
s2_model.sumamry()

steps_p_ep_tr = (total_train_images//2)//BATCH_SIZE
steps_p_ep_va = total_valid_images//BATCH_SIZE

def train():
	s1_model.fit_generator(train_gen_s1,
				steps_per_epoch = steps_p_ep_tr,
				validation_data = valid_gen,
				validation_steps = steps_p_ep_va,
				verbose = 1,
				epochs = EPOCHS,
				workers = 16,use_multiprocessing = True,
				max_queue_size = 20)
	s2_model.fit_generator(train_gen_s2,
				steps_per_epoch = steps_p_ep_tr,
				validation_data = valid_gen,
				validation_steps = steps_p_ep_va,
				verbose = 1,epochs= EPOCHS,
				workers = 16,use_multiprocessing = True,
				max_queue_size = 20)

csv_logger = CSVLogger("_model.csv",separator = ",",append=False)

checkpoint_fp = "vgg_model_best.h5"
checkpoint = ModelCheckpoint(checkpoint_fp,monitor='val_acc',
                             verbose=1,
                            save_best_only= True,mode='max')

learning_rate = ReduceLROnPlateau(monitor='acc',
                                 factor = 0.1,
                                 patience = 2,
                                 verbose = 1,
                                 mode = 'max')

callback = [checkpoint,learning_rate,csv_logger]

steps_p_ep_tr = total_train_images//BATCH_SIZE
steps_p_ep_va = total_valid_images//BATCH_SIZE

main_model.compile(optimizer = Adam(lr=0.0001), 
              loss = 'categorical_crossentropy', metrics=['accuracy'])

my_model = main_model.fit_generator(train_gen,
                                   steps_per_epoch = steps_p_ep_tr,
				   validation_data = valid_gen,
                                   validation_steps = steps_p_ep_va,
                                   verbose = 1,
                                   epochs = EPOCHS,
                                   callbacks = callback, workers=16, use_multiprocessing=True,max_queue_size = 20)

my_model.save_weights('vgg_model_epoch_10.h5')
