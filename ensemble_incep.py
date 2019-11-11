from keras.applications import inception_v3,mobilenet,vgg19,resnet50,xception,densenet
from keras.models import Model
from keras.layers import Dense
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,CSVLogger
from keras.optimizers import Adam

total_train_images = 58424
total_valid_images = 14606
IMG_SIZE = 299
BATCH_SIZE = 64
EPOCHS = 10

datagen = ImageDataGenerator(rescale=1.0/255,horizontal_flip = True,
vertical_flip = True)

#resizing all the images to 128 x 128
train_gen_s1 = datagen.flow_from_directory('tr_s1/' ,
                                        target_size = (IMG_SIZE,IMG_SIZE) ,
                                        batch_size = BATCH_SIZE,
                                       class_mode ='categorical',
                                       shuffle = True)
train_gen_s2 = datagen.flow_from_directory('tr_s2/',target_size = (IMG_SIZE,IMG_SIZE),
						batch_size=BATCH_SIZE,class_mode = 'categorical',
						shuffle = True)
valid_gen_s1 = datagen.flow_from_directory('va_s1/' ,
                                        target_size =(IMG_SIZE,IMG_SIZE) ,
                                        batch_size = BATCH_SIZE,
                                       class_mode ='categorical',
                                       shuffle = True)
valid_gen_s2 = datagen.flow_from_directory('va_s2/' ,
                                        target_size =(IMG_SIZE,IMG_SIZE) ,
                                        batch_size = BATCH_SIZE,
                                       class_mode ='categorical',
                                       shuffle = True)
def pretrained_model(model):
    if model == 'densenet':
        base_model = densenet.DenseNet121(include_top=False,weights='imagenet',input_shape = (IMG_SIZE,IMG_SIZE,3))
    elif model == 'inception':
        base_model = inception_v3.InceptionV3(include_top=True,weights='imagenet',input_shape = (IMG_SIZE,IMG_SIZE,3))
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
    base_model_1 = base_model.output
    base_model_2 = base_model.output
    #x = Dense(2048,activation='relu')(x)
    #x = Dropout(0.2)(x)
    predictions_1 = Dense(1108,activation='softmax')(base_model_1)
    predictions_2 = Dense(1108,activation='softmax')(base_model_2)
    return models.Model(base_model.input,predictions_1),models.Model(base_model.input,predictions_2)

s1_model,s2_model = pretrained_model('inception')
s1_model.summary()
s2_model.summary()

steps_p_ep_tr = (total_train_images//2)//BATCH_SIZE
steps_p_ep_va = (total_valid_images//2)//BATCH_SIZE

def train(mode):
	if mode == 's1':
		s1_model.compile(optimizer = Adam(lr=0.0001),
              		loss = 'categorical_crossentropy', metrics=['accuracy'])
		s1_model.fit_generator(train_gen_s1,
				steps_per_epoch = steps_p_ep_tr,
				validation_data = valid_gen_s1,
				validation_steps = steps_p_ep_va,
				verbose = 1,
				epochs = EPOCHS,
				workers = 16,use_multiprocessing = True,
				max_queue_size = 20,callbacks=[c_1,cp_1,lr_1])
	else:
		s2_model.compile(optimizer = Adam(lr=0.0001),
              		loss = 'categorical_crossentropy', metrics=['accuracy'])
		s2_model.fit_generator(train_gen_s2,
				steps_per_epoch = steps_p_ep_tr,
				validation_data = valid_gen_s2,
				validation_steps = steps_p_ep_va,
				verbose = 1,epochs= EPOCHS,
				workers = 16,use_multiprocessing = True,
				max_queue_size = 20,callbacks = [c_2,cp_2,lr_2])

c_1 = CSVLogger("s1_model_incep.csv",separator = ",",append=False)
c_2 = CSVLogger("s2_model_incep.csv",separator = ",",append=False)
checkpoint_fp_1 = "inc_model_best_1.h5"
checkpoint_fp_2 = "inc_model_best_2.h5"
cp_1 = ModelCheckpoint(checkpoint_fp_1,monitor='val_loss',
                             verbose=1,
                            save_best_only= True,mode='min')
cp_2 = ModelCheckpoint(checkpoint_fp_2,monitor='val_loss',
                             verbose=1,
                            save_best_only= True,mode='min')
lr_1 = ReduceLROnPlateau(monitor='loss',
                                 factor = 0.1,
                                 patience = 2,
                                 verbose = 1,
                                 mode = 'min')
lr_2 = ReduceLROnPlateau(monitor='loss',
                                 factor = 0.1,
                                 patience = 2,
                                 verbose = 1,
                                 mode = 'min')


train('s1')
print('S1 done')
train('s2')

print('Done')
