from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np

IMG_SIZE = 224
BATCH_SIZE = 64

datagen = ImageDataGenerator(rescale=1.0/255)
test_gen_s1 = datagen.flow_from_directory('test_data/s1/',
					target_size = (IMG_SIZE,IMG_SIZE),
					batch_size = BATCH_SIZE,
					class_mode='categorical')
test_gen_s2 = datagen.flow_from_directory('test_data/s2/',
                                        target_size = (IMG_SIZE,IMG_SIZE),
                                        batch_size = BATCH_SIZE,
                                        class_mode='categorical')
model_1 = load_model('vgg_model_best_1.h5')
model_2 = load_model('vgg_model_best_2.h5')
print('model loaded')
steps_ = (39794//2) // BATCH_SIZE
predictions_s1 = model_1.predict_generator(test_gen_s1,steps = steps_+1,verbose = 1)
print(predictions_s1.shape)

predictions_s2 = model_2.predict_generator(test_gen_s2,steps = steps_+1,verbose = 1)
print(predictions_s2.shape)
np.save('predictions_s1.npy',predictions_s1)
np.save('predictions_s2.npy',predictions_s2)

filenames_1 = test_gen_s1.filenames
filenames_2 = test_gen_s2.filenames

df = pd.DataFrame()
df['filename_s1'] = filenames_1
df['prediction_s1'] = predictions_s1
df['filename_s2'] = filenames_2
df['prediction_s2'] = predictions_s2
df.to_csv('predictions_ensemble.csv',index =False)
