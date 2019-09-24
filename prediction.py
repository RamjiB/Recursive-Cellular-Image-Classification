from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np

IMG_SIZE = 224
BATCH_SIZE = 64

datagen = ImageDataGenerator(rescale=1.0/255)
test_gen = datagen.flow_from_directory('test_data/',
					target_size = (IMG_SIZE,IMG_SIZE),
					batch_size = BATCH_SIZE,
					class_mode='categorical')
model = load_model('vgg_model_best.h5')
model.summary()
steps_ = 39794 // BATCH_SIZE
predictions = model.predict_generator(test_gen,steps = steps_+1,verbose = 1)
print(predictions.shape)

np.save('predictions.npy',predictions)
filenames = test_gen.filenames

df = pd.DataFrame()
df['filename'] = filenames
df['prediction'] = predictions
df.to_csv('predictions.csv',index =False)
