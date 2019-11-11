import numpy as np
import os,sys,tqdm
from progress.bar import Bar
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sys.path.append('rxrx1-utils/')
import rxrx.io as rio

import warnings
warnings.filterwarnings("ignore")

combined_df = rio.combine_metadata()
train_df = combined_df[combined_df['dataset'] == 'train']
train_df = train_df[train_df['well_type'] == 'treatment']
print('train_df shape: ',train_df.shape)

test_df = combined_df[combined_df['dataset'] == 'test']
test_df = test_df[test_df['well_type'] == 'treatment']
print(test_df.shape)

x_train_df,x_valid_df = train_test_split(train_df,test_size=0.20)

def create_folder(folderName):
    if not os.path.exists(folderName):
        try:
            os.makedirs(folderName)
        except OSError as exc:
            if exc.errno != exc.errno.EEXIST:
                raise
create_folder('train_data')
create_folder('valid_data')

def moving_files(df=x_train_df,mode='train'):
    indexes = df.index
    bar = Bar(mode+'_processing', max=len(df))
    for i in tqdm.tqdm(range(len(df))):
        row = df.iloc[i,:]
        class_name = str(int(row['sirna']))
        create_folder(mode +'_data/'+class_name)
        img = rio.load_site_as_rgb('train',
                                row['experiment'],
                                row['plate'],
                                row['well'],
                                row['site'])
        img = img.astype(np.uint8)
        dest = mode+'_data/'+class_name+'/'+indexes[i]+'_'+str(row['site'])+'.png'
        plt.imsave(dest,img)
        bar.next()
    bar.finish()

print('------------- valid data -------------------')
moving_files(x_valid_df,'valid')
print('------------- train data -------------------')
moving_files(x_train_df,'train')
