import numpy as np
import sys,tqdm
import matplotlib.pyplot as plt
from progress.bar import Bar
import threading

sys.path.append('rxrx1-utils/')
import rxrx.io as rio

import warnings
warnings.filterwarnings("ignore")
combined_df = rio.combine_metadata()
combined_df.head()
test_df = combined_df[combined_df['dataset'] == 'test']
test_df = test_df[test_df['well_type'] == 'treatment']
print(test_df.shape)
def moving_files(df,t,mode='test'):
    indexes = df.index
    bar = Bar(t, max=len(df))
    for i in tqdm.tqdm(range(len(df))):
        row = df.iloc[i,:]
        img = rio.load_site_as_rgb(mode,
                                    row['experiment'],
                                    row['plate'],
                                    row['well'],
                                    row['site'])
        img = img.astype(np.uint8)
        dest = mode+'_data/test/'+indexes[i]+'_'+str(row['site'])+'.png'
        bar.next()
        plt.imsave(dest,img)
    bar.finish()

t1 = threading.Thread(target=moving_files, args=(test_df[0:int(test_df.shape[0]/10)],'t1',))
t2 = threading.Thread(target=moving_files, args=(test_df[int(test_df.shape[0]/10):2*int(test_df.shape[0]/10)],'t2',))
t3 = threading.Thread(target=moving_files, args=(test_df[2*int(test_df.shape[0]/10):3*int(test_df.shape[0]/10)],'t3',))
t4 = threading.Thread(target=moving_files, args=(test_df[3*int(test_df.shape[0]/10):4*int(test_df.shape[0]/10)],'t4',))
t5 = threading.Thread(target=moving_files, args=(test_df[4*int(test_df.shape[0]/10):5*int(test_df.shape[0]/10)],'t5',))
t6 = threading.Thread(target=moving_files, args=(test_df[5*int(test_df.shape[0]/10):6*int(test_df.shape[0]/10)],'t6',))
t7 = threading.Thread(target=moving_files, args=(test_df[6*int(test_df.shape[0]/10):7*int(test_df.shape[0]/10)],'t7',))
t8 = threading.Thread(target=moving_files, args=(test_df[7*int(test_df.shape[0]/10):8*int(test_df.shape[0]/10)],'t8',))
t9 = threading.Thread(target=moving_files, args=(test_df[8*int(test_df.shape[0]/10):9*int(test_df.shape[0]/10)],'t9',))
t10 = threading.Thread(target=moving_files, args=(test_df[9*int(test_df.shape[0]/10):],'t10',))
t1.start()
# starting thread 2
t2.start()
t3.start()
t4.start()
t5.start()
t6.start()
t7.start()
t8.start()
t9.start()
t10.start()
# wait until thread 1 is completely executed
t1.join()
# wait until thread 2 is completely executed
t2.join()
t3.join()
t4.join()
t5.join()
t6.join()
t7.join()
t8.join()
t9.join()
t10.join()
print('Done')
