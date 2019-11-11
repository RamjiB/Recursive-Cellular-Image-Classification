import os
import threading
from progress.bar import Bar
import shutil

def create_folder(folderName):
	if not os.path.exists(folderName):
		try:
			os.makedirs(folderName)
		except OSError as exc:
			if exc.errno != exc.errno.EEXIST:
				raise

def move(path,thread,mode):
	folders = sorted(os.listdir(path))
	print(len(folders))
	bar = Bar(thread+'_processing',max = len(folders))
	for folder in folders:
		files = sorted(os.listdir(os.path.join(path,folder)))
		for file in files:
			f = file.split('.')[0]
			f = f.split('_')[3]
			src = os.path.join(path,folder,file)
			if f == '1':
				create_folder(mode+'_s1/'+folder)
				dest = mode+'_s1/'+folder+'/'+file
				shutil.move(src,dest)
			elif f =='2':
				create_folder(mode+'_s2/'+folder)
				dest = mode+'_s2/'+folder+'/'+file
				shutil.move(src,dest)
		bar.next()
	bar.finish()
t1 = threading.Thread(target = move,args = ('train_data','t1','tr',))
t2 = threading.Thread(target = move,args = ('valid_data','t2','va',))
t1.start()
t2.start()
t1.join()
t2.join()
print('Done')
