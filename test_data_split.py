import glob,shutil

files = sorted(glob.glob('test_data/test/*.png'))

for file in files:
	f = file.split('.')[0]
	f = f.split('_')
	if f[4] == '1':
		dst = 'test_data/s1/'+file.split('/')[2]
		shutil.move(file,dst)
	elif f[4] =='2':
		dst = 'test_data/s2/'+file.split('/')[2]
		shutil.move(file,dst)

print('Done')

