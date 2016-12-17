from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.utils import np_utils
from PIL import Image
import numpy
import pandas as pd
import pylab as p
import csv as csv
from scipy import misc
import os
import random
from matplotlib import pyplot as plt
import h5py



#dir3="/Users/shamitlal/Desktop/shamit/dtu/sem_7/major/testset"
dir1 = '/home/shiv/Desktop/Major1/test_frames'




def getResults():

	#test_x=[]
	test_y=[]  
	video_num = 0		
	#print(len(test_x))
	#dir1="/Users/shamitlal/Desktop/shamit/dtu/sem_7/major/testset"
	for folder in os.listdir(dir1):
		if folder[-1]=='e':
		    continue

		if(folder != 'v72'):
			continue		
				

		dir2=dir1+"/"+folder
		print( dir2)
		framelen=len(os.listdir(dir2))
		if ".DS_Store" in os.listdir(dir2):
		    framelen-=1

		#print("framelen=",framelen)
		num = 0
		test_x=[]
		for frame in os.listdir(dir2):
			if frame[-1]=='e':
				continue
		    	dir3=dir2+"/"+frame
		    	foo=Image.open(dir3)
					    
			test_x.append(list(foo.getdata()))
		
		
	
		X_test = numpy.asarray(test_x)
		#X_test = X_test.reshape(X_test.shape[0], 150, 150, 1)
		X_test = X_test.reshape(X_test.shape[0], 1, 150, 150)		
		X_test = X_test.astype('float32')
		X_test /= 255
		#X_test = X_test.transpose()		
		y_test = model.predict(X_test,batch_size=32,verbose=1)
		
		j = 0	
		threshold = 1e-10
		for frame in os.listdir(dir2):
			if frame[-1]=='e':
				continue
			print (y_test[j])
			if y_test[j]>=threshold:
				img = Image.open(dir2 + "/" + frame)
				img.save('/home/shiv/Desktop/Major1/result_folder/' + str(folder) + "/" + str(frame))
				
			j = j + 1				   	
			 	


    
if __name__ == "__main__":
	
	model = Sequential()
	#model.add(Convolution2D(32, 3, 3, border_mode='same',input_shape=(150,150,1)))
	model.add(Convolution2D(32, 3, 3, border_mode='same',input_shape=(1,150,150)))	
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
	model.add(Convolution2D(32, 3, 3, border_mode='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
	model.add(Convolution2D(32, 3, 3, border_mode='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
	model.add(Flatten())
	model.add(Dense(4096))
	model.add(Activation('relu'))
	model.add(Dense(1024))
	model.add(Activation('relu'))
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.load_weights("/home/shiv/Desktop/Major1/weights.hdf5")	
	model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])
	
	

	getResults()
	
