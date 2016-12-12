#export LC_ALL=en_US.UTF-8
#export LANG=en_US.UTF-8
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


def gettestdata():
    test_x=[]
    test_y=[]  
    li=[]
    print(len(test_x))
    dir1="/Users/shamitlal/Desktop/shamit/dtu/sem_7/major/testset"
    dir1y="/Users/shamitlal/Desktop/shamit/dtu/sem_7/major/labels"
    for folder in os.listdir(dir1):
        if folder[-1]=='e':
            continue

        dir2=dir1+"/"+folder
        dir2y=dir1y+"/"+folder
        print( dir2)
        framelen=len(os.listdir(dir2))
        if ".DS_Store" in os.listdir(dir2):
            framelen-=1
        
        
        with open(dir2y+".csv", 'rb') as f:
            reader=csv.reader(f)
            csvreader=list(reader)

        li=[]
        for ii in range(1,framelen+1):
            val=float(csvreader[ii][0])
            test_y.append(val)

        print("framelen=",framelen)

        for frame in os.listdir(dir2):
            if frame[-1]=='e':
                continue
            dir3=dir2+"/"+frame
            foo=Image.open(dir3)
            #plt.imshow(foo)
            test_x.append(list(foo.getdata()))

        print(len(test_x),len(test_y))


    print("test_y,test_x shape=",len(test_y),len(test_x))
    return test_x,test_y
    
    
def getTrainData(s1,s2,s3):
    train_x=[]
    train_y=[]  
    li=[]
    folders=[s1,s2,s3]
    dir1="/Users/shamitlal/Desktop/shamit/dtu/sem_7/major/trainset"
    dir1y="/Users/shamitlal/Desktop/shamit/dtu/sem_7/major/labels"
    for folder in folders:
        if folder[-1]=='e':
            continue

        dir2=dir1+"/"+folder
        dir2y=dir1y+"/"+folder
        print( dir2)
        framelen=len(os.listdir(dir2))
        if ".DS_Store" in os.listdir(dir2):
            framelen-=1
        
        with open(dir2y+".csv", 'rb') as f:
            reader=csv.reader(f)
            csvreader=list(reader)

        for ii in range(1,framelen+1):
            val=float(csvreader[ii][0])
            train_y.append(val)

        print("framelen=",framelen)

        for frame in os.listdir(dir2):
            if frame[-1]=='e':
                continue
            dir3=dir2+"/"+frame
            foo=Image.open(dir3)
            #plt.imshow(foo)
            train_x.append(list(foo.getdata()))
        print(len(train_x),len(train_y))

    print("train_y,train_x shape=",len(train_y),len(train_x))
    return train_x,train_y


x_axis = []

def drawErrorGraph(trainmetrics,testmetrics):
	cnt = len(trainmetrics)	
	fig = plt.figure()
	plt.plot(x_axis,trainmetrics,'r',x_axis,testmetrics,'b')
	fig.savefig('/home/shiv/Desktop/Major1/figures/epoch_'+str(cnt)+'.png', dpi=fig.dpi)


X_test,Y_test=gettestdata()
X_test=numpy.asarray(X_test)
Y_test=numpy.asarray(Y_test)
X_test = X_test.reshape(X_test.shape[0], 1, 150, 150)
X_test = X_test.astype('float32')
X_test /= 255

batch_size = 64
nb_epoch = 2000

model = Sequential()

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
model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])


#checkpointer = ModelCheckpoint(filepath="/Users/shamitlal/Desktop/shamit/dtu/sem_7/major/weights.hdf5", verbose=1, save_best_only=True)

dir1="/Users/shamitlal/Desktop/shamit/dtu/sem_7/major/trainset/"

testmetrics=[]
trainmetrics=[]
trainMetricsOneEpoch=[]
for i in range(nb_epoch):
    print("Processing epoch:",i+1);
    l=os.listdir(dir1)
    random.shuffle(l)
    j=0
    cnt=0
    for j in range(len(l)):
        if l[j]==".DS_Store":
            del(l[j])
            break

    j=0
    trainloss=0
    while j<39:
        #trainMetricsOneEpoch=[]   # clear the list
        X_train,Y_train=getTrainData(l[j],l[j+1],l[j+2])
        X_train=numpy.asarray(X_train)
        Y_train=numpy.asarray(Y_train)
        X_train = X_train.reshape(X_train.shape[0], 1, 150, 150)
        X_train = X_train.astype('float32')
        X_train /= 255
        j+=3
        ob=model.fit(X_train,Y_train,batch_size=64, nb_epoch=1)
        trainloss+=ob.history['loss'][0]
        #trainMetricsOneEpoch.append(ob.history['loss'][0])
        #remove these lines after testing
        #print(trainMetricsOneEpoch)
        #print(model.metrics_names) prints names of attributes returned by model.evaluate
        #score = model.evaluate(X_test, Y_test, batch_size=32)
        #print(score)
        #break
        #remove these lines after testing


    trainmetrics.append(float((trainloss)/13.0))
    score = model.evaluate(X_test, Y_test, batch_size=32)
    testmetrics.append(score[0])
    print(trainmetrics)
    print(testmetrics)
    x_axis.append(i+1)
    if i%5==0:
        drawErrorGraph(trainmetrics,testmetrics)

