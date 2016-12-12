from PIL import Image
import numpy
from scipy import misc
import os
def load_data():
    train_x=[]
    train_y=[]
    dir1="/Users/shamitlal/Desktop/shamit/dtu/sem_7/major/frames"
    for folder in os.listdir(dir1):
        if folder[-1]=='e':
            continue
        dir2=dir1+"/"+folder
        for image in os.listdir(dir2):
            if image[-1]=='e':
                continue
            dir3=dir2+"/"+image
            foo=Image.open(dir3)
            train_x.append(list(foo.getdata()))
            break
        break
    return train_x
    ()


def getdata():
    trainx=[]
    trainx=load_data()
    trainx=numpy.asarray(trainx) 
    trainx = trainx.reshape(trainx.shape[0],150,150)
    print trainx.shape

getdata()

