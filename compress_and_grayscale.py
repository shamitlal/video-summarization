

#dataset https://sites.google.com/site/vsummsite/download
# better method ----(http://www.scipy-lectures.org/advanced/image_processing/)

#run these two lines on bash first to prevent UTF-8 error
'''
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
'''
'''
 Download dataset from here - https://sites.google.com/site/vsummsite/download
 '''


'''
#loop over all images and do this
pil_im = Image.open('/Users/shamitlal/Desktop/thumb00247.jpg').convert('L') # convert to grayscale
pil_im.save('/Users/shamitlal/Desktop/thumb00247_grey.jpg')   #save image in greyscale
#loop over

#loop over all greyscale images to compress
foo = Image.open('/Users/shamitlal/Desktop/thumb00247_grey.jpg')
foo = foo.resize((100,100),Image.ANTIALIAS)
foo.save("/Users/shamitlal/Desktop/thumb00247_grey_comp_100.jpg",quality=95)
#loop ends


face = misc.imread('/Users/shamitlal/Desktop/thumb00247_grey_comp.jpg')  #numpy array 
f=face
#to print image from numpy array
plt.imshow(f,cmap='Greys_r')
plt.show()
'''
'''
run this command to extract frames from video
os.system('ffmpeg -i v100.avi /Users/shamitlal/Desktop/shamit/dtu/sem_7/major/frames/thumb%06d.jpg')
'''
#number of images=149141

from PIL import Image
import numpy
from scipy import misc
import matplotlib.pyplot as plt
import os
dir1 = '/Users/shamitlal/Desktop/shamit/dtu/sem_7/major/frames'
flag=1
cnt=0
os.chdir(dir1)
for filename in os.listdir(dir1):
    if flag==1:
        flag=0
        continue;
    os.chdir(dir1+'/'+filename)
    dir2=dir1+'/'+filename
    for fi in os.listdir(dir2): 
        if fi[-1]=='e':
            continue;   
        foo = Image.open(dir2+'/'+fi).convert('L')
        foo = foo.resize((150,150),Image.ANTIALIAS)
        foo.save(dir2+'/'+fi,quality=95)

