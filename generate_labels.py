import os
import numpy as np
import csv

def getFrameNumber(framename):
    frameNumber = 0
    for i in range(5,len(framename)):
        if summaryfile[i]=='.':
            break
        frameNumber = frameNumber*10 + ord(framename[i]) - 48

    return frameNumber

def importanceScore(dist):
    return (float)(1.0)/(dist)
    
dir1 = '/Users/shamitlal/Desktop/shamit/dtu/sem_7/major/UserSummary'
dir2 = '/Users/shamitlal/Desktop/shamit/dtu/sem_7/major/frames'
dir3 = ''
file_location = '/Users/shamitlal/Desktop/shamit/dtu/sem_7/major/labels'
os.chdir(dir1)
maxFrames = 50000
mini=1000000   # minimum framenumber with 1 as importance score
maxi=0         # maximum framenumber with 1 as importance score
frameScoreArray = np.zeros(50000)

for filename in os.listdir(dir1):
    print 'processing file',filename,'-------'
    if filename[-1]=='e':
        continue;
    dir2 = dir1 + '/' + filename
    print dir2
    if dir2[-1]=='t':    # skip last text file
        continue
    os.chdir(dir2)
    mini=1000000
    maxi=0
    frameScoreArray = np.zeros(50000)
    for subfile in os.listdir(dir2):
        if subfile[-1] == 'e':
            continue
        dir3 = dir2 + '/' + subfile

        os.chdir(dir3)
        for summaryfile in os.listdir(dir3):
            if summaryfile[-1] == 'e':
                continue
            frameNumber = getFrameNumber(summaryfile)
            mini=min(mini,frameNumber)
            maxi=max(maxi,frameNumber)
            frameScoreArray[frameNumber] = 1
            #print summaryfile,'  ', frameNumber, '  ', frameScoreArray[frameNumber]

    cnt=0
    importanceWidth=60
    print mini,'  ',maxi
    for i in range(mini,50000):
        cnt += 1

        if frameScoreArray[i] == 1:
            cnt=1
        elif cnt<=importanceWidth:
            frameScoreArray[i] = importanceScore(cnt)

    cnt=1
    for i in range(maxi,0,-1):
        cnt += 1
        if frameScoreArray[i] == 1:
            cnt=1
        elif cnt<=importanceWidth:
            value = importanceScore(cnt)
            frameScoreArray[i]=max(frameScoreArray[i], value)
    
    predictions_file = open(file_location + '/' + filename + '.csv', "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["Score"])


    for i in range(1,50000):
        open_file_object.writerow([frameScoreArray[i]])
    predictions_file.close()


    

    
