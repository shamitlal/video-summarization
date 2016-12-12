import os
dir1 = '/Users/shamitlal/Desktop/shamit/dtu/sem_7/major/database'
dir2 = '/Users/shamitlal/Desktop/shamit/dtu/sem_7/major/frames'
os.chdir(dir1)
flag=1;
for filename in os.listdir(dir1):
    if flag==1:
        flag=0
        continue
    file_name=dir1+'/'+filename
    print 'processing ', file_name , '...........\n'
    os.mkdir(dir2+'/'+filename[:3])
    os.system('ffmpeg -i ' + file_name + ' ' + dir2+'/'+filename[:3]+'/thumb%05d.jpg')
    
