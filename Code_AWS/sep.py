import sys
import csv
import numpy as np


def sep_random(N):
    file_in = sys.argv[1]
    f = open('/home/wqian/Data/Processed/train_test/500000/'+file_in,'r')
    ff = csv.reader(f)

    bucket = [range(20000) for i in range(N)]
    """ size_pointer keeps track of the size in each data bucket """
    size_pointer = [0 for i in range(N)]
    for line in ff:
        ind=np.random.randint(0,N-1)
        bucket[ind][size_pointer[ind]]=line
        size_pointer[ind]+=1

    for i in range(N):      
        new = open('/home/wqian/Data/Processed/train_test/500000/train_divide/'+str(i)+'.csv','w')
        newnew = csv.writer(new)
        for j in range(size_pointer[i]):
            newnew.writerow(bucket[i][j])
        new.close()

    f.close()

def sep_regular():
    window = 10000
    file_in = sys.argv[2]
    f = open('/home/wqian/Data/Processed/train_test/500000/'+file_in,'r')
    ff = csv.reader(f)


    ct = 0
    for line in ff:
        if ct%window ==0:
           ind = ct/window
           new = open('/home/wqian/Data/Processed/train_test/500000/divide/'+str(ind)+'.csv','w')
           newnew = csv.writer(new)
        newnew.writerow(line)
        ct+=1
        if ct%window ==0:
           new.close()
    f.close()


sep_random(int(sys.argv[3]))
#sep_regular()




