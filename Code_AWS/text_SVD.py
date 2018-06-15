import scipy.linalg as sla
import csv
import pandas as pd
import numpy as np
import gensim
import sys
from gensim.models.keyedvectors import KeyedVectors
word_vectors = KeyedVectors.load('/home/wqian/Data/Googlew2v')

embed = 300

#train_file= '/home/wqian/CTR/Data/Processed/all_info_1000.csv'

def all_number(ss):
    for s in ss:
        if not s in ['0','1','2','3','4','5','6','7','8','9']:
           return False
    return True
def select(W):
    ind = -1
    max_val = W[ind]
    while ind>=-embed and W[ind]>max_val*0.3:
          ind=ind-1
    return ind+1

def reduction(train_file):
    D_title={}
    D={}
    f = open(train_file,'r')

    ff = csv.reader(f)

    for line in ff:
        title_st = line[11]
        title_st = title_st.strip('\'[]').split(' ')
        for word in title_st:
           if not all_number(word):
             try:
               vec = word_vectors[word]
               D_title[word] =vec
             except KeyError:
               continue
    f.close()

    Key_List = [key for key in D_title]
    N = len(Key_List)

    mat = np.zeros((N,embed)) 

    ct = 0
    for key in D_title:
      mat[ct,:]=D_title[key]
      ct+=1

    assert  ct == N

    mat_mat = np.dot(np.transpose(mat),mat)
    [W,V]=sla.eigh(mat_mat)

    mat_transform = np.dot(mat,V)

    ct = 0
    ind = select(np.sqrt(W))
    for key in D_title:
       """ the value is chosen such that the eigenvalue is greater that 10.0, need to tune"""
       
       D[key]=mat_transform[ct,ind:]
       ct+=1
    return D,-ind
   
