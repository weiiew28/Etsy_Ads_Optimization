import csv
import numpy as np
import sys
columns=['query','ts','actual_click','actual_purchase','listing_id','shop_id','price','max_bid', 'budget','bid','score','cpc','quality','relevancy','pace','ctr','cvr']
type_dict={}
type_dict['actual_click']=int
type_dict['actual_purchase']=int
type_dict['ts']=int
type_dict['query']=str
type_dict['listing_id']=str
type_dict['shop_id']=str
type_dict['bid']=float
type_dict['cpc']=float
type_dict['quality']=float
type_dict['relevancy']=float
type_dict['pace']=float
type_dict['price']=float
type_dict['max_bid']=float
type_dict['budget']=float
type_dict['ctr']=float
type_dict['cvr']=float

file_name = sys.argv[1]
""" first component is the index of the shop, and second component is the number of listings it has """
shops = {}
candidates = set()

f = open("/home/ubuntu/auction/"+file_name,'r')
ff = csv.reader(f)
ind = 0
shop_index = 0
listing_index = 0
for line in ff:
    search_id = ind/8
    shop_id = line[5]
    listing_id = line[4]
    ctr = float(line[-2])/10
    cvr = float(line[-1])/10
    price = float(line[6])
    mcpc = float(line[7])
    budget = float(line[8])
    if shop_id in shops:
       candidates.add((search_id,shops[shop_id][0],ctr,cvr,price,mcpc,listing_id))
    else:
      shops[shop_id]=(shop_index,budget)
      candidates.add((search_id,shop_index,ctr,cvr,price,mcpc,listing_id))
      shop_index+=1
    ind+=1


""" construction of the sparse matrix """
S = ind/8
assert search_id+1==S
M = shop_index
total = 0

import cvxopt

from cvxopt import spmatrix
from cvxopt import matrix, solvers

budget_shop = [0 for i in range(M)]
for shop in shops:
    ind = shops[shop][0]
    bud = shops[shop][1]
    budget_shop[ind]=bud

for listing in candidates:
    total+=1
print 'total',total
print "S",S
print "M",M


I = [0 for i in range(2*total)]
J = [0 for i in range(2*total)]
val = [0 for i in range(2*total)]
b = [0 for i in range(total)]
ind = 0
for listing in candidates:
        I[2*ind]=ind
        J[2*ind]=listing[0]
        val[2*ind]=-1
        I[2*ind+1]=ind
        J[2*ind+1]=S+listing[1]
        ctr = listing[2]
        cvr = listing[3]
        price = listing[4]
        mcpc = listing[5]
        val[2*ind+1]=-ctr*mcpc
        b[ind]=-ctr*(mcpc+0.03*cvr*price)
        ind+=1

I_2 = [total+i for i in range(S)]
J_2 = [i for i in range(S)]
val_2 = [-1 for i in range(S)]
b_2 = [0 for i in range(S)]

I_3 = [total+S+i for i in range(M)]
J_3 = [S+i for i in range(M)]
val_3 = [-1 for i in range(M)]
b_3 = [0 for i in range(M)]
mat = spmatrix(val+val_2+val_3,I+I_2+I_3,J+J_2+J_3,(total+S+M,S+M))

c = matrix([1 for i in range(S)]+budget_shop)
h = matrix(b+b_2+b_3)
sol = solvers.lp(c, mat, h ,feastol=1e-4)

np.save('/home/ubuntu/dual_sol/solution',sol['x'])

    
