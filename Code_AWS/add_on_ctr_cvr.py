import pandas as pd
import numpy as np
import csv
import sys

""" existing field:
   
   query,impression_ts,click_label,purchase_label,listing_id,shop_id, price, max_bid, max_budget, bid,score,cpc,quality,relevancy,pace

"""

columns=['query','ts','actual_click','actual_purchase','listing_id','shop_id','price','max_bid', 'budget','bid','score','cpc','quality','relevancy','pace']
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

date = sys.argv[1]
N = int(float(sys.argv[2]))
f = pd.read_csv("/home/ubuntu/auction/auction_"+date+"_bid.csv",names=columns,dtype=type_dict)
ctr_predict = open('/home/ubuntu/ctr/auction_predict_ctr/out/predict_'+date+'_ctr','r')
ctr_col = np.zeros(N)
ind = 0
for line in ctr_predict:
    ctr_col[ind]=float(line.strip())
    ind+=1
assert ind == N
cvr_predict = open('/home/ubuntu/cvr/auction_predict_cvr/out/predict_'+date+'_cvr','r')
cvr_col = np.zeros(N)
ind = 0
for line in cvr_predict:
    cvr_col[ind]=float(line.strip())
    ind+=1
assert ind == N

print sum(f['actual_purchase'])
f['ctr']=pd.Series(ctr_col,index=f.index)
f['cvr']=pd.Series(cvr_col,index=f.index)


f.to_csv('/home/ubuntu/auction/auction_'+date+'_bid_ctrcvr.csv',index=False,header=False)
ctr_predict.close()
cvr_predict.close()
