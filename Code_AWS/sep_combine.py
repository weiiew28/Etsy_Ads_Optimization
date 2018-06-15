import csv
import pandas as pd

T_ctr = 10
T_cvr = 20
embed = 300

ctr_type_dict = {}
for i in range(T_ctr*embed):
    ctr_type_dict[str(i)]=float
for i in range(T_ctr):
    ctr_type_dict['Re'+str(i)]=float
ctr_type_dict['listing_id']=str
ctr_type_dict['shop_id']=str
ctr_type_dict['timestamp']=int
ctr_type_dict['label']=int
ctr_type_dict['position']=int
ctr_type_dict['smooth_ctr']=float
ctr_type_dict['smooth_fvr']=float
ctr_type_dict['smooth_cart']=float
ctr_type_dict['price']=float


cvr_type_dict={}
for i in range(T_cvr*embed):
    cvr_type_dict[str(i)]=float
for i in range(T_cvr):
    cvr_type_dict['Re'+str(i)]=float
cvr_type_dict['listing_id']=str
cvr_type_dict['shop_id']=str
cvr_type_dict['label']=int
cvr_type_dict['shop_history_fave']=float
cvr_type_dict['shop_history_carts']=float
cvr_type_dict['shop_history_purchases']=float
cvr_type_dict['listing_history_fave']=float
cvr_type_dict['listing_history_carts']=float
cvr_type_dict['listing_history_purchases']=float
cvr_type_dict['price']=float






def sep(file_in,out_path,mode=None):
    if mode== 'ctr':
       columns = ['listing_id', 'shop_id', 'timestamp', 'label', 'position', 'smooth_ctr', 'smooth_fvr', 'smooth_cart', 'price']+[str(i) for i in range(T_ctr*embed)]+ ['Re'+str(i) for i in range(T_ctr)]  
       df = pd.read_csv(file_in,names=columns,dtype=ctr_type_dict)
    else:
       columns = ['listing_id','shop_id','label','shop_history_fave','shop_history_carts','shop_history_purchases','listing_history_fave','listing_history_carts','listing_history_purchases','price']+[str(i) for i in range(T_cvr*embed)]+['Re'+str(i) for i in range(T_cvr)] 
       df = pd.read_csv(file_in,names=columns,dtype=cvr_type_dict)
    df_random = df.sample(frac=1)
    total_rows = df.shape[0]
    bucket = total_rows/5000+1
    for i in range(bucket):
        if i<bucket-1:
           inner_df = df.loc[i*5000:(i+1)*5000-1,:]
        else:
           inner_df = df.loc[i*5000:total_rows-1,:]
        inner_df.to_csv(out_path+str(i)+'.csv',index=False,header=False)
    return bucket

def combine(bucket_large,bucket_small,large_path,small_path,out_path,mode=None):
    if mode== 'ctr':
       columns = ['listing_id', 'shop_id', 'timestamp', 'label', 'position', 'smooth_ctr', 'smooth_fvr', 'smooth_cart', 'price']+[str(i) for i in range(T_ctr*embed)]+ ['Re'+str(i) for i in range(T_ctr)]
    else:
       columns = ['listing_id','shop_id','label','shop_history_fave','shop_history_carts','shop_history_purchases','listing_history_fave','listing_history_carts','listing_history_purchases','price']+[str(i) for i in range(T_cvr*embed)]+['Re'+str(i) for i in range(T_cvr)]
    for i in range(bucket_large):
        if mode == 'ctr':
           large = pd.read_csv(large_path+str(i)+'.csv',names=columns,dtype=ctr_type_dict)
           small = pd.read_csv(small_path+str(i%bucket_small)+'.csv',names=columns,dtype=ctr_type_dict)
        if mode == 'cvr':
           large = pd.read_csv(large_path+str(i)+'.csv',names=columns,dtype=cvr_type_dict)
           small = pd.read_csv(small_path+str(i%bucket_small)+'.csv',names=columns,dtype=cvr_type_dict)
        print i
        comb = pd.concat([large,small])
        comb = comb.sample(frac=1)
        comb.to_csv(out_path+str(i)+'.csv',index=False,header=False)
    return 0


#ctr_pos_buck=sep("/home/wqian/Data/train_test/ctr/test_ctr_pos_vec.csv","/home/wqian/Data/train_test/ctr/pos/",'ctr')
#ctr_neg_buck=sep("/home/wqian/Data/train_test/ctr/test_ctr_neg_vec.csv","/home/wqian/Data/train_test/ctr/neg/",'ctr')
#cvr_pos_buck=sep("/home/wqian/Data/train_test/cvr/test_cvr_pos_vec.csv","/home/wqian/Data/train_test/cvr/pos/",'cvr')
#cvr_neg_buck=sep("/home/wqian/Data/train_test/cvr/test_cvr_neg_vec.csv","/home/wqian/Data/train_test/cvr/neg/",'cvr')


ctr_neg_buck = 25
ctr_pos_buck = 25
cvr_neg_buck = 16
cvr_pos_buck = 16

#combine(ctr_neg_buck,ctr_pos_buck,"/home/wqian/Data/CTR/neg/","/home/wqian/Data/CTR/pos/","/home/wqian/Data/CTR/train_batch/",'ctr')
combine(cvr_neg_buck,cvr_pos_buck,"/home/wqian/Data/CVR/neg/","/home/wqian/Data/CVR/pos/","/home/wqian/Data/CVR/train_batch/",'cvr')
    
