import pandas as pd
import tensorflow as tf
import sys
import time
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

T = 10
embed = 300

COLUMNS = ['listing_id','shop_id','timestamp','label','position','smooth_ctr','smooth_fvr','smooth_cart','price']+[str(i) for i in range(T*embed)]+['Re'+str(i) for i in range(T)]
LABEL_COLUMN = 'label'
CONTINUOUS_COLUMNS = ['smooth_ctr','smooth_fvr','smooth_cart','price']+[str(i) for i in range(T*embed)]+['Re'+str(i) for i in range(T)]
CATEGORICAL_COLUMNS = ['listing_id','timestamp','position','shop_id']
type_dict = {}
for i in range(T*embed):
    type_dict[str(i)]=float
for i in range(T):
    type_dict['Re'+str(i)]=float
type_dict['listing_id']=str
type_dict['shop_id']=str
type_dict['timestamp']=int
type_dict['label']=int
type_dict['position']=int
type_dict['smooth_ctr']=float
type_dict['smooth_fvr']=float
type_dict['smooth_cart']=float
type_dict['price']=float

"""upper bound for number of fields"""
time_num = 7
position_num = 20
listing_id_num = 80000000
shop_id_num = 80000000


def input_fn(df):
    continuous_cols = {k: tf.constant(df[k].values,tf.float32) for k in CONTINUOUS_COLUMNS}
    categorical_cols = {k: tf.SparseTensor(
       indices=[[i, 0] for i in range(df[k].size)],
       values=[str(item) for item in df[k].values],
       dense_shape=[df[k].size, 1])
       for k in CATEGORICAL_COLUMNS}
    feature_cols = dict(continuous_cols.items()+categorical_cols.items())
    label = tf.constant(df[LABEL_COLUMN].values)
    return feature_cols,label

def train_input_fn():
    return input_fn(df_train)
def eval_input_fn():
    return input_fn(df_test)



def pack(L):
    ct = 0
    record = []
    for item in L:
        ct+=1
        record.append(item)
    A = np.zeros((ct,2))
    ind = 0
    for item in record:
        A[ind,:]=item
        ind+=1
    return A




""" defining the model """

vec_embed = [[] for i in range(T*embed)]
rel_vec = [[] for i in range(T)]
timestamp = tf.contrib.layers.sparse_column_with_hash_bucket("timestamp",time_num)
position = tf.contrib.layers.sparse_column_with_hash_bucket("position",position_num)
listing_id = tf.contrib.layers.sparse_column_with_hash_bucket("listing_id",listing_id_num)
shop_id = tf.contrib.layers.sparse_column_with_hash_bucket("shop_id",shop_id_num)
#category_id = tf.contrib.layers.sparse_column_with_hash_bucket("category_id",category_id_num)
smooth_ctr = tf.contrib.layers.real_valued_column("smooth_ctr")
smooth_fvr = tf.contrib.layers.real_valued_column("smooth_fvr")
smooth_cart = tf.contrib.layers.real_valued_column("smooth_cart")
price = tf.contrib.layers.real_valued_column("price")
for i in range(T*embed):
    vec_embed[i]=tf.contrib.layers.real_valued_column(str(i))
for i in range(T):
    rel_vec[i]=tf.contrib.layers.real_valued_column('Re'+str(i))



full_feature_cols = [position,listing_id,shop_id,timestamp,smooth_ctr, smooth_fvr, smooth_cart, price]+vec_embed+rel_vec

m = tf.contrib.learn.LinearClassifier(feature_columns=full_feature_cols,
    model_dir = "/home/wqian/saved_model/click_model_ftrl/",
    optimizer = tf.train.FtrlOptimizer(learning_rate=float(0.1),learning_rate_power = -0.5,
    initial_accumulator_value=0.5,
    l1_regularization_strength=4.0,
    l2_regularization_strength =2.0))


""" training detail"""
epoch = 30
batch_iter = 50


def train():
      with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
         for i in range(epoch):
             print "inner batch",i
             def batch_file_input():
                 batch_file='/home/wqian/Data/train_test/ctr/train_batch/'+str(120+i)+'.csv'
                 df_batch = pd.read_csv(batch_file,names=COLUMNS,dtype=type_dict)
                 return input_fn(df_batch)
             m.fit(input_fn=batch_file_input, steps=batch_iter)
def evaluation():
    M = 15
    root = '/home/wqian/Data/train_test/ctr/train_batch/'
    label_array =[]
    out_array = []
    total_instance = 0
    total_actual_click = 0
    total_predict_click = 0
    with tf.Session() as sess:
      for subfile in range(M):
        df_testing = pd.read_csv(root+str(180+subfile)+'.csv',names=COLUMNS,dtype=type_dict)
        label_array.append(df_testing['label'].values.reshape((df_testing.shape[0],1)))
        def pred_input_fn():
            return input_fn(df_testing)
        predict_prob = m.predict_proba(input_fn=pred_input_fn)
        predict_prob_matrix = pack(predict_prob)
        out_array.append(pack(predict_prob_matrix))
      #print tf.constant(np.concatenate(label_array)).shape
      #print tf.constant(np.concatenate(out_array)).shape 
      np.save(root+'testing_prob',np.concatenate(out_array)) 
      sess.close()   
    auc = tf.metrics.auc(tf.constant(np.concatenate(label_array)),tf.constant(np.concatenate(out_array)[:,1]))
    with tf.Session() as sess:
       sess.run(tf.global_variables_initializer())
       sess.run(tf.local_variables_initializer())
       auc_eval = sess.run(auc)
       print "auc",auc_eval
       sess.close()


def auction_prediction(N):
    root = '/home/wqian/Data/train_test/ctr/'
    predict_array=[]
    with tf.Session() as sess:
       for i in range(N):
           df_predict = pd.read_csv(root+str(i)+'.csv',names=COLUMNS,dtype=type_dict)
           def pred_input_fn():
              return input_fn(df_predict)
           predict_prob = m.predict_proba(input_fn=pred_input_fn)
           predict_prob_matrix = pack(predict_prob)
           predict_array.append(predict_prob_matrix)
       np.save(root+'auction_ctr_predict/predict_prob',np.concatenate(predict_array))
       sess.close()


#train()
#evaluation()
auction_prediction(53)
