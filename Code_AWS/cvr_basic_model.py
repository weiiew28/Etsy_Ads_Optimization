import pandas as pd
import tensorflow as tf
import numpy as np
import sys

tf.logging.set_verbosity(tf.logging.INFO)
window = 20
embed = 300

""" Basic linear model for the purchase model"""

"""features: listing_id(0),shop_id(1),label(2),shop_history_fave(3),shop_history_carts(4),shop_history_purchases(5),listing_history_fave(6),
                 listing_history_carts(7),listing_history_purchases(8),price(9), description_important_word_embed(10), relevance"""


COLUMNS=['listing_id','shop_id','label','shop_history_fave','shop_history_carts','shop_history_purchases','listing_history_fave','listing_history_carts','listing_history_purchases','price']+[str(i) for i in range(window*embed)]+['Re'+str(i) for i in range(window)]

CON_COL = ['shop_history_fave','shop_history_carts','shop_history_purchases','listing_history_fave','listing_history_carts','listing_history_purchases','price']+[str(i) for i in range(window*embed)]+['Re'+str(i) for i in range(window)]
CAT_COL = ['listing_id','shop_id']
LABEL_COL = 'label'


type_dict={}
for i in range(window*embed):
    type_dict[str(i)]=float
for i in range(window):
    type_dict['Re'+str(i)]=float
type_dict['listing_id']=str
type_dict['shop_id']=str
type_dict['label']=int
type_dict['shop_history_fave']=float
type_dict['shop_history_carts']=float
type_dict['shop_history_purchases']=float
type_dict['listing_history_fave']=float
type_dict['listing_history_carts']=float
type_dict['listing_history_purchases']=float
type_dict['price']=float



""" as I mentioned before, the following could cause problem if it becomes too large to fit in the memory """
listing_id_num = 80000000
shop_id_num = 80000000


def input_fn(df):
    continuous_cols = {k: tf.constant(df[k].values,tf.float32) for k in CON_COL}
    categorical_cols = {k: tf.SparseTensor(
       indices=[[i, 0] for i in range(df[k].size)],
       values=[str(item) for item in df[k].values],
       dense_shape=[df[k].size, 1])
       for k in CAT_COL}
    feature_cols = dict(continuous_cols.items()+categorical_cols.items())
    label = tf.constant(df[LABEL_COL].values)
    return feature_cols,label


listing_id = tf.contrib.layers.sparse_column_with_hash_bucket("listing_id",listing_id_num)
shop_id = tf.contrib.layers.sparse_column_with_hash_bucket("shop_id",shop_id_num)
shop_history_fave = tf.contrib.layers.real_valued_column("shop_history_fave")
shop_history_purchases = tf.contrib.layers.real_valued_column("shop_history_purchases")
shop_history_carts = tf.contrib.layers.real_valued_column("shop_history_carts")
listing_history_fave = tf.contrib.layers.real_valued_column("listing_history_fave")
listing_history_carts = tf.contrib.layers.real_valued_column("listing_history_carts")
listing_history_purchases = tf.contrib.layers.real_valued_column("listing_history_purchases")
price =  tf.contrib.layers.real_valued_column("price")
vec_embed =[tf.contrib.layers.real_valued_column(str(i)) for i in range(window*embed)]
query_relevance = [tf.contrib.layers.real_valued_column('Re'+str(i)) for i in range(window)]

feature_cols = [listing_id, shop_id, shop_history_fave,shop_history_purchases, shop_history_carts, listing_history_fave,listing_history_carts,listing_history_purchases,price]+vec_embed+query_relevance

m = tf.contrib.learn.LinearClassifier(feature_columns=feature_cols,
    model_dir = "/Users/wqian/Documents/Etsy/Saved_Model/cvr_ftrl/",
    optimizer = tf.train.FtrlOptimizer(learning_rate=0.01,learning_rate_power = -0.5,
    initial_accumulator_value=0.5,
    l1_regularization_strength=2.0,
    l2_regularization_strength =1.0))    

epoch = 35
batch_iter = 50

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



def training():
    with tf.Session() as sess:
         for i in range(epoch):
             def train_batch():
                 df_train_batch = pd.read_csv('/home/wqian/Data/train_test/cvr/train_batch/'+str(i)+'.csv', names=COLUMNS,dtype=type_dict)
                 return input_fn(df_train_batch)
             m.fit(input_fn=train_batch, steps=batch_iter)
         sess.close()

def evaluation():
    out_array = []
    label_array = []
    with tf.Session() as sess:
        for i in range(10):
            df_test_batch = pd.read_csv('/home/wqian/Data/train_test/cvr/train_batch/'+str(35+i)+'.csv',names=COLUMNS)
            def pred_input_fn():
               return input_fn(df_test_batch)
            num_points = df_test_batch.shape[0]
            actual_label = df_test_batch['label'].values.reshape((num_points,1))
            predict_prob = m.predict_proba(input_fn=pred_input_fn)
            predict_prob_matrix = pack(predict_prob)
            out_array.append(predict_prob_matrix)
            label_array.append(actual_label)
        predict_file = '/home/wqian/Data/train_test/cvr/train_batch/'+'testing_prob'
        np.save(predict_file,np.concatenate(out_array))
        sess.close()
    auc = tf.metrics.auc(tf.constant(np.concatenate(label_array)),tf.constant(np.concatenate(out_array)[:,1]))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        auc_eval = sess.run(auc)
        print "auc",auc_eval
        sess.close()

      

def auction_prediction(N):
    root = '/Users/wqian/Documents/Etsy/Data/cvr/'
    predict_array =[]
    with tf.Session() as sess:
       for i in range(N):
           df_predict = pd.read_csv(root+str(i)+'.csv',names=COLUMNS,dtype=type_dict)
           def pred_input_fn():
               return input_fn(df_predict)
           predict_prob = m.predict_proba(input_fn=pred_input_fn)
           predict_prob_matrix = pack(predict_prob)
           predict_array.append(predict_prob_matrix)
       np.save(root+'predict_prob',np.concatenate(predict_array))
       sess.close()


#training()
#evaluation()
auction_prediction(1)
