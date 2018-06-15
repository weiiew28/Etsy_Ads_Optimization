import pandas as pd
import tensorflow as tf
import load_tfrecord
import sys
import time
import numpy as np
""" we first try to embed these higer dimensional textual features """
""" embed the categorical information like shop id, listing id, category id """
"""before using the logisitical regression """

tf.logging.set_verbosity(tf.logging.INFO)
T = 10
embed = 300

""" we will later on add image features (high demensional) and textual features to the regression model"""
#listing_id, timestamp, label, position, num_impression, num_clicks, num_faves, num_carts, smooth_ctr,price, shop_id, title, tag, description, category_id


COLUMNS = ['listing_id','timestamp','position','num_impression','num_clicks','num_faves','num_carts','smooth_ctr','smooth_fvr','smooth_cart','price','shop_id','category_id','label']+[str(i) for i in range(T*embed)]

LABEL_COLUMN = 'label'
train_file = '/home/wqian/Data/Processed/train_test/500000/'+sys.argv[2]
test_file = '/home/wqian/Data/Processed/train_test/500000/'+sys.argv[3]

df_train = pd.read_csv(train_file,names=COLUMNS)
df_test = pd.read_csv(test_file,names=COLUMNS)
df = pd.concat([df_train,df_test])

#df_train = df_train.sample(frac=1)
#df_test = df_test.sample(frac=1)


CONTINUOUS_COLUMNS = ['num_impression','num_clicks','num_faves','num_carts','price','smooth_ctr','smooth_fvr','smooth_cart']+[str(i)  for i in range(T*embed)]
CATEGORICAL_COLUMNS = ['listing_id','timestamp','position','shop_id','category_id']
#INT_COLUMNS = ['num_impression','num_clicks','num_faves','num_carts']
""" need to fix here later, since it is possible that the training csv could be too large to read"""
time_num = df.timestamp.nunique()
position_num = df.position.nunique()
listing_id_num = df.listing_id.nunique()
shop_id_num = df.shop_id.nunique()
category_id_num = df.category_id.nunique()


def input_fn(df):
    continuous_cols = {k: tf.constant(df[k].values,tf.float32) for k in CONTINUOUS_COLUMNS}
    #int_cols = {k: tf.constant(df[k].values,tf.int64) for k in INT_COLUMNS}
    ###  why do we need to create the sparsetensor here
    categorical_cols = {k: tf.SparseTensor(
       indices=[[i, 0] for i in range(df[k].size)],
       values=[str(item) for item in df[k].values],
       dense_shape=[df[k].size, 1])
       for k in CATEGORICAL_COLUMNS}
    #categorical_cols = {k: tf.constant([str(item) for item in df[k].values]) for k in CATEGORICAL_COLUMNS}
    """ feature_cols here is a dictionary of tensors and sparse tensors"""
    label = tf.constant(df[LABEL_COLUMN].values)
    feature_cols = dict(continuous_cols.items()+categorical_cols.items())
    return feature_cols,label

def train_input_fn():
    return input_fn(df_train)
def eval_input_fn():
    return input_fn(df_test)


""" define the model """
vec_embed = [[] for i in range(T*embed)]
timestamp = tf.contrib.layers.sparse_column_with_hash_bucket("timestamp",time_num)
position = tf.contrib.layers.sparse_column_with_hash_bucket("position",position_num)
listing_id = tf.contrib.layers.sparse_column_with_hash_bucket("listing_id",listing_id_num)
shop_id = tf.contrib.layers.sparse_column_with_hash_bucket("shop_id",shop_id_num)
category_id = tf.contrib.layers.sparse_column_with_hash_bucket("category_id",category_id_num)
num_impression = tf.contrib.layers.real_valued_column("num_impression")
num_click = tf.contrib.layers.real_valued_column("num_clicks")
smooth_ctr = tf.contrib.layers.real_valued_column("smooth_ctr")
num_faves = tf.contrib.layers.real_valued_column("num_faves")
num_carts = tf.contrib.layers.real_valued_column("num_carts")
price = tf.contrib.layers.real_valued_column("price")
for i in range(T*embed):
    vec_embed[i]=tf.contrib.layers.real_valued_column(str(i))

wide_columns = [listing_id,shop_id,category_id,timestamp,position]
deep_columns = [
               tf.contrib.layers.embedding_column(listing_id, dimension=10),
               tf.contrib.layers.embedding_column(shop_id, dimension=10),
               tf.contrib.layers.embedding_column(category_id, dimension=10),
               num_impression,num_click,num_faves,num_carts,price]+vec_embed

""" train """
lr_ftrl = float(sys.argv[4])
lr_dnn = float(sys.argv[5])
m = tf.contrib.learn.DNNLinearCombinedClassifier(
    model_dir = "/home/wqian/CTR/Code/serve_savedmodel/wide/",
    linear_feature_columns=wide_columns,
    linear_optimizer=tf.train.FtrlOptimizer(learning_rate=lr_ftrl,learning_rate_power = -0.5,
    initial_accumulator_value=0.5,
    l1_regularization_strength=2.0,
    l2_regularization_strength =1.0),
    dnn_feature_columns=deep_columns,
    dnn_optimizer = tf.train.AdagradOptimizer(learning_rate=lr_dnn),
    dnn_hidden_units=[100, 50])

""" batch_input """
batch_size = 10000
epoch = 24
batch_iter = int(sys.argv[1])

#pass_of_data = int(sys.argv[1])

flag = time.time()

#file_name = '/home/wqian/CTR/Data/Processed/tfrecords/'+sys.argv[1]
#file_queue = tf.train.string_input_producer([file_name])
#information = load_tfrecord.read_decode(file_queue,batch_size)

read_listing_id = tf.placeholder(tf.string,shape=(batch_size))
read_timestamp = tf.placeholder(tf.string,shape=(batch_size))
read_position = tf.placeholder(tf.string,shape=(batch_size))
read_num_impression = tf.placeholder(tf.int64,shape=(batch_size))
read_num_clicks = tf.placeholder(tf.int64,shape=(batch_size))
read_num_faves = tf.placeholder(tf.int64,shape=(batch_size))
read_num_carts = tf.placeholder(tf.int64,shape=(batch_size))
read_smooth_ctr=tf.placeholder(tf.float32,shape=(batch_size))
read_price = tf.placeholder(tf.float32,shape=(batch_size))
read_shop_id = tf.placeholder(tf.string,shape=(batch_size))
read_cont = tf.placeholder(tf.float32,shape=(batch_size,T*embed))
read_category_id = tf.placeholder(tf.string,shape=(batch_size))
batch_D={'listing_id':read_listing_id,'timestamp':read_timestamp,
   'position':read_position,'num_impression':read_num_impression,
   'num_clicks':read_num_clicks,'num_faves':read_num_faves,
    'num_carts':read_num_carts, 'smooth_ctr':read_smooth_ctr,
    'price':read_price, 'shop_id':read_shop_id, 'category_id':read_category_id}
for i in range(T*embed):
    batch_D[str(i)]=read_cont[:,i]
batch_label = tf.placeholder(tf.int64,shape=(batch_size))


f = lambda x: tf.SparseTensor(indices=[[i, 0] for i in range(batch_size)],values=[item for item in x],dense_shape=[batch_size, 1])

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
      for i in range(epoch-1):
         def batch_input():
             cols = sess.run(information)
             cont = np.zeros((batch_size,T*embed))
             for k in range(T*embed):
                 cont[:,k]=cols[12+k]
             bd = sess.run(batch_D,feed_dict={read_listing_id:cols[0],read_timestamp:cols[1],read_position:cols[2],
                                          read_num_impression:cols[3],read_num_clicks:cols[4],read_num_faves:cols[5],
                                         read_num_carts:cols[6],read_smooth_ctr:cols[7],read_price:cols[8],read_shop_id:cols[9],
                                         read_category_id:cols[10],read_cont:cont,batch_label:cols[11]})
             for item in CONTINUOUS_COLUMNS+INT_COLUMNS:
                 bd[item]=tf.convert_to_tensor(bd[item])
             for item in CATEGORICAL_COLUMNS:
                 bd[item]=f(bd[item])
             bl = sess.run(batch_label,feed_dict={batch_label:cols[11]})
             bl = tf.convert_to_tensor(bl)
             return bd,bl
         """ gradient step """
         def batch_file_input():
             batch_file='/home/wqian/Data/Processed/train_test/500000/train_divide/'+str(i+1)+'.csv'
             df_batch = pd.read_csv(batch_file,names=COLUMNS)
             return input_fn(df_batch)
         def validation_input():
             validation_file = '/home/wqian/Data/Processed/train_test/500000/train_divide/'+str(0)+'.csv'
             df = pd.read_csv(validation_file,names=COLUMNS)
             return input_fn(df)
         m.fit(input_fn=batch_file_input, steps=batch_iter)
         m.evaluate(input_fn=validation_input,steps=1)
    sess.close()

def evaluation():
    print "start evaluation"
    M = 6
    root = '/home/wqian/Data/Processed/train_test/500000/divide/'
    out_array = []
    total_instance = 0
    total_actual_click = 0
    total_predict_click = 0
    with tf.Session() as sess:
      for subfile in range(M):
        df_testing = pd.read_csv(root+str(subfile)+'.csv',names=COLUMNS)
        def pred_input_fn():
            return input_fn(df_testing)
        #prediction = m.predict_classes(input_fn=pred_input_fn)
        total_actual_click+=np.sum(df_testing['label'])
        total_instance+=df_testing.shape[0]
        predict_prob = m.predict_proba(input_fn=pred_input_fn)
        predict_prob_matrix = pack(predict_prob)
        total_predict_click+=np.sum(predict_prob_matrix[:,1])
        out_array.append(pack(predict_prob))
      print ("total number of actual clicks", total_actual_click)
      print ("total number of predicted clicks", total_predict_click)
      print ("average actual", float(total_actual_click)/total_instance)
      print ("average predict", float(total_predict_click)/total_instance)
      predict_file = '/home/wqian/Data/Result/500000/'+sys.argv[6]
      np.save(predict_file,np.concatenate(out_array))
      sess.close() 


#training()
evaluation()
