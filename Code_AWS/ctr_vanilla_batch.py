import pandas as pd
import tensorflow as tf
import load_tfrecord
import sys
import time
import numpy as np
import multiprocessing
### Input [1] train tfrecord [2] test tfrecord [3]total dataset size [4]pass of data [5] batch iterations [6]train csv [7]test csv

tf.logging.set_verbosity(tf.logging.INFO)

""" check input """

try:
   #print "train tfrecord:", sys.argv[1]
   #print "test tfrecord:", sys.argv[2]
   #print "total dataset size", sys.argv[3]
   #print "pass of data", sys.argv[4]
   print "batch iteration", sys.argv[1]
   print "train csv", sys.argv[2]
   print "test csv", sys.argv[3]
except IndexError:
   print "please check your input"
   sys.exit()





""" using mini batch training """
""" using the tensorflow implementaiton of FTRL """
T = 10
embed = 300

""" we will later on add image features (high demensional) and textual features to the regression model"""

#listing_id(0), timestamp(1), label(2), position(3), num_impression(4), num_clicks(5), num_faves(6), num_carts(7), smooth_ctr(8),smooth_fvr(9),smooth_cart(10),price(11), ship_id(12), title(13), tag(14), description(15), category_id(16)

COLUMNS = ['listing_id','timestamp','position','num_impression','num_clicks','num_faves','num_carts','smooth_ctr','smooth_fvr','smooth_cart','price','shop_id','category_id','label']+[str(i) for i in range(T*embed)]
LABEL_COLUMN = 'label'
CONTINUOUS_COLUMNS = ['num_clicks','num_faves','num_carts','smooth_ctr','smooth_fvr','smooth_cart','price']+[str(i) for i in range(T*embed)]
CATEGORICAL_COLUMNS = ['listing_id','timestamp','position','shop_id','category_id']
INT_COLUMNS = ['num_impression']

train_file = '/home/wqian/Data/Processed/train_test/500000/'+ sys.argv[2]
test_file = '/home/wqian/Data/Processed/train_test/500000/'+sys.argv[3]

df_train = pd.read_csv(train_file,names=COLUMNS)
df_test = pd.read_csv(test_file,names=COLUMNS)
df = pd.concat([df_train,df_test])


""" need to fix here later, since it is possible that the training csv could be too large to read"""
time_num = df.timestamp.nunique()
position_num = df.position.nunique()
listing_id_num = df.listing_id.nunique()
shop_id_num = df.shop_id.nunique()
category_id_num = df.category_id.nunique()


def input_fn(df):
    continuous_cols = {k: tf.constant(df[k].values,tf.float32) for k in CONTINUOUS_COLUMNS}
    int_cols = {k: tf.constant(df[k].values,tf.int64) for k in INT_COLUMNS} 
    categorical_cols = {k: tf.SparseTensor(
       indices=[[i, 0] for i in range(df[k].size)],
       values=[str(item) for item in df[k].values],
       dense_shape=[df[k].size, 1])
       for k in CATEGORICAL_COLUMNS}
    feature_cols = dict(continuous_cols.items()+categorical_cols.items()+int_cols.items())
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
smooth_fvr = tf.contrib.layers.real_valued_column("smooth_fvr")
smooth_cart = tf.contrib.layers.real_valued_column("smooth_cart")
num_faves = tf.contrib.layers.real_valued_column("num_faves")
num_carts = tf.contrib.layers.real_valued_column("num_carts")
price = tf.contrib.layers.real_valued_column("price")
for i in range(T*embed):
    vec_embed[i]=tf.contrib.layers.real_valued_column(str(i))


""" train """
single_cols = [listing_id,shop_id,smooth_ctr]
feature_cols = [position,listing_id,shop_id,category_id,num_impression,num_click, num_faves,num_carts,price]
feature_cols_ctr = [position,listing_id,shop_id,category_id,smooth_ctr,num_faves,num_carts,price]+vec_embed
full_feature_cols = [position,listing_id,shop_id,category_id,num_impression,num_click,num_faves,num_carts,price]+vec_embed

#validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
#    test_set.data,
#    test_set.target,
#    every_n_steps=50)

m = tf.contrib.learn.LinearClassifier(feature_columns=full_feature_cols,
    model_dir = "/home/wqian/CTR/Code/serve_savedmodel/full/", 
    optimizer = tf.train.FtrlOptimizer(learning_rate=float(sys.argv[5]),learning_rate_power = -0.5,
    initial_accumulator_value=0.5,
    l1_regularization_strength=4.0,
    l2_regularization_strength =2.0))



""" preparing for the batch data """
batch_size = 10000
#epoch = int(sys.argv[3])/batch_size
""" manually set"""
epoch = 24
batch_iter = int(sys.argv[1])

#pass_of_data = int(sys.argv[4])

flag = time.time()

#file_name = '/home/wqian/Data/Processed/tfrecords/'+sys.argv[1]
#file_queue = tf.train.string_input_producer([file_name])
#information = load_tfrecord.read_decode(file_queue,batch_size)

#print "read_decode finished, taking: ", time.time()-flag

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

flag = time.time()
#infos = [0 for i in range(pass_of_data*epoch)]

""" batch data in the queue """
#with tf.Session() as sess_retrieve_data:
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(coord=coord)
#         for i in range(pass_of_data*epoch):
#            print "dealing with queue",i
#            cols = sess_retrieve_data.run(information)
#            infos[i]=cols
#            if i ==1:
#              break
#         coord.request_stop()
#         coord.join(threads)
#         sess_retrieve_data.close()
     
#print "batch data retrieve finished, using", time.time()-flag
#model_saver = tf.train.Saver(m.get_variable_names())
#var_list = m.get_variable_names
#model_saver = tf.train.Saver(var_list)

def valid_input():
    file_valid = '/home/wqian/Data/Processed/train_test/500000/train_divide/'+str(0)+'.csv'
    df = pd.read_csv(file_valid,names=COLUMNS)
    return input_fn(df)

def train_eval():
      with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
         for i in range(epoch-1):
             print "inner batch",i
             def batch_input():
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
             flag = time.time()
             m.fit(input_fn=batch_file_input, steps=batch_iter)
             print "single batch training takes:", time.time()-flag
             """ test on the validation set, tuning for the hyperparameter """
             results = m.evaluate(input_fn=valid_input, steps=1)
             for key in sorted(results):
                 print("%s: %s" % (key, results[key]))
             #if i==0:
             #     break
         #""" export model """
         #""" for serving purpose"""
         sess.close()
         #""" source: https://github.com/MtDersvan/tf_playground/blob/master/wide_and_deep_tutorial/wide_and_deep_basic_serving.md"""
         #from tensorflow.contrib.layers import create_feature_spec_for_parsing
         #feature_spec = create_feature_spec_for_parsing(feature_cols)
         #from tensorflow.contrib.learn.python.learn.utils import input_fn_utils
         #serving_input_fn = input_fn_utils.build_parsing_serving_input_fn(feature_spec)
         #servable_model_dir = "serve_savedmodel"
         #servable_model = m.export_savedmodel(servable_model_dir, serving_input_fn)
        
def evaliation():
    """ eval """
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
      predict_file = '/home/wqian/Data/Result/500000/'+sys.argv[4]
      np.save(predict_file,np.concatenate(out_array))
      sess.close()

#flag = time.time()
#train_eval()
#print "training a pass: ", time.time()-flag
evaliation()
#p1 = multiprocessing.Process(target = train_eval)
#p2 = multiprocessing.Process(target = train_eval)

#p1.start()
#p2.start()

#p1.join()
#p2.join()
