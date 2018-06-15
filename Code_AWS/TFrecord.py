""" convert the data to tfrecords"""

import tensorflow as tf
import sys

train_file = '/home/wqian/CTR/Data/Processed/train_test/'+sys.argv[1]

T=10
embed = int(sys.argv[3])

COLUMNS = ['listing_id','timestamp','position','num_impression','num_clicks','num_faves','num_carts','smooth_ctr','price','shop_id','category_id','label']+[str(i) for i in range(T*embed)]

LABEL_COLUMN = 'label'

#df_train = pd.read_csv(train_file,names=COLUMNS)
#df_test = pd.read_csv(test_file,names=COLUMNS)
#df = pd.concat([df_train,df_test])



def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


import csv

f_train = open(train_file,'r')
f_train_reader = csv.reader(f_train)

#CHECK_ID = []

writer = tf.python_io.TFRecordWriter('/home/wqian/CTR/Data/Processed/tfrecords/'+sys.argv[2])
for line in f_train_reader:
    listing_id = line[0]
    timestamp = line[1]
    position = line[2]
    num_impression = int(float(line[3]))
    num_clicks = int(float(line[4]))
    num_faves = int(float(line[5]))
    num_carts = int(float(line[6]))
    smooth_ctr = float(line[7])
    price = float(line[8])
    shop_id = line[9]
    category_id = line[10]
    label = int(float(line[11]))
    vec_embed = [float(line[12+i]) for i in range(T*embed)]
    F_D = {}
    F_D['listing_id']=_bytes_feature(listing_id)
    F_D['timestamp']=_bytes_feature(timestamp)
    F_D['position']=_bytes_feature(position)
    F_D['num_impression']=_int64_feature(num_impression)
    F_D['num_clicks']=_int64_feature(num_clicks)
    F_D['num_faves']=_int64_feature(num_faves)
    F_D['num_carts']=_int64_feature(num_carts)
    F_D['smooth_ctr']=_float_feature(smooth_ctr)
    F_D['price']=_float_feature(price)
    F_D['shop_id']=_bytes_feature(shop_id)
    F_D['category_id']=_bytes_feature(category_id)
    F_D['label']=_int64_feature(label)
    for i in range(T*embed):
        F_D[str(i)]=_float_feature(vec_embed[i])
    example = tf.train.Example(features = tf.train.Features(feature=F_D))
    writer.write(example.SerializeToString())

writer.close()
f_train.close()


### Sanity Check
#record_iterator = tf.python_io.tf_record_iterator('/home/wqian/CTR/Data/Processed/train_1000.tfrecords')
#L_ID =[]
#for string_record in record_iterator:
#    example = tf.train.Example()
#    example.ParseFromString(string_record)
#    l_id = (example.features.feature['price']
#                                  .float_list
#                                  .value[0])
#    L_ID.append(l_id)

#import numpy as np

#print np.allclose(CHECK_ID,L_ID)
