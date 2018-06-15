import tensorflow as tf
import numpy as np
import sys
import pandas as pd

T = 10
embed = 300

predict = np.load(sys.argv[1])
m = predict.shape[0]

prediction = tf.constant(predict[:,1:2])
labels = np.zeros((m,1))

COLUMNS = ['listing_id','timestamp','position','num_impression','num_clicks','num_faves','num_carts','smooth_ctr','price','shop_id','category_id','label']+[str(i) for i in range(T*embed)]
df_test = pd.read_csv(sys.argv[2],names=COLUMNS)

labels[:,0]=df_test['label']
labels = tf.constant(labels)

auc = tf.metrics.auc(labels,prediction)

with tf.Session() as sess:
     sess.run(tf.global_variables_initializer())
     sess.run(tf.local_variables_initializer())
     auc_eval = sess.run(auc)
print "auc:", auc_eval
