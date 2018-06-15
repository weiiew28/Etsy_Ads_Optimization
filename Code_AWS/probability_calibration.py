import pandas as pd 
import numpy as np
import sklearn.calibration as cali
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
import sys

columns = ['query','ts','actual_click','actual_purchase','listing_id','shop_id','price','max_bid','budget','bid','score','cpc','quality','relevancy','pace','ctr','cvr']

def cali(fname,predict_name,out_name,mode='ctr'):
    if mode == 'ctr':
       true_col = 'actual_click'
       prob_col = 'ctr'
    if mode == 'cvr':
       true_col = 'actual_purchase'
       prob_col = 'cvr'
    pred_df = pd.read_csv(predict_name,names=columns)
    nn = pred_df.shape[0]
    df = pd.read_csv(fname,names=columns)
    n = df.shape[0]
    y_true = df[true_col].values
    y_prob = df[prob_col].values
    #fraction_of_positives, mean_predicted_value = cali.calibration_curve(y_true, y_prob, normalize=False, n_bins=10)
    #plt.figure()
    #plt.plot(mean_predicted_value,fraction_of_positives)
    #plt.show()
    #plt.close()
    ir = IsotonicRegression()
    y = ir.fit_transform(y_prob,y_true)
    y_pred = ir.predict(pred_df[prob_col].values)
    nn = y_pred.shape[0]
    h = open(out_name,'w')
    for i in range(nn):
        if i<nn-1:
           h.write(str(y_pred[i])+'\n')
        else:
           h.write(str(y_pred[i]))
    h.close()



date_train = sys.argv[1]
date_test = sys.argv[2]
mode = sys.argv[3]
cali("/home/ubuntu/auction/auction_"+date_train+"_bid_ctrcvr.csv","/home/ubuntu/auction/auction_"+date_test+"_bid_ctrcvr.csv","/home/ubuntu/"+mode+"/auction_predict_"+mode+"/out/calibrated_"+date_test+'_'+mode)

