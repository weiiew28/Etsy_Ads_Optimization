""" This script read from the pos and neg example of the CTR and CVR respectively
and combine them into the vw training input format """
import csv
import numpy as np
import sys

""" features: listing_id(0),shop_id(1),label(2),shop_history_fave(3),shop_history_carts(4),shop_history_purchases(5),listing_history_fave(6),listing_history_carts(7),listing_history_purchases(8),price(9), description_important_word_embed(10), relevance"""
def to_vw_cvr(file_in_pos,num_pos,file_in_neg,num_neg,file_out):
    shuffle = np.random.permutation(num_pos+num_neg)
    f_pos = open(file_in_pos,'r')
    ff_pos = csv.reader(f_pos)
    f_neg = open(file_in_neg,'r')
    ff_neg = csv.reader(f_neg)
    training_list = [0 for i in range(num_pos+num_neg)]
    ind = 0
    for line in ff_pos:
        """ pos constitutes 2% of the training examples, thus, weight 0.2 """
        instance = "1 0.1 | "
        listing_id = line[0]
        shop_id = line[1]
        shop_history_fave = line[3]
        shop_history_carts = line[4]
        shop_history_purchases = line[5]
        listing_history_fave = line[6]
        listing_history_carts = line[7]
        listing_history_purchases = line[8]
        price = line[9]
        instance += 'listing_id_'+str(listing_id)+' '+'shop_id_'+str(shop_id)+' '+'shop_fave:'+str(shop_history_fave)+' '+'shop_carts:'+str(shop_history_carts)+' '+'shop_purchases:'+str(shop_history_purchases)+' '+'listing_fave:'+str(listing_history_fave)+' '+'listing_carts'+str(listing_history_carts)+' '+'listing_purchases:'+str(listing_history_purchases)+' '
        for num in line[10:]:
            instance+=str(num)+' '
        training_list[shuffle[ind]]=instance
        ind+=1
        if ind==num_pos:
            break
    f_pos.close()
    for line in ff_neg:
        """ pos constitutes 98% of the training examples, thus, weight 10 """
        instance = "-1 4 | "
        listing_id = line[0]
        shop_id = line[1]
        shop_history_fave = line[3]
        shop_history_carts = line[4]
        shop_history_purchases = line[5]
        listing_history_fave = line[6]
        listing_history_carts = line[7]
        listing_history_purchases = line[8]
        price = line[9]
        instance += 'listing_id_'+str(listing_id)+' '+'shop_id_'+str(shop_id)+' '+'shop_fave:'+str(shop_history_fave)+' '+'shop_carts:'+str(shop_history_carts)+' '+'shop_purchases:'+str(shop_history_purchases)+' '+'listing_fave:'+str(listing_history_fave)+' '+'listing_carts'+str(listing_history_carts)+' '+'listing_purchases:'+str(listing_history_purchases)+' '
        for num in line[10:]:
            instance+=str(num)+' '
        training_list[shuffle[ind]]=instance
        ind+=1
        if ind ==num_pos+num_neg:
            break
    f_neg.close()
    g = open(file_out,'w')
    for i in range(num_pos+num_neg):
        if i < num_pos+num_neg-1:
           g.write(training_list[i]+'\n')
        else:
           g.write(training_list[i])
    g.close()




 

""" ctr:
    ['listing_id'(0),'shop_id'(1),'timestamp'(2),'label'(3),'position'(4),'smooth_ctr'(5),'smooth_fvr'(6),'smooth_cart'(7),'price'(8)]+[str(i) for i in range(T*embed)]+['Re'+str(i) for i in range(T)] """


def to_vw_ctr(file_in_pos,num_pos,file_in_neg,num_neg,file_out):
    """ vw is best for online training, shuffle the positive and negative examples """
    shuffle = np.random.permutation(num_pos+num_neg)
    f_pos = open(file_in_pos,'r')
    ff_pos = csv.reader(f_pos)
    f_neg = open(file_in_neg,'r')
    ff_neg = csv.reader(f_neg)
    training_list = [0 for i in range(num_pos+num_neg)]
    ind = 0
    for line in ff_pos:
        """ pos constitutes 1% of the training examples, thus, weight 0.1 """
        instance = "1 0.1 | "
        listing_id = line[0]
        shop_id = line[1]
        timestamp = line[2]
        position = line[4]
        smooth_ctr = line[5]
        smooth_fvr = line[6]
        smooth_cart = line[7]
        price = line[8]
        instance+= 'listing_id_'+str(listing_id)+' '+'shop_id_'+str(shop_id)+' '+'timestamp_'+str(timestamp)+' '+'position:'+str(position)+' '+'smooth_ctr:'+str(smooth_ctr)+' '+'smooth_fvr:'+str(smooth_fvr)+' '+'smooth_cart:'+str(smooth_cart)+' '+'price:'+str(price)+' '
        for num in line[9:]:
            instance+=str(num)+' '
        training_list[shuffle[ind]]=instance
        ind+=1
        if ind == num_pos:
            break
    f_pos.close()
    for line in ff_neg:
        """ neg constitutes 99% of the training examples, thus, weight 10 """
        instance = "-1 5.0 | "
        listing_id = line[0]
        shop_id = line[1]
        timestamp = line[2]
        position = line[4]
        smooth_ctr = line[5]
        smooth_fvr = line[6]
        smooth_cart = line[7]
        price = line[8]
        instance+= 'listing_id_'+str(listing_id)+' '+'shop_id_'+str(shop_id)+' '+'timestamp_'+str(timestamp)+' '+'position:'+str(position)+' '+'smooth_ctr:'+str(smooth_ctr)+' '+'smooth_fvr:'+str(smooth_fvr)+' '+'smooth_cart:'+str(smooth_cart)+' '+'price:'+str(price)+' '
        for num in line[9:]:
            instance+=str(num)+' '
        assert len(line)== 3019
        training_list[shuffle[ind]]=instance
        ind+=1
        if ind == num_pos+num_neg:
            break
    f_neg.close()
    g = open(file_out,'w')
    for i in range(num_pos+num_neg):
        if i < num_pos+num_neg-1:
            g.write(training_list[i]+'\n')
        else:
            g.write(training_list[i])
    g.close()

for i in range(18):
   file_num_pos = str(i)
   file_num_neg = str(19+i)
   print i
   file_in_pos = "/Users/Wei/Documents/Research/Ads_optimization/cvr/pos/"+file_num_pos+".csv"
   file_in_neg = "/Users/Wei/Documents/Research/Ads_optimization/cvr/neg/"+file_num_neg+".csv"
   file_out = "/Users/Wei/Documents/Research/Ads_optimization/cvr/train_batch_vw/"+file_num_neg
   to_vw_cvr(file_in_pos,5000,file_in_neg,5000,file_out)

   file_in_pos = "/Users/Wei/Documents/Research/Ads_optimization/ctr/pos/"+file_num_pos+".csv"
   file_in_neg = "/Users/Wei/Documents/Research/Ads_optimization/ctr/neg/"+file_num_neg+".csv"
   file_out = "/Users/Wei/Documents/Research/Ads_optimization/ctr/train_batch_vw/"+file_num_neg
   to_vw_ctr(file_in_pos,5000,file_in_neg,5000,file_out)
