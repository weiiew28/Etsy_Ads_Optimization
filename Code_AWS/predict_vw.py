""" This script converts the prediction csv file of ctr and cvr to vw form """
import csv
import numpy as np
import sys


""" features: listing_id(0),shop_id(1),label(2),shop_history_fave(3),shop_history_carts(4),shop_history_purchases(5),listing_history_fave(6),listing_history_carts(7),listing_history_purchases(8),price(9), description_important_word_embed(10), relevance"""
def cvr_to_vw(file_in,file_out,N):
    f = open(file_in,'r')
    ff = csv.reader(f)
    g = open(file_out,'w')
    ind = 0
    for line in ff:
        instance = str(line[2])+" | "
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
        if ind<N-1:
           g.write(instance+'\n')
        else:
           g.write(instance)
    f.close()
    g.close()


""" features: ['listing_id'(0),'shop_id'(1),'timestamp'(2),'label'(3),'position'(4),'smooth_ctr'(5),'smooth_fvr'(6),'smooth_cart'(7),'price'(8)]+[str(i) for i in range(T*embed)]+['Re'+str(i) for i in range(T)] """
def ctr_to_vw(file_in,file_out,N):
    f = open(file_in,'r')
    ff = csv.reader(f)
    g = open(file_out,'w')
    ind = 0
    for line in ff:
        instance = str(line[3])+" | "
        listing_id = line[0]
        shop_id = line[1]
        timestamp = line[2]
        position = line[4]
        smooth_ctr = line[5]
        smooth_fvr = line[6]
        smooth_cart = line[7]
        price = line[8]
        instance += 'listing_id_'+str(listing_id)+' '+'shop_id_'+str(shop_id)+' '+'timestamp_'+str(timestamp)+' '+'position:'+str(position)+' '+'smooth_ctr:'+str(smooth_ctr)+' '+'smooth_fvr:'+str(smooth_fvr)+' '+'smooth_cart:'+str(smooth_cart)+' '+'price:'+str(price)+' '
        for num in line[9:]:
            instance+=str(num)+' '
        if ind<N-1:
            g.write(instance+'\n')
        else:
            g.write(instance)
    f.close()
    g.close()




ctr_to_vw('/Users/Wei/Documents/Research/Ads_optimization/ctr/auction_predict/auction_test_ctr_vec.csv','/Users/Wei/Documents/Research/Ads_optimization/ctr/auction_predict/auction_test_ctr_vw',int(float(sys.argv[1])))
cvr_to_vw('/Users/Wei/Documents/Research/Ads_optimization/cvr/auction_predict/auction_test_cvr_vec.csv','/Users/Wei/Documents/Research/Ads_optimization/cvr/auction_predict/auction_test_cvr_vw',int(float(sys.argv[1])))
