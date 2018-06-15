import re
import csv
import sys

"""

Field of the raw data:

'query(0), 'impression_ts(1), 'visit_guid(2), 'listing_id(3), 'ctr_label(4), 'click_ts(5), 'pos(6), 'purchase_label(7), 'bidding_list(8), 'listing_detail(9), 'num_impressions_total(10), 'listing_history(11), 'shop_history(12), 'shop_id(13), 'max_bid(14), 'max_budget(15)

"""

""" bidding_list: 'bid(0), 'score(1), 'cpc(2), 'quality(3), 'relevancy(4), 'pace(5)"""
""" listing_hostory: 'num_impressions_total(0), 'num_impressions(1), 'num_clicks(2), 'num_faves(3), 'num_carts(4), 'num_purchases(5) """
""" shop_history: 'num_impressions_total(0), 'num_impressions(1), 'num_clicks(2), 'num_faves, 'num_carts, 'num_purchases"""


""" click_impression (0), fave_impression (1), cart_impression(2), listing_fave_click (4), listing_cart_click(5), listing_purchase_click (6), shop_fave_click (7), shop_fave_click (8), shop_purchase_click(9) """    

smooth_values =[0.01922933056775076, 3.28753800130818e-05, 0.0014605131694920596, 0.06733077732544526, 0.0017096476602371502, 0.07595236684637716, 0.031445491466818284, 0.010973286791042735, 0.056776172852402565] 
total_values = [18433247, 354459.0, 606.0, 23866.0, 176219805.0, 1426373.0, 15652.0, 80984.0, 44853.0]



""" generate training features for ctr and cvr model """

def extract_number(st,num_field):
    pattern = '"_[0-9]":([0-9|.|e|-]+)'
    numbers = re.findall(pattern,st)
    try:
       assert len(numbers)==num_field
    except AssertionError:
       print st
    out_num = [0 for i in range(num_field)]
    for i in range(num_field):
        out_num[i]=float(numbers[i])
    return out_num 


def extract_listing_info(st):
    pattern = '"_1":"([\s|\S]+?)"'
    try:
      tag = re.findall(pattern,st)[0]
    except IndexError:
      print st
    pattern = '"_2":"([\s|\S]+?)"'
    try:
      title = re.findall(pattern,st)[0]
    except IndexError:
      title = ''
      print st
    pattern = '"_3":([0-9|.]+),'
    try:
       price = float(re.findall(pattern,st)[0])
    except IndexError:
       price = 0.0  
    pattern = '"_4":"([\s|\S]+?)"'
    try:
       category = re.findall(pattern,st)[0]
    except IndexError:
       category = ''
       print st
    pattern = '"_5":"([\s|\S]+?)"},"tl":{}'
    try:
       description = re.findall(pattern,st)[0]
    except IndexError:
       print st
       description = ''
    return [tag,title,price,category,description]

def generate_csv(raw_file_in,ctr_file_out,cvr_file_out,mode=None,bid_out=None):
    raw_file = open(raw_file_in,'r')
    ctr_f = open(ctr_file_out,'w')
    ctr_ff = csv.writer(ctr_f)
    cvr_f = open(cvr_file_out,'w')
    cvr_ff = csv.writer(cvr_f)
    if mode == 'auction_real' or mode == 'auction_fake':
        bid_f = open(bid_out,'w')
        bid_ff = csv.writer(bid_f)
    for line in raw_file:
        fields = line.strip().split('\t')
        query = fields[0]
        impression_ts = fields[1]
        timestamp = int(float(impression_ts))/(24*60*60)%7
        listing_id = fields[3]
        shop_id = fields[13]
        click_label = int(float(fields[4]))
        position = int(float(fields[6]))
        if fields[7] == 'null':
           purchase_label=0
        else:
           purchase_label = int(float(fields[7]))
        if fields[11] == 'null':
           smooth_ctr = smooth_values[0]
           smooth_fave = smooth_values[1]
           smooth_cart = smooth_values[2]
           smooth_listing_fave_click = smooth_values[3]
           smooth_listing_cart_click = smooth_values[4]
           smooth_listing_purchase_click = smooth_values[5]
        else:
           listing_hist = extract_number(fields[11],6)
           total_num_impression = listing_hist[0]
           total_clicks = listing_hist[2]
           total_faves = listing_hist[3]
           total_carts = listing_hist[4]
           total_purchases = listing_hist[5]
           smooth_ctr = (total_clicks+smooth_values[0])/(total_num_impression+1)
           smooth_fave = (total_faves+smooth_values[1])/(total_num_impression+1)
           smooth_cart = (total_carts+smooth_values[2])/(total_num_impression+1)
           smooth_listing_fave_click = (total_faves+smooth_values[4])/(total_clicks+1)
           smooth_listing_cart_click = (total_carts+smooth_values[5])/(total_clicks+1)
           smooth_listing_purchase_click = (total_purchases+smooth_values[6])/(total_clicks+1)
       
        if fields[12] == 'null':
           smooth_shop_fave_click = smooth_values[6]
           smooth_shop_cart_click = smooth_values[7]
           smooth_shop_purchase_click = smooth_values[8]
        else:
           shop_hist = extract_number(fields[12],6)
           total_num_impression = shop_hist[0]
           total_clicks = shop_hist[2]
           total_faves = shop_hist[3]
           total_carts = shop_hist[4]
           total_purchases = shop_hist[5]
           smooth_shop_fave_click = (total_faves+smooth_values[6])/(total_clicks+1)
           smooth_shop_cart_click = (total_carts+smooth_values[7])/(total_clicks+1)
           smooth_shop_purchase_click = (total_purchases+smooth_values[8])/(total_clicks+1)

        if fields[9]!='null':
           [tag,title,price,category,description] = extract_listing_info(fields[9])
        else:
           tag = ''
           title = ''
           price = 0
           category = ''
           description = ''
        if mode=='training':
           ctr_ff.writerow([listing_id,shop_id,timestamp,click_label,position,smooth_ctr,smooth_fave,smooth_cart,price,tag,query,title])
           cvr_ff.writerow([listing_id,shop_id,purchase_label,smooth_listing_fave_click,smooth_listing_cart_click,smooth_listing_purchase_click,smooth_shop_fave_click,smooth_shop_cart_click,smooth_shop_purchase_click,query,description,tag,title])
        if mode=='auction_real':
           ctr_ff.writerow([listing_id,shop_id,timestamp,click_label,0,smooth_ctr,smooth_fave,smooth_cart,price,tag,query,title])
           cvr_ff.writerow([listing_id,shop_id,purchase_label,smooth_listing_fave_click,smooth_listing_cart_click,smooth_listing_purchase_click,smooth_shop_fave_click,smooth_shop_cart_click,smooth_shop_purchase_click,query,description,tag,title])
           bidding_list = extract_number(fields[8],6)
           """ bidding_list: 'bid(0), 'score(1), 'cpc(2), 'quality(3), 'relevancy(4), 'pace(5)"""
           bid = bidding_list[0]
           score = bidding_list[1]
           cpc = bidding_list[2]
           quality = bidding_list[3]
           relevancy = bidding_list[4]
           pace = bidding_list[5]
           max_bid = float(fields[14])
           if fields[15]!='null':
              max_budget = float(fields[15])
           else:
              max_budget = 1000.0
           bid_ff.writerow([query,impression_ts,click_label,purchase_label,listing_id,shop_id, price, max_bid, max_budget, bid,score,cpc,quality,relevancy,pace])
        if mode == 'auction_fake':
           ctr_ff.writerow([listing_id,shop_id,timestamp,click_label,0,smooth_ctr,smooth_fave,smooth_cart,price,tag,query,title])
           cvr_ff.writerow([listing_id,shop_id,purchase_label,smooth_listing_fave_click,smooth_listing_cart_click,smooth_listing_purchase_click,smooth_shop_fave_click,smooth_shop_cart_click,smooth_shop_purchase_click,query,description,tag,title])
           max_bid = float(fields[14])
           if fields[15]!='null':
              max_budget = float(fields[15])
           else:
              max_budget = 1000.0
           bid_ff.writerow([query,impression_ts,listing_id,shop_id,price,max_bid,max_budget])

    if mode== 'auction_real' or mode=='auction_fake':
        bid_f.close()
    ctr_f.close()
    cvr_f.close()
    raw_file.close()
 

print "raw file: 1, ctr file: 2, cvr file: 3"
generate_csv("/Users/Wei/Documents/Research/Ads_optimization/raw_data/"+sys.argv[1],"/Users/Wei/Documents/Research/Ads_optimization/ctr/auction_predict/"+sys.argv[2],"/Users/Wei/Documents/Research/Ads_optimization/cvr/auction_predict/"+sys.argv[3],'auction_real',"/Users/Wei/Documents/Research/Ads_optimization/auction/"+sys.argv[4])

