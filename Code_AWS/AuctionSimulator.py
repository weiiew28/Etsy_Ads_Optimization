import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys


""" Input Format
query,impression_ts,click_label,purchase_label,listing_id,shop_id, price, max_bid, max_budget, bid,score,cpc,quality,relevancy,pace,ctr,cvr
"""
columns = ['query','ts','actual_click','actual_purchase','listing_id','shop_id','price','max_bid','budget','bid','score','cpc','quality','relevancy','pace','ctr','cvr']


type_dict={}
type_dict['ts']=int
type_dict['query']=str
type_dict['actual_click']=int
type_dict['actual_purchase']=int
type_dict['listing_id']=str
type_dict['shop_id']=str
type_dict['bid']=float
type_dict['score']=float
type_dict['cpc']=float
type_dict['quality']=float
type_dict['relevancy']=float
type_dict['pace']=float
type_dict['price']=float
type_dict['max_bid']=float
type_dict['budget']=float
type_dict['ctr']=float
type_dict['cvr']=float




auction_book = {}
old_auction_book={}   
shop_book={}
old_shop_book={}

class shops():
   def __init__(self,shop_id,budget):
       self.listings = {}
       self.name = shop_id
       self.budget = budget
       self.cost = 0
       self.revenue = 0
       self.alpha = 0
       self.click_count = 0

   def add_listing(self,listing_id,price,max_bid):
       init_bid = min(max_bid,self.budget-self.cost)
       self.listings[listing_id]=(price, max_bid,init_bid)

   def bidding(self,listing_id,plan_pace=None):
       if plan_pace!=None:
          """ update at some check point """
          gamma = 1.0
          self.alpha = min(0.9,self.alpha*np.exp(gamma*(self.cost/self.budget-plan_pace)))
       bid = min(int((1-self.alpha)*self.listings[listing_id][1]),self.budget-self.cost)
       assert bid<=self.budget-self.cost
       return bid

   def attribution(self,listing_id):
       self.revenue+=self.listings[listing_id][0]*0.03

   def click(self,cpc):
       self.click_count+=1
       self.cost+=cpc
       assert self.cost<=self.budget
   
   def budget_usage(self):
       return float(self.cost)/self.budget

class prolist():

   def __init__(self,total_impression,listing_id,budget, price, max_bid):
       self.name = listing_id
       """ ratio will be adjusted dependent on the test size and average daily impression """ 
       self.budget = max_bid
       """ subject to optimize here: self.goal, self.alpha """
       self.max_bid = min(self.budget,max_bid)
       self.goal = min(max(int(self.budget/(0.5*self.max_bid)),1),total_impression)
       self.goal = 3
       self.cost = 0.0
       self.revenue = 0.0
       self.price = price
       self.bid = self.max_bid
       assert self.bid<=self.budget
       self.alpha = 0.5
       self.impression_count = 0
       self.click_count = 0

   def impression(self):
       self.impression_count+=1


   def click(self,cpc):
       self.cost = self.cost+cpc
       self.click_count+=1

   def attribution(self):
       self.revenue = self.revenue+self.price
   
   def bidding(self,current_percentage):
       """ tuning gamma: optimize here """
       gamma = 1.0
       self.alpha = np.min([self.alpha*np.exp(gamma*(float(self.cost)/self.budget-current_percentage)), 0.9])
       self.bid = np.min([int(self.max_bid*(1-self.alpha)),self.budget-self.cost])
       assert self.bid <= self.budget-self.cost
       return self.bid

   def rate_of_return(self):
       if self.cost == 0:
          self.roi = 0
       else:
          self.roi = self.revenue/self.cost    
       return self.roi

   def budget_usage(self):
       self.utilization = self.cost/self.budget
       return self.utilization


def proposed_shop_level(file_in):
    auction_df = pd.read_csv(file_in, names=columns,dtype=type_dict)
    auction_df = pd.read_csv(file_in, names=columns,dtype=type_dict)
    impressions = auction_df.shape[0]/8
    #bidding_revenue_record = np.zeros(impressions/100+2)
    #purchase_revenue_record = np.zeros(impressions/100+2)
    total_bidding_revenue = 0
    total_attribute_revenue = 0
    for i in range(impressions):
        update_point = False
        if (i+1)%10000 == 0:
            update_point = True
        ctr_cvr=['' for j in range(8)]
        for j in range(8):
            listing = auction_df.loc[i*8+j,:]
            ctr = listing.ctr/10
            cvr = listing.cvr/10
            listing_id = listing.listing_id
            shop_id = listing.shop_id
            if shop_id in auction_book:
                if listing_id in auction_book[shop_id].listings:
                    if update_point:
                       pacing = float(i+1)/impressions
                       bid = auction_book[shop_id].bidding(listing_id,pacing)
                    else:
                       bid = auction_book[shop_id].bidding(listing_id)
                else:
                    auction_book[shop_id].add_listing(listing_id,listing.price,listing.max_bid)
                    bid =  auction_book[shop_id].listings[listing_id][2]
            else:
                auction_book[shop_id]=shops(shop_id,listing.budget)
                auction_book[shop_id].add_listing(listing_id,listing.price,listing.max_bid)
                bid =  auction_book[shop_id].listings[listing_id][2]
            score = ctr*(bid+cvr*listing.price*0.03)*100
            ctr_cvr[j]=(listing_id,score,ctr,cvr,bid,listing.price,listing.actual_purchase,listing.actual_click,shop_id)
        sorted_list = sorted(ctr_cvr,key=lambda item: item[1],reverse=True)

        """ end of auction """
        for j in range(4):
            if sorted_list[j][7]==1:
               """ first price auction """
               #auction_book[sorted_list[j][8]].click(sorted_list[j][4])
               """ second price auction """
               cpc = max(1,int(sorted_list[j+1][1]/sorted_list[j][2]/100-sorted_list[j][3]*sorted_list[j][5]*0.03))
               auction_book[sorted_list[j][8]].click(cpc)
               total_bidding_revenue+=cpc
               if sorted_list[j][6]==1:
                  auction_book[sorted_list[j][8]].attribution(sorted_list[j][0])
                  total_attribute_revenue+=sorted_list[j][5]*0.03
        #if (i+1)%100 == 0:
        #   print "finish query {0}".format(i+1)
    return [total_bidding_revenue,total_attribute_revenue]

def proposed_auction(file_in):
    """ we assume that we have preprocessed the auction data such that each impression has only 8 positions prolists"""
    auction_df = pd.read_csv(file_in, names=columns,dtype=type_dict)
    impressions = auction_df.shape[0]/8
    """ log the progressive bidding revenue and purchase revenue """

    bidding_revenue_record = np.zeros(impressions/100+2)
    purchase_revenue_record = np.zeros(impressions/100+2)
    
    total_bidding_revenue = 0
    total_attribute_revenue = 0
    for i in range(impressions):
        """ auction for each search query """
        ctr_cvr=['' for j in range(8)]
        for j in range(8):
            listing = auction_df.loc[i*8+j,:]
            """ this is adjustment due to the biased CTR and CVR prediction """
            ctr = listing.ctr/10
            cvr = listing.cvr/10
            if listing.listing_id in auction_book:
               interval = impressions/auction_book[listing.listing_id].goal
               bid = auction_book[listing.listing_id].bidding((float(i+1)/interval)/auction_book[listing.listing_id].goal)
            else:
               auction_book[listing.listing_id] = prolist(impressions,listing.listing_id,listing.budget,listing.price,listing.max_bid)
               bid = auction_book[listing.listing_id].bid
            auction_book[listing.listing_id].impression()
            score = ctr*(bid+cvr*listing.price*0.03)*100
            """ first price auction here """
            """ will modify for second price auction in the future """
            ctr_cvr[j]=(listing.listing_id,score,ctr,cvr,bid,listing.price,listing.actual_purchase,listing.actual_click)
        """ end of auction """
        sorted_list = sorted(ctr_cvr, key=lambda item: item[1],reverse=True)
        #sorted_ctr = [sorted_list[k][2] for k in range(8)]
        #sorted_cvr = [sorted_list[k][3] for k in range(8)]
        #click = [np.random.binomial(1,p) for p in sorted_ctr]
        #purchase = [np.random.binomial(1,sorted_cvr[k]) if click[k]==1 else 0 for k in range(8)]
        for j in range(4):
            if sorted_list[j][7]==1:
               """ first price auction """
               #auction_book[ctr_cvr[j][0]].click(sorted_list[j][4])
               """ second price auction """
               cpc = max(1,int(sorted_list[j+1][1]/sorted_list[j][2]/100-sorted_list[j][3]*sorted_list[j][5]*0.03))
               auction_book[sorted_list[j][0]].click(cpc)
               total_bidding_revenue+=cpc
               if sorted_list[j][6]==1:
                  auction_book[sorted_list[j][0]].attribution()
                  total_attribute_revenue+=sorted_list[j][5]*0.03
        if (i+1)%100==0:
            print "finish {0} queries".format(i+1)
            bidding_revenue_record[(i+1)/100]=total_bidding_revenue
            purchase_revenue_record[(i+1)/100]=total_attribute_revenue
    bidding_revenue_record[-1]=total_bidding_revenue
    purchase_revenue_record[-1]=total_attribute_revenue
    np.save('/Users/Wei/Documents/Research/Ads_optimization/etsy-ads-optimization/Data/0716_proposed_bidding',bidding_revenue_record)
    np.save('/Users/Wei/Documents/Research/Ads_optimization/etsy-ads-optimization/Data/0716_proposed_purchase',purchase_revenue_record)
    np.save('/Users/Wei/Documents/Research/Ads_optimization/etsy-ads-optimization/Data/0716_proposed_total',bidding_revenue_record+purchase_revenue_record)
    return [total_bidding_revenue,total_attribute_revenue]


def old_shop_level(file_in):
    auction_df = pd.read_csv(file_in, names=columns)
    impressions = auction_df.shape[0]/8
    total_bidding_revenue = 0
    total_attribute_revenue = 0
    for i in range(impressions):
        ctr_cvr=['' for j in range(8)]
        for j in range(8):
            listing = auction_df.loc[i*8+j,:]
            listing_id = listing.listing_id
            shop_id = listing.shop_id
            relevancy = listing.relevancy
            quality = listing.quality
            ctr = listing.ctr/10
            cvr = listing.cvr/10
            if shop_id in old_auction_book:
                if not listing_id in old_auction_book[shop_id].listings:
                   old_auction_book[shop_id].add_listing(listing_id,listing.price,listing.max_bid)
            else:
                old_auction_book[shop_id]=shops(shop_id,listing.budget)
                old_auction_book[shop_id].add_listing(listing_id,listing.price,listing.max_bid)
            bid = min(listing.bid,old_auction_book[shop_id].budget-old_auction_book[shop_id].cost)
            pacing = float(i+1)/impressions+float(old_auction_book[shop_id].budget-old_auction_book[shop_id].cost-bid/2)/old_auction_book[shop_id].budget
            score = bid*quality**1.5*relevancy*pacing
            ctr_cvr[j]=(listing_id,score,ctr,cvr,bid,listing.price,listing.actual_purchase,pacing,quality,relevancy,listing.actual_click,shop_id)
        sorted_list = sorted(ctr_cvr,key=lambda item: item[1],reverse=True)
        """ end of auction"""
        for j in range(4):  
            if sorted_list[j][10]==1:
              """ first price auction """
              #old_auction_book[ctr_cvr[j][0]].click(sorted_list[j][4])
              """ second price auction """
              cpc = max(1,int(sorted_list[j+1][1]/(sorted_list[j][7]*abs(sorted_list[j][8])**1.5*sorted_list[j][9])))
              old_auction_book[sorted_list[j][11]].click(cpc)
              total_bidding_revenue+=cpc
              if sorted_list[j][6]==1:
                 old_auction_book[sorted_list[j][11]].attribution(sorted_list[j][0])
                 total_attribute_revenue+=sorted_list[j][5]*0.03
        
        #if (i+1)%100 == 0:
        #    print "finish query {0}".format(i+1)
    return [total_bidding_revenue,total_attribute_revenue]

    

def old_auction(file_in):
    auction_df = pd.read_csv(file_in, names=columns)
    impressions = auction_df.shape[0]/8
    bidding_revenue_record = np.zeros(impressions/100+2)
    purchase_revenue_record = np.zeros(impressions/100+2)
    upperbound_purchase_revenue = 0
    total_bidding_revenue = 0
    total_attribute_revenue = 0
    original_bidding_revenue = 0
    original_purchase = 0
    for i in range(impressions):
        """ auction for each search query """
        purchase_history = [0 for j in range(8)]
        ctr_cvr=['' for j in range(8)]
        original_ctr_cvr = ['' for j in range(8)]
        for j in range(8):
            listing = auction_df.loc[i*8+j,:]
            relevance = listing.relevancy
            quality = listing.quality
            ctr = listing.ctr/10
            cvr = listing.cvr/10
            if not listing.listing_id in old_auction_book:
               old_auction_book[listing.listing_id] = prolist(impressions,listing.listing_id, listing.budget, listing.price, listing.max_bid)
            old_auction_book[listing.listing_id].impression()
            bid = min(listing.bid,old_auction_book[listing.listing_id].budget-old_auction_book[listing.listing_id].cost)
            pacing = float(i+1)/impressions+float(old_auction_book[listing.listing_id].budget-old_auction_book[listing.listing_id].cost-bid/2)/old_auction_book[listing.listing_id].budget
            score = bid*float(abs(listing.quality))**1.5*relevance*pacing
            ctr_cvr[j]=(listing.listing_id,score,ctr,cvr,bid,listing.price,listing.actual_purchase,pacing,quality,relevance,listing.actual_click)
            original_ctr_cvr[j]=(listing.listing_id,listing.score,listing.cpc,listing.price,listing.actual_click,listing.actual_purchase)
            purchase_history[j] = listing.actual_purchase*listing.price*0.03
        """ end of auction """
        sorted_purchase = sorted(purchase_history,reverse=True)
        sorted_list = sorted(ctr_cvr, key=lambda item: item[1],reverse=True)
        sorted_original_list = sorted(original_ctr_cvr,key=lambda item: item[1],reverse=True)
        #sorted_ctr = [sorted_list[k][2] for k in range(8)]
        #sorted_cvr = [sorted_list[k][3] for k in range(8)]
        #click = [np.random.binomial(1,p) for p in sorted_ctr]
        #purchase = [np.random.binomial(1,sorted_cvr[k]) if click[k]==1 else 0 for k in range(8)]
        for j in range(4):
           if sorted_list[j][10]==1:
              """ first price auction """
              #old_auction_book[ctr_cvr[j][0]].click(sorted_list[j][4])
              """ second price auction """
              cpc = max(1,int(sorted_list[j+1][1]/(sorted_list[j][7]*abs(sorted_list[j][8])**1.5*sorted_list[j][9])))
              old_auction_book[sorted_list[j][0]].click(cpc)
              total_bidding_revenue+=cpc
              if sorted_list[j][6]==1:
                 old_auction_book[sorted_list[j][0]].attribution()
                 total_attribute_revenue+=sorted_list[j][5]*0.03
        """ actual click and purchase in the logged data"""
        for j in range(4):
            if sorted_original_list[j][4]==1:
                original_bidding_revenue+=sorted_original_list[j][2]
                if sorted_original_list[j][5]==1:
                    original_purchase+=sorted_original_list[j][3]*0.03
            upperbound_purchase_revenue+=sorted_purchase[j]
        if (i+1)%100==0:
            print "finish {0} queries".format(i+1)
            bidding_revenue_record[(i+1)/100]=total_bidding_revenue
            purchase_revenue_record[(i+1)/100]=total_attribute_revenue
    bidding_revenue_record[-1]=total_bidding_revenue
    purchase_revenue_record[-1]=total_attribute_revenue
    print "in the logged data, the total bidding revenue for the first 4 slots is", original_bidding_revenue
    print "in the logged data, the total attribution revenue for the first 4 slots is", original_purchase
    np.save('/Users/Wei/Documents/Research/Ads_optimization/etsy-ads-optimization/Data/0716_current_bidding',bidding_revenue_record)
    np.save('/Users/Wei/Documents/Research/Ads_optimization/etsy-ads-optimization/Data/0716_current_purchase',purchase_revenue_record)
    np.save('/Users/Wei/Documents/Research/Ads_optimization/etsy-ads-optimization/Data/0716_current_total',bidding_revenue_record+purchase_revenue_record)
    return [total_bidding_revenue,total_attribute_revenue,upperbound_purchase_revenue]

def average_use(book):
    s=0
    n = len([key for key in book])
    for listing in book:
        s+=book[listing].budget_usage()
    return s/n

def num_impression(book):
    m = 0
    for listing in book:
        if book[listing].impression_count>m:
           m = book[listing].impression_count
    bucket = [0 for i in range(m+1)]
    for listing in book:
        ct = book[listing].impression_count
        bucket[ct]+=1
    return bucket

def num_click(book):                                                                                                       
    m = 0
    for listing in book:
        if book[listing].click_count >m:
           m = book[listing].click_count
    bucket = [0 for i in range(m+1)]
    for listing in book:
        ct = book[listing].click_count
        bucket[ct]+=1
    return bucket



date = sys.argv[1]

[total_bidding_revenue,total_attribute_revenue] = old_shop_level("/Users/Wei/Documents/Research/Ads_optimization/auction/auction_"+date+"_bid_ctrcvr.csv")
print "total_bidding_revenue for proposed model",total_bidding_revenue
print "total_attribute_revenue for proposed model",total_attribute_revenue
print "budget usage for the proposed model", average_use(old_auction_book)
print "budget usage for the old model", average_use(old_auction_book)
click_bucket = num_click(old_auction_book)
print "old model click bucket: ",click_bucket

[total_bidding_revenue,total_attribute_revenue] = proposed_shop_level("/Users/Wei/Documents/Research/Ads_optimization/auction/auction_"+date+"_bid_ctrcvr.csv")
print "total_bidding_revenue for proposed model",total_bidding_revenue
print "total_attribute_revenue for proposed model",total_attribute_revenue
print "budget usage for the proposed model", average_use(auction_book)
[total_bidding_revenue,total_attribute_revenue] = proposed_auction("/Users/Wei/Documents/Research/Ads_optimization/auction/auction_"+date+"_bid_ctrcvr.csv")
click_bucket = num_click(auction_book)
print "proposed model click bucket: ",click_bucket
#print "total_bidding_revenue for proposed model",total_bidding_revenue
#print "total_attribute_revenue for proposed model",total_attribute_revenue 
#print "budget usage for the proposed model", average_use(auction_book)
#[total_bidding_revenue,total_attribute_revenue,upperbound_purchase] = old_auction("/Users/Wei/Documents/Research/Ads_optimization/auction/auction_"+date+"_bid_ctrcvr.csv")
#print "total_bidding_revenue for old model",total_bidding_revenue
#print "total_attribute_revenue for old model",total_attribute_revenue
#print "budget usage for the old model", average_use(old_auction_book)
#print "upper bound for the purchase revenue is", upperbound_purchase
