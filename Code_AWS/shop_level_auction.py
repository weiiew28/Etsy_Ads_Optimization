import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import csv

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
com_pos = int(float(sys.argv[3]))
alpha = float(sys.argv[6])
gamma_ratio = float(sys.argv[7])
set_budget = sys.argv[8]
if set_budget != "False":
   set_budget = float(sys.argv[8])
else:
   set_budget = False

click_penalty = float(sys.argv[9])
max_ratio = float(sys.argv[10])

class shops():
   def __init__(self,shop_id,budget,check_points):
       self.listings = {}
       self.name = shop_id
       """ for small scale experiment, everybody is subject to the same amount of budget """
       if set_budget:
          self.budget = int(set_budget*budget)
          """ for large scale experiment, will use the logged budget """
       else:
          self.budget = budget
       self.last_cost = 0
       self.cost = 0
       self.revenue = 0
       self.init_alpha = alpha
       self.alpha = alpha
       self.click_count = 0
       self.purchase_count = 0
       self.usage = [0 for i in range(check_points)]
       self.exhaust = 0 
       self.gamma = gamma_ratio*float(1)/(self.init_alpha*check_points)
   
   def budget_resest(budget):
       if set_budget:
          self.budget = int(set_budget*budget)
          """ for large scale experiment, will use the logged budget """
       else:
          self.budget = budget
   def check_point_reset(self,check_points):
       self.gamma = gamma_ratio*float(1)/(self.init_alpha*check_points)
       self.usage = [0 for i in range(check_points)]

   def daily_reset(self,init_alpha=None):
       """ average of final alpha of the previous day and the init alpha of the previous day """
       if init_alpha:
          self.alpha = init_alpha
          self.init_alpha = init_alpha
       else:
          self.alpha = 0.2*self.alpha+0.8*self.init_alpha
          self.init_alpha = self.alpha
       self.last_cost = 0
       self.cost = 0
       self.revenue = 0
       self.click_count = 0
       self.purchase_count = 0
       self.exahust = 0

   def add_listing(self,listing_id,price,max_bid):
       init_bid = min(max_bid,self.budget-self.cost)
       self.listings[listing_id]=(price, max_bid,init_bid)

   def update(self,plan_pace):
       """ if the budget is exhaust, then there is no need to update """
       if plan_pace>0:
          self.alpha = min(0.9,self.alpha*np.exp(self.gamma*((self.cost-self.last_cost-plan_pace)/plan_pace)))
       self.last_cost = self.cost
       #self.alpha = min(0.9,self.alpha*np.exp(gamma*(self.cost/self.budget-plan_pace)))

   def bidding(self,listing_id):
       bid = min(int((1-self.alpha)*self.listings[listing_id][1]),self.budget-self.cost)
       assert bid<=self.budget-self.cost
       return bid

   def ecpc(self):
       """empirical cost per click"""
       return self.cost/self.click_count

   def attribution(self,listing_id):
       self.purchase_count+=1
       self.revenue+=self.listings[listing_id][0]*0.03

   def click(self,cpc):
       if self.cost+cpc<=self.budget:
          self.click_count+=1
          self.cost+=cpc
          return True
       else:
          return False
       #try:
       #   assert self.cost<=self.budget
       #except AssertionError:
       #    self.exhaust = self.cost-self.budget
       #    print "out of budget", self.name
       #    print "cost", self.cost
       #    print "budget", self.budget
       #    print "cpc", cpc
   
   def budget_usage(self):
       return float(self.cost)/self.budget


def proposed_shop_level(file_in,auction_book,ranking_prop):
    auction_df = pd.read_csv(file_in, names=columns,dtype=type_dict)
    auction_df = pd.read_csv(file_in, names=columns,dtype=type_dict)
    impressions = auction_df.shape[0]/8
    """ competitive_ration evaluates how competitive the auction is """
    first_bidding_revenue = 0
    total_bidding_revenue = 0
    top_score = 0
    actual_score = 0
    total_attribute_revenue = 0
    check_frequency = int(float(sys.argv[2]))
    check_points = impressions/check_frequency
    for shop in auction_book:
        auction_book[shop].check_point_reset(check_points)
    for i in range(impressions):
        """ gloabally update all the alphas at the check point """
        if (i+1)%check_frequency == 0:
            for shop in auction_book:
                remaining_budget = auction_book[shop].budget-auction_book[shop].cost
                remaining_checkpoints = check_points-(i+1)/check_frequency+1
                auction_book[shop].update(float(remaining_budget)/remaining_checkpoints)
                auction_book[shop].usage[(i+1)/check_frequency-1]=auction_book[shop].cost/auction_book[shop].budget
        ctr_cvr=['' for j in range(8)]
        for j in range(8):
            listing = auction_df.loc[i*8+j,:]
            """ using production model ctr training """
            ctr = abs(listing.quality)/10
            #ctr = listing.ctr/10
            cvr = listing.cvr/10
            listing_id = listing.listing_id
            shop_id = listing.shop_id
            if shop_id in auction_book:
                """ check if budget is consistent """
                if listing.budget !=auction_book[shop_id].budget:
                   auction_book[shop_id].budget_resest(listing.budget) 
                auction_book[shop_id].listings[listing_id]=(listing.price,listing.max_bid*max_ratio)
                bid = auction_book[shop_id].bidding(listing_id)
            else:
                auction_book[shop_id]=shops(shop_id,listing.budget,check_points)
                auction_book[shop_id].add_listing(listing_id,listing.price,listing.max_bid*max_ratio)
                bid = auction_book[shop_id].bidding(listing_id)
            origin_bid = min(listing.bid,auction_book[shop_id].budget-auction_book[shop_id].cost) 
            score = (ctr*(bid+click_penalty)+ctr**(float(sys.argv[5]))*cvr*listing.price*ranking_prop)*100
            ctr_cvr[j]=(listing_id,score,ctr,cvr,bid,listing.price,listing.actual_purchase,listing.actual_click,shop_id)
        sorted_list = sorted(ctr_cvr,key=lambda item: item[1],reverse=True) 
        """ end of auction """
        for j in range(com_pos):
            local_clicking=set()
            if sorted_list[j][7]==1:
               """ first price auction """
               #auction_book[sorted_list[j][8]].click(sorted_list[j][4])
               """ second price auction """
               cpc = min(sorted_list[j][4],max(1,int(sorted_list[j+1][1]/sorted_list[j][2]/100-sorted_list[j][3]*sorted_list[j][5]*ranking_prop*sorted_list[j][2]**(float(sys.argv[5])-1)-click_penalty)))
               """ since it could happen that the click correspond to the same shop, then the actual cost will change"""
               if sorted_list[j][8] in local_clicking:
                  print "same shop prolist being clicked at least twice"
               else:
                  local_clicking.add(sorted_list[j][8])
               status = auction_book[sorted_list[j][8]].click(cpc)
               if status:
                  total_bidding_revenue+=cpc
                  first_bidding_revenue+=sorted_list[j][4]
               top_score+=sorted_list[j][1]
               actual_score+=sorted_list[j+1][1] 
               if status and sorted_list[j][6]==1:
                  auction_book[sorted_list[j][8]].attribution(sorted_list[j][0])
                  total_attribute_revenue+=sorted_list[j][5]*0.03
    cr = float(total_bidding_revenue)/first_bidding_revenue
    #print "competative ratio in terms of score for the proposed model", float(actual_score)/top_score
    return [auction_book,total_bidding_revenue,total_attribute_revenue,cr]


def old_shop_level(file_in,old_auction_book):
    auction_df = pd.read_csv(file_in, names=columns)
    impressions = auction_df.shape[0]/8
    first_bidding_revenue = 0
    total_bidding_revenue = 0
    top_score = 0
    actual_score = 0
    total_attribute_revenue = 0
    upperbound_purchase_revenue = 0
    upperbound_bidding_revenue = 0
    check_frequency = int(float(sys.argv[2]))
    check_points = impressions/check_frequency
    for shop in old_auction_book:
        old_auction_book[shop].check_point_reset(check_points)
    for i in range(impressions):
        if (i+1)%check_frequency == 0:
            for shop in old_auction_book:
                old_auction_book[shop].usage[(i+1)/check_frequency-1]=old_auction_book[shop].cost/old_auction_book[shop].budget
        ctr_cvr=['' for j in range(8)]
        purchase_history = [0 for j in range(8)]
        bid_history = [0 for j in range(8)]
        for j in range(8):
            listing = auction_df.loc[i*8+j,:]
            listing_id = listing.listing_id
            shop_id = listing.shop_id
            relevancy = listing.relevancy
            quality = abs(listing.quality)
            ctr = listing.ctr/10
            cvr = listing.cvr/10
            if shop_id in old_auction_book:
                if not listing_id in old_auction_book[shop_id].listings:
                   old_auction_book[shop_id].add_listing(listing_id,listing.price,listing.max_bid*max_ratio)
            else:
                old_auction_book[shop_id]=shops(shop_id,listing.budget,check_points)
                old_auction_book[shop_id].add_listing(listing_id,listing.price,listing.max_bid*max_ratio)
            bid = min(listing.bid,old_auction_book[shop_id].budget-old_auction_book[shop_id].cost)
            pacing = float(i+1)/impressions+float(old_auction_book[shop_id].budget-old_auction_book[shop_id].cost-bid/2)/old_auction_book[shop_id].budget
            score = bid*quality**1.5*relevancy*pacing
            ctr_cvr[j]=(listing_id,score,ctr,cvr,bid,listing.price,listing.actual_purchase,pacing,quality,relevancy,listing.actual_click,shop_id)
            purchase_history[j] = listing.actual_purchase*listing.price*0.03
            bid_history[j]=listing.actual_click*listing.cpc
        sorted_list = sorted(ctr_cvr,key=lambda item: item[1],reverse=True)
        sorted_purchase = sorted(purchase_history,reverse=True)
        sorted_bidding = sorted(bid_history, reverse=True)
        """ end of auction"""
        for j in range(com_pos):
            local_click = set()
            upperbound_purchase_revenue+=sorted_purchase[j]
            upperbound_bidding_revenue+=sorted_bidding[j]
            if sorted_list[j][10]==1:
              """ first price auction """
              #old_auction_book[ctr_cvr[j][0]].click(sorted_list[j][4])
              """ second price auction """
              if sorted_list[j][11] in local_click:
                 print "shop prolists being clicked at least twice"
              else:
                 local_click.add(sorted_list[j][11])
              cpc = max(1,int(sorted_list[j+1][1]/(sorted_list[j][7]*abs(sorted_list[j][8])**1.5*sorted_list[j][9])))
              status = old_auction_book[sorted_list[j][11]].click(cpc)
              
              if status:
                 total_bidding_revenue+=cpc
                 first_bidding_revenue+=sorted_list[j][4]
              top_score+=sorted_list[j][1]
              actual_score+=sorted_list[j+1][1]
              if status and sorted_list[j][6]==1:
                 old_auction_book[sorted_list[j][11]].attribution(sorted_list[j][0])
                 total_attribute_revenue+=sorted_list[j][5]*0.03

    cr =float(total_bidding_revenue)/first_bidding_revenue
    #print "competitive ratio in terms score for the old model", float(actual_score)/top_score
    up = upperbound_purchase_revenue
    #print "upperbound for the bidding revenue", upperbound_bidding_revenue
    return [old_auction_book,total_bidding_revenue,total_attribute_revenue,cr,up]

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

def global_alpha(book):
    s = 0
    m = 0
    for shop in book:
        s+=book[shop].alpha
        m+=1
    return float(s)/m

def global_budget_utilization(book):
    s = 0
    m = 0
    for shop in book:
        s+=book[shop].budget_usage()
        m+=1
    return float(s)/m
        

def num_shop_budget(alpha,book):
    n = 0
    for shop in book:
        if book[shop].budget_usage()>=alpha:
           n+=1
    return n

def num_clicks(book):
    s = 0
    for shop in book:
        s+=book[shop].click_count
    return s

def num_purchases(book):
    s = 0
    for shop in book:
        s+=book[shop].purchase_count
    return s

def num_of_outofbudget(book):
    s = 0
    for shop in book:
        if book[shop].exhaust>0:
           s+=1
    return s


def global_ecpc(book):
    s = 0
    n = 0
    for shop in book:
        s+=book[shop].cost
        n+=book[shop].click_count
    return s/n


mode = sys.argv[1]
ranking_prop = float(sys.argv[4])



print " date: argv[1]; update_alpha frequnect: argv[2]; top k: argv[3]; shared percentage: argv[4]; ctr exponent: argv[5]; alpha init: argv[6]; gamma: argv[7]; budet: argv[8]; click_penalty: argv[9]; max infation ratio: argv[10]" 

""" goal of click is 2% """
#play_sequence=['0711','0712','0717','0720','0724']
play_sequence=['0730']


mode = sys.argv[1]
record = open('daily_default_'+mode+'.csv','w')
recorder = csv.writer(record)

recorder.writerow(['date','mode','bidding revenue','shared purchase revenue','CR','alpha','clicks','purchases','ecpc'])

T = len(play_sequence)
for i in range(T):
    """ memoryless """
    auction_book = {}
    old_auction_book={}
    print "start iter: "+play_sequence[i]
    print "##### proposed model #####"
    date = play_sequence[i]
    [auction_book,total_bidding_revenue,total_attribute_revenue,cr] = proposed_shop_level("/home/ubuntu/auction/auction_"+date+"_bid_ctrcvr.csv",auction_book,ranking_prop)
    print "total_bidding_revenue for proposed model",total_bidding_revenue
    print "total_attribute_revenue for proposed model",total_attribute_revenue
    #click_bucket = num_click(auction_book)
    #print "proposed model click bucket: ",click_bucket
    print "global budget utilization is", global_budget_utilization(auction_book)
    print "global alpha is", global_alpha(auction_book)
    print "global ecpc under the proposed model is", global_ecpc(auction_book)
    print "total number of clicks under the proposed model is",num_clicks(auction_book)
    print "total number of purchases under the proposed model is",num_purchases(auction_book)
    print "number of shops that running out of budget",num_of_outofbudget(auction_book)
    print "number of shops that exhaust a certain percentage of budget in the proposed model",[num_shop_budget(float(a)/10,auction_book) for a in range(10)]
    for shop in auction_book:
        auction_book[shop].daily_reset(init_alpha=0.5)
    [auction_book,total_bidding_revenue,total_attribute_revenue,cr] = proposed_shop_level("/home/ubuntu/auction/auction_"+date+"_bid_ctrcvr.csv",auction_book,ranking_prop)
    recorder.writerow([date,'proposed',total_bidding_revenue,total_attribute_revenue,cr,global_alpha(auction_book),num_clicks(auction_book),num_purchases(auction_book),global_ecpc(auction_book)])
    print " ##### current model #####"

    """ current model """
    [old_auction_book,total_bidding_revenue,total_attribute_revenue,cr,up] = old_shop_level("/home/ubuntu/auction/auction_"+date+"_bid_ctrcvr.csv",old_auction_book)
    #print "total_bidding_revenue for old model",total_bidding_revenue
    #print "total_attribute_revenue for old model",total_attribute_revenue
    #print "global budget utilization is", global_budget_utilization(old_auction_book)
    #print "total number of clicks under the current model is",num_clicks(old_auction_book)
    #print "global ecpc under the current model is", global_ecpc(old_auction_book)
    #click_bucket = num_click(old_auction_book)
    #print "old model click bucket: ",click_bucket
    #print "total number of purchases under the current model is",num_purchases(old_auction_book)
    print "number of shops that running out of budget",num_of_outofbudget(old_auction_book)
    print "number of shops that exhaust a certain percentage of budget in the current model",[num_shop_budget(float(a)/10,old_auction_book) for a in range(10)]
    recorder.writerow([date,'current',total_bidding_revenue,total_attribute_revenue,cr,global_alpha(old_auction_book),num_clicks(old_auction_book),num_purchases(old_auction_book),global_ecpc(old_auction_book),up])

record.close()   
