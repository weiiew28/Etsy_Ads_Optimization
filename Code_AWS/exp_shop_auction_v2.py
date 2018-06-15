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


class shops():
   def __init__(self,shop_id,budget,check_points,alpha):
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

   def budget_reset(self,budget):
       if set_budget:
          self.budget = int(set_budget*budget)
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
          self.init_alpha = init_alpha
       self.last_cost = 0
       self.cost = 0
       self.revenue = 0
       self.click_count = 0
       self.purchase_count = 0
       self.exhaust = 0

   def add_listing(self,listing_id,price,max_bid):
       init_bid = min(max_bid,self.budget-self.cost)
       self.listings[listing_id]=(price, max_bid, init_bid)

   def update(self,plan_pace):
       """ if the budget is exhausted, then there is no need to update """
       if plan_pace>0:
          self.alpha = min(0.9,self.alpha*np.exp(self.gamma*(self.cost/plan_pace-1)))
       self.last_cost = self.cost
    

   def bidding(self,listing_id):
       bid = min(self.listings[listing_id][1],self.budget-self.cost)
       assert bid<=self.budget-self.cost
       return bid

   def ecpc(self):
       """empirical cost per click"""
       return self.cost/self.click_count

   def attribution(self,listing_id):
       self.purchase_count+=1
       self.revenue+=self.listings[listing_id][0]*0.03

   def click(self,cpc):
       self.click_count+=1
       self.cost+=cpc
       try:
          assert self.cost<=self.budget
       except AssertionError:
           self.exhaust = self.cost-self.budget
           print "out of budget", self.name
           print "cost", self.cost
           print "budget", self.budget
           print "cpc", cpc

   def budget_usage(self):
       return float(self.cost)/self.budget


def proposed_shop_level(file_in,auction_book,ranking_prop,click_penalty,check_frequency):
    auction_df = pd.read_csv(file_in, names=columns,dtype=type_dict)
    impressions = auction_df.shape[0]/8
    """ competitive_ratio evaluates how competitive the auction is: it is the second price revenue divided by the first price revenue, if it is large        , then it implies that the second price is close to first one, not very competitive.  """
    first_bidding_revenue = 0
    total_bidding_revenue = 0
    top_score = 0
    actual_score = 0
    total_attribute_revenue = 0
    count_bid_equal = 0
    check_points = impressions/check_frequency
    for shop in auction_book:
        auction_book[shop].check_point_reset(check_points)
        auction_book[shop].daily_reset(init_alpha=0.5)

    for i in range(impressions):
        """ gloabally update all the alphas at the check point """
        if (i+1)%check_frequency == 0:
            for shop in auction_book:
                remaining_budget = auction_book[shop].budget-auction_book[shop].cost
                remaining_checkpoints = check_points-(i+1)/check_frequency+1
                #auction_book[shop].update(float(remaining_budget)/remaining_checkpoints)
                auction_book[shop].update(float((i+1)/check_frequency)/check_points*auction_book[shop].budget)
                auction_book[shop].usage[(i+1)/check_frequency-1]=auction_book[shop].cost/auction_book[shop].budget
        ctr_cvr=['' for j in range(8)]
        for j in range(8):
            listing = auction_df.loc[i*8+j,:]
            """ using production model ctr training """
            if listing.bid==listing.max_bid:
               count_bid_equal+=1
            ctr = abs(listing.quality)/10
            cvr = listing.cvr/10
            listing_id = listing.listing_id
            shop_id = listing.shop_id
            if shop_id in auction_book:
                """ check if budget is consistent """
                if listing.budget !=auction_book[shop_id].budget:
                     auction_book[shop_id].budget_reset(listing.budget)
                auction_book[shop_id].listings[listing_id]=(listing.price,listing.max_bid*max_ratio)
                bid = auction_book[shop_id].bidding(listing_id)
            else:
                auction_book[shop_id]=shops(shop_id,listing.budget,check_points,set_alpha)
                auction_book[shop_id].add_listing(listing_id,listing.price,listing.max_bid*max_ratio)
                bid = auction_book[shop_id].bidding(listing_id)
            origin_bid = min(listing.bid,auction_book[shop_id].budget-auction_book[shop_id].cost)
            score = (ctr*((1-auction_book[shop_id].alpha)*bid+click_penalty)+ctr*cvr*listing.price*ranking_prop)*100
            ctr_cvr[j]=(listing_id,score,ctr,cvr,bid,listing.price,listing.actual_purchase,listing.actual_click,shop_id,auction_book[shop_id].alpha)
        sorted_list = sorted(ctr_cvr,key=lambda item: item[1],reverse=True)
        """ end of auction """
        for j in range(com_pos):
            local_clicking=set()
            if sorted_list[j][7]==1:
               """ first price auction """
               #auction_book[sorted_list[j][8]].click(sorted_list[j][4])
               """ second price auction """
               cpc = min(sorted_list[j][4],max(1,int((sorted_list[j+1][1]/sorted_list[j][2]/100-sorted_list[j][3]*sorted_list[j][5]*ranking_prop-click_penalty)/(1-sorted_list[j][9]))))
               """ since it could happen that the click correspond to the same shop, then the actual cost will change"""
               if sorted_list[j][8] in local_clicking:
                  print "same shop prolist being clicked at least twice"
               else:
                  local_clicking.add(sorted_list[j][8])
               auction_book[sorted_list[j][8]].click(cpc)
               total_bidding_revenue+=cpc
               first_bidding_revenue+=sorted_list[j][4]
               top_score+=sorted_list[j][1]
               actual_score+=sorted_list[j+1][1]
               if sorted_list[j][6]==1:
                  auction_book[sorted_list[j][8]].attribution(sorted_list[j][0])
                  total_attribute_revenue+=sorted_list[j][5]*0.03
    cr=float(total_bidding_revenue)/first_bidding_revenue
    #print "percentge of maxbid equal to logged bid",float(count_bid_equal)/(impressions*8)
    return [auction_book,total_bidding_revenue,total_attribute_revenue,cr]


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
com_pos = 4
set_alpha = 0.5
gamma_ratio = 10
set_budget = sys.argv[2]
if set_budget != "False":
   set_budget = float(sys.argv[2])
else:
   set_budget = False
max_ratio = float(sys.argv[3])



#date=mode
#[starting_auction_book,total_bidding_revenue,total_attribute_revenue,cr] = proposed_shop_level("/home/ubuntu/auction/auction_"+date+"_bid_ctrcvr.csv",{},0.03,0,20000)

#for shop in starting_auction_book:
#    starting_auction_book[shop].daily_reset(init_alpha=0.5)

#print " description: argv[1], set_budget: argv[2] max_bid_ratio: argv[3] "

""" goal of click is 2% """
purchase_ratio=[0.01,0.03,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
click_penalty=[22,24,26,28,30]


def experiment_default():
    record = open('/home/ubuntu/etsy-ads-optimization/Data/KDD_submission/basic_uniform.csv','w')
    recordd = csv.writer(record)
    recordd.writerow(['date','bidding revenue','purchase revenue','total revenue','ecpc','click','purchase','competitive ratio'])
    T = ['0711','0712','0717','0720','0724','0726','0730']
    for date in T:
        [starting_auction_book,total_bidding_revenue,total_attribute_revenue,cr] = proposed_shop_level("/home/ubuntu/auction/auction_"+date+"_bid_ctrcvr.csv",{},0.03,0,20000)
        for shop in starting_auction_book:
              starting_auction_book[shop].daily_reset(init_alpha=0.5)
        [auction_book,total_bidding_revenue,total_attribute_revenue,cr] = proposed_shop_level("/home/ubuntu/auction/auction_"+date+"_bid_ctrcvr.csv",starting_auction_book,0.03,0,20000)
        recordd.writerow([date,total_bidding_revenue,total_attribute_revenue,total_bidding_revenue+total_attribute_revenue, global_ecpc(auction_book),num_clicks(auction_book),num_purchases(auction_book),cr])

    record.close()

 
def experiment_click(date):
    T = len(click_penalty)
    record = open('/home/ubuntu/etsy-ads-optimization/Data/KDD_submission/tuning_click_'+date+'_extra.csv','w')
    recordd = csv.writer(record)
    recordd.writerow(['click_penalty','bidding revenue','pruchase revenue','total revenue','ecpc','click','purchase','competitive ratio'])
    [starting_auction_book,total_bidding_revenue,total_attribute_revenue,cr] = proposed_shop_level("/home/ubuntu/auction/auction_"+date+"_bid_ctrcvr.csv",{},0.03,0,20000)
    for shop in starting_auction_book:
        starting_auction_book[shop].daily_reset(init_alpha=0.5)
    for i in range(T):
        """ memoryless """
        print "click penalty",click_penalty[i]
        [auction_book,total_bidding_revenue,total_attribute_revenue,cr] = proposed_shop_level("/home/ubuntu/auction/auction_"+date+"_bid_ctrcvr.csv",starting_auction_book,0.03,click_penalty[i],20000)
        recordd.writerow([float(click_penalty[i]),total_bidding_revenue,total_attribute_revenue,total_bidding_revenue+total_attribute_revenue, global_ecpc(auction_book),num_clicks(auction_book),num_purchases(auction_book),cr])

    record.close()

def experiment_purchase(date):
    T = len(purchase_ratio)

    record = open('/home/ubuntu/etsy-ads-optimization/Data/KDD_submission/tuning_purchase_ratio_'+date+'.csv','w')
    recordd = csv.writer(record)
    recordd.writerow(['click_penalty','bidding revenue','pruchase revenue','total revenue','ecpc','click','purchase','competitive ratio'])
    [starting_auction_book,total_bidding_revenue,total_attribute_revenue,cr] = proposed_shop_level("/home/ubuntu/auction/auction_"+date+"_bid_ctrcvr.csv",{},0.03,0,20000)
    for shop in starting_auction_book:
        starting_auction_book[shop].daily_reset(init_alpha=0.5)
    for i in range(T):
        """ memoryless """
        print "purchase_ratio",purchase_ratio[i]
        [auction_book,total_bidding_revenue,total_attribute_revenue,cr] = proposed_shop_level("/home/ubuntu/auction/auction_"+date+"_bid_ctrcvr.csv",starting_auction_book,purchase_ratio[i],0,20000)
        recordd.writerow([float(purchase_ratio[i]),total_bidding_revenue,total_attribute_revenue,total_bidding_revenue+total_attribute_revenue, global_ecpc(auction_book),num_clicks(auction_book),num_purchases(auction_book),cr])
    record.close()

experiment_default()
#experiment_click('0730')
#experiment_purchase('0711')
