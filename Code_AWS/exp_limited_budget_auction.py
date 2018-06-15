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
   def __init__(self,shop_id,budget,check_points,alpha=0.5):
       self.listings = {}
       self.name = shop_id
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
       self.history=[]
       self.impression_history={}
   
   def impression(self,ts):
       if ts in self.impression_history:
          self.impression_history[ts]+=1
       else:
          self.impression_history[ts]=1

   def budget_resest(budget,set_budget):
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
          self.init_alpha = self.alpha
       self.exhaust = 0
       self.last_cost = 0
       self.cost = 0
       self.revenue = 0
       self.click_count = 0
       self.purchase_count = 0
       self.history=[]
       self.impression_history={}


   def add_listing(self,listing_id,price,max_bid):
       self.listings[listing_id]=(price, max_bid)

   def update(self,plan_pace):
       """ if the budget is exhaust, then there is no need to update """
       #if plan_pace>0:
       #   self.alpha = min(0.9,self.alpha*np.exp(self.gamma*(float(self.cost-self.last_cost-plan_pace)/plan_pace)))
       #self.last_cost = self.cost
       """ look at the discrepancy of planned budget and actual spending """
       self.alpha = min(0.9,self.alpha*np.exp(self.gamma*(float(self.cost-plan_pace)/plan_pace)))
       
   def bidding(self,listing_id):
       if self.budget<=self.cost:
          return 0
       assert self.cost == sum([record[2] for record in self.history])
       # bid = min(int((1-self.alpha)*self.listings[listing_id][1]),self.budget-self.cost)
       bid = min(self.listings[listing_id][1],self.budget-self.cost) 
       assert bid<=self.budget-self.cost
       return bid

   def mehta_method_bidding(self,listing_id):
       if self.budget<=self.cost:
          return 0
       assert self.cost == sum([record[2] for record in self.history])
       bid = min(self.budget-self.cost,self.listings[listing_id][1]*(1-np.exp(float(self.cost)/self.budget-1)))
       assert bid<=self.budget-self.cost
       return bid

   def ecpc(self):
       """empirical cost per click"""
       return self.cost/self.click_count

   def attribution(self,listing_id):
       self.purchase_count+=1
       self.revenue+=self.listings[listing_id][0]*0.03

   def click(self,cpc,query,ts_bucket):
       if self.cost+cpc<=self.budget:
          self.click_count+=1
          self.cost+=cpc
          self.history.append([ts_bucket,query,cpc,self.alpha])
          return True
       else:
          return False

   def budget_usage(self):
       return float(self.cost)/self.budget

def proposed_shop_level(file_in,auction_book,ranking_prop,click_penalty,check_frequency,mbud,pacing_mode,set_alpha):
    auction_df = pd.read_csv(file_in, names=columns,dtype=type_dict)
    auction_df = pd.read_csv(file_in, names=columns,dtype=type_dict)
    impressions = auction_df.shape[0]/8
    """ competitive_ration evaluates how competitive the auction is """
    first_bidding_revenue = 0
    total_bidding_revenue = 0
    top_score = 0
    actual_score = 0
    total_attribute_revenue = 0
    count_bid_equal = 0
    
    check_points = impressions/check_frequency
    for shop in auction_book:
        auction_book[shop].check_point_reset(check_points)
    for i in range(impressions):
        """ gloabally update all the alphas at the check point """
        if (i+1)%check_frequency == 0:
            for shop in auction_book:
                remaining_budget = auction_book[shop].budget-auction_book[shop].cost
                remaining_checkpoints = check_points-(i+1)/check_frequency+1
                current = (i+1)/check_frequency
                #if pacing_mode == 'conditional uniform':
                #    auction_book[shop].update(float(remaining_budget)/remaining_checkpoints)
                if pacing_mode == 'uniform':
                    auction_book[shop].update(auction_book[shop].budget*float(current)/check_points)
                if pacing_mode == 'convex':
                    auction_book[shop].update((float(current)/check_points)**6*auction_book[shop].budget)
                if pacing_mode == 'concave':
                    current = (i+1)/check_frequency
                    auction_book[shop].update((float(current)/check_points)**0.1*auction_book[shop].budget)
                auction_book[shop].usage[(i+1)/check_frequency-1]=auction_book[shop].cost
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
                auction_book[shop_id].listings[listing_id]=(listing.price,listing.max_bid)
                " proposed method "
                bid = auction_book[shop_id].bidding(listing_id)
                "mehta method"
                #bid = auction_book[shop_id].mehta_method_bidding(listing_id)
            else:
                auction_book[shop_id]=shops(shop_id,mbud[shop_id],check_points,set_alpha)
                auction_book[shop_id].add_listing(listing_id,listing.price,listing.max_bid)
                bid = auction_book[shop_id].bidding(listing_id)
                #bid = auction_book[shop_id].mehta_method_bidding(listing_id)
            auction_book[shop_id].impression((i+1)/check_frequency)
            origin_bid = min(listing.bid,auction_book[shop_id].budget-auction_book[shop_id].cost)
            score = (ctr*(bid*(1-auction_book[shop_id].alpha)+click_penalty)+ctr*cvr*listing.price*ranking_prop)*100
            ctr_cvr[j]=(listing_id,score,ctr,cvr,bid,listing.price,listing.actual_purchase,listing.actual_click,shop_id,auction_book[shop_id].budget-auction_book[shop_id].cost,listing.query,(i+1)/check_frequency,auction_book[shop_id].alpha)
        sorted_list = sorted(ctr_cvr,key=lambda item: item[1],reverse=True)
        """ end of auction """
        for j in range(com_pos):
            local_clicking=set()
            if sorted_list[j][7]==1:
               """ first price auction """
               #auction_book[sorted_list[j][8]].click(sorted_list[j][4])
               """ second price auction """
               try:
                  assert sorted_list[j][4]<=auction_book[sorted_list[j][8]].budget-auction_book[sorted_list[j][8]].cost
               except AssertionError:
                  print "bid exceed remaining budget"
                  print local_clicking
                  print sorted_list[j][8]
                  print "at time of bidding, bid and remaining budget are",(sorted_list[j][4],sorted_list[j][9]) 
               cpc = min(sorted_list[j][4],max(1,int((sorted_list[j+1][1]/sorted_list[j][2]/100-sorted_list[j][3]*sorted_list[j][5]*ranking_prop-click_penalty)/(1-sorted_list[j][-1]))))
               """ since it could happen that the click correspond to the same shop, then the actual cost will change"""
               if sorted_list[j][8] in local_clicking:
                  print "same shop prolist being clicked at least twice"
               else:
                  local_clicking.add(sorted_list[j][8])
               status = auction_book[sorted_list[j][8]].click(cpc,sorted_list[j][10],sorted_list[j][11])
               if status:
                  total_bidding_revenue+=cpc
                  first_bidding_revenue+=sorted_list[j][4]
               top_score+=sorted_list[j][1]
               actual_score+=sorted_list[j+1][1]
               if status and sorted_list[j][6]==1:
                  auction_book[sorted_list[j][8]].attribution(sorted_list[j][0])
                  total_attribute_revenue+=sorted_list[j][5]*0.03
    cr=float(total_bidding_revenue)/first_bidding_revenue
    print "percentge of maxbid equal to logged bid",float(count_bid_equal)/(impressions*8)
    return [auction_book,total_bidding_revenue,total_attribute_revenue,cr,check_points]

def old_shop_level(file_in,old_auction_book,check_frequency,mbud=None):
    auction_df = pd.read_csv(file_in, names=columns)
    impressions = auction_df.shape[0]/8
    first_bidding_revenue = 0
    total_bidding_revenue = 0
    top_score = 0
    actual_score = 0
    total_attribute_revenue = 0
    upperbound_purchase_revenue = 0
    upperbound_bidding_revenue = 0
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
                if not mbud:
                   old_auction_book[shop_id]=shops(shop_id,listing.budget,check_points)
                else:
                   old_auction_book[shop_id]=shops(shop_id,mbud[shop_id],check_points)
                old_auction_book[shop_id].add_listing(listing_id,listing.price,listing.max_bid*max_ratio)
            bid = min(listing.bid,old_auction_book[shop_id].budget-old_auction_book[shop_id].cost)
            pacing = float(i+1)/impressions+float(old_auction_book[shop_id].budget-old_auction_book[shop_id].cost-bid/2)/old_auction_book[shop_id].budget
            score = bid*quality**1.5*relevancy*pacing
            ctr_cvr[j]=(listing_id,score,ctr,cvr,bid,listing.price,listing.actual_purchase,pacing,quality,relevancy,listing.actual_click,shop_id,listing.query,(i+1)/check_frequency)
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
              status = old_auction_book[sorted_list[j][11]].click(cpc,sorted_list[j][12],sorted_list[j][13])
              if status:
                 total_bidding_revenue+=cpc
                 first_bidding_revenue+=sorted_list[j][4]
              top_score+=sorted_list[j][1]
              actual_score+=sorted_list[j+1][1]
              if status and sorted_list[j][6]==1:
                 old_auction_book[sorted_list[j][11]].attribution(sorted_list[j][0])
                 total_attribute_revenue+=sorted_list[j][5]*0.03

    print "upperbound for the purchase revenue",upperbound_purchase_revenue
    print "upperbound for the bidding revenue", upperbound_bidding_revenue
    cr = float(total_bidding_revenue)/first_bidding_revenue
    return [old_auction_book,total_bidding_revenue,total_attribute_revenue,cr]


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

def budget_utilization(book,subset,checkpoint):
    cc = [0 for i in range(checkpoint)]
    for i in range(checkpoint):
        s = 0
        for shop in subset:
            s+=book[shop].usage[i]
        cc[i] = s
    return cc


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

def bid_purchase(book,subset):
    budget = 0
    bidding_revenue = 0
    purchase_revenue = 0
    clicks = 0
    purchases = 0
    for shop in subset:
        budget+=book[shop].budget
        bidding_revenue+=book[shop].cost
        purchase_revenue+=book[shop].revenue
        clicks+=book[shop].click_count
        purchases+=book[shop].purchase_count
    return [budget,bidding_revenue,float(bidding_revenue)/budget,purchase_revenue,clicks,purchases]

mode = sys.argv[1]
com_pos = 4
gamma_ratio = 1
set_budget = sys.argv[2]
max_ratio = float(sys.argv[3])

             
date = '0720'
[old_auction_book,total_bidding_revenue,total_attribute_revenue,cr] = old_shop_level("/home/ubuntu/auction/auction_"+date+"_bid_ctrcvr.csv",{},20000)
modified_budget = {}
mcpc = {}
competitive_shops=set()
for shop in old_auction_book:
    modified_budget[shop]=max(int(old_auction_book[shop].cost*0.8),max([old_auction_book[shop].listings[item][1] for item in old_auction_book[shop].listings]))

for shop in old_auction_book:
    if old_auction_book[shop].click_count>=10:
       competitive_shops.add(shop)

print "there are {0} competitive shops that has been clicked at least 10 times".format(len(competitive_shops))

if mode != 'current':
        [starting_auction_book,total_bidding_revenue,total_attribute_revenue,cr,check_points] = proposed_shop_level("/home/ubuntu/auction/auction_"+date+"_bid_ctrcvr.csv",{},0.03,0,20000,modified_budget,'conditional uniform', 0.5)

        for shop in starting_auction_book:
            starting_auction_book[shop].daily_reset(init_alpha=0.5)

alphas = [0.1,0.5,0.8]
p_mode = ['uniform', 'convex','concave']


def experiment_alpha_adjust():
    date = '0720'
    record = open('/home/ubuntu/etsy-ads-optimization/Data/KDD_submission/tuning_update_'+mode+'.csv','w')
    budget_use = open('/home/ubuntu/etsy-ads-optimization/Data/KDD_submission/comp_shop_bu_update_'+mode+'.csv','w')
    budget_usee = csv.writer(budget_use)
    recordd = csv.writer(record)
    recordd.writerow(['update rule','bidding revenue','pruchase revenue','total revenue','ecpc','click','purchase','competitive ratio'])
    for m in p_mode:
        for shop in starting_auction_book:
            starting_auction_book[shop].daily_reset(init_alpha=0.5)
        [auction_book,total_bidding_revenue,total_attribute_revenue,cr,check_points] = proposed_shop_level("/home/ubuntu/auction/auction_"+date+"_bid_ctrcvr.csv",starting_auction_book,0.03,0,20000,modified_budget,m,0.5)
        recordd.writerow([m,total_bidding_revenue,total_attribute_revenue,total_bidding_revenue+total_attribute_revenue, global_ecpc(auction_book),num_clicks(auction_book),num_purchases(auction_book),cr])
        [budget,bidding_revenue,percentage,purchase_revenue,clicks,purchases]=bid_purchase(auction_book,competitive_shops)
        recordd.writerow(['comp shops',budget,bidding_revenue,percentage,purchase_revenue,bidding_revenue+purchase_revenue,clicks,purchases])
        usage = budget_utilization(auction_book,competitive_shops,check_points)
        usage = [float(item)/budget for item in usage]
        budget_usee.writerow(['update rule']+range(1,check_points))
        budget_usee.writerow([m]+usage)
    record.close()
    budget_use.close()

def experiment_alpha():
    T = len(alphas)
    date = '0720'
    record = open('/home/ubuntu/etsy-ads-optimization/Data/KDD_submission/tuning_alpha_'+mode+'.csv','w')
    budget_use = open('/home/ubuntu/etsy-ads-optimization/Data/KDD_submission/comp_shop_bu_'+mode+'.csv','w')
    budget_usee = csv.writer(budget_use)
    recordd = csv.writer(record)
    recordd.writerow(['alpha','bidding revenue','pruchase revenue','total revenue','ecpc','click','purchase','competitive ratio'])
    for i in range(T):
        """ memoryless """
        print "alpha", alphas[i]
        comp_shop = open('/home/ubuntu/etsy-ads-optimization/Data/KDD_submission/competitive_0720_'+mode+'_'+str(alphas[i])+'.csv','w')
        comp_shop_info = csv.writer(comp_shop)
        budget_usee.writerow([float(alphas[i])])
        for shop in starting_auction_book:
            starting_auction_book[shop].daily_reset(init_alpha=alphas[i])
        [auction_book,total_bidding_revenue,total_attribute_revenue,cr,check_points] = proposed_shop_level("/home/ubuntu/auction/auction_"+date+"_bid_ctrcvr.csv",starting_auction_book,0.03,0,20000,modified_budget,'conditional uniform',alphas[i])
        recordd.writerow([float(alphas[i]),total_bidding_revenue,total_attribute_revenue,total_bidding_revenue+total_attribute_revenue, global_ecpc(auction_book),num_clicks(auction_book),num_purchases(auction_book),cr])
        [budget,bidding_revenue,percentage,purchase_revenue,clicks,purchases]=bid_purchase(auction_book,competitive_shops)
        recordd.writerow(['comp shops',float(alphas[i]),budget,bidding_revenue,percentage,purchase_revenue,bidding_revenue+purchase_revenue,clicks,purchases])
        usage = budget_utilization(auction_book,competitive_shops,check_points)
        usage = [float(item)/budget for item in usage]
        budget_usee.writerow(usage)
        for shop in competitive_shops:
            comp_shop_info.writerow([shop])
            for detail in auction_book[shop].history:
                comp_shop_info.writerow([' ']+detail)
        comp_shop.close()
    [old_auction_book,total_bidding_revenue,total_attribute_revenue,cr] = old_shop_level("/home/ubuntu/auction/auction_"+date+"_bid_ctrcvr.csv",{},20000,modified_budget)
    recordd.writerow(['current',total_bidding_revenue,total_attribute_revenue,total_bidding_revenue+total_attribute_revenue, global_ecpc(old_auction_book),num_clicks(old_auction_book),num_purchases(old_auction_book),cr])
    
    [budget,bidding_revenue,percentage,purchase_revenue,clicks,purchases]=bid_purchase(old_auction_book,competitive_shops)
    recordd.writerow(['comp shops','current', budget, bidding_revenue, percentage, purchase_revenue, bidding_revenue_+purchase_revenue,clicks,purchases])
    usage = budget_utilization(old_auction_book,competitive_shops,check_points)
    usage = [float(item)/budget for item in usage]
    budget_usee.writerow('current')
    budget_usee.writerow(usage)
    comp_shop = open('/home/ubuntu/etsy-ads-optimization/Data/KDD_submission/competitive_0720_'+mode+'current.csv','w')
    comp_shop_info = csv.writer(comp_shop)
    for shop in competitive_shops:
           comp_shop_info.writerow([shop])
           for detail in old_auction_book[shop].history:
               comp_shop_info.writerow([' ']+detail)
    comp_shop.close() 
    record.close()
    budget_use.close()

def default_performance():
    date = '0720'
    comp_shop_info = open('competitive_0720_cpc_current','w')
    [old_auction_book,total_bidding_revenue,total_attribute_revenue,cr] = old_shop_level("/home/ubuntu/auction/auction_"+date+"_bid_ctrcvr.csv",{},20000,modified_budget)
    for shop in competitive_shops:
        comp_shop_info.write(shop+': \n')
        for detail in old_auction_book[shop].history:
            comp_shop_info.write(str(detail)+'  ')
            comp_shop_info.write('\n')
    comp_shop_info.close()

experiment_alpha_adjust()
#experiment_alpha()
#default_performance()
