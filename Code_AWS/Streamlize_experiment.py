import os

date = '0727'
Im = '82472'
print """ step 1: parse from raw data """
os.system('python parse_from_raw.py '+date+' auction_'+date+'_ctr.csv auction_'+date+'_cvr.csv auction_'+date+'_bid.csv')
print """ step 2: vectorize the features """
os.system('python convert_to_feature.py '+date)
print """ step 3: convert to vw form """
os.system('python predict_vw.py '+date+' '+Im)
print """ step 4: predict cvr """
os.system('vw -i /Users/Wei/Documents/Research/Ads_optimization/cvr/cvr.model -t /Users/Wei/Documents/Research/Ads_optimization/cvr/auction_predict/auction_'+date+'_cvr_vw -p /Users/Wei/Documents/Research/Ads_optimization/cvr/auction_predict/out/predict_'+date+'_cvr --link=logistic')
print """ step 5: predict ctr """
os.system('vw -i /Users/Wei/Documents/Research/Ads_optimization/ctr/ctr.model -t /Users/Wei/Documents/Research/Ads_optimization/ctr/auction_predict/auction_'+date+'_ctr_vw -p /Users/Wei/Documents/Research/Ads_optimization/ctr/auction_predict/out/predict_'+date+'_ctr --link=logistic')
print """ step 6:final add on """ 
os.system('python add_on_ctr_cvr.py '+Im+' '+date)
