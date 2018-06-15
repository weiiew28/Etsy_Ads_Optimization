train the logistic regression model using vw:
https://github.com/JohnLangford/vowpal_wabbit
vw training_file -l (learning rate) -c -passes -b (hasing bit) -f model 
vw -i model -t predict file --link=logistic -t out prediction file

run the auction simulation:

shop_level_auction.py -- default experiment
exp_shop_auction.py -- change click penalty and purchase percentage
exp_limited_budget_auction.py -- limited budget experiment

