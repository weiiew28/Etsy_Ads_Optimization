import numpy as np

""" first pass of data to select the listings and query """

""" second pass of data to attach the query with all the candidate """


f = open('test_auction_dataset','r')

campaign_set=set()
impression_set = set()

cap = 600
campaign = 100

for line in f:

    field = line.strip().split('\t')
    impression_set.add((field[0],field[1]))
    campaign_set.add((field[3],field[9], field[10], field[11], field[12],field[13], field[14], field[15]))

f.close()

impression_set = list(impression_set)
campaign_set = list(campaign_set)

ind_rand = np.random.permutation(len(impression_set))[:cap]
ind_rand2 = np.random.permutation(len(campaign_set))[:campaign]

impression_set_filter = [impression_set[i] for i in ind_rand]
campaign_set_filter = [campaign_set[i] for i in ind_rand2]

out = open('fake_auction_dataset','w')

for query in impression_set_filter:

    for cam in campaign_set_filter:

        line = query[0]+'\t'+query[1]+'\t'
        line+= '0'+'\t'+cam[0]+'\t'+'0'+'\t'+'0'+'\t'+'0'+'\t'+'0'+'\t'+'0'+'\t'
        line+= cam[1]+'\t'+cam[2]+'\t'+cam[3]+'\t'+cam[4]+'\t'+cam[5]+'\t'+cam[6]+'\t'+cam[7]+'\n'
        out.write(line)

out.close()
