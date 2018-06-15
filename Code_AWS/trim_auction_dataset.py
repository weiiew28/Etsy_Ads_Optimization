""" this script trim each impression to 8 position of prolists exactly"""

f = open("/home/wqian/Data/Auction_Dataset/0711","r")
g = open("/home/wqian/Data/RAW/trimmed_0711","w")
current = None
ram_record = set()
ct = 0

for line in f:
    fields = line.strip().split('\t')
    query = fields[0]
    impression_timestamp = fields[1]
    pos = int(fields[6])
    if ct == 0:
       inner_dict = []
       current = (query,impression_timestamp)
    if ct >0 and current!=(query,impression_timestamp):
       """ dealing with the previous search session"""
       sort_inner_dict = sorted(list(set(inner_dict)),key=lambda s:s[2])
       length_prolist = len(sort_inner_dict)
       if length_prolist >=8:
          inner_ct = 0
          for key in sort_inner_dict:
              g.write(key[3])
              inner_ct+=1
              if inner_ct==8:
                 break
          #g.write("### END ###\n")
       inner_dict = []
       current = (query,impression_timestamp)
    inner_dict.append((query,impression_timestamp,pos,line))
    ct+=1

f.close()
g.close()
        



