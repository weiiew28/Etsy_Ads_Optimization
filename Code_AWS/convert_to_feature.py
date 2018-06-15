# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words



import re
import pandas
import csv
import gensim
import numpy as np

from gensim.models.keyedvectors import KeyedVectors
word_vectors = KeyedVectors.load('/Users/Wei/Documents/Research/Ads_optimization/Googlew2v')
original_embed = 300


def sort_by_frequency(word_list):
    word_dict = {}
    for word in word_list:
        if word in word_dict:
           word_dict[word]+=1
        else:
           word_dict[word]=1
    sorted_word_dict = sorted(word_dict,key=word_dict.get)
    return [key for key in sorted_word_dict]

def filtered_words(word_list):
    ind = []
    for word in word_list:
        try:
           tmp = word_vectors[word]
           ind.append(word)
        except KeyError:
           continue
    return ind


def look_up_vector(st,T,embed):
    ### T is the window size, and embed is the embedding dimension 
    vec = np.zeros((embed*T,1))
    """ pointer for the string """
    ct = 0
    """ pointet for the word window """
    ind = 0
    m = len(st)
    while ind<min(m,T):
      word = word_vectors[st[ind]]
      vec[ind*embed:(ind+1)*embed,0]=word
      ind+=1
    return [vec[i,0] for i in range(T*embed)]

def clean_text(st):
    """ remove \n"""
    st= st.decode('utf-8')
    st = st.strip('"')
    st = re.sub('http://[\s|\S]+?\\n','',st)
    st = re.sub('\\n',' ',st)
    """ remove special unicode character """
    st = re.sub('\\u0026[\s|\S]+?;',' ',st)
    st = re.sub(u'\u2606','',st)
    escape_char = re.compile(r'\\x[0123456789abcdef][0123456789abcdef]')
    st = re.sub('[*]+','',st)
    st = re.sub('[-]+','.',st)
    st = re.sub('[.]+','.',st)
    st = re.sub('\\\\ s',' ',st)
    st = re.sub('\\x80',' ',st)
    st = re.sub('\\x93',' ',st)
    st = re.sub('\\xe2',' ',st)
    st = re.sub('\\xc2',' ',st)
    st = re.sub('\\xa0',' ',st)
    st = re.sub('\\x9d',' ',st)
    st = re.sub('\\\\','',st)
    return st.lower()


def query_word_relevancy(query,word_list,T):
    vec = [-1.0 for i in range(T)]
    if len(query)==0:
       return vec
    if len(word_list)==0:
       return vec
    m = len(query)
    mm = len(word_list)
    ind_out = 0
    vec = [0 for i in range(T)]
    while ind_out<min(T,mm):
       s = 0
       for word in query:
          match = word_list[ind_out]
          s+=np.sum(word_vectors[word]*word_vectors[match])/(np.linalg.norm(word_vectors[word])*np.linalg.norm(word_vectors[match]))
       vec[ind_out]=s/m
       ind_out+=1
    return vec
           

""" input fields:

listing_id(0),shop_id(1),timestamp(2),click_label(3),pos(4),smooth_ctr(5),smooth_fave(6),smooth_cart(7),price(8),tag(9),query(10),title(11) """


""" out fields:
['listing_id','shop_id','timestamp','label','position','smooth_ctr','smooth_fvr','smooth_cart','price']+[str(i) for i in range(T*embed)]+['Re'+str(i) for i in range(T)] """

def ctr_2f(ctr_in,ctr_out):
    f = open(ctr_in,'r')
    ff = csv.reader(f)
    g = open(ctr_out,'w')
    gg = csv.writer(g)
    for line in ff:
        query = filtered_words(line[10].split('_'))
        tag = line[9].decode('utf-8').split('.')
        tags = []
        for item in tag:
            tags+=item.split(' ')
        title = clean_text(line[11]).split(' ')
        """ tag and title are complimentary to each other in describing a listing """
        important_words = sort_by_frequency(filtered_words(tags+title))
        embed_word_vec = look_up_vector(important_words,10,300)
        relevance_vec = query_word_relevancy(query,important_words,10)
        out_vec = [line[0],line[1],line[2],line[3],line[4],line[5],line[6],line[7],line[8]]+embed_word_vec+relevance_vec
        assert len(out_vec)==3019
        gg.writerow(out_vec)
    f.close()
    g.close()



def sum_description(s):
    parser = PlaintextParser.from_string(s, Tokenizer('english'))
    stemmer = Stemmer('english')
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words('english')
    summary = []
    try:
      for sentence in summarizer(parser.document, 2):
           summary+=sentence.words
    except np.linalg.linalg.LinAlgError:
      print (s)
    return summary


""" input fields:

listing_id(0),shop_id(1),purchase_label(2),smooth_listing_fave_click(3),smooth_listing_cart_click(4),smooth_listing_purchase_click(5), smooth_shop_fave_click(6),smooth_shop_cart_click(7),smooth_shop_purchase_click(8),price(9),query(10),description(11),tag(12),title(13)"""

"""features: listing_id(0),shop_id(1),label(2),shop_history_fave(3),shop_history_carts(4),shop_history_purchases(5),listing_history_fave(6),listing_history_carts(7),listing_history_purchases(8),price(9), description_important_word_embed(10), relevance"""

def c2f(cvr_in,cvr_out):
    f = open(cvr_in,'r')
    ff = csv.reader(f)
    g = open(cvr_out,'w')
    gg = csv.writer(g)
    for line in ff:
        try:
           query = filtered_words(line[9].split('_'))
        except IndexError:
           print (line)
        tag = line[11].decode('utf-8').split('.')
        tags = []
        for item in tag:
            tags+=item.split(' ')
        description = clean_text(line[10])
        """ obtain a summary """
        sum_words = sum_description(description)
        try:
          important_words = filtered_words(sum_words+tags)
        except TypeError:
          print (sum_words)
        embed_word_vec = look_up_vector(important_words,20,300)
        relevance_vec = query_word_relevancy(query,important_words,20)
        gg.writerow([line[0],line[1],line[2],line[6],line[7],line[8],line[4],line[5],line[6],line[9]]+embed_word_vec+relevance_vec)


ctr_2f("/Users/Wei/Documents/Research/Ads_optimization/ctr/auction_predict/auction_test_ctr.csv","/Users/Wei/Documents/Research/Ads_optimization/ctr/auction_predict/auction_test_ctr_vec.csv")
c2f("/Users/Wei/Documents/Research/Ads_optimization/cvr/auction_predict/auction_test_cvr.csv","/Users/Wei/Documents/Research/Ads_optimization/cvr/auction_predict/auction_test_cvr_vec.csv")
