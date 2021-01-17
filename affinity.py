import os
import re
import json
import anew
import string
import nltk
import spacy
import pandas as pd
from nltk.corpus import stopwords
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import MinMaxScaler

from flask import json

exclude_punctuation = set([p for p in string.punctuation if p not in ["!",",",".","?",";"]])
stop_words_eng = set(stopwords.words('english'))
te = TransactionEncoder()
scaler = MinMaxScaler(feature_range=(1,10))
nlp = spacy.load('en_core_web_sm')

def extract_hashtag(x):
  ht_ls = re.findall(r"#(\w+)", x)     # find hashtag
  return list(set(ht_ls))

# remove mentions, url, hashtag
def strip_text(t):
  #t = re.sub(r'\.?@\w+\:?',"",t)
  t = re.sub(r'\d+','',t)
  t = re.sub('https?://.+',"",t)             # url
  t = re.sub('pic\.twitter\.com.+',"",t)     # url
  t = re.sub("[#]", " ", t)                  # remove hashtag
  #t = t.lower()                              # lower case
  return ''.join(ch for ch in t if ch not in exclude_punctuation)   # remove unnessary punctuation

# extract entities
LABEL_INCLUDED = ["PERSON","NORP","ORG","GPE","PRODUCT","EVENT","WORK_OF_ART","LAW","LANGUAGE"]
def extract_NER(x):
  doc = nlp(str(x))
  return [(ent.text,ent.label_) for ent in doc.ents if ent.label_ in LABEL_INCLUDED and ent.text.strip() != ""]


# combine key entities, mentions and hashtag for affinity analysis
def entity_combined(row):
  #ents = ["@"+x.strip() for x in row["mentions"].split(",")]
  #ents = ["#"+x.strip() for x in row["hashtags"]]
  ents = []
  ents.append(row["source"])
  for e in row["entities"]:
    ents.append(e[0].strip())
  return list(set(ents))

# remove multiple items
def is_single(row):
    if row["antecedents"] is not None and len(row["antecedents"])==1:
        if row["consequents"] is not None and len(row["consequents"])==1:
            return "y"
    return "n"

def frange(start, stop, step):
    i = start
    while i > stop:
        yield i
        i -= step

def apriori_w_adjusted_min_support(ent_df):
    full_range = list(frange(0.9,0.1,0.1)) + list(frange(0.09,0.01,0.01)) + list(frange(0.009,0.001,0.001))
    for i in full_range:
        threshold = round(i,3)
        frequent_ents = apriori(ent_df,min_support=threshold,use_colnames=True)
        if len(frequent_ents) > 20:
            print("min_support:{}".format(round(i,3)))
            return frequent_ents
    return frequent_ents

def retrieve_data(filename):
  response = []

  input_doc_path = "./"+filename
  with open(input_doc_path,"r") as f:
    if input_doc_path.endswith(".json"):
      doc = [json.loads(line) for line in f]
    elif input_doc_path.endswith(".csv"):
      doc = pd.read_csv(f, delimiter=',')

  doc = pd.DataFrame(doc)

  #doc[["tweet","mentions"]].head(10)

  # rename column names to standardize data schema
  doc = doc.rename(columns={"id_reference_num":"id","delivery":"source","username":"source","tweet":"text"})
  doc = doc[["id","source","date","text"]]

  #doc["hashtags"] = doc["tweet"].apply(lambda x: extract_hashtag(x))

  doc["text_clean"] = doc["text"].apply(lambda x: strip_text(x))

  """
  # SpaCy entity labels
  PERSON	People, including fictional.
  NORP	Nationalities or religious or political groups.
  FAC	Buildings, airports, highways, bridges, etc.
  ORG	Companies, agencies, institutions, etc.
  GPE	Countries, cities, states.
  LOC	Non-GPE locations, mountain ranges, bodies of water.
  PRODUCT	Objects, vehicles, foods, etc. (Not services.)
  EVENT	Named hurricanes, battles, wars, sports events, etc.
  WORK_OF_ART	Titles of books, songs, etc.
  LAW	Named documents made into laws.
  LANGUAGE	Any named language.
  DATE	Absolute or relative dates or periods.
  TIME	Times smaller than a day.
  PERCENT	Percentage, including "%".
  MONEY	Monetary values, including unit.
  QUANTITY	Measurements, as of weight or distance.
  ORDINAL	"first", "second", etc.
  CARDINAL	Numerals that do not fall under another type.
  """
  doc["entities"] = doc["text_clean"].apply(lambda x: extract_NER(x))

  #doc[["tweet_clean","mentions","hashtags","entities"]].head(10)
  doc["entity_combined"] = doc.apply(entity_combined,axis=1)

  #doc[["tweet_clean","mentions","hashtags","entities","entity_hashtag_mention"]].head(10)

  ent_list = list(doc["entity_combined"])

  te_ary = te.fit(ent_list).transform(ent_list)

  ent_df = pd.DataFrame(te_ary,columns=te.columns_)
  #ent_df.head()

  #frequent_ents = apriori(ent_df,min_support=0.6,use_colnames=True)
  frequent_ents = apriori_w_adjusted_min_support(ent_df)

  as_rules = association_rules(frequent_ents, metric="lift",min_threshold=1)

  key_rules = as_rules[(as_rules["confidence"]>0.7)&(as_rules["lift"]>=1)]

  key_rules["is_single"] = key_rules.apply(is_single,axis=1)
  key_rules = key_rules[key_rules["is_single"]=="y"]

  key_rules["source"] = key_rules["antecedents"].apply(lambda x: ','.join(x))
  key_rules["target"] = key_rules["consequents"].apply(lambda x: ','.join(x))

  """
  graph.node = [
      { ID:"@apple", group:1, index:0,name:"apple", px:100, py:100, size:40, weight:100, x:41, y:95},
      { ID:"@orange", group:2, index:1,name:"orange", px:50, py:50, size:10, weight:100, x:200, y:300},
      { ID:"@grape", group:1, index:1,name:"grape", px:200, py:200, size:3, weight:100, x:250, y:150}
    ];
    graph.edge = [
      { 
        source: 1, 
        target: 2,
        value:5
      }
    ];
  """  

  antecedents_nodes = key_rules[["source","antecedent support"]]
  antecedents_nodes.rename(columns={"source":"node","antecedent support":"support"},inplace=True)
  consequents_nodes = key_rules[["target","consequent support"]]
  consequents_nodes.rename(columns={"target":"node","consequent support":"support"},inplace=True)
  graph_nodes = pd.concat([antecedents_nodes,consequents_nodes])
  graph_nodes = graph_nodes.reset_index(drop=True)
  #graph_nodes

  graph_nodes_dedup = graph_nodes.drop_duplicates(["node"],keep="first")
  graph_nodes_dedup = graph_nodes_dedup.reset_index(drop=True)
  #graph_nodes_dedup

  """
  { ID:"@apple", group:1, index:0,name:"apple", px:100, py:100, size:40, weight:100, x:41, y:95},
  { ID:"@orange", group:2, index:1,name:"orange", px:50, py:50, size:10, weight:100, x:200, y:300},
  { ID:"@grape", group:1, index:1,name:"grape", px:200, py:200, size:3, weight:100, x:250, y:150}
  """  
  graph_nodes_dedup["ID"] = graph_nodes_dedup["node"]
  graph_nodes_dedup["group"] = 2
  graph_nodes_dedup["index"] = graph_nodes_dedup.index
  graph_nodes_dedup["px"] = 100
  graph_nodes_dedup["py"] = 100
  graph_nodes_dedup["x"] = 100
  graph_nodes_dedup["y"] = 100
  #scale support to range between 1 and 10
  graph_nodes_dedup["support"] = scaler.fit_transform(graph_nodes_dedup[["support"]])
  graph_nodes_dedup["weight"] = 10
  graph_nodes_dedup.rename(columns={"node":"name","support":"size"},inplace=True)
  #graph_nodes_dedup

  graph_node_json = graph_nodes_dedup.to_json(orient='records')
  #graph_node_json

  # ## Create graph link response
  graph_edges = key_rules[["antecedents","consequents","lift"]]
  graph_edges.rename(columns={"antecedents":"source","consequents":"target","lift":"value"},inplace=True)
  #graph_edges

  """
    source: 1, 
    target: 2,
    value:5
  """
  graph_edges["source"] = graph_edges["source"].apply(lambda x: ','.join(x))
  graph_edges["target"] = graph_edges["target"].apply(lambda x: ','.join(x))

  #graph_edges

  # replace source name with source index
  graph_edges_idx = graph_edges.merge(graph_nodes_dedup[["ID","index"]],how="left",left_on="source",right_on="ID")
  graph_edges_idx.drop(["source","ID"],axis=1,inplace=True)
  graph_edges_idx.rename(columns={"index":"source"},inplace=True)

  # replace target name with target index
  graph_edges_idx = graph_edges_idx.merge(graph_nodes_dedup[["ID","index"]],how="left",left_on="target",right_on="ID")
  graph_edges_idx.drop(["target","ID"],axis=1,inplace=True)
  graph_edges_idx.rename(columns={"index":"target"},inplace=True)
  #graph_edges_idx

  graph_edge_json = graph_edges_idx.to_json(orient='records')

  response.append(graph_node_json)
  response.append(graph_edge_json)
  print("affinity response finished")
  return response


"""
def retrieve_data_tw_test(filename):
  response = []
  nodes_json = json.dumps([{"name":"@PepsiCo","size":6.9599299821,"ID":"@PepsiCo","group":2,"index":0,"px":100,"py":100,"x":100,"y":100,"weight":10},{"name":"Indofood","size":6.1303702576,"ID":"Indofood","group":2,"index":1,"px":100,"py":100,"x":100,"y":100,"weight":10},{"name":"Tell Nestl\\u00e9","size":6.07582734,"ID":"Tell Nestl\\u00e9","group":2,"index":2,"px":100,"py":100,"x":100,"y":100,"weight":10},{"name":"@onetoughnerd","size":6.4550908837,"ID":"@onetoughnerd","group":2,"index":3,"px":100,"py":100,"x":100,"y":100,"weight":10},{"name":"@Nestle,Michigan","size":5.6711949973,"ID":"@Nestle,Michigan","group":2,"index":4,"px":100,"py":100,"x":100,"y":100,"weight":10},{"name":"@Nestle,@onetoughnerd","size":5.7193957152,"ID":"@Nestle,@onetoughnerd","group":2,"index":5,"px":100,"py":100,"x":100,"y":100,"weight":10},{"name":"Tell Nestl\\u00e9,@Nestle","size":6.0745589,"ID":"Tell Nestl\\u00e9,@Nestle","group":2,"index":6,"px":100,"py":100,"x":100,"y":100,"weight":10},{"name":"Tell Nestl\\u00e9,Indofood","size":6.0669482603,"ID":"Tell Nestl\\u00e9,Indofood","group":2,"index":7,"px":100,"py":100,"x":100,"y":100,"weight":10},{"name":"@Nestle,Indofood","size":6.1265649378,"ID":"@Nestle,Indofood","group":2,"index":8,"px":100,"py":100,"x":100,"y":100,"weight":10},{"name":"@Nestle","size":20.6604766797,"ID":"@Nestle","group":2,"index":9,"px":100,"py":100,"x":100,"y":100,"weight":10},{"name":"Michigan","size":8.4554206781,"ID":"Michigan","group":2,"index":10,"px":100,"py":100,"x":100,"y":100,"weight":10}])
  links_json = json.dumps([{"value":1.1022123871,"source":0,"target":9},{"value":1.1023318034,"source":1,"target":9},{"value":1.1027862067,"source":2,"target":9},{"value":11.2155260132,"source":3,"target":10},{"value":16.2883900999,"source":2,"target":1},{"value":16.2883900999,"source":1,"target":2},{"value":14.7432263526,"source":4,"target":3},{"value":11.1605117598,"source":5,"target":10},{"value":14.7432263526,"source":3,"target":4},{"value":16.2883851222,"source":6,"target":1},{"value":1.1027858697,"source":7,"target":9},{"value":16.2950995215,"source":8,"target":2},{"value":16.2950995215,"source":2,"target":8},{"value":16.2883851222,"source":1,"target":6}])
  response.append(nodes_json)
  response.append(links_json)
  return response

def retrieve_data_fda_test(filename):
  response = []
  nodes_json = json.dumps([{"name":"Act","size":7.2727272727,"ID":"Act","group":2,"index":0,"px":100,"py":100,"x":100,"y":100,"weight":10},{"name":"CFR","size":10.0,"ID":"CFR","group":2,"index":1,"px":100,"py":100,"x":100,"y":100,"weight":10},{"name":"Cosmetic Act","size":9.0909090909,"ID":"Cosmetic Act","group":2,"index":2,"px":100,"py":100,"x":100,"y":100,"weight":10},{"name":"FDA","size":10.0,"ID":"FDA","group":2,"index":3,"px":100,"py":100,"x":100,"y":100,"weight":10},{"name":"U.S.C.","size":10.0,"ID":"U.S.C.","group":2,"index":4,"px":100,"py":100,"x":100,"y":100,"weight":10},{"name":"the Federal Food, Drug","size":9.0909090909,"ID":"the Federal Food, Drug","group":2,"index":5,"px":100,"py":100,"x":100,"y":100,"weight":10},{"name":"Compliance Officer","size":6.3636363636,"ID":"Compliance Officer","group":2,"index":6,"px":100,"py":100,"x":100,"y":100,"weight":10}])
  links_json = json.dumps([{"value":1.0,"source":0,"target":1},{"value":1.0,"source":1,"target":0},{"value":1.1,"source":2,"target":0},{"value":1.1,"source":0,"target":2},{"value":1.0,"source":3,"target":0},{"value":1.0,"source":0,"target":3},{"value":1.0,"source":4,"target":0},{"value":1.0,"source":0,"target":4},{"value":1.1,"source":0,"target":5},{"value":1.1,"source":5,"target":0},{"value":1.0,"source":6,"target":1},{"value":1.0,"source":2,"target":1},{"value":1.0,"source":1,"target":2},{"value":1.0,"source":3,"target":1},{"value":1.0,"source":1,"target":3},{"value":1.0,"source":4,"target":1},{"value":1.0,"source":1,"target":4},{"value":1.0,"source":1,"target":5},{"value":1.0,"source":5,"target":1},{"value":1.1,"source":2,"target":6},{"value":1.1,"source":6,"target":2},{"value":1.0,"source":6,"target":3},{"value":1.0,"source":6,"target":4},{"value":1.1,"source":6,"target":5},{"value":1.1,"source":5,"target":6},{"value":1.0,"source":3,"target":2},{"value":1.0,"source":2,"target":3},{"value":1.0,"source":4,"target":2},{"value":1.0,"source":2,"target":4},{"value":1.1,"source":2,"target":5},{"value":1.1,"source":5,"target":2},{"value":1.0,"source":3,"target":4},{"value":1.0,"source":4,"target":3},{"value":1.0,"source":3,"target":5},{"value":1.0,"source":5,"target":3},{"value":1.0,"source":4,"target":5},{"value":1.0,"source":5,"target":4}])
  response.append(nodes_json)
  response.append(links_json)
  return response
"""

