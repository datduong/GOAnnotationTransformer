


import os,sys,re,pickle
import numpy as np
import pandas as pd

from tqdm import tqdm


#### get the motifs seen in traind ata only
## we originally get the motif-count for all data sets, and not just train/dev/test
## we need to return to this step and recount for train data

#### need to strip names just to be safe
# for onto_type in ['mf','cc','bp']:
#   prot_label_type = pickle.load(open(onto_type+'_prot_annot_type.pickle','rb'))
#   prot_label_type_strip = {} ## 
#   for key,val in prot_label_type.items(): 
#     try: 
#       prot_label_type_strip[key.strip()] = val
#     except: 
#       print (key)
#   #
#   pickle.dump(prot_label_type_strip,open(onto_type+'_prot_annot_type.pickle','wb'))


for onto_type in ['mf','cc','bp']:
  prot_label_type = pickle.load(open(onto_type+'_prot_annot_type.pickle','rb'))
  # read in train data again.
  counter_in_train = {}
  fin = open ("/u/scratch/d/datduong/UniprotJan2020/AddIsA/"+onto_type+"-input.tsv","r")
  for index,line in enumerate(fin):
    motif = line.strip().split('\t')[-1] ## last col
    motif = motif.split(';') # COILED 272-319;DOMAIN ig-like v-type 30-139
    if motif[0] == 'nan': 
      continue
    for m in motif: 
      # if (line.split('\t')[0] == 'Q8ZPP9'):
      m = m.strip().split() 
      m = [i.strip() for i in m]
      m = m[0:(len(m)-1)] ## ignore last position 
      m = " ".join(i.strip() for i in m) ## put back into full name ?
      if m not in prot_label_type: 
        print (line.split('\t')[0])
        print (motif)
        print ('check why we see new motif not found before')
        break
      if (m in counter_in_train):
        counter_in_train[m] = counter_in_train[m] + 1
      else: 
        counter_in_train[m] = 1
  #
  #
  pickle.dump(counter_in_train, open('train_'+onto_type+'_prot_annot_type.pickle',"wb"))
  ## write out in text anyway to see ? ... not really needed. 





#### get some data like zinc fingers etc..

path = '/u/scratch/d/datduong/UniprotJan2020/AddIsA/'
os.chdir(path)

annot_name_arr = ['_prot_annot_type'] # , '_prot_annot_type_topo'

for annot_name in annot_name_arr:
  for onto_type in ['mf','cc','bp']:
    train = pickle.load(open('train_'+onto_type+annot_name+'.pickle','rb'))
    train_count = {}
    ## must select something to be 'UNK' so that we can handle unseen domain
    fout = open('train_'+onto_type+annot_name+'.txt','w')
    fout.write('Type\tCount\tUnkChance\n')
    for key in sorted (train.keys()) :  # key, value in sorted(train.items(), key=lambda kv: kv[1], reverse=True)
      value = train[key]
      chance = value // 10
      if chance > 100: ##!! at least 100 turned into UNK, so we can estimate UNK effect
        chance = 100
      fout.write(key + "\t"+ str(value) + "\t" + str(chance)+'\n')
      train_count[key] = [value,chance]
    #
    fout.close()
    pickle.dump(train_count,open('train_'+onto_type+annot_name+'_count.pickle','wb'))

