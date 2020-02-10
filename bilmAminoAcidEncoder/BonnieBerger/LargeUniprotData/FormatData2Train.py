

import pickle,re,sys,os
import numpy as np
import pandas as pd


def Format2Train (fin_name,fout_name):
  # for onto in ['mf','cc','bp']:
  fin = open(fin_name,"r")
  fout = open(fout_name,"w")
  max_len = 0 
  num_at500 = 0
  for index,line in enumerate(fin):
    if index == 0: ## skip header
      continue
    line = line.strip().split("\t") # Entry Gene ontology IDs Sequence  Prot Emb  Type
    if len(line[2]) < 20:
      continue ## skip short sequences?
    if len(line[2]) > 350:
      if np.random.uniform() < .5: ## take half of the long range
        continue ## skip long sequences otherwise too hard to fit on gpu
    if len(line[2]) > max_len: 
      max_len = len(line[2])
    #
    if len(line[2]) == 500:
      num_at500 = num_at500 + 1
    seq = " ".join(line[2]) # we split the squence into single letter
    # want output: name, seq, label, vec, motif
    if line[0] == 'A0A2U3Y4D7': ## see example
      print (line)
    ##
    fout.write( line[0]+"\t"+seq+"\t"+line[1]+"\t"+line[3]+"\t"+line[4]+"\n" )
  #
  fin.close()
  fout.close()
  print ('max_len {}, and num at 500, {}'.format(max_len,num_at500))


####

print ('remove short and long sequences')
os.chdir('/u/scratch/d/datduong/UniprotJan2020/AddIsA')
Format2Train('bp-prot-annot.tsv','bp-input.tsv')
Format2Train('mf-prot-annot.tsv','mf-input.tsv')
Format2Train('cc-prot-annot.tsv','cc-input.tsv')


#### how should we split the data ??

## constraint so that count>2 ... at the end, should we do zeroshot. if we do zeroshot, then random split works.
## if we want to eval rare labels, we need rare labels in the test data and also train

## very costly to train 2 models. so what should we do ??
## try to even split as many rare as possible.

for onto in [ 'mf', 'cc', 'bp' ] : 

  print ('\nonto {}'.format(onto))
  rare_label_array = {}
  label_to_test = {}
  fin = open(onto+"-label-rm20.tsv","r") # cc-label-rm10p.tsv
  for line in fin:
    line = re.sub("GO:","GO",line) ## because input doesn't use GO:
    line = line.strip().split("\t")
    num = float(line[1])
    if num > 2: ##!! filter by occ ####
      label_to_test[ line[0] ] = num
      if num <= 5:
        rare_label_array[line[0]] = num
  fin.close()
  print ('number of label {} is {}'.format(onto,len(label_to_test)))
  print ('number of rare {} is {}'.format(onto,len(rare_label_array)))

  #### need to design a good test set.
  # name, seq, label, vec, motif
  rare_label_array_in_test = {}
  in_test = {}
  in_train = {}
  fin = open(onto+'-input.tsv',"r")
  for index,line in enumerate (fin) :
    line = line.split("\t")
    labels = line[2].split()
    for l in labels:
      if l in rare_label_array:
        ## if rare label, we add 2 observations into test
        if l not in rare_label_array_in_test:
          rare_label_array_in_test[l] = 1 ## not seen this rare label, so add it immediately into testset
          in_test[line[0]] = 1
        else:
          #
          if rare_label_array_in_test[l] > 2:
            ## we care about the rare labels
            in_train[line[0]] = 1 ## if we add 2 obs into test already, then add the rest into train
          else:
            rare_label_array_in_test[l] = rare_label_array_in_test[l] + 1
            in_test[line[0]] = 1

  #
  fin.close()
  #
  print ('len of test {}'.format(len(in_test)))
  print ('len of train {}'.format(len(in_train)))
  pickle.dump(in_test, open(onto+'_in_test_name.pickle','wb'))
  pickle.dump(in_train, open(onto+'_in_train_name.pickle','wb'))





# #### split using int(n * 0.8)
# def train_validate_test_split(df, train_percent=.80, seed=1234):
#   np.random.seed(seed)
#   perm = np.random.permutation(df.index)
#   m = len(df.index)
#   train_end = int(train_percent * m)
#   train = df.ix[perm[:train_end]]
#   validate = df.ix[perm[train_end:]]
#   return train, validate


# main_dir = ''
# os.chdir(main_dir)
