

import pickle,re,sys,os
import numpy as np
import pandas as pd


def Format2Train (fin_name,fout_name):
  # for onto in ['mf','cc','bp']:
  fin = open(fin_name,"r")
  # fout = open(fout_name,"w")
  for index,line in enumerate(fin):
    if index == 0: ## skip header
      continue
    line = line.strip().split("\t") # Entry Gene ontology IDs Sequence  Prot Emb  Type
    if len(line[2]) < 20:
      continue ## skip short sequences?
    # P0CK34  GO0006810 GO0006913 GO0016032 GO0030260 GO0044403 GO0044409 GO0044419 GO0044766 GO0046718 GO0046794 GO0046907 GO0051169 GO0051170 GO0051179 GO0051234 GO0051641 GO0051701 GO0051704 GO0075732 GO0075733 GO1902579 GO1902581 GO1902594 MPKRDAPWRHMAGTSKVSRSGNYSPSGGMGSKSNKANAWVNRPMYRKPRIYRMYKSPDVPKGCEGPCKVQSYEQRHDISHVGKVMCISDITRGNGITHRVGKRFCVKSVYILGKIWMDENIMLKNHTNSVIFWLVRDRRPYGTPMDFGQVFNMFDNEPSTATVKNDLRDRYQVMHRFNAKVSGGQYASNEQALVRRFWKVNNHVVYNHQEAGKYENHTENALLLYMACTHASNPVYATLKIRIYFYDSITN 2.0127356 2.3204482 -2.5727956 0.05463714 2.1264694 2.2226157 2.6598582 -1.3360673 0.47345287 -0.24576506 -1.5927457 0.5761297 0.30144203 1.1199504 -0.9677993 0.030553874 -1.190126 -0.6435121 1.3716252 1.8186872 -0.31875175 1.231084 -2.0255122 0.71411294 -3.0143094 -2.398779 -0.4160399 -0.5778183 0.9321428 2.1335223 1.7670814 2.252809 1.6116441 0.54933906 -1.9151797 1.6856685 -0.1230594 3.1120727 1.6652839 3.6802979 2.256363 1.3106073 -0.010380058 -3.0786116 -1.2762203 2.6824052 -0.9896302 0.9047133 -1.9570969 1.9116216 2.7822735 0.26710334 -0.019561993 -1.410613 -1.262366 1.628542 0.54004043 -0.8422738 -1.1750352 -0.5469422 1.089183 3.1632643 -1.871679 -2.4472032 2.0087004 0.6778927 -0.7896352 0.012454153 -1.3751782 -2.1542118 -1.4835027 0.82873815 -0.3728072 -1.6215867 1.0487263 -1.1527256 0.3341928 0.9411456 -1.2120696 -0.6230893 1.4279802 -1.1323034 -2.0770915 -1.1950109 1.3365928 -1.109156 1.1522198 0.3157159 -0.7521503 -1.8183339 -0.620338 1.6962907 -0.5209034 1.7124537 0.4186317 -1.9933925 2.2028923 -0.026782999 -0.52311456 -2.375578  ZN_FING 54-71;MOTIF bipartite nuclear localization signal 3-20;MOTIF nuclear localization signal 35-49;MOTIF nuclear export signal 96-117;MOTIF bipartite nuclear localization signal 195-242
    # print (line[2])
    # exit()
    seq = " ".join(line[2]) # we split the squence into single letter
    # print (line[2])
    # print (seq)
    # print (line[1])
    # want output: name, seq, label, vec, motif
    if line[0] == 'A0A2U3Y4D7': 
      print (line)
      print (line[1])
      print (seq)
      exit()
    # fout.write( line[0]+"\t"+seq+"\t"+line[1]+"\t"+line[3]+"\t"+line[4]+"\n" )
    # exit()
  #
  fin.close()
  # fout.close()


####
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
