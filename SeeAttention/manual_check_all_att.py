import pickle,os,sys,re
import numpy as np
import pandas as pd


os.chdir('/u/scratch/d/datduong/deepgo/data/BertNotFtAARawSeqGO/fold_1mf/')
GO2all = pickle.load(open("GO2AA_attention_H1to2.pickle","rb"))


# obs = 0 ## prot index number 

for head in range (2) :
  this_obs = GO2all[24]
  # head = 0
  go_index = 421
  x = this_obs [ head ][go_index]
  print ( np.quantile ( np.round(x,6), q=[.25,.5,.75,.95] ) )


fout = open('check_x2.txt','w')
fout.write(",".join(str(z) for z in x)+"\n")
fout.close() 



##

m = read.table('check_x2.txt',header=F)
k = strsplit(as.character(m[1,1]),",")
k = sapply(k,as.numeric)
windows()
plot(1:length(k), k,type='l')

