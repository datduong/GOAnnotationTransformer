
import sys,re,os,pickle
import numpy as np
import pandas as pd

## take prot already have vec and override

os.chdir("/local/datdb/deepgo/dataExpandGoSet/train/fold_1/")


## create an array in the exact order as file
for data_type in ['test','train','dev']:
  for onto in ['cc','bp','mf']:
    ## read in a file which we already computed vector
    map_vec ={}
    know_file = open("/local/datdb/deepgo/data/train/fold_1/ProtAnnotTypeData/"+data_type+"-"+onto+"-input-bonnie.tsv","r")
    for line in know_file: ## no header
      line = line.split('\t')
      map_vec[line[0]] = re.sub(" ",";",line[-2]) ## 2nd to last
    know_file.close()
    ##
    ## COMMENT now get open file to replace
    fin = open(data_type+"-"+onto+"-same-origin.tsv","r")
    fout = open(data_type+"-"+onto+"-same-origin-bonnie.tsv","w")
    for index,line in enumerate(fin):
      if index == 0:
        fout.write(line)
      else:
        line = line.split("\t")
        line = line[0:(len(line)-1)] ## remove vec
        try:
          new_line = "\t".join(l for l in line) + "\t" + map_vec[line[0]] + "\n"
          fout.write(new_line)
        except: 
          print ('skip {}'.format(line[0]))
          continue ## skip ?? 
    fout.close()
    fin.close()

