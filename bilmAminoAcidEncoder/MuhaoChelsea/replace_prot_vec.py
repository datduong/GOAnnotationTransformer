
import sys,re,os,pickle
import numpy as np
import pandas as pd

#### take prot already have vec and override

os.chdir("/local/datdb/deepgo/dataExpandGoSet/train/fold_1/ProtAnnotTypeData")

for data_type in ['test','train','dev']:
  for onto in ['cc','bp','mf']:
    ## read in a file which we already computed vector
    map_vec = pickle.load(open("/local/datdb/deepgo/data/MuhaoProteinVec/ProtVecDict.pickle","rb"))
    ##
    ## COMMENT now get open file to replace
    fin = open(data_type+"-"+onto+"-input.tsv","r")
    fout = open(data_type+"-"+onto+"-input-muhao.tsv","w")
    for index,line in enumerate(fin):
      if index == 0:
        fout.write(line)
      else:
        line = line.split("\t")
        annot = line[-1] ##!! annotation is at the end.
        line = line[0:(len(line)-2)] ## remove vec ##!!##!! need -2
        vec = "0.0 "*256
        vec = vec.strip()
        ##!!##!!
        if line[0] in map_vec:
          vec = " ".join(s for s in map_vec[line[0]].split(';'))
        else:
          print ('in {} {} skip {}'.format(data_type,onto,line[0]))
        #
        new_line = "\t".join(l for l in line) + "\t" + vec + "\t" + annot + "\n"
        fout.write(new_line)
    #
    fout.close()
    fin.close()

