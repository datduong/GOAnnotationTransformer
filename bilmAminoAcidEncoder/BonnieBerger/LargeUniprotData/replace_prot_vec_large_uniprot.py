
import sys,re,os,pickle
import numpy as np
import pandas as pd

## take prot already have vec and override

os.chdir("/local/datdb/deepgo/dataExpandGoSet16Jan2020/train/fold_1/ProtAnnotTypeData")


## create an array in the exact order as file
for data_type in ['test','train','dev']:
  for onto in ['cc','bp','mf']:
    #### load a known protein-vector pickle, and then simply replace into the new file
    ## read in a file which we already computed vector
    map_vec ={}
    know_file = open("/local/datdb/deepgo/dataExpandGoSet/train/fold_1/ProtAnnotTypeData/"+data_type+"-"+onto+"-input-bonnie.tsv","r") ##!!##!! okay to use these vectors, they are designed based on sequence, not GO labels
    for line in know_file: ## no header
      line = line.split('\t')
      map_vec[line[0]] = re.sub(" ",";",line[-2]) ## 2nd to last
    know_file.close()
    ##
    #### COMMENT now get open file to replace
    fin = open(data_type+"-"+onto+"-input.tsv","r") ## has name seq go prot_vec domain
    fout = open(data_type+"-"+onto+"-input-bonnie.tsv","w")
    for index,line in enumerate(fin):
      # if index == 0:
      #   fout.write(line)
      # else:
      line = line.strip().split("\t")
      annot = line[-1] ##!! annotation is at the end.
      line = line[0:(len(line)-2)] ## remove vec ##!!##!! need -2
      vec = "0.0 "*100 ## has 100 by default
      vec = vec.strip()
      ##!!##!!
      if line[0] in map_vec:
        vec = " ".join(s for s in map_vec[line[0]].split(';'))
      else:
        print ('in {} {} skip {}'.format(data_type,onto,line[0]))
      #
      new_line = "\t".join(l for l in line) + "\t" + vec + "\t" + annot + "\n"
      fout.write(new_line)
    fout.close()
    fin.close()

