


import sys,re,os,pickle
import numpy as np
import pandas as pd

## take prot already have vec and override

os.chdir("/u/scratch/d/datduong/UniprotJan2020")


## create an array in the exact order as file
# for data_type in ['test','train','dev']:

for onto in ['cc','bp','mf']:
  #### read in data

  label_to_test = {}
  fin = open(onto+"-label-rm10p.tsv","r") # cc-label-rm10p.tsv
  for line in fin:
    label_to_test[ line.strip().split("\t")[0] ] = 1
  fin.close()

  ## read in a file which we already computed vector
  fout = open("uniprot-filter3rm10-"+onto+"-bonnie.tsv","w") ## has name seq go prot_vec domain
  fin = open("uniprot-filter3-"+onto+"-bonnie.tsv","r")
  for index,line in enumerate(fin):
    line = line.strip().split("\t")
    name = line[0]
    annot = line[2] ##!! name seq label
    annot = re.sub(":","",annot)
    annot = annot.strip().split()
    #
    annot = sorted( [ a for a in annot if a in label_to_test ] )
    annot = " ".join(a for a in annot)

    # want output: name, seq, label, vec, motif
    fout.write( name + "\t" + line[1] + "\t" + annot + "\t" + line[3] + '\n' )

  #
  fin.close()
  fout.close()



GO0005575 GO0005622 GO0009579 GO0016020 GO0034357 GO0042651
