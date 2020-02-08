


import sys,re,os,pickle
import numpy as np
import pandas as pd

## take prot already have vec and override

os.chdir("/u/scratch/d/datduong/UniprotJan2020/AddIsA")


## create an array in the exact order as file
# for data_type in ['test','train','dev']:

for onto in ['cc','bp','mf']:
  #### read in data

  label_to_test = {}
  fin = open(onto+"-label-rm20.tsv","r") # cc-label-rm10p.tsv
  for line in fin:
    line = line.strip().split("\t")
    num = float(line[1])
    if num > 2: ##!! filter by occ ####
      label_to_test[ line[0] ] = num
  fin.close()

  ## read in a file which we already computed vector
  fout = open("uniprot-"+onto+"-isa-rm20-bonnie.tsv","w") ## has name seq go prot_vec domain
  fin = open("uniprot-"+onto+"-isa-bonnie.tsv","r")
  for index,line in enumerate(fin):

    if index==0:
      continue ## skip header

    line = line.strip().split("\t")
    name = line[0]
    annot = line[1] ##!! name seq label
    annot = annot.strip().split(';')
    #
    annot = sorted( [ a for a in annot if a in label_to_test ] )
    annot = " ".join(a for a in annot)
    annot = re.sub(":","",annot)
    if len(annot) == 0: 
      continue #### no empty label

    # want output: name, seq, label, vec, motif
    # seq = " ".join(line[2])
    seq = line[2] ## don't do this yet.
    fout.write( name + "\t" + seq + "\t" + annot + "\t" + line[3] + '\n' )

  #
  fin.close()
  fout.close()



