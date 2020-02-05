
import sys,re,os,pickle
import numpy as np
import pandas as pd

## take prot already have vec and override

os.chdir("/local/datdb/UniprotJan2020")


## create an array in the exact order as file
# for data_type in ['test','train','dev']:

for onto in ['cc','bp','mf']:
  #### load a known protein-vector pickle, and then simply replace into the new file
  ## read in a file which we already computed vector
  map_vec ={}
  know_file = open("/local/datdb/UniprotJan2020/uniprot-"+onto+"-bonnie.tsv","r") ##!!##!! name label seq vector
  for line in know_file: ## no header
    line = line.split('\t')
    map_vec[line[0]] = re.sub(" ",";",line[-1]) ## take only vec, and it's at the end
  know_file.close()
  ##
  #### COMMENT now get open file to replace
  fin = open("uniprot-filter3-"+onto+".csv","r") ## has name seq go prot_vec domain
  fout = open("uniprot-filter3-"+onto+"-bonnie.tsv","w")
  for index,line in enumerate(fin):
    # if index == 0:
    #   fout.write(line)
    # else:
    line = line.strip().split("\t")
    name = line[0]
    annot = line[1] ##!! name label seq vector
    annot = re.sub(":","",annot)
    annot = re.sub(";"," ",annot).strip()
    #
    if (len(line[2])<20) or (len(line[2])>500): #### skip long length? and super short one?
      continue
    seq = " ".join(s for s in line[2]) ## split seq by space
    vec = "0.0 "*100 ## has 100 by default
    vec = vec.strip()
    ##!!##!!
    if name in map_vec:
      vec = " ".join(s for s in map_vec[name].split(';'))
    else:
      print ('in {} skip {}'.format(onto,name))
    #
    # want output: name, seq, label, vec, motif
    new_line = name + "\t" + seq +"\t" + label + "\t" + vec + "\t" + annot + "\n"
    fout.write(new_line)
  fout.close()
  fin.close()

