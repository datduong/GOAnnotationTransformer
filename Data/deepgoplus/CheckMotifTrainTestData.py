
#! same motif in prosite and uniprot?
#! possible that uniprot has more than just prosite. but prosite should be all in uniprot.

import pickle,os,sys,re
import pandas as pd
import numpy as np
from difflib import SequenceMatcher

prosite_uniprot_name_map = pickle.load(open('/u/scratch/d/datduong/UniprotSeqTypeOct2019/prosite_uniprot_name_map.pickle','rb'))
prosite_download_data_name = pickle.load(open('/u/scratch/d/datduong/UniprotSeqTypeOct2019/prosite_download_data_name.pickle','rb'))


def format_motif_name (name):
  return re.sub(r"_[0-9]{1,}$","",name) #.lower()

#
motif_in_test = {}
prot_in_test = []
test_data = open("/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000/test-mf-motif.tsv","r")
test_data2= open("/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000/test-mf-motif-rename.tsv","w")
for line in test_data:
  line2 = line.split('\t')
  prot_in_test.append( line2[0] )
  for motif in line2[1::] :
    # if ('(-1)' in motif) or ('(0)' in motif): #! skip match by pattern ? https://prosite.expasy.org/scanprosite/scanprosite_doc.html#of_miniprofiles
    #   continue
    motif = motif.split(';')
    # if motif[0] == 'ABC_TRANSPORTER_1':
    #   print (line)
    #   break
    try:
      #! map back into naming used by uniprot
      if prosite_download_data_name[motif[0]] in prosite_uniprot_name_map:
        motif_name_new = prosite_uniprot_name_map [ prosite_download_data_name[motif[0]] ] # name-->rule-->uniprot
        line = re.sub(motif[0],motif_name_new,line)
        base_name = format_motif_name ( motif[0] )
        base_name = re.compile ( base_name + "_[0-9]{1,}") ## add back numbering, this might be used in prosite_rule.dat
        line = re.sub(base_name,motif_name_new,line)
    except:
      ##!! match key in @prosite_download_data_name to name found
      base_name = format_motif_name ( motif[0] )
      best_key = 'none'
      for key in prosite_download_data_name:
        if re.match(base_name,key):
          best_key = key
      if (best_key != 'none') and (prosite_download_data_name[best_key] in prosite_uniprot_name_map):
        motif_name_new = prosite_uniprot_name_map [ prosite_download_data_name[best_key] ] # name-->rule-->uniprot
        line = re.sub(motif[0],motif_name_new,line)
    #
    # if motif_name in motif_in_test:
    #   motif_in_test[motif_name] = motif_in_test[motif_name] + 1
    # else:
    #   motif_in_test[motif_name] = 1
  #! write out
  test_data2.write(line)

# DOMAIN 262 505 ABC transporter 2. {ECO:0000255|PROSITE-ProRule:PRU00434}
# prosite_uniprot_name_map['PRU00434']

prosite_uniprot_name_map [ prosite_download_data_name['ABC_TRANSPORTER_2'] ]
# T96060017039  ABC_TRANSPORTER_1;229-243;(-1)
#
test_data2.close()
test_data.close()

#! load data found in training (after using uniprot)
in_train = pickle.load(open("/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000/bonnie+motif/mf_all_prot_annot_type.pickle","rb"))

name_in_train = {}
for name,count in in_train.items():
  major_type = name.split()[0]
  name = re.sub (major_type,"",name).strip() # remove words like DOMAIN, MOTIFS
  name_in_train.setdefault(name,count)

#

def make_name_variation (name):
  if ('_' in name):
    name1 = re.sub('_',' ',name)
    name2 = re.sub('_','-',name)
    return [name,name1,name2]
  else:
    return [name]

name_map = {} ## map old name to new name
name_found = {}
for motif,count in motif_in_test.items():
  #### different variation in name in uniprot and prosite
  name_variation = make_name_variation(motif)
  for name in name_variation:
    if name in name_in_train:
      name_found.setdefault(name,count)
      name_map [name_variation[0]] = name #! replace name later


#! find by pattern matching substring ?


name_map_pattern = {}
name_found_pattern = {}
for motif,count in motif_in_test.items():
  if motif_name in name_map:
    continue
  #
  name_variation = make_name_variation(motif)
  for name in name_variation:
    if name in name_in_train:
      name_found.setdefault(name,count)
      name_map [name_variation[0]] = name #! replace name later
