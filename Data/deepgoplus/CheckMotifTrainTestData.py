
#! same motif in prosite and uniprot?
#! possible that uniprot has more than just prosite. but prosite should be all in uniprot.

import pickle,os,sys,re
import pandas as pd
import numpy as np
from difflib import SequenceMatcher

def format_motif_name (name):
  return re.sub(r"_[0-9]{1,}$","",name).lower()

#
motif_in_test = {}
prot_in_test = []
test_data = open("/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000/test-mf-motif.tsv","r")
for line in test_data:
  line = line.split('\t')
  prot_in_test.append( line[0] )
  for motif in line[1::] :
    if ('(-1)' in motif) or ('(0)' in motif): #! skip match by pattern ? https://prosite.expasy.org/scanprosite/scanprosite_doc.html#of_miniprofiles
      continue
    motif = motif.split(';')
    motif_name = format_motif_name ( motif[0] )
    if motif_name in motif_in_test:
      motif_in_test[motif_name] = motif_in_test[motif_name] + 1
    else:
      motif_in_test[motif_name] = 1


#
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

string1 = "cyt dcmp deaminases"
string2 = "cmp/dcmp-type deaminase"

match = SequenceMatcher(None, string1, string2).find_longest_match(0, len(string1), 0, len(string2))

print(match)  # -> Match(a=0, b=15, size=9)
print(string1[match.a: match.a + match.size])  # -> apple pie
print(string2[match.b: match.b + match.size])  # -> apple pie



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
