
#! same motif in prosite and uniprot?
#! possible that uniprot has more than just prosite. but prosite should be all in uniprot.

import pickle,os,sys,re
import pandas as pd
import numpy as np

def format_motif_name (name):
  return re.sub(r"_[0-9]{1,}$","",name).lower()

#
motif_in_test = {}
prot_in_test = []
test_data = open("","r")
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
