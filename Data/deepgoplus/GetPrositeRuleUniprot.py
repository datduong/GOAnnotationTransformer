
import re,sys,os,pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

##! we need get prosite rule to find all motifs
##! uniprot naming convention is annoying
os.chdir ('/u/scratch/d/datduong/UniprotSeqTypeOct2019/')

def get_1type (string,what_type=None): ## @what_type tells use sub category of Domain
  front = string.split('{') ## split by the name convention
  front = front[0].split()
  # where = "-".join(front[1:3]) # 2nd and 3rd
  ## ignore the numbering ?? https://www.uniprot.org/uniprot/Q8K3W3#family_and_domains
  ## https://www.uniprot.org/uniprot/P19327#family_and_domains
  if 'COILED' in string:
    name = 'COILED'
  elif 'ZN_FING' in string:
    name = 'ZN_FING'
  else:
    name = " ".join(front[3::]).lower()
    ## check for cases like MOTIF dry motif 133-135;important conformation changes for-ligand-induced
    # if ';' in name: ## keep only relevant parts
    #   name = name.split(';')[0] ## return only the front
    name = re.sub(r"\.$","",name)
    name = front[0] + " " + re.sub(r" [0-9]+$","",name) # front[0] + " " + 
  #
  if len(name)==0: print (string); exit()
  #
  name = re.sub(r';',' ',name)
  return name.strip()  #, where ## type and location ### remove blank from names


def map_1_motif_prosite (string): 
  # @string is 'DOMAIN 12 47 EF-hand 1. {ECO:0000255|PROSITE-ProRule:PRU00448}.
  if 'PROSITE-ProRule:' in string: 
    rule = string.split('PROSITE-ProRule:')[-1].strip() 
    rule = re.sub(r"\}\.","",rule)
    rule = rule.split(',')[0] ## get first rule only, don't need ECO
    english_name = get_1type(string)
    return {rule:english_name}
  else: 
    return None


uniprot = open('/u/scratch/d/datduong/UniprotSeqTypeOct2019/uniprot-reviewed_yes.tab','r')

prosite_name_map = {}
for index, line in tqdm(enumerate(uniprot)): #! go through uniprot data
  if index == 0:
    continue ## header
  #
  line = line.split('\t')
  # if 'P0CU67' in line: 
  #   print (line)
  # 'DOMAIN 12 47 EF-hand 1. {ECO:0000255|PROSITE-ProRule:PRU00448}.; DOMAIN 49 84 EF-hand 2. {ECO:0000255|PROSITE-ProRule:PRU00448}.; DOMAIN 86 121 EF-hand 3. {ECO:0000255|PROSITE-ProRule:PRU00448}.; DOMAIN 123 158 EF-hand 4. {ECO:0000255|PROSITE-ProRule:PRU00448}.'
  for where_ in [13]: #
    if len ( line[where_] ) == 0 : 
      continue
    #
    all_motif = line[where_].split(';')
    for motif in all_motif: 
      out = map_1_motif_prosite ( motif )
      if out is not None: 
        prosite_name_map.update(out)
  #
  # if index > 2000: 
  #   break 

#
uniprot.close()
pickle.dump(prosite_name_map,open('prosite_uniprot_name_map.pickle','wb'))


##!! now we read in prosite data (downloaded from prosite)
prosite_download_data_name = {}
fin = open("prorule.dat","r")
for line in fin: 
  if re.match('^AC',line): 
    rule = re.sub ( ";" , "", line.split()[-1] )
    start = 1 ## only record if we found a name
  if re.match('^TR',line) and (start ==1): 
    name = line.split(';')[2].strip()
    prosite_download_data_name[name] = rule
    start = 0 ## reset


#
pickle.dump(prosite_download_data_name,open('prosite_download_data_name.pickle','wb'))








