

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
  #! write out
  test_data2.write(line)


# prosite_uniprot_name_map [ prosite_download_data_name['ABC_TRANSPORTER_2'] ]

test_data2.close()
test_data.close()


#### #! now we will merge new name into sequence data, and possibly keep similarity by seq??

#! load data found in training (after using uniprot)
in_train = pickle.load(open("/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000/bonnie+motif/mf_all_prot_annot_type.pickle","rb"))

name_in_train = {} #! remove words like DOMAIN, MOTIFS
for name,count in in_train.items():
  major_type = name.split()[0]
  name = re.sub (major_type,"",name).strip() # remove words like DOMAIN, MOTIFS
  name_in_train.setdefault(name,count)

#
def make_name_variation (name):
  name = name.lower()
  if ('_' in name):
    name1 = re.sub('_',' ',name)
    name2 = re.sub('_','-',name)
    return [name,name1,name2]
  else:
    return [name]

#### #! read the new file in again.

motif_in_test = {}
test_data= open("/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000/test-mf-motif-rename.tsv","r")
for line in test_data:
  line2 = line.split('\t')
  for motif in line2[1::] :
    # if ('(-1)' in motif) or ('(0)' in motif): #! skip match by pattern ? https://prosite.expasy.org/scanprosite/scanprosite_doc.html#of_miniprofiles
    #   continue
    motif_name = motif.split(';')[0] ## just take name
    if motif_name in motif_in_test:
      motif_in_test[motif_name] = motif_in_test[motif_name] + 1
    else:
      motif_in_test[motif_name] = 1

#
test_data.close()

name_map = {} ## map old name to new name
name_found = {}
for motif,count in motif_in_test.items():
  #! we checked only DOMAIN, so that it has prosite-rule
  if motif in in_train:
    name_found.setdefault(motif,count) # don't need to map
  #! different variation in name in uniprot and prosite, but we didn't find as "prosite_rule"
  name_variation = make_name_variation(motif)
  for name in name_variation:
    if name in name_in_train:
      name_found.setdefault(name,count)
      name_map [motif] = name #! replace name later


#
pickle.dump(name_map, open('/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000/domain_name_map.pickle','wb'))


#### #! now we replace the name, must add in MOTIF and so forth

# ZN_FING

def make_entry (motif):
  motif = motif.split(';') # name, pos, score
  #? must format name to remove the "_1" "_2" at the end?
  #! have to do this, otherwise the names don't match, and we will not remove enough duplicated regions
  # T96060019016  WD_REPEATS_2;39-79;8.804  WD_REPEATS_REGION;39-128;16.269 WD_REPEATS_2;80-128;10.508  WD_REPEATS_1;106-120;(-1) WD_REPEATS_REGION;319-359;11.102  WD_REPEATS_1;337-351;(-1)
  name = format_motif_name (motif[0])
  if ('_REPEAT' in name) or ('_REP' in name): 
    name = format_repeat_name (name)
  if ('ZINC_FINGER' in name) or ('ZF_' in name): 
    name = format_zincfinger_name (name)
  return { motif[1] : [name, motif[2]] } # return a dict so we can look up quick


def check_overlap(range1,range2): #! @range1 is a string 123-456
  # https://stackoverflow.com/a/6821298/7239347
  range1 = range1.split('-')
  x = [int(range1[0]),int(range1[1])]
  range2 = range2.split('-')
  y = [int(range2[0]),int(range2[1])]
  overlap = range(max(x[0], y[0]), min(x[-1], y[-1])+1)
  if len(overlap) == 0:
    return 0
  else:
    return 1 ## yes overlap


def format_zincfinger_name(string): 
  # we just use single zinc finger... oh well... we don't care what type of zinc finger
  return 'ZN_FING'


def format_repeat_name(string): 
  # notice that we use "REPEAT wd", but uniprot uses WD_REPEATS_2;39-79;8.804  WD_REPEATS_REGION;39-128;16.269
  # check with map
  if string in name_map: 
    name = name_map[name]
  else: 
    name = string.split('_REP')[0].lower() ## just get first ?
  return 'REPEAT ' + name

def remove_overlap(motif_dict):
  to_remove = []
  position = list ( motif_dict.keys() )
  for i in range(0,len(position)-1):
    for j in range(i+1,len(position)):
      # check range overlap
      if check_overlap(position[i],position[j]) == 1: ## yes overlap
        ## check subset in name? WD_REPEATS;39-79;8.804  WD_REPEATS_REGION ??
        if motif_dict[position[i]][0] == motif_dict[position[j]][0]: ## same name
          if ('(0)' in motif_dict[position[i]][1]) or ('(-1)' in motif_dict[position[i]][1]):
            to_remove.append(position[i])
          if ('(0)' in motif_dict[position[j]][1]) or ('(-1)' in motif_dict[position[j]][1]):
            to_remove.append(position[j])
  ## end.
  return list ( set(to_remove) )



motif_in_test = {}
test_data= open("/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000/test-mf-motif-rename.tsv","r")
fout = open("/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000/test-mf-motif-rename-pass2.tsv","w")
for line in test_data:
  # {1-2:name, 1-4:name} if overlap, then take domain over the match by pattern
  line2 = line.split('\t')
  #
  # if line2[0] == 'T96060007054':
  #   break
  if len(line2[1::]) == 1: # has exactly 1 len, then don't remove?
    motif = line2[1::][0]
    motif = motif.strip().split(';')
    if motif[0] in name_map: 
      new_name = name_map [motif[0]] ## new name
      fout.write( line2[0] + '\t' + new_name + "\t"+ motif[1] + "\t" + motif[2] + "\n" )
    else: 
      fout.write(line) ## write the same line
    continue # skip the rest
  #
  motif_this_line = {} # more than 1 motifs
  for motif in line2[1::] :
    #! skip match by pattern ? https://prosite.expasy.org/scanprosite/scanprosite_doc.html#of_miniprofiles
    if re.search (r'_REGION',motif): 
      continue ## skip
    motif_this_line.update ( make_entry (motif) )
  # now we check overlap
  to_remove = remove_overlap (motif_this_line)
  this_line_out = ""
  for key,val in motif_this_line.items():
    if key in to_remove:
      continue
    #
    this_line_out = this_line_out + val[0] + ';' + key + ';' + val[1] + '\t'
  #
  if len(this_line_out) == 0: 
    this_line_out = 'none'
  fout.write ( line2[0] + '\t' + this_line_out.strip() + '\n' )

#
fout.close()
test_data.close() #? close. 


# T96060019016  WD_REPEATS_2;39-79;8.804  WD_REPEATS_REGION;39-128;16.269 WD_REPEATS_2;80-128;10.508  WD_REPEATS_1;106-120;(-1) WD_REPEATS_REGION;319-359;11.102  WD_REPEATS_1;337-351;(-1)

# T96060019016  WD_REPEATS;39-79;8.804  WD_REPEATS_REGION;39-128;16.269 WD_REPEATS;80-128;10.508  WD_REPEATS_REGION;319-359;11.102  WD_REPEATS;337-351;(-1)