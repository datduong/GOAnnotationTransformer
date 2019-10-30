

import os,sys,re,pickle
import numpy as np
import pandas as pd

from tqdm import tqdm

def get_1type (string,what_type=None): ## @what_type tells use sub category of Domain
  front = string.split('{') ## split by the name convention
  front = front[0].split()
  where = "-".join(front[1:3]) # 2nd and 3rd
  ## ignore the numbering ?? https://www.uniprot.org/uniprot/Q8K3W3#family_and_domains
  ## https://www.uniprot.org/uniprot/P19327#family_and_domains
  
  if 'COILED' in string:
    name = 'COILED'
  elif 'ZN_FING' in string:
    name = 'ZN_FING'
  else:
    name = " ".join(front[3::]).lower() ## check for cases like MOTIF dry motif 133-135;important conformation changes for-ligand-induced
    name = re.sub(r"\.$","",name)
    name = front[0] + " " + re.sub(r" [0-9]+$","",name)
  if len(name)==0: 
    print (string)
    exit() 
  name = re.sub(r';',' ',name)
  return name , where ## type and location

def get_location (string) :
  # 'REGION 46 69 Basic motif. {ECO:0000255|PROSITE-ProRule:PRU00978}.; REGION 71 99 Leucine-zipper. {ECO:0000255|PROSITE-ProRule:PRU00978}.; REGION 356 387 c-CRD. {ECO:0000250|UniProtKB:P19880}.'
  string = string.split(';')
  out = []
  type_out = []
  for s in string:
    this = get_1type(s)
    out.append(this)
    type_out.append(this[0])
  return out , type_out

def format_write(tup): ## tuple
  if len(tup)==0:
    return 'nan'
  out = ""
  for t in tup:
    out = out+t[0]+" "+t[1]+";"
  return re.sub(r';$','',out)

## get some data like zinc fingers etc..
path = '/u/scratch/d/datduong/deepgo/data/train/fold_1/'
os.chdir(path)

data_type = "test"
onto_type = 'mf'

for data_type in ['train','dev','test']:
  for onto_type in ['mf']:

    print ('\n\n')
    print (data_type)
    print (onto_type)
    print ('\n\n')

    fin = pd.read_csv(path+data_type+'-'+onto_type+'.tsv',sep='\t') # Entry Gene ontology IDs Sequence  Prot Emb
    prot_name = list (fin['Entry'])

    uniprot = open('/u/scratch/d/datduong/UniprotSeqTypeOct2019/uniprot-reviewed_yes.tab','r')
    # Entry Entry name  Status  Length  Gene ontology IDs
    # Region 5
    # Zinc finger 6
    # Sequence similarities -- not use
    # Repeat 8
    # Protein families -- not use
    # Motif 10
    # Compositional bias 11
    # Coiled coil 12
    # Domain [FT] 13 ## feature 
    # Domain [CC] Sequence 14 ## comment

    fout = open ( path+data_type+'-'+onto_type+'-prot-annot.tsv', 'w' )
    fout.write('Entry\tGene ontology IDs\tSequence\tProt Emb\tType\n')
    col = 'Entry\tGene ontology IDs\tSequence\tProt Emb'.split('\t')
    prot_label_type = {} ## total annotation type on the protein sequence

    for index, line in tqdm(enumerate(uniprot)):
      if index == 0:
        continue ## header
      line = line.split('\t')
      if line[0] not in prot_name :
        continue
      prot_annot = []
      row_found_in_data = fin.loc[fin['Entry'] == line[0]]
      if line[15].strip() != list ( row_found_in_data['Sequence'] )[0]:
        print ('found but not match sequence {}'.format(line[0]))
        # print (line[15].strip())
        # print (list ( fin.loc[fin['Entry'] == line[0]]['Sequence'] )[0])
        # break
        fout.write( "\t".join(row_found_in_data[i].tolist()[0] for i in col) + "\t" + format_write(prot_annot)+'\n' )
        continue
      for where_ in [6,8,10,11,12,13]:
        if len (line[where_]) > 0 :
          out, type_out = get_location (line[where_])
          prot_annot = prot_annot + out
          for t in type_out :
            # if t == 'head':
            #   print (line[0]) # https://www.uniprot.org/uniprot/P20152#family_and_domains
            if t in prot_label_type:
              prot_label_type[t] = 1 + prot_label_type[t]
            else:
              prot_label_type[t] = 1
      # if line[0]=='Q2V3L3': # Q99653
      #   break
      # if index > 40000:
      #   break
      # Entry Gene ontology IDs Sequence  Prot Emb
      fout.write( "\t".join(row_found_in_data[i].tolist()[0] for i in col) + "\t" + format_write(prot_annot)+'\n' )

    fout.close()
    uniprot.close()
    print ('total type {}'.format(len(prot_label_type)))

    pickle.dump(prot_label_type,open(data_type+'_'+onto_type+'_prot_annot_type.pickle','wb'))

    # for i in range(len(z)):
    #   if (z[i] != x[i]) :
    #     print (i)


onto_type = 'mf'
train = pickle.load(open('train_'+onto_type+'_prot_annot_type.pickle','rb'))
not_in_train = {} ## what can we do if domain not seen in train data ? 
all_prot_annot = {}
for data_type in ['dev','train','test']: #,'dev','test'
  prot_label_type = pickle.load(open(data_type+'_'+onto_type+'_prot_annot_type.pickle','rb'))
  print ('data type {} len {}'.format(data_type,len(prot_label_type)))
  ## get top domain only... may be too much to fit all types?
  # https://able.bio/rhett/sort-a-python-dictionary-by-value-or-key--84te6gv
  counter = 0
  for key, value in sorted(prot_label_type.items(), key=lambda kv: kv[1], reverse=True):
    # print("%s: %s" % (key, value))
    if key not in all_prot_annot: 
      all_prot_annot[key] = value
    else: 
      all_prot_annot[key] = value + all_prot_annot[key]
    counter = counter + 1 
    # if counter > 10: 
    #   break 
    ## what if not in train ?? 
    if key not in train: 
      if key not in not_in_train: 
        not_in_train[key] = value
      else: 
        not_in_train[key] = value + not_in_train[key]


print ('not in train {}'.format(len(not_in_train)))

pickle.dump(all_prot_annot,open(onto_type+'all_prot_annot.pickle','wb'))
print ('total unique type {}'.format(len(all_prot_annot)))

