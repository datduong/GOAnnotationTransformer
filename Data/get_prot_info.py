

import os,sys,re,pickle
import numpy as np
import pandas as pd

from tqdm import tqdm

def find_change(string1,string2):
  # if same len, then try to assign domain if change is not in range
  # else just not use domain
  if string1==string2:
    return 0, None ## dont need to do the check at @same_len

  string1 = np.array([s for s in string1])
  string2 = np.array([s for s in string2])
  if len(string1) != len(string2):
    return 1, None ## @string2 is the "old" data, and @1 is used to indicate not same len
  else: ## if come to here, then string1!=string2, but the len are the same
    where_not_same = string1!=string2
    where_not_same = np.where (where_not_same)[0]
    return 0, where_not_same ## @0 says that strings are same len


def is_in_interval(a,b,c):
  if (b<a) and (a<c):
    return True
  else:
    return False


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
    name = " ".join(front[3::]).lower()
    ## check for cases like MOTIF dry motif 133-135;important conformation changes for-ligand-induced
    # if ';' in name: ## keep only relevant parts
    #   name = name.split(';')[0] ## return only the front
    name = re.sub(r"\.$","",name)
    name = front[0] + " " + re.sub(r" [0-9]+$","",name)

  if len(name)==0: print (string); exit()

  name = re.sub(r';',' ',name)
  return name , where ## type and location

def get_location (string,where_change=None) :
  # P08909  5HT2C_RAT       reviewed        460     GO:0001587; GO:0001662; GO:0004930; GO:0004993; GO:0005887; GO:0007187; GO:0007200; GO:0007208; GO:0007268; GO:0007626; GO:0007631; GO:0009897; GO:0009986; GO:0014054; GO:0014057; GO:0030425; GO:0030594; GO:0031100; GO:0031583; GO:0031644; GO:0032098; GO:0035095; GO:0040013; GO:0042493; GO:0043397; GO:0045907; GO:0045963; GO:0048016; GO:0051209; GO:0051482; GO:0051930    REGION 135 140 Agonist binding. {ECO:0000250}.; REGION 326 330 Agonist binding. {ECO:0000250}.          SIMILARITY: Belongs to the G-protein coupled receptor 1 family. {ECO:0000255|PROSITE-ProRule:PRU00521}.            G-protein coupled receptor 1 family     MOTIF 152 154 DRY motif; important for ligand-induced conformation changes. {ECO:0000250}.; MOTIF 366 370 NPxxY motif; important for ligand-induced conformation changes and signaling. {ECO:0000250}.; MOTIF 458 460 PDZ-binding.                         DOMAIN: The PDZ domain-binding motif is involved in the interaction with MPDZ.  MVNLGNAVRSLLMHLIGLLVWQFDISISPVAAIVTDTFNSSDGGRLFQFPDGVQNWPALSIVVIIIMTIGGNILVIMAVSMEKKLHNATNYFLMSLAIADMLVGLLVMPLSLLAILYDYVWPLPRYLCPVWISLDVLFSTASIMHLCAISLDRYVAIRNPIEHSRFNSRTKAIMKIAIVWAISIGVSVPIPVIGLRDESKVFVNNTTCVLNDPNFVLIGSFVAFFIPLTIMVITYFLTIYVLRRQTLMLLRGHTEEELANMSLNFLNCCCKKNGGEEENAPNPNPDQKPRRKKKEKRPRGTMQAINNEKKASKVLGIVFFVFLIMWCPFFITNILSVLCGKACNQKLMEKLLNVFVWIGYVCSGINPLVYTLFNKIYRRAFSKYLRCDYKPDKKPPVRQIPRVAATALSGRELNVNIYRHTNERVARKANDPEPGIEMQVENLELPVNPSNVVSERISSV
  string = string.split(';')
  out = []
  type_out = []
  for s in string:
    # notice some string is an extended description of the other strings
    if re.findall(' [0-9]+ [0-9]+ ', s):

      skip = False
      if where_change is not None:
        # check if it contain the change
        bound = re.findall(' [0-9]+ [0-9]+ ', s)[0].strip().split()
        low = float(bound[0])
        high = float(bound[1])
        for z in where_change: ## found 1 change that occur in this region. so we skip this region
          if is_in_interval (z,low,high):
            skip = True
            break

      if skip:
        continue

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
  for onto_type in ['cc','bp']:

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

      same_len, where_change_index = find_change( line[15].strip(), list(row_found_in_data['Sequence'])[0] )
      if same_len == 1:
        print ('found but not match sequence {}'.format(line[0]))
        fout.write( "\t".join(row_found_in_data[i].tolist()[0] for i in col) + "\t" + format_write(prot_annot)+'\n' )
        continue

      # # if line[15].strip() != list ( row_found_in_data['Sequence'] )[0]:
      #   print ('found but not match sequence {}'.format(line[0]))
      #   # print (line[15].strip())
      #   # print (list ( fin.loc[fin['Entry'] == line[0]]['Sequence'] )[0])
      #   # break
      #   fout.write( "\t".join(row_found_in_data[i].tolist()[0] for i in col) + "\t" + format_write(prot_annot)+'\n' )
      #   continue

      for where_ in [6,8,10,11,12,13]:
        if len (line[where_]) > 0 :
          out, type_out = get_location (line[where_], where_change=where_change_index)
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

