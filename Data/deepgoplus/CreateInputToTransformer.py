
#### format the data so that our code can read it quickly

## want something like this
# P24813  M G N I L R K G Q Q I Y L A G D M K K Q M L L N K D G T P K R K V G R P G R K R I D S E A K S R R T A Q N R A A Q R A F R D R K E A K M K S L Q E R V E L L E Q K D A Q N K T T T D F L L C S L K S L L S E I T K Y R A K N S D D E R I L A F L D D L Q E Q Q K R E N E K G T S T A V S K A A K E L P S P N S D E N M T V N T S I E V Q P H T Q E N E K V M W N I G S W N A P S L T N S W D S P P G N R T G A V T I G D E S I N G S E M P D F S L D L V S N D R Q T G L E A L D Y D I H N Y F P Q H S E R L T A E K I D T S A C Q C E I D Q K Y L P Y E T E D D T L F P S V L P L A V G S Q C N N I C N R K C I G T K P C S N K E I K C D L I T S H L L N Q K S L A S V L P V A A S H T K T I R T Q S E A I E H I S S A I S N G K A S C Y H I L E E I S S L P K Y S S L D I D D L C S E L I I K A K C T D D C K I V V K A R D L Q S A L V R Q L L GO0000975 GO0000976 GO0000977 GO0000978 GO0000981 GO0000982 GO0000987 GO0001012 GO0001067 GO0001071 GO0001077 GO0001159 GO0001228 GO0003676 GO0003677 GO0003690 GO0003700 GO0005488 GO0043565 GO0044212 GO0097159 GO1901363 GO1990837 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 MOTIF bipartite nuclear localization signal 17-24;MOTIF bipartite nuclear localization signal 47-54;MOTIF nuclear export signal 372-379;DOMAIN bzip 43-106


import pickle,re,sys,os
from tqdm import tqdm
import networkx
import obonet


os.chdir('/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data') # /local/datdb

graph = obonet.read_obo('go.obo') # https://github.com/dhimmel/obonet

# graph.node['GO:0006909']
# {'def': '"An endocytosis process that results in the engulfment of external particulate material by phagocytes. The particles are initially contained within phagocytic vacuoles (phagosomes), which then fuse with primary lysosomes to effect digestion of the particles." [ISBN:0198506732]',
#  'name': 'phagocytosis',
#  'namespace': 'biological_process',
#  'xref': ['Wikipedia:Phagocytosis']}

ontology_map = {'mf':'molecular_function','bp':'biological_process','cc':'cellular_component'}

LEN_CUTOFF = 1500

for data_type in ['test','train']: #'test','train'
  #### we need to filter by category otherwise too much. can't run it.

  LongLenCounter = 0

  for ontology in ['mf','cc','bp']:
    # Entry Gene ontology IDs Sequence  Prot Emb  Type
    fin = open('FullLen/deepgoplus.cafa3.'+data_type+'-bonnie.tsv',"r") # test-mf-prot-annot.tsv
    fout = open('SeqLenLess'+str(LEN_CUTOFF)+'/deepgoplus.cafa3.'+data_type+'-bonnie-'+ontology+'.tsv',"w") # test-mf-input.tsv
    for index,line in tqdm ( enumerate(fin) ) :
      if index == 0 :
        continue ## skip header

      line = line.strip().split('\t')

      #### remove long sequences, transformer will not handle it
      if len( line[2] ) > LEN_CUTOFF:
        print ('LongLen '+line[0])
        LongLenCounter = LongLenCounter + 1
        continue

      line[1] = line[1].split(";") ## split by space, and not ";"
      ##!! filter out by ontology
      try:
        line[1] = [ lab for lab in line[1] if graph.nodes[lab]['namespace'] == ontology_map[ontology] ]
        if len(line[1]) == 0: ## may not have all categories
          line[1] = [ 'none' ]
      except:
        print (line[0])
        line[1] = [ 'none' ]

      #### do not record proteins without any labels?
      if 'none' in line[1]:
        continue

      fout.write( line[0] + "\t" + " ".join(a for a in line[2]) + "\t") ## name and seq

      line[1] = sorted(line[1]) ## just sort to read easier
      line[1] = [ re.sub(":","",lab) for lab in line[1] ]  ## remove GO:xyz style

      fout.write(" ".join(g for g in line[1]) + "\t")
      fout.write(" ".join(str(g) for g in line[3].split(';')) + "\n") ## protein-protein interaction or 3D vectors vectors

      # if line[4]=='nan': ##!! skip motif for now
      #   fout.write('none'+"\n")
      # else:
      #   fout.write(line[4]+"\n")

    ## next data input
    fin.close()
    fout.close()
    print ('LongLenCounter '+str(LongLenCounter))



