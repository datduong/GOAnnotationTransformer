
import sys,re,os,pickle
import numpy as np
import pandas as pd

## take prot already have vec and override

os.chdir("/local/datdb/deepgo/dataExpandGoSet16Jan2020/train/fold_1/ProtAnnotTypeData")


## create an array in the exact order as file
for data_type in ['test','train','dev']:
  for onto in ['cc','bp','mf']:
    #### load a known protein-vector pickle, and then simply replace into the new file
    ## read in a file which we already computed vector
    map_vec ={}
    know_file = open("/local/datdb/deepgo/dataExpandGoSet/train/fold_1/ProtAnnotTypeData/"+data_type+"-"+onto+"-input-bonnie.tsv","r") ##!!##!! okay to use these vectors, they are designed based on sequence, not GO labels
    for line in know_file: ## no header
      line = line.split('\t')
      map_vec[line[0]] = re.sub(" ",";",line[-2]) ## 2nd to last
    know_file.close()
    ##
    #### COMMENT now get open file to replace
    fin = open(data_type+"-"+onto+"-input.tsv","r") ## has name seq go prot_vec domain
    fout = open(data_type+"-"+onto+"-input-bonnie.tsv","w")
    for index,line in enumerate(fin):
      if index == 0:
        fout.write(line)
      else:
        line = line.strip().split("\t")
        annot = line[-1] ##!! annotation is at the end.
        line = line[0:(len(line)-2)] ## remove vec ##!!##!! need -2
        vec = "0.0 "*100 ## has 100 by default
        vec = vec.strip()
        ##!!##!!
        if line[0] in map_vec:
          vec = " ".join(s for s in map_vec[line[0]].split(';'))
        else:
          print ('in {} {} skip {}'.format(data_type,onto,line[0]))
        #
        new_line = "\t".join(l for l in line) + "\t" + vec + "\t" + annot + "\n"
        fout.write(new_line)
    fout.close()
    fin.close()



# P24813  M G N I L R K G Q Q I Y L A G D M K K Q M L L N K D G T P K R K V G R P G R K R I D S E A K S R R T A Q N R A A Q R A F R D R K E A K M K S L Q E R V E L L E Q K D A Q N K T T T D F L L C S L K S L L S E I T K Y R A K N S D D E R I L A F L D D L Q E Q Q K R E N E K G T S T A V S K A A K E L P S P N S D E N M T V N T S I E V Q P H T Q E N E K V M W N I G S W N A P S L T N S W D S P P G N R T G A V T I G D E S I N G S E M P D F S L D L V S N D R Q T G L E A L D Y D I H N Y F P Q H S E R L T A E K I D T S A C Q C E I D Q K Y L P Y E T E D D T L F P S V L P L A V G S Q C N N I C N R K C I G T K P C S N K E I K C D L I T S H L L N Q K S L A S V L P V A A S H T K T I R T Q S E A I E H I S S A I S N G K A S C Y H I L E E I S S L P K Y S S L D I D D L C S E L I I K A K C T D D C K I V V K A R D L Q S A L V R Q L L       GO0000975 GO0000976 GO0000977 GO0000978 GO0000981 GO0000982 GO0000987 GO0001012 GO0001067 GO0001071 GO0001077 GO0001159 GO0001228 GO0003676 GO0003677 GO0003690 GO0003700 GO0005488 GO0043565 GO0044212 GO0097159 GO1901363 GO1990837   0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 MOTIF bipartite nuclear localization signal 17-24;MOTIF bipartite nuclear localization signal 47-54;MOTIF nuclear export signal 372-379;DOMAIN bzip 43-106

# P24813  M G N I L R K G Q Q I Y L A G D M K K Q M L L N K D G T P K R K V G R P G R K R I D S E A K S R R T A Q N R A A Q R A F R D R K E A K M K S L Q E R V E L L E Q K D A Q N K T T T D F L L C S L K S L L S E I T K Y R A K N S D D E R I L A F L D D L Q E Q Q K R E N E K G T S T A V S K A A K E L P S P N S D E N M T V N T S I E V Q P H T Q E N E K V M W N I G S W N A P S L T N S W D S P P G N R T G A V T I G D E S I N G S E M P D F S L D L V S N D R Q T G L E A L D Y D I H N Y F P Q H S E R L T A E K I D T S A C Q C E I D Q K Y L P Y E T E D D T L F P S V L P L A V G S Q C N N I C N R K C I G T K P C S N K E I K C D L I T S H L L N Q K S L A S V L P V A A S H T K T I R T Q S E A I E H I S S A I S N G K A S C Y H I L E E I S S L P K Y S S L D I D D L C S E L I I K A K C T D D C K I V V K A R D L Q S A L V R Q L L       GO0000981 GO0000982 GO0001071 GO0001077 GO0001228 GO0003676 GO0003677 GO0003700 GO0005488 GO0043565 GO0097159 GO1901363 1.7086899 2.9371815 -2.1639278 0.73574066 2.0095403 1.5610785 1.4358937 -1.8453534 0.8764646 -0.7049832 -1.9137208 -0.2609287 -0.3465057 1.3764834 -1.5572621 -0.6707294 -0.7015525 -1.2138381 1.8691612 1.4134102 -0.41737497 1.56581 -1.3325707 0.44410166 -2.500504 -2.2543368 0.41753787 -1.4663787 1.0536602 2.7899048 1.5239813 1.7779022 2.1339169 -0.4003217 -2.1823595 1.1433305 0.5599026 3.436931 1.5368394 3.1181211 2.7305949 0.24831374 -1.091658 -2.5408394 -1.6933004 1.0487809 0.6708526 0.5780349 -1.88304 1.5802318 3.5720003 0.47251526 -1.1604546 -2.0393517 -0.469036 0.85834485 -0.78384554 -0.061952595 -0.49874583 -1.455135 2.0247009 2.3927796 -1.8992213 -1.529914 2.1460564 1.0341935 -1.1371604 0.52162987 -1.5785094 -1.330142 -1.7133 0.92283225 -1.9550374 -2.9450953 0.93020606 -1.3986292 0.98889095 0.8285901 -2.5398433 -0.6010293 1.1799023 -1.1386067 -1.1376082 -0.22008406 1.9607855 -0.7613414 2.0002909 1.053337 -0.9155342 -1.4088498 -1.3588339 2.0324843 -0.2437854 2.403235 0.070868835 -1.952719 2.4803252 -0.79034823 -0.63242435 -1.8576721   MOTIF bipartite nuclear localization signal 17-24;MOTIF bipartite nuclear localization signal 47-54;MOTIF nuclear export signal 372-379;DOMAIN bzip 43-106

