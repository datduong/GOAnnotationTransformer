
import re,os,sys,pickle

#### create a list of label with format GO:1234567
## backtrack will be much easier

os.chdir('/u/scratch/d/datduong/UniprotJan2020/AddIsA/TrainDevTest')

for onto in ['mf','cc','bp']:
  label_found = {}
  for data_type in ['train','test','dev']: 
    fin = open(onto+"-"+data_type+".tsv","r") ## file that we used to split the 3 datasets 
    for line in fin:
      label = line.split('\t')[2]
      if "GO" not in label :
        continue ## because some empty label ??
      label = re.sub("GO","GO:",label)
      label = [l.strip() for l in label.split()]
      for lab in label:
        if lab not in label_found:
          label_found[lab] = 1
        else:
          label_found[lab] = 1 + label_found[lab]
    ##
    fin.close()
  ##
  fout = open(onto+"-label.tsv","w")
  keys = sorted ( list( label_found.keys() ) )
  for k in keys:
    fout.write(k+"\t"+str(label_found[k])+"\n")
  fout.close()


