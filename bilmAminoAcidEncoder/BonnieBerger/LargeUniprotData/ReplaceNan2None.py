
import pickle,re,sys,os

os.chdir('/u/scratch/d/datduong/UniprotJan2020/AddIsA/TrainDevTest/')

for onto in ['mf','cc','bp']:
  for data_type in ['train','test','dev']:
    fin = open(onto+"-"+data_type+".tsv","r")
    fout = open(onto+"-"+data_type+"-input.tsv","w")
    for line in fin: 
      line = line.strip().split('\t')
      last = line[-1]
      if last == 'nan':
        last = 'none'
      # name seq label prot_vec motif
      fout.write(line[0]+"\t"+line[1]+"\t"+line[2]+"\t"+line[3]+"\t"+last+"\n")
    #
    fin.close()
    fout.close()
    
    