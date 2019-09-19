
## don't need all the check points 
import os,sys,re,pickle

main_path = '/local/datdb/deepgo/data/BertNotFtAAseqGO/fold_1mf'
last_need = 10000

files = os.listdir(main_path)
for f in files : 
  if 'checkpoint-' in f: 
    num = int ( f.split('checkpoint-')[-1] ) ## last number
    if num < last_need: 
      print (f)
      os.system ('rm -rf '+f)


