
import os, sys, re, pickle

path='/local/datdb/deepgo/data/BertNotFtAARawSeqGO/bp/fold_1/2embPpiAnnotE256H1L12I512Set0/YesPpiNoTypeScaleFreezeBert12Ep10e10Drop0.1'
os.chdir(path)
checkpoint = os.listdir(path)
checkpoint = [c for c in checkpoint if 'checkpoint' in c]
for c in checkpoint: 
  os.system('scp config.json '+c)


