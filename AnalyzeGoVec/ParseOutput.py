
import sys,re,os,pickle


#### tally result

def submitJobs (file_name):

  # file_name = '/local/datdb/deepgo/data/BertNotFtAARawSeqGO/EvalLabelByGroup/DeepGOFlatSeqProtBase/output_count.txt'
  fin = open(file_name,"r")
  record = False
  for line in fin:
    line = line.strip()
    # if ('low' in line) or ('middle' in line) or ('high' in line):
    #   print (line)
    if 'hamming' in line:
      record = False
    if 'recall at' in line:
      record = True
    if record:
      if ('recall at' in line) or ('precision at' in line):
        continue
      else: 
        print (line)



if len(sys.argv)<1: #### run script
	print("Usage: \n")
	sys.exit(1)
else:
	submitJobs ( sys.argv[1] )



