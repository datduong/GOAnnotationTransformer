
import sys,re,os,pickle
import numpy as np
import pandas as pd

#### merge blast and transformer. or some other method

import util as util

def submitJobs(test_file1,prediction1,test_file2,prediction2,header1,header2,save_file):

  if header1 == 'none':
    header1=None

  if header2 == 'none':
    header2=None

  if save_file == 'none':
    save_file=None

  util.Ensemble(test_file1,prediction1,test_file2,prediction2,header1,header2,save_file)



if len(sys.argv)<1: #### run script
	print("Usage: \n")
	sys.exit(1)
else:
	submitJobs ( sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7] )



