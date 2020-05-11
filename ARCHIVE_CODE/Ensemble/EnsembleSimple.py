

import sys,re,os,pickle
import numpy as np
import pandas as pd

#### assume the output are in correct format

#### same row and col ordering

#### merge blast and transformer. or some other method

import util as util

def submitJobs(prediction1,prediction2,save_file):

  p1 = pickle.load(open(prediction1,'rb'))
  p2 = pickle.load(open(prediction2,'rb'))

  for choice in ['mean','max']:
    mix = util.CombineResult (p1['prediction'],p2['prediction'],option=choice)
    p1['prediction'] = mix ## just override
    pickle.dump( p1 , open( save_file + choice + '.pickle' , 'wb' ) )


if len(sys.argv)<1: #### run script
	print("Usage: \n")
	sys.exit(1)
else:
	submitJobs ( sys.argv[1], sys.argv[2], sys.argv[3] )




