
import os,sys,re,pickle
import numpy as np

from scipy.spatial import distance



def PairWiseDistance (coordinate): ## create position-position matrix
  # coordinate is obs x dim
  pwdist = distance.pdist(coordinate) ## use distance.squareform(z) to get back squareform
  ## should we get summary stat ?
  ## may be we can do it over all the data
  return pwdist

def SummaryStatDistance (pwdist): # @pwdist should be some very long list
  ## should do some rounding, so we don't really model true continuous distance ??
  split = np.quantile(pwdist,q=np.arange(.1,1.05,.05)) ## 10 20 or 50 chunks ?? 
