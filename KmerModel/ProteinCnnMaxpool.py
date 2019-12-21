

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string, re, sys, os, math
from tqdm import tqdm
import numpy as np
from collections import namedtuple
from tempfile import TemporaryDirectory

from copy import deepcopy
import logging
import json

from scipy.special import softmax
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.init import xavier_uniform_
from torch.utils.data import DataLoader, Dataset, RandomSampler


class CnnMaxpool (nn.Module):
  ## simple CNN maxpool layer on protein matrix (batch x len x dim)
  def __init__(self, args, **kwargs):
    super(CnnMaxpool, self).__init__()
    self.args = args
    self.CnnMaxpoolLayer = nn.Sequential ( nn.Conv1d( self.args.protein_dim, self.args.protein_dim, 3 ) )
    



