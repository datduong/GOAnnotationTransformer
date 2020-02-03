
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


sys.path.append('/local/datdb/ProteinEmbMethodGithub/protein-sequence-embedding-iclr2019/')
from src.alphabets import Uniprot21
import src.scop as scop
from src.utils import pack_sequences, unpack_sequences
from src.utils import PairedDataset, AllPairsDataset, collate_paired_sequences
from src.utils import MultinomialResample
import src.models.embedding
import src.models.comparison


## model how to handel Kmer.
## there are several ways. we can do CNN or ELMo style.

class KmerBase (nn.Module):
  def __init__(self, KmerEncoder, args, **kwargs):
    super(KmerBase, self).__init__()
    ## @KmerEncoder is something like CNN or some known encoder that take sequence --> vector , or sequence --> matrix ? 
    self.KmerEncoder = KmerEncoder 
    


