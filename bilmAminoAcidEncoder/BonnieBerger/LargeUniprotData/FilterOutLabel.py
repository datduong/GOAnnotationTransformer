


import sys,re,os,pickle
import numpy as np
import pandas as pd

## take prot already have vec and override

os.chdir("/u/scratch/d/datduong/UniprotJan2020/AddIsA")

FILTER = 7
## create an array in the exact order as file
# for data_type in ['test','train','dev']:

for onto in ['cc','bp','mf']:
  #### read in data

  label_to_test = {}
  fin = open(onto+"-label-rm20.tsv","r") # remove the 20% most common terms
  for line in fin:
    line = line.strip().split("\t")
    num = float(line[1])
    if num > FILTER: ##!! filter by occ ####
      label_to_test[ line[0] ] = num
  fin.close()

  ## read in a file which we already computed vector
  ## should read in file that already has motif

  # fout = open("uniprot-"+onto+"-isa-rm20-bonnie.tsv","w") ## has name seq go prot_vec domain
  # fout.write('Entry\tGene ontology IDs\tSequence\tProt Emb\n')

  fout = open(onto+"-input-ge5.tsv","w")

  fin = open(onto+"-input.tsv","r") ## no header

  for index,line in enumerate(fin):

    # if index==0:
    #   continue ## skip header

    line = line.strip().split("\t")

    # P27335  M Q S R P A Q E S G S A S E T P A R G R P T P S D A P R D E P T N Y N N N A E S L L E Q R L T R L I E K L N A E K H N S N L R N V A F E I G R P S L E P T S A M R R N P A N P Y G R F S I D E L F K M K V G V V S N N M A T T E Q M A K I A S D I A G L G V P T E H V A S V I L Q M V I M C A C V S S S A F L D P E G S I E F E N G A V P V D S I A A I M K K H A G L R K V C R L Y A P I V W N S M L V R N Q P P A D W Q A M G F Q Y N T R F A A F D T F D Y V T N Q A A I Q P V E G I I R R P T S A E V I A H N A H K Q L A L D R S N R N E R L G S L E T E Y T G G V Q G A E I V R N H R Y A N N G GO0019028 GO0019029 GO0044423 1.5463756 2.9005387 -1.1741593 0.9369323 2.7192078 1.1784599 1.9212605 -2.8906393 0.9127741 -1.3154982 -1.182249 -1.1324608 -0.7327065 1.7803066 -2.161174 -0.28980905 -0.81540585 -1.1852144 1.9869368 1.5117079 -1.0665057 0.54105425 -2.454377 -0.051022004 -2.773206 -2.5653675 0.31161124 -0.8099746 1.4180896 2.3519127 0.59064204 1.8964572 1.9294692 -0.050120942 -1.9244108 2.1167254 0.9742994 2.7331626 1.0219752 2.583562 2.7231383 0.81110334 -0.63380736 -2.9161737 -1.6338533 1.6548984 0.07407284 0.8532791 -1.7122338 1.8206428 3.1731336 1.2016163 -1.5412264 -2.022008 -0.16574445 0.818345 -0.86529833 0.19809315 -2.4262104 -1.3317852 2.1013107 2.1360078 -1.5562685 -1.9706916 2.7079108 2.022486 -1.0627599 0.48498315 -1.5838302 -1.4607569 -2.8126953 0.15295382 -1.9479179 -2.2898736 2.6450284 -1.4479096 0.76208353 1.3034091 -2.4091575 -0.66898894 0.687213 -0.3218196 -1.0825852 0.074827485 1.8712511 -1.0582259 2.0502405 1.0933572 -1.8598036 -0.79720473 -0.55315083 1.9050505 -0.049637605 1.3064883 1.1435237 -2.3594713 3.0859892 -1.1288643 -0.84344864 -1.1305472  none

    seq = re.sub(" ","",line[1]).strip()
    if len(seq) < 20:
      continue ## skip short sequences?
    
    name = line[0]
    annot = line[2] ##!! name seq label protvec motif

    annot = re.sub('GO','GO:',annot) ## put back the GO:
    annot = annot.strip().split()

    annot = sorted( [ a for a in annot if a in label_to_test ] )
    annot = " ".join(a for a in annot)
    annot = re.sub(":","",annot) ## remove the : in GO so that we can read it in as vocab if needed

    if len(annot) == 0:
      continue #### no empty label

    # want output: name, seq, label, vec, motif
    # seq = " ".join(line[2])
    seq = line[2] ## don't do this yet.
    fout.write( name + "\t" + line[1] + "\t" + annot + "\t" + line[3] + "\t" + line[4] + '\n' )

  #
  fin.close()
  fout.close()



#### how should we split the data ??

## constraint so that count>2 ... at the end, should we do zeroshot. if we do zeroshot, then random split works.
## if we want to eval rare labels, we need rare labels in the test data and also train

## very costly to train 2 models. so what should we do ??
## try to even split as many rare as possible.

for onto in [ 'mf', 'cc', 'bp' ] : 

  print ('\nonto {}'.format(onto))
  rare_label_array = {}
  label_to_test = {}
  fin = open(onto+"-label-rm20.tsv","r") # cc-label-rm10p.tsv
  for line in fin:
    line = re.sub("GO:","GO",line) ## because input doesn't use GO:
    line = line.strip().split("\t")
    num = float(line[1])
    if num > FILTER: ##!! filter by occ ####
      label_to_test[ line[0] ] = num
      if num <= 10:
        rare_label_array[line[0]] = num
  fin.close()
  print ('number of label {} is {}'.format(onto,len(label_to_test)))
  print ('number of rare {} is {}'.format(onto,len(rare_label_array)))

  #### need to design a good test set.
  # name, seq, label, vec, motif
  rare_label_array_in_test = {}
  in_test = {}
  in_train = {}
  fin = open(onto+'-input-ge5.tsv',"r")
  for index,line in enumerate (fin) :
    line = line.strip().split("\t")
    labels = line[2].strip().split()
    for l in labels:
      if l in rare_label_array:
        ## if rare label, we add 3 observations into test
        if l not in rare_label_array_in_test:
          rare_label_array_in_test[l] = 1 ## not seen this rare label, so add it immediately into testset
          ## what if 1 protein has 2 rare GO ... but why we see more in_train?
          in_test[line[0]] = 1
        else:
          #
          if rare_label_array_in_test[l] >= 3:
            ## we care about the rare labels
            in_train[line[0]] = 1 ## if we add 2 obs into test already, then add the rest into train
          else:
            rare_label_array_in_test[l] = rare_label_array_in_test[l] + 1
            in_test[line[0]] = 1

  #
  fin.close()
  #
  print ('len of test {}'.format(len(in_test)))
  print ('len of train {}'.format(len(in_train)))
  pickle.dump(in_test, open(onto+'_in_test_name.pickle','wb'))
  pickle.dump(in_train, open(onto+'_in_train_name.pickle','wb'))


print (rare_label_array_in_test) # sanity

