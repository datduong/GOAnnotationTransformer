

#! create test data with motif 

import re,sys,os,pickle
import pandas as pd 
import numpy as np 

os.chdir('/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000/bonnie+motif')

motif_in_train = pickle.load(open("/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000/bonnie+motif/train_mf_prot_annot_type_count.pickle","rb"))
motif_in_train_dict = {}
for key,val in motif_in_train.items():
    key = key.split()
    motif_in_train_dict[ " ".join(key[1::]) ] = key[0]


#
name_map = pickle.load(open('/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000/domain_name_map.pickle',"rb"))

# make a dict {prot:[domain1,domain2]}
motif_by_prosite = {}
fin = open("/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000/test-mf-motif-rename-pass2.tsv","r")
for line in fin:
    line_out = ""
    line = line.split("\t")
    for motif in line[1::]:
        motif = motif.strip().split(';')
        # MOTIF akap cam-binding 607-627;MOTIF akap cam-binding 756-776;
        #! skip 'same by similarity'
        if ('(-1)' in motif[0]) or ('(0)' in motif[0]):
            continue #! skip
        if motif[0] in name_map: 
            motif[0] = name_map[motif[0]]
        if motif[0] in motif_in_train_dict: 
            motif[0] = motif_in_train_dict[motif[0]] + " " + motif[0] # add in MOTIF, or DOMAIN
        line_out = line_out + " ".join(motif[0:2]) + ";"
    #
    line_out = re.sub(r';$',"",line_out)
    if len(line_out)==0: 
        line_out = 'none'
    motif_by_prosite[line[0]] = line_out


#
fin.close()


#! read in original file formated for our method, and then append motif at the end. 
fin = open ('/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000/deepgoplus.cafa3.test-bonnie-mf.tsv','r')
fout = open ('/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000/bonnie+motif/test-mf.tsv','w')
for line in fin: 
    protein = line.strip().split('\t')[0]
    if protein in motif_by_prosite: 
        line_out = line.strip() + "\t" + motif_by_prosite[protein] + "\n"
    else: 
        line_out = line.strip() + "\tnone\n" 
    fout.write(line_out)

#
fout.close()
fin.close() 

