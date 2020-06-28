import re,sys,pickle,os
import pandas as pd
import numpy as np
from copy import deepcopy

####

#! uniprot has prosite pattern, but doesn't say location, we can use prosite to get location of pattern
#! will have extra information, but we may have duplicated info, so we need to filter out??

####

def reorder_motif(motif1_array):
  # A;1-2;PS123 --> what??
  motif1_array = motif1_array.split(';')
  return [ motif1_array[1], motif1_array[0], motif1_array[2] ]


def check_overlap(range1,range2): #! @range1 is a string 123-456
  # check_overlap do union?
  # https://stackoverflow.com/a/6821298/7239347
  range1 = range1.split('-')
  x = [int(range1[0]),int(range1[1])]
  range2 = range2.split('-')
  y = [int(range2[0]),int(range2[1])]
  overlap = range(max(x[0], y[0]), min(x[-1], y[-1])+1)
  if len(overlap) == 0:
    return 0, None
  else:
    return 1, [ min(x[0],y[0]), max(x[1],y[1]) ] ## yes overlap


def check_overlap_do_union (motif_array):
  # @motif_array [LEUCINE_ZIPPER;293-314;PS00029,  NEUROTR_ION_CHANNEL;162-176;PS00236]
  motif_keep = []
  track_motif = deepcopy (motif_array)
  while len (track_motif) > 0:
    this_motif = track_motif[0].strip().split(';') ## take first one
    if len (track_motif) == 1:
      motif_keep.append(';'.join(this_motif)) #! because we split it out
      break #! exit while
    else:
      # there are 2.
      to_remove = [0] ## we remove the 1st one for sure
      # LEUCINE_ZIPPER;293-314;PS00029  NEUROTR_ION_CHANNEL;162-176;PS00236
      # TYR_PHOSPHO_SITE_1;454-461 TYR_PHOSPHO_SITE_2;454-461
      for i,motif_i in enumerate(track_motif):
        if i == 0: 
          continue ## skip first one
        motif_i = motif_i.strip().split(';')
        if this_motif[0] == motif_i[0]: #? same name
          yes_overlap, range_overlap = check_overlap(this_motif[1],motif_i[1])
          if yes_overlap == 1: #! join 2 range interval
            this_motif[1] = "-".join(str(s) for s in range_overlap)
            to_remove.append(i)
      #
      to_remove = list (set (to_remove))
      print (to_remove)
      for index in sorted(to_remove, reverse=True):
        del track_motif[index] # https://stackoverflow.com/questions/11303225/how-to-remove-multiple-indexes-from-a-list-at-the-same-time
      motif_keep.append (';'.join(this_motif))
  #
  return motif_keep

# create dict {prot: [m1 m2 m3]}
# merge into train data by prot-name
protein_prosite = {}
fin = open("/u/scratch/d/datduong/UniprotSeqTypeOct2019/train-mf-prosite.tsv","r")
fout = open("/u/scratch/d/datduong/UniprotSeqTypeOct2019/train-mf-prosite-union.tsv","w")
for line in fin:
  line = line.strip().split('\t') ## split name, motif1, motif2
  new_motif = check_overlap_do_union (line[1::])
  fout.write ( "\t".join( [ line[0] ] + new_motif ) + '\n' )


#
fin.close()
fout.close()

#! PROTEIN_KINASE_DOM dominates the PROTEIN_KINASE_ATP PROTEIN_KINASE_TYR
#! --> possibly just use PROTEIN_KINASE_DOM??

# O04905  ASN_GLYCOSYLATION;8-11;PS00001  ASN_GLYCOSYLATION;59-62;PS00001 CAMP_PHOSPHO_SITE;12-15;PS00004 PKC_PHOSPHO_SITE;10-12;PS00005  PKC_PHOSPHO_SITE;25-27;PS00005  PKC_PHOSPHO_SITE;78-80;PS00005  PKC_PHOSPHO_SITE;148-150;PS00005  CK2_PHOSPHO_SITE;44-47;PS00006  CK2_PHOSPHO_SITE;55-58;PS00006  CK2_PHOSPHO_SITE;113-116;PS00006  MYRISTYL;2-7;PS00008  MYRISTYL;21-26;PS00008  MYRISTYL;24-29;PS00008  MYRISTYL;28-33;PS00008 MYRISTYL;56-61;PS00008  AMIDATION;10-13;PS00009 ADENYLATE_KINASE;94-105;PS00113

# O04905  ASN_GLYCOSYLATION;8-11;PS00001  ASN_GLYCOSYLATION;59-62;PS00001 CAMP_PHOSPHO_SITE;12-15;PS00004 PKC_PHOSPHO_SITE;10-12;PS00005  PKC_PHOSPHO_SITE;25-27;PS00005  PKC_PHOSPHO_SITE;78-80;PS00005  PKC_PHOSPHO_SITE;148-150;PS00005  CK2_PHOSPHO_SITE;44-47;PS00006  CK2_PHOSPHO_SITE;55-58;PS00006  CK2_PHOSPHO_SITE;113-116;PS00006  MYRISTYL;2-7;PS00008  MYRISTYL;21-33;PS00008  MYRISTYL;56-61;PS00008  AMIDATION;10-13;PS00009 ADENYLATE_KINASE;94-105;PS00113


# G5ECJ6  ASN_GLYCOSYLATION;81-84;PS00001 ASN_GLYCOSYLATION;218-221;PS00001 ASN_GLYCOSYLATION;251-254;PS00001 ASN_GLYCOSYLATION;512-515;PS00001 PKC_PHOSPHO_SITE;161-163;PS00005  PKC_PHOSPHO_SITE;301-303;PS00005  PKC_PHOSPHO_SITE;312-314;PS00005  PKC_PHOSPHO_SITE;437-439;PS00005  PKC_PHOSPHO_SITE;456-458;PS00005  PKC_PHOSPHO_SITE;534-536;PS00005  CK2_PHOSPHO_SITE;123-126;PS00006  CK2_PHOSPHO_SITE;260-263;PS00006  CK2_PHOSPHO_SITE;279-282;PS00006  CK2_PHOSPHO_SITE;322-325;PS00006  CK2_PHOSPHO_SITE;445-448;PS00006  CK2_PHOSPHO_SITE;457-460;PS00006  CK2_PHOSPHO_SITE;518-521;PS00006  MYRISTYL;4-9;PS00008  MYRISTYL;98-103;PS00008 MYRISTYL;130-135;PS00008  MYRISTYL;205-210;PS00008  MYRISTYL;206-211;PS00008  MYRISTYL;257-262;PS00008  MYRISTYL;275-280;PS00008  MYRISTYL;300-305;PS00008  MYRISTYL;318-323;PS00008  PROTEIN_KINASE_ATP;289-310;PS00107  PROTEIN_KINASE_TYR;399-411;PS00109  SH2;151-241;PS50001 SH3;43-110;PS50002  PROTEIN_KINASE_DOM;283-535;PS50011

# G5ECJ6  ASN_GLYCOSYLATION;81-84;PS00001 ASN_GLYCOSYLATION;218-221;PS00001 ASN_GLYCOSYLATION;251-254;PS00001 ASN_GLYCOSYLATION;512-515;PS00001 PKC_PHOSPHO_SITE;161-163;PS00005  PKC_PHOSPHO_SITE;301-303;PS00005  PKC_PHOSPHO_SITE;312-314;PS00005  PKC_PHOSPHO_SITE;437-439;PS00005  PKC_PHOSPHO_SITE;456-458;PS00005  PKC_PHOSPHO_SITE;534-536;PS00005  CK2_PHOSPHO_SITE;123-126;PS00006  CK2_PHOSPHO_SITE;260-263;PS00006  CK2_PHOSPHO_SITE;279-282;PS00006  CK2_PHOSPHO_SITE;322-325;PS00006  CK2_PHOSPHO_SITE;445-448;PS00006  CK2_PHOSPHO_SITE;457-460;PS00006  CK2_PHOSPHO_SITE;518-521;PS00006  MYRISTYL;4-9;PS00008  MYRISTYL;98-103;PS00008 MYRISTYL;130-135;PS00008  MYRISTYL;205-211;PS00008  MYRISTYL;257-262;PS00008  MYRISTYL;275-280;PS00008  MYRISTYL;300-305;PS00008  MYRISTYL;318-323;PS00008  PROTEIN_KINASE_ATP;289-310;PS00107  PROTEIN_KINASE_TYR;399-411;PS00109  SH2;151-241;PS50001 SH3;43-110;PS50002  PROTEIN_KINASE_DOM;283-535;PS50011


