



#!/bin/bash
## make data for training on hoffman
. /u/local/Modules/default/init/modules.sh
module load python/3.7.2

# cd /u/scratch/d/datduong/BertGOAnnotation/Data/
# python3 get_prot_info.py > log.txt

cd /u/scratch/d/datduong/BertGOAnnotation/AnalyzeGoVec
python3 Combine2blast.py > CombineZeroshot2Blast.txt


