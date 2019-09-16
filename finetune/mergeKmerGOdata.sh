

# need to merge the Kmer data for finetune (who cares what sequence, as long as we capture the context of the sequence)
# and the GO branch (can be important, because of the year)

## let's say we stick with deepgo for now. 

/u/scratch/d/datduong/deepgo/data
/u/scratch/d/datduong/UniprotAllReviewGoAnnot/seq_finetune.txt
/u/scratch/d/datduong/deepgo/data/GO_branch_split_half.txt
mkdir /u/scratch/d/datduong/deepgo/data/DataToFinetuneBertTokenPredict
cd /u/scratch/d/datduong/deepgo/data/DataToFinetuneBertTokenPredict
cat /u/scratch/d/datduong/UniprotAllReviewGoAnnot/seq_finetune.txt /u/scratch/d/datduong/deepgo/data/GO_branch_split_half.txt > AAseq+GObranch.txt 


/local/datdb/BERTPretrainedModel/cased_L-12_H-768_A-12Kmer2016/ ## make sure no duplication 
awk '!seen[$0]++' vocab+3kmer+GO.txt > vocab.txt



## now we make pregenerated data. 


#!/bin/bash
## make data for training (about 30 mins) on hoffman
. /u/local/Modules/default/init/modules.sh
module load python/3.7.2
# conda activate tensorflow_gpuenv
server='/u/scratch/d/datduong'
data_dir=$server/'deepgo/data'
output_dir=$data_dir/'DataToFinetuneBertTokenPredict/FinetunePhaseData'
mkdir $output_dir
cd $server/BertGOAnnotation/finetune/lm_finetuning
bert_vocab=$server/'BERTPretrainedModel/cased_L-12_H-768_A-12Kmer2016/vocab+3kmer+GO.txt'
train_corpus=$data_dir/'DataToFinetuneBertTokenPredict/AAseq+GObranch.txt'
python3 pregenerate_training_data.py --bert_vocab $bert_vocab --train_corpus $train_corpus --bert_model bert-base-cased --output_dir $output_dir --epochs_to_generate 10 --max_seq_len 4096

