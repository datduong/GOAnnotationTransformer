

#!/bin/bash

. /u/local/Modules/default/init/modules.sh
module load python/3.7.2

# conda activate tensorflow_gpuenv
server='/u/scratch/d/datduong'
data_dir=$server/'goAndGeneAnnotationMar2017'
output_dir=$data_dir/'BertFineTuneGOEmb'
mkdir $output_dir

cd $server/BertGOAnnotation/finetune/lm_finetuning

bert_vocab=$server/'BERTPretrainedModel/cased_L-12_H-768_A-12GO2017'

python3 pregenerate_training_data.py --bert_vocab $bert_vocab --train_corpus $data_dir/GO_branch_split_half.txt --bert_model bert-base-cased --output_dir $output_dir --epochs_to_generate 100 --max_seq_len 512



