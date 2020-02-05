


## generate data for fine tune LM model

#!/bin/bash

## make data for training (about 30 mins) on hoffman
. /u/local/Modules/default/init/modules.sh
module load python/3.7.2

conda activate tensorflow_gpuenv
server='/local/datdb'
data_dir=$server/'deepgo/data'
output_dir=$data_dir/'BertFineTuneAAseq'
mkdir $output_dir

cd $server/GOAnnotationTransformer/TrainModel/lm_finetuning

bert_vocab=$server/'BERTPretrainedModel/cased_L-12_H-768_A-12Kmer'

python3 pregenerate_training_data.py --bert_vocab $bert_vocab --train_corpus $data_dir/prot_go_finetune.txt --bert_model bert-base-cased --output_dir $output_dir --epochs_to_generate 10 --max_seq_len 512



## LM tune the data
conda activate tensorflow_gpuenv
server='/local/datdb'
data_dir=$server/'deepgo/data'
output_dir=$data_dir/'BertFineTuneAAseq'
mkdir $output_dir
bert_vocab=$server/'BERTPretrainedModel/cased_L-12_H-768_A-12Kmer'
bert_model='/local/datdb/BERTPretrainedModel/cased_L-12_H-768_A-12/'

cd $server/GOAnnotationTransformer/TrainModel/lm_finetuning
CUDA_VISIBLE_DEVICES=1 /local/datdb/anaconda3/envs/tensorflow_gpuenv/bin/python -u finetune_on_pregenerated.py --bert_vocab $bert_vocab --pregenerated_data $output_dir --bert_model $bert_model --output_dir $output_dir --epochs 10 --train_batch_size 12 --fp16



