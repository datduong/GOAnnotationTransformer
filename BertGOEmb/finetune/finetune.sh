
#!/bin/bash

. /u/local/Modules/default/init/modules.sh
module load python/3.7.2

# conda activate tensorflow_gpuenv
server='/u/scratch/d/datduong'
# data_dir=$server/'goAndGeneAnnotationMar2017'
# server='/local/datdb'
data_dir=$server/'BERTPretrainedModel/cased_L-12_H-768_A-12GO+vocab2017'
output_dir=$data_dir/'BertFineTuneGOEmb'
mkdir $output_dir

cd $server/BertGOAnnotation/finetune/lm_finetuning

bert_vocab=$server/'BERTPretrainedModel/cased_L-12_H-768_A-12GO+vocab2017'

python3 pregenerate_training_data.py --bert_vocab $bert_vocab --train_corpus $data_dir/GO_def_branch.txt --bert_model bert-base-cased --output_dir $output_dir --epochs_to_generate 100 --max_seq_len 512 




## train LM and next-sentence

module load python/3.7.2

conda activate tensorflow_gpuenv
server='/local/datdb'
data_dir=$server/'goAndGeneAnnotationMar2017'
pregenerated_data=$data_dir/BertFineTuneGOEmb
# data_dir=$server/'BERTPretrainedModel/cased_L-12_H-768_A-12GO+vocab2017'
output_dir=$data_dir/'BertFineTuneGOEmb768'
mkdir $output_dir

cd $server/BertGOAnnotation/finetune/lm_finetuning

bert_vocab=$server/'BERTPretrainedModel/cased_L-12_H-768_A-12GO2017'
# bert_vocab=$server/'BERTPretrainedModel/cased_L-12_H-768_A-12GO+vocab2017'
config_name=$bert_vocab/'config768.json'

CUDA_VISIBLE_DEVICES=6 python3 -u finetune_on_pregenerated.py --bert_vocab $bert_vocab --pregenerated_data $pregenerated_data --bert_model bert-base-cased --output_dir $output_dir --epochs 50 --train_batch_size 6 --config_name $config_name --config_override 




