
#!/bin/bash

. /u/local/Modules/default/init/modules.sh
module load python/3.7.2

# conda activate tensorflow_gpuenv
server='/u/scratch/d/datduong'
# data_dir=$server/'goAndGeneAnnotationMar2017'
# server='/local/datdb'
data_dir=$server/'BERTPretrainedModel/cased_L-12_H-768_A-12GO2017' # +vocab
output_dir=$data_dir/'BertFineTuneGOEmb'
mkdir $output_dir

cd $server/BertGOAnnotation/finetune/lm_finetuning

bert_vocab=$server/'BERTPretrainedModel/cased_L-12_H-768_A-12GO2017' # +vocab

python3 pregenerate_training_data.py --bert_vocab $bert_vocab --train_corpus $data_dir/GO_branch_split_half.txt --bert_model bert-base-cased --output_dir $output_dir --epochs_to_generate 100 --max_seq_len 128 


## train LM and next-sentence
# module load python/3.7.2
conda activate tensorflow_gpuenv
server='/local/datdb'
pregenerated_data='/local/datdb/BERTPretrainedModel/cased_L-12_H-768_A-12GO2017/BertFineTuneGOEmb'

data_dir=$server/'goAndGeneAnnotationMar2017'
output_dir=$data_dir/'BertFineTuneGOEmb768'
mkdir $output_dir

cd $server/BertGOAnnotation/finetune/lm_finetuning

bert_vocab=$server/'BERTPretrainedModel/cased_L-12_H-768_A-12GO2017'
# bert_vocab=$server/'BERTPretrainedModel/cased_L-12_H-768_A-12GO+vocab2017'
config_name=$bert_vocab/'config.json'

CUDA_VISIBLE_DEVICES=2 python3 -u finetune_on_pregenerated.py --bert_vocab $bert_vocab --pregenerated_data $pregenerated_data --bert_model bert-base-cased --output_dir $output_dir --epochs 100 --train_batch_size 24 --config_name $config_name --config_override 




## !!! use deepgo 2016 go.obo
## train LM and next-sentence

#!/bin/bash
# . /u/local/Modules/default/init/modules.sh
module load python/3.7.2

# conda activate tensorflow_gpuenv
server='/u/scratch/d/datduong'
data_dir=$server/'deepgo/data'
output_dir=$data_dir/'BertFineTuneGOEmb768'
mkdir $output_dir
cd $server/BertGOAnnotation/finetune/lm_finetuning
bert_vocab=$server/'BERTPretrainedModel/cased_L-12_H-768_A-12GO2016'
python3 pregenerate_training_data.py --bert_vocab $bert_vocab --train_corpus $data_dir/GO_branch_split_half.txt --bert_model bert-base-cased --output_dir $output_dir --epochs_to_generate 100 --max_seq_len 128 


## do masked language model + next sentence prediction
conda activate tensorflow_gpuenv
server='/local/datdb'
data_dir=$server/'deepgo/data'
pregenerated_data=$data_dir/BertFineTuneGOEmb768
bert_model=$data_dir/'BertFineTuneGOEmb768Result'

output_dir=$data_dir/'BertFineTuneGOEmb768ResultContinue'
mkdir $output_dir

cd $server/BertGOAnnotation/finetune/lm_finetuning

bert_vocab=$server/'BERTPretrainedModel/cased_L-12_H-768_A-12GO2016'
config_name=$bert_vocab/'config.json'

CUDA_VISIBLE_DEVICES=5 python3 -u finetune_on_pregenerated.py --bert_vocab $bert_vocab --pregenerated_data $pregenerated_data --output_dir $output_dir --epochs 50 --train_batch_size 16 --shift_epoch 50 --bert_model $bert_model

# when continue, need to shift the epoch because of pregenerated data
# do not override config if we want to continue training 
# bert-base-cased --config_name $config_name --config_override 


