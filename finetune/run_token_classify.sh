
## using the newer code ... hopefully the fp16 works


## LM tune the data
# conda activate tensorflow_gpuenv
server='/local/datdb'
data_dir=$server/'deepgo/data/DataToFinetuneBertTokenPredict/FinetunePhaseData'
mkdir $server/'deepgo/data/BertNotFtAAseqGO'
last_save=$server/'deepgo/data/BertNotFtAAseqGO/fold_1'
output_dir=$server/'deepgo/data/BertNotFtAAseqGO/fold_1' # Continue
mkdir $output_dir

# bert_vocab=$server/'BERTPretrainedModel/cased_L-12_H-768_A-12Kmer2016/vocab+3kmer+GO.txt'
bert_vocab=$output_dir/'vocab.txt'
config_name=$server/'BERTPretrainedModel/cased_L-12_H-768_A-12Kmer2016/config.json'

train_masklm_data='/local/datdb/deepgo/data/train/fold_1/TokenClassify/train-mf.csv'
eval_masklm_data='/local/datdb/deepgo/data/train/fold_1/TokenClassify/dev-mf.csv'

label_2test='/local/datdb/deepgo/data/train/deepgo.mf.csv'

cd $server/BertGOAnnotation/finetune/


## continue training 
CUDA_VISIBLE_DEVICES=7 python3 -u run_token_classify.py --block_size 2048 --mlm --bert_vocab $bert_vocab --train_data_file $train_masklm_data --output_dir $output_dir --num_train_epochs 50 --per_gpu_train_batch_size 4 --per_gpu_eval_batch_size 4 --config_name $config_name --do_train --model_type bert --overwrite_output_dir --save_steps 300 --logging_steps 10 --evaluate_during_training --eval_data_file $eval_masklm_data --label_2test $label_2test --config_override # --no_cuda


