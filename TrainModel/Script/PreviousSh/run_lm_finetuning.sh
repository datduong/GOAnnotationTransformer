
## using the newer code ... hopefully the fp16 works


## LM tune the data
# conda activate tensorflow_gpuenv
server='/local/datdb'
data_dir=$server/'deepgo/data/DataToFinetuneBertTokenPredict/FinetunePhaseData'
last_save=$server/'deepgo/data/BertFineTuneAAseqGO/MaskLmOnlyContinue'
output_dir=$server/'deepgo/data/BertFineTuneAAseqGO/MaskLmOnlyContinue' # Continue
mkdir $output_dir

# bert_vocab=$server/'BERTPretrainedModel/cased_L-12_H-768_A-12Kmer2016/vocab+3kmer+GO.txt'
bert_vocab=$output_dir/'vocab.txt'
config_name=$server/'BERTPretrainedModel/cased_L-12_H-768_A-12Kmer2016/config.json'

train_masklm_data='/local/datdb/deepgo/data/DataToFinetuneBertTokenPredict/AAseq+GObranchMaskLM.txt'
eval_data_file='/local/datdb/deepgo/data/DataToFinetuneBertTokenPredict/seq_finetune_test.txt'

cd $server/BertGOAnnotationTrainModel/

## only run Mask Language ?? 
# CUDA_VISIBLE_DEVICES=1 python3 -u run_lm_finetuning.py --block_size 524 --mlm --bert_vocab $bert_vocab --train_data_file $train_masklm_data --output_dir $output_dir --num_train_epochs 20 --per_gpu_train_batch_size 8 --config_name $config_name --config_override --do_train --model_type bert --overwrite_output_dir --save_steps 20000

# eval_data_file
# CUDA_VISIBLE_DEVICES=2 python3 -u run_lm_finetuning.py --block_size 524 --mlm --bert_vocab $bert_vocab --train_data_file $train_masklm_data --eval_data_file $eval_data_file --output_dir $output_dir --num_train_epochs 20 --per_gpu_eval_batch_size 8 --config_name $config_name --do_eval --model_type bert --overwrite_output_dir --save_steps 20000 --model_name_or_path $last_save


## continue training 
CUDA_VISIBLE_DEVICES=7 python3 -u run_lm_finetuning.py --block_size 524 --mlm --bert_vocab $bert_vocab --train_data_file $train_masklm_data --output_dir $output_dir --num_train_epochs 50 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 8 --config_name $config_name --do_train --model_type bert --overwrite_output_dir --save_steps 30000 --logging_steps 10000 --model_name_or_path $last_save --evaluate_during_training --eval_data_file $eval_data_file



# CUDA_VISIBLE_DEVICES=7 python3 -u lm_finetuning/finetune_on_pregenerated.py --bert_vocab $bert_vocab --pregenerated_data $data_dir --bert_model bert-base-cased --output_dir $output_dir --epochs 10 --train_batch_size 8 --config_name $config_name --config_override --local_rank 2


# --fp16



