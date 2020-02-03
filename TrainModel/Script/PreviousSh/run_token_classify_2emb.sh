
## using the newer code ... hopefully the fp16 works


## LM tune the data
# conda activate tensorflow_gpuenv
server='/local/datdb'
data_dir=$server/'deepgo/data/DataToFinetuneBertTokenPredict/FinetunePhaseData'
mkdir $server/'deepgo/data/BertNotFtAARawSeqGO'

choice='2embPpiAnnotE256H1L12I512Set0/NoPpiEp10e10Drop0.1'

block_size=2048 #1792 2048
## cc input256hidden512 no ppi no label encoder checkpoint=42654

for ontology in 'bp' ; do # 'mf' 'cc'

  checkpoint=58200 ## bp 58200

  last_save=$server/'deepgo/data/BertNotFtAARawSeqGO/'$ontology/'fold_1/'$choice
  output_dir=$server/'deepgo/data/BertNotFtAARawSeqGO/'$ontology/'fold_1/'$choice
  mkdir $output_dir

  # bert_vocab=$server/'BERTPretrainedModel/cased_L-12_H-768_A-12Kmer2016/vocab+3kmer+GO.txt'
  bert_vocab=$output_dir/'vocabAA.txt'
  # config_name=$server/'BERTPretrainedModel/cased_L-12_H-768_A-12Kmer2016/config.json'
  config_name=$output_dir/config.json
  model_name_or_path=$output_dir

  train_masklm_data='/local/datdb/deepgo/data/train/fold_1/TokenClassify/TwoEmb/train-'$ontology'-aa.csv'
  eval_data_file='/local/datdb/deepgo/data/train/fold_1/TokenClassify/TwoEmb/dev-'$ontology'-aa.csv'
  label_2test='/local/datdb/deepgo/data/train/deepgo.'$ontology'.csv'

  cd $server/BertGOAnnotationTrainModel/

  # 5040 batches train
  ## continue training use @model_name_or_path and turn off @config_override
  # CUDA_VISIBLE_DEVICES=6 python3 -u run_token_classify_2emb.py --block_size $block_size --mlm --bert_vocab $bert_vocab --train_data_file $train_masklm_data --output_dir $output_dir --num_train_epochs 100 --per_gpu_train_batch_size 4 --per_gpu_eval_batch_size 6 --config_name $config_name --do_train --model_type bert --overwrite_output_dir --save_steps 5000 --logging_steps 5000 --evaluate_during_training --eval_data_file $eval_data_file --label_2test $label_2test --config_override --learning_rate 0.0001 --seed 2019 --fp16 > $output_dir/train_point.txt # --no_cuda

  # ## testing phase
  eval_data_file='/local/datdb/deepgo/data/train/fold_1/TokenClassify/TwoEmb/test-'$ontology'-aa.csv'
  CUDA_VISIBLE_DEVICES=5 python3 -u run_token_classify_2emb.py --block_size $block_size --mlm --bert_vocab $bert_vocab --train_data_file $train_masklm_data --output_dir $output_dir --num_train_epochs 50 --per_gpu_eval_batch_size 8 --config_name $config_name --do_eval --model_type bert --overwrite_output_dir --evaluate_during_training --eval_data_file $eval_data_file --label_2test $label_2test --config_override --eval_all_checkpoints --fp16 --checkpoint $checkpoint > $output_dir/eval_test_check_point.txt

  ## view weights ?? 

  # cd $server/GOAnnotationTransformer/SeeAttention/
  # eval_data_file='/local/datdb/deepgo/data/train/fold_1/TokenClassify/TwoEmb/train-'$ontology'-aa.csv'

  # model_name_or_path='/local/datdb/deepgo/data/BertNotFtAARawSeqGO/fold_1mf2emb'$choice'/checkpoint-80000'
  # model_name_or_path='/local/datdb/deepgo/data/BertNotFtAARawSeqGO/fold_1cc2emb'$choice'/checkpoint-105000' # 

  # CUDA_VISIBLE_DEVICES=6 python3 -u view_weight_2emb.py --block_size 1792 --mlm --bert_vocab $bert_vocab --train_data_file $train_masklm_data --output_dir $output_dir --per_gpu_eval_batch_size 6 --config_name $config_name --do_eval --model_type bert --overwrite_output_dir --evaluate_during_training --eval_data_file $eval_data_file --label_2test $label_2test --model_name_or_path $model_name_or_path > $output_dir/view_weights.txt

  # CUDA_VISIBLE_DEVICES=6 python3 -u view_weight_aa_2emb.py --block_size 1792 --mlm --bert_vocab $bert_vocab --train_data_file $train_masklm_data --output_dir $output_dir --per_gpu_eval_batch_size 6 --config_name $config_name --do_eval --model_type bert --overwrite_output_dir --evaluate_during_training --eval_data_file $eval_data_file --label_2test $label_2test --model_name_or_path $model_name_or_path > $output_dir/view_aa_weights.txt


done


