
## using the newer code ... hopefully the fp16 works


## LM tune the data
# conda activate tensorflow_gpuenv
server='/local/datdb'
data_dir=$server/'deepgo/data/DataToFinetuneBertTokenPredict/FinetunePhaseData'
mkdir $server/'deepgo/data/BertNotFtAARawSeqGO'

choice='AsIsPpiE768I768H6L8Drop0.2'
for ontology in 'mf' 'cc' ; do 
  last_save=$server/'deepgo/data/BertNotFtAARawSeqGO/'$ontology/'fold_1'/$choice
  output_dir=$server/'deepgo/data/BertNotFtAARawSeqGO/'$ontology/'fold_1'/$choice
  mkdir $output_dir

  # bert_vocab=$server/'BERTPretrainedModel/cased_L-12_H-768_A-12Kmer2016/vocab+3kmer+GO.txt'
  bert_vocab=$output_dir/'vocabAA+GO.txt'
  # config_name=$server/'BERTPretrainedModel/cased_L-12_H-768_A-12Kmer2016/config.json'
  config_name=$output_dir/config.json
  model_name_or_path=$output_dir

  train_data_file='/local/datdb/deepgo/data/train/fold_1/TokenClassify/train-'$ontology'-ppi.csv'
  eval_data_file='/local/datdb/deepgo/data/train/fold_1/TokenClassify/dev-'$ontology'-ppi.csv'
  label_2test='/local/datdb/deepgo/data/train/deepgo.'$ontology'.csv'

  cd $server/GOAnnotationTransformer/TrainModel/

  # 5040 batches train 2048
  ## continue training 
  CUDA_VISIBLE_DEVICES=5 python3 -u run_token_classify_ppi.py --block_size 1792 --mlm --bert_vocab $bert_vocab --train_data_file $train_data_file --output_dir $output_dir --num_train_epochs 50 --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 8 --config_name $config_name --do_train --model_type bert --overwrite_output_dir --save_steps 10000 --logging_steps 10000 --evaluate_during_training --eval_data_file $eval_data_file --label_2test $label_2test --config_override --fp16 > $output_dir/train_point.txt # --no_cuda


  ## testing phase 
  for test_data in 'dev' 'test'; do
    eval_data_file='/local/datdb/deepgo/data/train/fold_1/TokenClassify/'$test_data'-'$ontology'-ppi.csv'
    CUDA_VISIBLE_DEVICES=5 python3 -u run_token_classify_ppi.py --block_size 1792 --mlm --bert_vocab $bert_vocab --train_data_file $train_data_file --output_dir $output_dir --per_gpu_eval_batch_size 12 --config_name $config_name --do_eval --model_type bert --overwrite_output_dir --evaluate_during_training --eval_data_file $eval_data_file --label_2test $label_2test --config_override --eval_all_checkpoints --fp16 > $output_dir/'eval_'$test_data'_check_point.txt'
  done 

done 


