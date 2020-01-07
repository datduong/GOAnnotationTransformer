#!/bin/bash

. /u/local/Modules/default/init/modules.sh
module load python/3.7.2

# conda activate tensorflow_gpuenv
server='/u/scratch/d/datduong'
# data_dir=$server/'deepgo/data/DataToFinetuneBertTokenPredict/FinetunePhaseData'
mkdir $server/'deepgo/data/BertNotFtAARawSeqGO'

pretrained_label_path='/u/scratch/d/datduong/deepgo/data/cosine.AveWordClsSep768.Linear768.Layer12/label_vector.pickle'

choice='2embPpiAnnotE768H1L12I768PreLab' # Lr5e-5
block_size=1792 # mf and cc 1792 but bp has more term  2048
save_every=7000 # 9500 10000

for ontology in 'mf' 'cc' ; do # 'mf' 'cc'
  # last_save=$server/'deepgo/data/BertNotFtAARawSeqGO/'$ontology/'fold_1'/$choice
  output_dir=$server/'deepgo/data/BertNotFtAARawSeqGO/'$ontology/'fold_1'/$choice
  mkdir $output_dir

  # bert_vocab=$server/'BERTPretrainedModel/cased_L-12_H-768_A-12Kmer2016/vocab+3kmer+GO.txt'
  bert_vocab=$output_dir/'vocabAA.txt'
  # config_name=$server/'BERTPretrainedModel/cased_L-12_H-768_A-12Kmer2016/config.json'
  config_name=$output_dir/config.json
  model_name_or_path=$output_dir

  aa_type_file='/u/scratch/d/datduong/deepgo/data/train/fold_1/train_'$ontology'_prot_annot_count.pickle'

  train_masklm_data='/u/scratch/d/datduong/deepgo/data/train/fold_1/TokenClassify/TwoEmb/train-'$ontology'-prot-annot.tsv' ## okay to call it as long as it has ppi
  eval_data_file='/u/scratch/d/datduong/deepgo/data/train/fold_1/TokenClassify/TwoEmb/dev-'$ontology'-prot-annot.tsv'
  label_2test='/u/scratch/d/datduong/deepgo/data/train/deepgo.'$ontology'.csv'

  cd $server/BertGOAnnotation/finetune/

  python3 -u RunTokenClassifyLoadDataOnly.py --block_size $block_size --bert_vocab $bert_vocab --train_data_file $train_masklm_data --output_dir $output_dir --eval_data_file $eval_data_file --label_2test $label_2test --aa_type_emb --aa_type_file $aa_type_file > $output_dir/load_data_log.txt

  # eval_data_file='/u/scratch/d/datduong/deepgo/data/train/fold_1/TokenClassify/TwoEmb/test-'$ontology'-prot-annot.tsv'

  # python3 -u RunTokenClassifyLoadDataOnly.py --block_size $block_size --bert_vocab $bert_vocab --train_data_file $train_masklm_data --output_dir $output_dir --eval_data_file $eval_data_file --label_2test $label_2test --aa_type_emb --aa_type_file $aa_type_file > $output_dir/load_data_log.txt

done


