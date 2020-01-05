
main_dir='/local/datdb/deepgo/data/BertNotFtAARawSeqGO/'
method='/fold_1/2embPpiAnnotE256H1L12I512Set0/YesPpi100YesTypeScaleFreezeBert12Ep10e10Drop0.1/'
code_dir='/local/datdb/BertGOAnnotation/AnalyzeGoVec'
cd $code_dir
python3 AnalyzeGoTypeAccuracy.py $main_dir $method > $main_dir/zeroshotProt100YesType.txt

