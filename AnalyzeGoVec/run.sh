
/usr/local/cuda-10.1/bin
export PATH="/usr/local/cuda-10.1/bin:$PATH"


main_dir='/local/datdb/deepgo/data/BertNotFtAARawSeqGO/'
method='/fold_1/2embPpiAnnotE256H1L12I512Set0/NoPpiNoTypeScaleFreezeBert12Ep10e10Drop0.1/'
code_dir='/local/datdb/BertGOAnnotation/AnalyzeGoVec'
cd $code_dir
python3 AnalyzeGoTypeAccuracy.py $main_dir $method > $main_dir/zeroshotNoProtNoType.txt



#### use blast to eval added term... not pure zeroshot
out_dir='/u/scratch/d/datduong/deepgo/dataExpandGoSet/train/fold_1/blastPsiblastResultEval100'
main_dir='/u/scratch/d/datduong/deepgo/data/BertNotFtAARawSeqGO/'
## COMMENT method doesn't really matter here.
method='/fold_1/2embPpiAnnotE256H1L12I512Set0/YesPpi100YesTypeScaleFreezeBert12Ep10e10Drop0.1/'
code_dir='/u/scratch/d/datduong/BertGOAnnotation/AnalyzeGoVec'
cd $code_dir
python3 AnalyzeGoTypeAccuracy.py $main_dir $method > $out_dir/ZeroshotBlastPsiblastResultEval100.txt
cd $out_dir

