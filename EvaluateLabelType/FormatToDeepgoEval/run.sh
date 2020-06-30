
cd /u/scratch/d/datduong/GOAnnotationTransformer/EvaluateLabelType/FormatToDeepgoEval

where=/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000

python3 MakeInputPandaPickle.py $where/deepgoplus.cafa3.test-bonnie-mf.tsv $where/deepgoplus.cafa3.test-bonnie-mf.pickle

####

cd $SCRATCH/deepgoplus
where=/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000
model=NoPpiYesAaTypeLabelBertAveL12Epo1000bz6
onto=mf
python3 evaluate_deepgoplus.py -tsdf $where/$model/$onto/test_panda_deepgo_format.pickle -o $onto > $onto.Transformer+Motif.NoAddAnc.txt
# python3 evaluate_naive.py -trdf data-cafa/train_data.pkl -tsdf $where/$model/$onto/test_panda_deepgo_format.pickle -o $onto > $onto.Transformer+Motif.naive.txt


#### run editted code on the original prediction.plk
cd $SCRATCH/deepgoplus
where=/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000
model=NoPpiYesAaTypeLabelBertAveL12Epo1000bz6
onto=mf
# python3 evaluate_deepgoplus.py -o $onto > $onto.OriginalDeepgoCnn.NoAddAnc.DefaultFilterMf.txt
# python3 evaluate_naive.py -trdf data-cafa/train_data.pkl -tsdf data-cafa/predictions.pkl -o $onto > $onto.OriginalDeepgoCnn.Naive.DefaultFilterMf.txt


#### run editted code on the original prediction.plk, after keeping only mf
cd $SCRATCH/deepgoplus
where=/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000
onto=mf
python3 evaluate_deepgoplus.py -tsdf /u/scratch/d/datduong/deepgoplus/data-cafa/predictions_filter_by_mf.pkl -o $onto > $onto.OriginalDeepgoCnn.NoAddAnc.FilterMf.txt
# python3 evaluate_naive.py -trdf data-cafa/train_data.pkl -tsdf data-cafa/predictions_filter_by_mf.pkl -o $onto > $onto.OriginalDeepgoCnn.Naive.FilterMf.txt


#### run editted code on the original prediction.plk, after keeping only mf
cd $SCRATCH/deepgoplus
where=/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000
onto=mf
python3 evaluate_deepgoplus.py -tsdf /u/scratch/d/datduong/deepgoplus/data-cafa/predictions_filter_by_mf.pkl -o $onto > debug.txt
# python3 evaluate_naive.py -trdf data-cafa/train_data.pkl -tsdf data-cafa/predictions_filter_by_mf.pkl -o $onto > $onto.OriginalDeepgoCnn.Naive.FilterMf.txt


