
cd /u/scratch/d/datduong/GOAnnotationTransformer/EvaluateLabelType/FormatToDeepgoEval

where=/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000

python3 MakeInputPandaPickle.py $where/deepgoplus.cafa3.test-bonnie-mf.tsv $where/deepgoplus.cafa3.test-bonnie-mf.pickle

####

cd $SCRATCH/deepgoplus
where=/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000
model=NoPpiYesAaTypeLabelBertAveL12Epo1000bz6
onto=mf
python3 evaluate_deepgoplus.py -tsdf $where/$model/$onto/test_panda_deepgo_format.pickle -o $onto > $onto.Transformer+Motif.NoAddAnc.txt


#### run editted code on the original prediction.plk
cd $SCRATCH/deepgoplus
where=/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000
model=NoPpiYesAaTypeLabelBertAveL12Epo1000bz6
onto=mf
python3 evaluate_deepgoplus.py -o $onto > $onto.OriginalDeepgoCnn.NoAddAnc.txt

