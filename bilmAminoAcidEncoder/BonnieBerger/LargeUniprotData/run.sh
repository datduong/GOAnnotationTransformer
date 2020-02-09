


#### add in is-a nodes
. /u/local/Modules/default/init/modules.sh
module load python/3.7.2
cd /u/scratch/d/datduong/GOAnnotationTransformer/bilmAminoAcidEncoder/BonnieBerger/LargeUniprotData
python3 KeepIsA.py

#### count labels

#### filter label

#### add motifs data
. /u/local/Modules/default/init/modules.sh
module load python/3.7.2
cd /u/scratch/d/datduong/GOAnnotationTransformer/bilmAminoAcidEncoder/BonnieBerger/LargeUniprotData
python3 GetProtDomain.py

#### format data into form to train model
. /u/local/Modules/default/init/modules.sh
module load python/3.7.2
cd /u/scratch/d/datduong/GOAnnotationTransformer/bilmAminoAcidEncoder/BonnieBerger/LargeUniprotData
python3 FormatData2Train.py > track_rare_count_in_data.txt

#### split train/dev/test
. /u/local/Modules/default/init/modules.sh
module load python/3.7.2
cd /u/scratch/d/datduong/GOAnnotationTransformer/bilmAminoAcidEncoder/BonnieBerger/LargeUniprotData
python3 SplitTrainDevTest.py

#### get motifs seen only in train set. need both name+count


#### go through and see what labels will be used. 
