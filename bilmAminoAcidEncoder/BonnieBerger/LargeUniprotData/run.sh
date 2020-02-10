


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
python3 GetProtDomainMf.py

#### format data into form to train model
#### also remove very long and very short sequences
. /u/local/Modules/default/init/modules.sh
module load python/3.7.2
cd /u/scratch/d/datduong/GOAnnotationTransformer/bilmAminoAcidEncoder/BonnieBerger/LargeUniprotData
python3 FormatData2Train.py > track_rare_count_in_data.txt
python3 SplitTrainDevTest.py
python3 GetMotifCountInTrain.py
python3 GetLabel2Train.py



#### split train/dev/test
. /u/local/Modules/default/init/modules.sh
module load python/3.7.2
cd /u/scratch/d/datduong/GOAnnotationTransformer/bilmAminoAcidEncoder/BonnieBerger/LargeUniprotData
python3 SplitTrainDevTest.py

#### get motifs seen only in train set. need both name+count
cd /u/scratch/d/datduong/GOAnnotationTransformer/bilmAminoAcidEncoder/BonnieBerger/LargeUniprotData
python3 GetMotifCountInTrain.py


#### go through and see what labels will be used.
cd /u/scratch/d/datduong/GOAnnotationTransformer/bilmAminoAcidEncoder/BonnieBerger/LargeUniprotData
python3 GetLabel2Train.py


#### replace nan with none .... don't really need this.


