


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

