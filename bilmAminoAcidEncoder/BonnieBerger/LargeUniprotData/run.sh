

. /u/local/Modules/default/init/modules.sh
module load python/3.7.2

#### add in is-a nodes
cd /u/scratch/d/datduong/GOAnnotationTransformer/bilmAminoAcidEncoder/BonnieBerger/LargeUniprotData
python3 KeepIsA.py 

#### count labels 

