

cd /local/datdb/GOAnnotationTransformer/bilmAminoAcidEncoder/BonnieBerger
CUDA_VISIBLE_DEVICES=5 python3 get_vector.py bp 

cd /local/datdb/GOAnnotationTransformer/bilmAminoAcidEncoder/BonnieBerger
CUDA_VISIBLE_DEVICES=6 python3 get_vector.py cc 

cd /local/datdb/GOAnnotationTransformer/bilmAminoAcidEncoder/BonnieBerger
CUDA_VISIBLE_DEVICES=7 python3 get_vector.py mf 



#### larger data 

cd /local/datdb/GOAnnotationTransformer/bilmAminoAcidEncoder/BonnieBerger
CUDA_VISIBLE_DEVICES=7 python3 get_vector_large_uniprot.py mf 

