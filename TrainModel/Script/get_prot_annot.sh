

#!/bin/bash
## make data for training on hoffman
. /u/local/Modules/default/init/modules.sh
module load python/3.7.2

cd /u/scratch/d/datduong/GOAnnotationTransformer/Data/
python3 GetProtDomain.py > log.txt

