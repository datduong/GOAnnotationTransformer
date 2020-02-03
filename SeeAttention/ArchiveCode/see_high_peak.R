



# setwd('C:/Users/dat/Documents/BertNotFtAARawSeqGO/fold_1mf_relu/')

setwd('/u/scratch/d/datduong/deepgo/data/BertNotFtAARawSeqGO/fold_1mf/')

count_cooccur = function(name,cooccur){
  where2get = which (rownames(cooccur)==gsub(":","",name))
  return ( sum ( cooccur[where2get,]>0 )/ncol(cooccur) ) 
}

go = read.csv("/u/scratch/d/datduong/deepgo/data/deepgo.mf.csv",header=F,stringsAsFactors=F)
count_file = read.csv("/u/scratch/d/datduong/deepgo/data/train/fold_1/CountGoInTrain-mf.tsv",header=T,sep='\t')

cooccur = read.csv("/u/scratch/d/datduong/deepgo/data/train/fold_1/GO2GO_mf_count.csv",header=T)
cooccur = as.matrix (cooccur)
rownames(cooccur) = colnames(cooccur)

for (num in 0:5){
  print ('')
  print (paste('head',num, sep = " "))
  df = read.csv ( paste0('GO2GO_attention_head',num,'.csv'), header=T )
  df = as.matrix (df)
  rownames(df) = paste0('',colnames(df))
  out = apply(df,1,sort,decreasing=T,index.return=T)
  best_place = NULL
  for (j in 1:length(out)){
    best = out[[j]]$ix[1:10]
    best = go[best,]
    if (is.null(best_place)) { best_place = best }
    else{ best_place = rbind(best_place,best)}
  }
  most = names (sort ( table(best_place), decreasing=T )) [1:5] ## take some top 10 only over all row ?
  this = subset ( count_file,count_file$GO %in% most )
  # rownames(this)=NULL
  # print ( this[ order(this$count,decreasing=T), ] )
  print ( sort ( sapply ( most, count_cooccur, cooccur=cooccur ) ), decreasing=T )

}


[1] "head 0"
          GO count
2 GO:0004857   312
1 GO:0072509   148
[1] ""
[1] "head 1"
          GO count
1 GO:0005125   106
2 GO:0005179    99
[1] ""
[1] "head 2"
          GO count
1 GO:0005125   106
2 GO:0005179    99
[1] ""
[1] "head 3"
          GO count
2 GO:0004857   312 748 co-occurring terms
1 GO:0016538    50 
[1] ""
[1] "head 4"
          GO count
1 GO:0005125   106 
2 GO:0005179    99 
[1] ""
[1] "head 5"
          GO count
1 GO:0000166   946
2 GO:0005125   106