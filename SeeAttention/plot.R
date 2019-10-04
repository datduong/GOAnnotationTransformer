
library('dendextend', help, pos = 2, lib.loc = NULL)
library(RColorBrewer)
graphics.off() 
coul <- colorRampPalette(brewer.pal(8, "PiYG"))(25)

get_cluster = function(df,cut_k=2) {
  dendrogram = hclust(dist(df))
  dendrogram_group = cutree(dendrogram,k=cut_k) ## get group 
  dendrogram_group = sort(dendrogram_group)
  dendrogram_name = names(dendrogram_group)
  return (list(dendrogram,dendrogram_group,dendrogram_name))
}

get_top_contributor = function (df,row){
  z = df[row,]
  return ( rownames(df) [ which ( z > quantile ( z, .9) ) ] ) 
}

# setwd('C:/Users/dat/Documents/BertNotFtAARawSeqGO/fold_1mf_relu/')

setwd('/u/scratch/d/datduong/deepgo/data/BertNotFtAARawSeqGO/fold_1mf_relu/')

# for (num in 0:5){
#   # num=5
#   df = read.csv ( paste0('GO2GO_attention_head',num,'.csv'), header=T )
#   # df = read.csv ( paste0('GO2GO_attention_ave_head.csv'), header=T )  
#   df = as.matrix (df)
#   rownames(df) = paste0('',colnames(df))
#   # df = df[1:50,1:50] ## subset just to see 
#   pdf (paste0('Head',num,'.pdf'),height=7,width=8) # first50
#   heatmap(df,scale="none",col = coul,Colv=NA,main=paste('Head',num))
#   dev.off() 
#   # p1 = heatmap(df,scale="none",col = coul,Colv=NA,main=paste('ave',num))
# }


# rownames(df)[10]
# get_top_contributor(df,10)

# graphics.off() 


# cooccur = read.csv ( 'C:/Users/dat/Documents/GO2GO_mf_count.csv', header=T )  
cooccur = read.csv ( '/u/scratch/d/datduong/deepgo/data/train/fold_1/TokenClassify/GO2GO_mf_count.csv', header=T )  

cooccur = as.matrix (cooccur)
rownames(cooccur) = colnames(cooccur)
cooccur = cooccur / colSums(cooccur) ## R divide down the row
cooccur[is.nan(cooccur)] = 0 
# cooccur = cooccur[1:50,1:50]
# pdf (paste0('coocurrFirst50.pdf'),height=7,width=8)
# heatmap(cooccur,scale="none",col = coul,Colv=NA,main='Co-occurrence')
# dev.off() 
cooccur_group = get_cluster(cooccur,4)


for (num in c(0:5)) {
df = read.csv ( paste0('GO2GO_attention_head',num,'.csv'), header=T )
df = as.matrix (df)
rownames(df) = paste0('',colnames(df))
# df = df[1:50,1:50] ## subset just to see 
df_group = get_cluster(df,4)
df_group[[2]]
df_group[[3]]

print (dendlist(as.dendrogram(df_group[[1]]), as.dendrogram(cooccur_group[[1]])) %>% untangle(method = "step1side") %>% entanglement() )

# ent = entanglement (as.dendrogram(df_group[[1]]), as.dendrogram(cooccur_group[[1]]))
# print (ent)

# print (cor_cophenetic(as.dendrogram(df_group[[1]]), as.dendrogram(cooccur_group[[1]])))

}
# tanglegram(dl)


relu 
[1] 0.6135558
[1] 0.6103813
[1] 0.6778977
[1] 0.7205237
[1] 0.6783651
[1] 0.7430645

[1] 0.3535921
[1] 0.3005707
[1] 0.2775146
[1] 0.3023639
[1] 0.2899833
[1] 0.2807402

[1] 0.296925 correlation 
[1] 0.2987238
[1] 0.2409891
[1] 0.3854294
[1] 0.3730905
[1] 0.5327675


gelu 
[1] 0.3525503
[1] 0.4241359
[1] 0.7189362
[1] 0.6376364
[1] 0.75702
[1] 0.7171842

[1] 0.2930331
[1] 0.3140639
[1] 0.2800639
[1] 0.4119118
[1] 0.3083879
[1] 0.3585995

[1] 0.4988198 correlation
[1] 0.2338982
[1] 0.2963654
[1] 0.02816023
[1] 0.5428462
[1] 0.01679081


# Get lower triangle of the correlation matrix
get_lower_tri<-function(cormat){
  cormat[upper.tri(cormat)] <- NA
  return(cormat)
}
# Get upper triangle of the correlation matrix
get_upper_tri <- function(cormat){
  cormat[lower.tri(cormat)]<- NA
  return(cormat)
}

library('ggplot2')
library('reshape2')
make_heatmap_ggplot = function (df) {
  melted_cormat <- melt(df)
  ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) + geom_tile() + theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

windows()
make_heatmap_ggplot(df)
windows()
make_heatmap_ggplot(cooccur)
