
library(RColorBrewer)

graphics.off() 

coul <- colorRampPalette(brewer.pal(8, "PiYG"))(15)

setwd('C:/Users/dat/Documents/BertNotFtAARawSeqGO/fold_1mf/')

# for (num in 0:5){
num=1
df = read.csv ( paste0('GO2GO_attention_head',num,'.csv'), header=T )
# df = read.csv ( paste0('GO2GO_attention_ave_head.csv'), header=T )  
df = as.matrix (df)
rownames(df) = paste0('',colnames(df))

df = df[1:50,1:50] ## subset just to see 

get_top_contributor = function (df,row){
  z = df[row,]
  return ( rownames(df) [ which ( z > quantile ( z, .9) ) ] ) 
}

# rownames(df)[10]
# get_top_contributor(df,10)

# graphics.off() 
windows()
# heatmap(df,scale="none",col = coul,Colv=NA,main=paste('head',num))
heatmap(df,scale="none",col = coul,Colv=NA,main=paste('ave',num))
# }


cooccur = read.csv ( 'C:/Users/dat/Documents/GO2GO_mf_count.csv', header=T ) 
cooccur = as.matrix (cooccur)
rownames(cooccur) = colnames(cooccur)
cooccur = cooccur / colSums(cooccur) ## R divide down the row
cooccur[is.nan(cooccur)] = 0 
cooccur = cooccur[1:50,1:50]
windows()
heatmap(cooccur,scale="none",col = coul,Colv=NA,main='occurr')



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
