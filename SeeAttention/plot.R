
df = read.csv ( 'GO2GO_attention.csv', header=T ) 

df = as.matrix (df)
rownames(df) = colnames(df)

get_top_contributor = function (df,row){
  z = df[row,]
  return ( rownames(df) [ which ( z > quantile ( z, .9) ) ] ) 
}

# rownames(df)[10]
# get_top_contributor(df,10)
# windows()
# heatmap(df, scale="none",Colv=NA)


cooccur = read.csv ( 'GO2GO_mf_count.csv', header=T ) 
cooccur = as.matrix (cooccur)
rownames(cooccur) = colnames(cooccur)
cooccur = cooccur / colSums(cooccur) ## R divide down the row
cooccur[is.nan(cooccur)] = 0 

# heatmap(cooccur, scale="none",Colv=NA)
# plot(hclust(dist(cooccur)))

# windows()
# plot(hclust(dist(df)))


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

melted_cormat <- melt(cooccur)
ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile()


windows() 
melted_cormat <- melt(df*10)
ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile()

