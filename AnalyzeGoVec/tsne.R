library(Rtsne)
library('ggrepel', help, pos = 2, lib.loc = NULL)
library('ggplot2', help, pos = 2, lib.loc = NULL)

set.seed(1)

fin = read.csv("GOvecFromBert12.tsv",sep="\t",header=F,stringsAsFactors=F)
this_title = 'Bert12'

numcol = ncol(fin)
GOvec = as.matrix(fin[ , 2:(numcol-2) ])

tsne <- Rtsne(GOvec, dims = 2, perplexity=30, verbose=TRUE, max_iter = 500)
tsne_out = tsne$Y
tsne_out = data.frame( cbind ( fin[,1], tsne_out, fin[,(numcol-1):numcol] ) )
colnames(tsne_out) = c('name','dim1','dim2','ic','color1')

# windows() 

ggplot(tsne_out, aes(x = dim1, y = dim2, color=factor(color1), size=100/ic)) + 
  geom_point(alpha=.8) + 
  theme_bw() + 
  geom_text_repel(
    data = subset(tsne_out, color1 > 0),
    aes(label = name),
    size = 5,
    box.padding = unit(0.35, "lines"),
    point.padding = unit(0.3, "lines")
  ) + 
  ggtitle (this_title) + 
  theme(
    plot.title = element_text(size=20, face="bold") )

