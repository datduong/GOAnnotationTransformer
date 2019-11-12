library(Rtsne)
library('ggrepel', help, pos = 2, lib.loc = NULL)
library('ggplot2', help, pos = 2, lib.loc = NULL)

set.seed(1)

setwd('C:/Users/dat/Documents/BertNotFtAARawSeqGO/bp/fold_1/2embPpiAnnotE256H1L12I512Set0/YesPpiYesTypeScaleFreezeBert12Ep10e10Drop0.1/')


# GOvecFromModel
# fin = read.csv("GOvecFromBert12.tsv",sep="\t",header=F,stringsAsFactors=F)
# this_title = 'Bert12'

fin = read.csv("GOvecFromModel.tsv",sep="\t",header=F,stringsAsFactors=F)
this_title = 'GOvecFromModel'

fin2 = read.csv("GOvecFromModelHiddenLayer12test.tsv",sep="\t",header=F,stringsAsFactors=F)
this_title = 'GOvecFromModelHiddenLayer12 initbert'
numcol = ncol(fin)
fin = cbind(fin2, fin[,(numcol-1):numcol] ) ## append ic and color


numcol = ncol(fin)
GOvec = as.matrix(fin[ , 2:(numcol-2) ])

tsne <- Rtsne(GOvec, dims = 2, perplexity=30, verbose=TRUE, max_iter = 500)
tsne_out = tsne$Y
tsne_out = data.frame( cbind ( fin[,1], tsne_out, fin[,(numcol-1):numcol] ) )
colnames(tsne_out) = c('name','dim1','dim2','ic','color1')

windows() 

ggplot(tsne_out, aes(x = dim1, y = dim2, color=factor(color1), size=100*ic)) + 
  geom_point(alpha=.8, color='gray40' ) + 
  theme_linedraw() + theme_light() + 
  geom_point(alpha=.8, data=subset(tsne_out, color1 > 0) , aes(x = dim1, y = dim2, color=factor(color1), size=100*ic ) ) + 
  geom_text_repel(
    data = subset(tsne_out, color1 > 0),
    aes(label = name),
    size = 4,
    box.padding = unit(0.35, "lines"),
    point.padding = unit(0.3, "lines")
  ) + 
  # geom_text_repel(
  #   data = subset(tsne_out, color1 == 0),
  #   aes(label = name),
  #   size = 2,
  #   box.padding = unit(0.35, "lines"),
  #   point.padding = unit(0.3, "lines")
  # ) + 
  ggtitle (this_title) + 
  # labs(size = "100IC") +
  # theme(legend.position="left",plot.title = element_text(hjust = 0.5)) +
  guides(colour=FALSE, size=FALSE) #+ 
  # guides(size = guide_legend(override.aes = list(size=5))) 


