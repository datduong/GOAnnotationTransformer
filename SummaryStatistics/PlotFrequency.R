

library(e1071)

for (onto in c('mf','cc','bp')) {
  print (onto)
  df = read.table(paste0("CountGoInTrain-",onto,".tsv"),sep="\t",header=T)
  print (quantile(df$count,seq(0,1,by=.05)))
  print ( skewness (df$count) )
}
