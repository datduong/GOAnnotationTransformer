

library(taRifx)
library(xtable)
library(knitr)
library(kableExtra)

fin2 = read.table('GOannotationJan15.txt',sep='\t',header=T,stringsAsFactors=F)
fin2[,2:ncol(fin2)] = fin2[,2:ncol(fin2)] * 100
# fin2$Row = 1:nrow(fin2)
# colnames(fin2)=NULL
kable( fin2, "latex", longtable = F, booktabs = T, caption = "blank", row.names=TRUE ) %>% #row.names=c(1:nrow(fin2))
add_header_above(c(" "=2, "BP AUC" = 2, "MF AUC" = 2, "CC AUC"=2)) %>%
kable_styling(latex_options = c("hold_position","scale_down"), position = "center") 


pack_rows("Train and test DeepGOSeqFlat on DeepGO data", 1, 14, label_row_css = "background-color: #666; color: #fff;")
%>% 

add_indent( c(5:10)) %>%
add_indent( c(12:14)) %>%


