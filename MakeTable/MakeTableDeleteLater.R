

library(taRifx)
library(xtable)
library(knitr)
library(kableExtra)

# Case19OptionSnpInfo4eQtleGene
fin2 = read.table('eGeneFoundByHistone.txt',header=T,stringsAsFactors=F)
fin2 = fin2[ order(fin2$eGenes,decreasing=T), ]

kable( fin2, "latex", longtable = F, booktabs = T, caption = "blank", row.names=FALSE ) 

%>% #row.names=c(1:nrow(fin2))
add_header_above(c(" "=2, "BP"=3, "MF"=3, "CC"=3)) %>%
kable_styling(latex_options = c("hold_position","scale_down"), position = "center") %>%
pack_rows("BLAST Psi-BLAST", 1, 2, label_row_css = "background-color: #666; color: #fff;") %>% 
pack_rows("DeepGO", 3, 4, label_row_css = "background-color: #666; color: #fff;") %>% 
pack_rows("Transformer", 5, 8, label_row_css = "background-color: #666; color: #fff;")

