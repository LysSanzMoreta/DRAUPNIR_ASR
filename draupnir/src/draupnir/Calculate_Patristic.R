#!/usr/bin/Rscript
library(ape)
exit <- function() { invokeRestart("abort") }
print("suscessfully running R script")
inputs <- commandArgs(trailingOnly = TRUE)
tree<-read.tree(inputs[1])
print(tree)
#PatristicDistMatrix<-cophenetic.phylo(tree) #Only for patristic distance between tips
dist <- dist.nodes(tree)
nodes_list <- strsplit(tree$node.label," ")
#the library has an error, therefore we have to manually re-assign the root_name
last_internal_node_name <- nodes_list[2][[1]]
alpha_part <- gsub("[0-9.-]", "", last_internal_node_name)
numeric_part <- as.character(strtoi(gsub("[^0-9.-]", "", last_internal_node_name)) - 1)
root_name <- paste(alpha_part,numeric_part,sep="")
nodes_list <- lapply(nodes_list, function(x) if(identical(x, character(0))) root_name else x) #seems to work
#nodes_list <- paste(nodes_list,collapse=' ')
row.names(dist) <- c(tree$tip.label,nodes_list)
#row.names(dist) <- c(tree$tip.label, tree$node.label) #old, was skipping the root name
#colnames(dist) <- c(tree$tip.label, tree$node.label)
colnames(dist) <- c(tree$tip.label, nodes_list)

write.csv(dist,inputs[2],quote=FALSE)



