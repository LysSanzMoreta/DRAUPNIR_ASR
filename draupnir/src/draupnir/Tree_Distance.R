#!/usr/bin/Rscript
library(TreeDist)
library(ape)
trees <- commandArgs(trailingOnly = TRUE)
# tree1 <- ape::read.tree('/home/lys/Dropbox/PhD/DRAUPNIR/Mixed_info_Folder/benchmark_randall.mafft.treefile')
# tree2 <- ape::read.tree("/home/lys/Dropbox/PhD/DRAUPNIR/AncestralResurrectionStandardDataset/RandallBenchmarkTree.tree")

tree1 <- ape::read.tree(trees[1])
tree2 <- ape::read.tree(trees[2])
distance <- TreeDistance(tree1, tree2)
distance2 <- ClusteringInfoDistance(
  tree1,
  tree2 = NULL,
  normalize = 1,
  reportMatching = FALSE
)
cat(distance)

