from Draupnir_datasets import *


available_datasets()

alignment_file = ""
tree_file = ""
fasta_file = "/home/lys/Dropbox/PhD/DRAUPNIR_Evolution/datasets/custom/PF0096/PF00096.fasta"
name = "blah"
create_dataset(name,use_custom=True,build=True,fasta_file=fasta_file,tree_file=None,alignment_file=None)


