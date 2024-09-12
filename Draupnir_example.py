#!/usr/bin/env python3
"""
=======================
2022: Lys Sanz Moreta
Draupnir : Ancestral protein sequence reconstruction using a tree-structured Ornstein-Uhlenbeck variational autoencoder
=======================
"""
import warnings

import pyro
import torch
import argparse
import os,sys
from argparse import RawTextHelpFormatter
local_repository=True
if local_repository:
    sys.path.insert(1,"/home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src")
    import draupnir
else:#pip installed module
    import draupnir
from draupnir import str2bool,str2None
print("Loading draupnir module from {}".format(draupnir.__file__))

def main():
    """Executes the Draupnir pipeline:
    a) Generate a dataset in the appropiate form
    b) Run Draupnir model and generate results
    c) Produce additional results with the output from the -run- step"""

    draupnir.available_datasets(print_dict=True)

    #Highlight: Creates the dataset configuration and the dataset tensor
    build_config,settings_config, root_sequence_name = draupnir.create_draupnir_dataset(name=args.dataset_name,
                                                           use_custom=args.use_custom,
                                                           script_dir=script_dir,
                                                           args=args,
                                                           build=args.build_dataset, # True: construct the dataset, False: use the stored dataset
                                                           fasta_file=args.fasta_file,
                                                           tree_file=args.tree_file,
                                                           alignment_file=args.alignment_file)

    #Highlight: Creates image of the estimated tree colured by clades
    draw_tree = False
    if draw_tree:
        draupnir.draw_tree_simple(args.dataset_name,settings_config) #only colours shown
        draupnir.draw_tree_facets(args.dataset_name,settings_config) #coloured panels and names

    #Highlight: Runs draupnir
    draupnir.run(args.dataset_name,root_sequence_name,args,settings_config,build_config,script_dir)

    # Highlight: Calculate mutual information---> Only use AFTER at least the model has been run at least once with the variational guide
    run_mutual_information = False
    if run_mutual_information:
        draupnir.calculate_mutual_information(args,
                                              results_dir = "Mutual_info_dir",
                                              draupnir_folder_variational = ".../DRAUPNIR_ASR/PLOTS_Draupnir_simulations_src_sh3_1_2022_03_22_20h23min14s337405ms_5epochs_variational", #example
                                              draupnir_folder_MAP=".../DRAUPNIR_ASR/PLOTS_Draupnir_simulations_src_sh3_1_2022_03_22_20h19min54s739903ms_5epochs_delta_map",
                                              draupnir_folder_marginal=".../DRAUPNIR_ASR/PLOTS_Draupnir_simulations_src_sh3_1_2022_03_22_20h19min54s739903ms_5epochs_delta_map")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Draupnir args",formatter_class=RawTextHelpFormatter)

    parser.add_argument('-name','--dataset-name', type=str, nargs='?',
                        default="simulations_blactamase_1",
                        #default="ABO", #TODO: fix fasta and tree file to have same names
                        help='Dataset project name, look at draupnir.available_datasets()')
    parser.add_argument('-use-custom','--use-custom', type=str2bool, nargs='?',
                        default=False,
                        help='True: Use a custom dataset (create your own dataset). First create a folder with the same name as args.dataset_name where to store the necessary files here: draupnir/src/draupnir/data) '
                             'False: Use a default dataset (those shown in the paper) (they will automatically be downloaded at draupnir/src/draupnir/data)')
    parser.add_argument('-n', '--num-epochs', default=15000, type=int, help='number of training epochs')
    parser.add_argument('--alignment-file', type=str2None, nargs='?',
                        #default="/home/lys/Dropbox/PhD/DRAUPNIR_ASR/PF0096/PF0096.mafft",
                        default=None,
                        help='Path to alignment in fasta format (use with args.use_custom = True), with ALIGNED sequences. '
                             'PLEASE make sure that the fasta header names and the names in the tree are the same')
    parser.add_argument('--tree-file', type=str2None, nargs='?',
                        #default="/home/lys/Dropbox/PhD/DRAUPNIR_ASR/PF0096/PF0096.fasta.treefile",
                        default=None,
                        help='Path to newick tree (in format 1 from ete3) (use with args.use_custom = True).'
                             'PLEASE make sure that the fasta header names and the names in the tree are the same')
    parser.add_argument('--fasta-file', type=str2None, nargs='?',
                        default=None,
                        help='Path to fasta file (use with args.use_custom = True) with UNALIGNED sequences and NO tree (tree is inferred using IQtree). '
                             'PLEASE make sure that the fasta header names and the names in the tree are the same')
    parser.add_argument('--leaf-embeddings', type=str2None, nargs='?',
                        default=None,
                        help='Path to dataframe containing pre-computed embeddings for the leaf sequences (i.e ESMB embeddings)') #TODO: IMPLEMENT

    parser.add_argument('-build', '--build-dataset', default=False, type=str2bool,
                        help='True: Create and store the dataset from a given alignment file/tree or the unaligned sequences;'
                             'False: Use previously stored data files under folder with -dataset-name or at draupnir/src/draupnir/data. '
                             'Once you have built once the dataset you do not have to do it again (if everything went fine)'
                             'Further customization can be found under draupnir/src/draupnir/datasets.py')

    parser.add_argument('-bsize','--batch-size', default=1, type=str2None,nargs='?',help='set batch size.\n '
                                                                'Set to 1 to NOT batch (batch_size == 1 batch == entire dataset).\n '
                                                                'Set to None it automatically suggests a batch size and activates batching (it is slow, only use for very large datasets).\n '
                                                                'If batch_by_clade=True: 1 batch= 1 clade (size given by clades_dict).'
                                                                'Else set the batchsize to the given number')
    parser.add_argument('-aa-probs', default=21, type=int, help='21: 20 amino acids,1 gap probabilities \n '
                                                                ' 24: 23 amino acids, 1 gap')
    parser.add_argument('-n-samples','-n_samples', default=10, type=int, help='Number of samples (sequences sampled) per node')
    parser.add_argument('-use-blosum','--use-blosum', type=str2bool, nargs='?',default=True,help='Use blosum matrix embedding')
    parser.add_argument('-subs_matrix', default="BLOSUM62", type=str, help='blosum matrix to create blosum embeddings, choose one from ~/anaconda3/pkgs/biopython-1.76-py37h516909a_0/lib/python3.7/site-packages/Bio/Align/substitution_matrices/data')
    parser.add_argument('-embedding-dim', default=50, type=int, help='Blosum embedding dim')
    parser.add_argument('-use-cuda', type=str2bool, nargs='?', default=True,
                        help='True: Use GPU; False: Use CPU')
    parser.add_argument('-use-scheduler', type=str2bool, nargs='?', default=False, help='Use learning rate scheduler, to modify the learning rate during training. Only used with 1 large dataset in the paper')
    parser.add_argument('-test-frequency', default=100, type=int, help='sampling frequency (in epochs) during training, every <n> epochs, sample')
    parser.add_argument('-guide', '--select_guide', default="delta_map", type=str,help='choose a guide, available types: "delta_map" , "diagonal_normal" or "variational"')
    #Highlight: Sample from a pre-trained model
    parser.add_argument('-load-pretrained-path', type=str, nargs='?',default="/home/lys/Dropbox/PhD/DRAUPNIR_ASR/PLOTS_Draupnir_simulations_jj_1_2022_03_25_18h42min36s777683ms_10000epochs_delta_map",
                        help='Load pretrained Draupnir Checkpoints (folder path) to generate samples')
    parser.add_argument('-generate-samples', type=str2bool, nargs='?', default=False,help='Load fixed pretrained parameters (stored in Draupnir Checkpoints) and generate new samples')

    #Highlight: EXPERIMENTAL FEATURES
    parser.add_argument('-one-hot','--one-hot-encoded', type=str2bool, nargs='?',
                        default=False,
                        help='Build a one-hot-encoded dataset. Do not use, for now, Draupnir works with blosum-encoded and integers as amino acid representations, '
                             'so this is not needed for Draupnir inference at the moment')
    parser.add_argument('-bbc','--batch-by-clade', type=str2bool, nargs='?', default=False, help='Experimental. Use the leaves divided by their corresponding clades into batches. Do not use with leaf-testing')
    parser.add_argument('-pdb_folder', default=None, type=str,
                        help='Path to folder of PDB structures. The engine can read them and parse them into a dataset that the model can use.')
    parser.add_argument('-angles','--infer-angles', type=str2bool, nargs='?', default=False,help='Experimental. Additional Inference of angles. Use only with sequences associated PDB structures and their angles.')
    parser.add_argument('-kappa-addition', default=5, type=int, help='lower bound on the angles distribution parameters')
    parser.add_argument('-plate','--plating',  type=str2bool, nargs='?', default=False, help='Plating/Subsampling the mapping of the sequences (ONLY the sequences, not the latent space, '
                                                                                             'see example in DRAUPNIRModel_classic_plating under models.py).\n'
                                                                                             ' Remember to set plating/subsampling size, otherwise it is done automatically')
    parser.add_argument('-plate-size','--plating_size', type=str2None, nargs='?',default=None,help='Set plating/subsampling size:\n '
                                                                    'If set to None it automatically suggests a plate size, only if args.plating is TRUE!. Otherwise it remains as None and no plating occurs\n '
                                                                    'Else it sets the plate size to a given integer')
    parser.add_argument('-plate-idx-shuffle','--plate-unordered', type=str2bool, nargs='?',const=None, default=False,help='When subsampling/plating, shuffle (True) or not (False) the idx of the sequences which are given in tree level order')
    parser.add_argument('-position-embedding-dim', default=30, type=int, help='Tree position embedding dimension size')
    parser.add_argument('-max-indel-size', default=5, type=int, help='maximum insertion deletion size (not used)')
    parser.add_argument('-activate-elbo-convergence', default=False, type=bool, help='extends the running time until a convergence criteria in the elbo loss is met')
    parser.add_argument('-activate-entropy-convergence', default=False, type=bool, help='extends the running time until a convergence criteria in the sequence entropy is met')

    #TODO: Ray HPO? Would need to do for each protein family
    parser.add_argument('-d', '--config-dict', default=None,type=str, help="Used with parameter search")
    parser.add_argument('--parameter-search', type=str2bool, default=False, help="Activates a mini grid search for parameter search. TODO: Improve") #TODO: Change to something that makes more sense
    args = parser.parse_args()
    if args.use_cuda:
        #torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        torch.set_default_dtype(torch.float64)
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device= "cpu"
            raise warnings.warn("Cuda not found, falling back to cpu")
        torch.set_default_device(device)
    else:
        torch.set_default_tensor_type(torch.DoubleTensor)
        device = "cpu"
    args.__dict__["device"] = device
    #pyro.set_rng_seed(0) # torch is already running with different seeds
    #torch.manual_seed(0)
    pyro.enable_validation(False)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main()
