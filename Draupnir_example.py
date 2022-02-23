#!/usr/bin/env python3
"""
2022: Lys Sanz Moreta
Draupnir : Ancestral protein sequence reconstruction using a tree-structured Ornstein-Uhlenbeck variational autoencoder
=======================
"""
import sys
import pyro
import torch
sys.path.append("./draupnir/src")
import draupnir
import argparse
import os
from draupnir import str2bool,str2None



def main():

    draupnir.available_datasets(print_dict=True)
    build_config,settings_config, root_sequence_name = draupnir.create_draupnir_dataset(args.dataset_name,
                                                           use_custom=args.use_custom,
                                                           script_dir=script_dir,
                                                           build=args.build_dataset, # True: construct the dataset, False: use the stored dataset
                                                           fasta_file=args.fasta_file,
                                                           tree_file=args.tree_file,
                                                           alignment_file=args.alignment_file)

    if args.parameter_search:
        draupnir.manual_random_search()
    else:
        #params_config = draupnir.config_build(args)
        #draupnir.draupnir_main(args.dataset_name,args,device,settings_config,build_config,script_dir)
        draupnir.run(args.dataset_name,root_sequence_name,args,device,settings_config,build_config,script_dir)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Draupnir args")
    parser.add_argument('-name','--dataset-name', type=str, nargs='?',
                        default="PF00400",
                        help='Dataset project name')
    parser.add_argument('-use-custom','--use-custom', type=str2bool, nargs='?',
                        default=False,
                        help='Use a custom dataset (found at datasets/custom) or a default dataset (those shown in the paper) (found at datasets/default)')
    parser.add_argument('--alignment-file', type=str, nargs='?',
                        default="/home/lys/Dropbox/PhD/DRAUPNIR_ASR/datasets/custom/PF0096/PF0096.fasta",
                        help='Path to alignment in fasta format (use with custom dataset)')
    parser.add_argument('--tree-file', type=str, nargs='?',
                        default="/home/lys/Dropbox/PhD/DRAUPNIR_ASR/datasets/custom/PF0096/PF0096.fasta.treefile",
                        help='Path to newick tree (format 1 from ete3) (use with custom dataset)')
    parser.add_argument('--fasta-file', type=str, nargs='?',
                        default="/home/lys/Dropbox/PhD/DRAUPNIR_ASR/datasets/custom/PF0096/PF00096.fasta",
                        help='Path to fasta file (use with custom dataset)')
    parser.add_argument('-n', '--num-epochs', default=2, type=int, help='number of training epochs')
    parser.add_argument('-build', '--build-dataset', default=False, type=str2bool,
                        help='True: Create and store the dataset from an alignment file/tree or just sequences;'
                             ' False: use stored data files under datasets/custom or datasets/default. '
                             'Further customization under Draupnir_Datasets.py')
    parser.add_argument('-bsize','--batch-size', default=1, type=str2None,nargs='?',help='set batch size. '
                                                                'If set to 1 to NOT batch (batch_size = 1 = 1 batch = 1 entire dataset). '
                                                                'If set to None it automatically suggests a batch size and activates batching (it is slow, only use for very large datasets). '
                                                                'If batch_by_clade=True: 1 batch= 1 clade (given by clades_dict).'
                                                                'Else set the batchsize to the given number')
    parser.add_argument('-guide', '--select_guide', default="delta_map", type=str,help='choose a guide, available "delta_map" or "diagonal_normal" or "variational"')
    parser.add_argument('-bbc','--batch-by-clade', type=str2bool, nargs='?',const=False, default=False, help='Use the leaves divided by their corresponding clades into batches. Do not use with leaf-testing')
    parser.add_argument('-angles','--infer-angles', type=str2bool, nargs='?',const=False, default=False,help='Additional Inference of angles')
    parser.add_argument('-plate','--plating',  type=str2bool, nargs='?', default=False, help='Plating/Subsampling the mapping of the sequences (ONLY, not the latent space). Remember to set plating size, otherwise it is done automatically')
    parser.add_argument('-plate-size','--plating_size', type=str2None, nargs='?',default=None,help='Set plating/subsampling size '
                                                                    'If set to None it automatically suggests a plate size, only if args.plating is TRUE!. Otherwise it remains as None and no plating occurs '
                                                                    'Else it sets the plate size to a given integer')
    parser.add_argument('-plate-idx-shuffle','--plate-unordered', type=str2bool, nargs='?',const=None, default=False,help='When subsampling/plating, shuffle (True) or not (False) the idx of the sequences which are given in tree level order')

    parser.add_argument('-aa-prob', default=21, type=int, help='20 amino acids,1 gap probabilities')
    parser.add_argument('-n-samples','-n_samples', default=10, type=int, help='Number of samples')
    parser.add_argument('-kappa-addition', default=5, type=int, help='lower bound on angles')
    parser.add_argument('-use-blosum','--use-blosum', type=str2bool, nargs='?',default=True,help='Use blosum matrix embedding')

    parser.add_argument('-generate-samples','--generate-samples', type=str2bool, nargs='?', default=False,help='Load fixed pretrained parameters (stored in Draupnir Checkpoints) and generate new samples')
    #parser.add_argument('-use-trained-logits','--use-trained-logits', type=str2bool, nargs='?', default=False,help='Load fixed pretrained logits (i.e train_info_dict.torch) and generate new samples')
    parser.add_argument('--load-pretrained-path', type=str, nargs='?',default="/home/lys/Dropbox/PhD/DRAUPNIR_ASR/PLOTS_GP_VAE_simulations_blactamase_1_2022_02_09_20h12min18s525670ms_1000epochs_delta_map",help='Load pretrained Draupnir Checkpoints (folder path) to generate samples')
    parser.add_argument('-subs_matrix', default="BLOSUM62", type=str, help='blosum matrix to create blosum embeddings, choose one from /home/lys/anaconda3/pkgs/biopython-1.76-py37h516909a_0/lib/python3.7/site-packages/Bio/Align/substitution_matrices/data')
    parser.add_argument('-embedding-dim', default=50, type=int, help='Blosum embedding dim')
    parser.add_argument('-position-embedding-dim', default=30, type=int, help='Tree position embedding dim')
    parser.add_argument('-max-indel-size', default=5, type=int, help='maximum insertion deletion size (not used)')
    parser.add_argument('-use-cuda', type=str2bool, nargs='?', default=True, help='Use GPU') #not working for encoder
    parser.add_argument('-use-scheduler', type=str2bool, nargs='?', default=False, help='Use learning rate scheduler, to modify the learning rate during training')
    parser.add_argument('-activate-elbo-convergence', default=False, type=bool, help='extends the running time until a convergence criteria in the elbo loss is met')
    parser.add_argument('-activate-entropy-convergence', default=False, type=bool, help='extends the running time until a convergence criteria in the sequence entropy is met')
    parser.add_argument('-test-frequency', default=100, type=int, help='sampling frequency (in epochs) during training')
    parser.add_argument('-d', '--config-dict', default=None,type=str, help="Used with parameter search")
    parser.add_argument('--parameter-search', type=str2bool, default=False, help="Activates a mini grid search for parameter search") #TODO: Change to something that makes more sense
    args = parser.parse_args()
    if args.use_cuda:
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch.set_default_tensor_type(torch.DoubleTensor)
        device = "cpu"
    pyro.set_rng_seed(0) #TODO: Different seeds---> not needed, torch is already running with different seeds
    #torch.manual_seed(0)
    pyro.enable_validation(False)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main()
