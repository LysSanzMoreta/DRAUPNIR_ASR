#!/usr/bin/env python3
"""
2021: aleatoryscience
Lys Sanz Moreta
Draupnir : GP prior VAE for Ancestral Sequence Resurrection
=======================
"""
import argparse
import time
import warnings
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import re
import ast
import subprocess
import ntpath
import pandas as pd
from collections import defaultdict
from Bio import AlignIO
import torch
import pyro
from pyro.infer import SVI, config_enumerate,infer_discrete
from pyro.infer.autoguide import AutoMultivariateNormal, AutoDiagonalNormal,AutoDelta,AutoNormal
from pyro.infer import Trace_ELBO, JitTrace_ELBO,TraceMeanField_ELBO,JitTraceMeanField_ELBO,TraceEnum_ELBO
sys.path.append("./draupnir/src/draupnir")
import Draupnir_utils as DraupnirUtils
import Draupnir_OrdinaryVAEmodels as DraupnirModels
import Draupnir_plots as DraupnirPlots
import Draupnir_train as DraupnirTrain
import Draupnir_datasets as DraupnirDatasets
import Draupnir_models_utils as DraupnirModelsUtils
import Draupnir_load_utils as DraupnirLoadUtils
import datetime
import pickle
import json
# PRESETS
torch.set_printoptions(threshold=None)
np.set_printoptions(None)
torch.autograd.set_detect_anomaly(True)
now = datetime.datetime.now()

TrainLoad = namedtuple('TrainLoad', ['dataset_train', 'evolutionary_matrix_train', 'patristic_matrix_train','cladistic_matrix_train'])
TestLoad = namedtuple('TestLoad',
                      ['dataset_test', 'evolutionary_matrix_test', 'patristic_matrix_test','cladistic_matrix_test',"leaves_names_test",
                       "position_test", "internal_nodes_indexes"])
AdditionalLoad = namedtuple("AdditionalLoad",
                            ["patristic_matrix_full", "cladistic_matrix_full","children_array", "ancestor_info_numbers", "alignment_length",
                             "tree_levelorder_names", "clades_dict_leaves", "closest_leaves_dict","clades_dict_all","linked_nodes_dict","descendants_dict","aa_frequencies",
                             "correspondence_dict","special_nodes_dict","full_name"])
SettingsConfig = namedtuple("SettingsConfig",["one_hot_encoding", "model_design","aligned_seq","data_folder","full_name"])
ModelLoad = namedtuple("ModelLoad",["z_dim","align_seq_len","device","args","build_config","leaves_nodes","n_tree_levels","gru_hidden_dim","pretrained_params","aa_frequencies","blosum",
                                    "blosum_max","blosum_weighted","dataset_train_blosum","variable_score","internal_nodes","graph_coo","nodes_representations_array","dgl_graph","children_dict",
                                    "closest_leaves_dict","descendants_dict","clades_dict_all","leaves_testing","plate_unordered","one_hot_encoding"])
BuildConfig = namedtuple('BuildConfig',['alignment_file','use_ancestral','n_test','build_graph',"aa_probs","triTSNE","align_seq_len","leaves_testing","batch_size","plate_subsample_size","script_dir","no_testing"])

SamplingOutput = namedtuple("SamplingOutput",["aa_sequences","latent_space","logits","phis","psis","mean_phi","mean_psi","kappa_phi","kappa_psi"])

def load_data(name,settings_config,build_config,param_config,results_dir,script_dir,args):
    """
    Reads and prepares the stored dataset and other files (created by DraupnirUtils.create_draupnir_dataset()) to be split into train (leaves) and test (internal nodes) sets. Additionally
    reads and stores in namedtuples some other information related to how the tree is organized
    :param str name: dataset name
    :param namedtuple settings_config: namedtuple containing dataset information
    :param namedtuple build_config: namedtuple containing information on how to perform interence
    :param str results_dir: path to folder where all the results of the run will be stored
    :param str script_dir: path from which draupnir is being executed
    :param namedtuple args: customized configuration arguments

    :out namedtuple train_load: contains train related tensors. dataset_train has shape [n_seqs, max_len + 2, 30], where in the second dimension
                                0  = [seq_len, position in tree, distance to root,ancestor, ..., 0]
                                1  = Git vector (30 dim) if available
                                2: = [(1 integer + 0*29) or (one hot encoded amino acid sequence (21 slots)+0*9)]"
    :out namedtuple test_load:
    :out namedtuple additional_load
    :out namedtuple build_config
    """


    aligned = ["aligned" if settings_config.aligned_seq else "NOT_aligned"]
    one_hot = ["OneHotEncoded" if settings_config.one_hot_encoding else "Integers"]

    dataset = np.load("{}/{}/{}/{}_dataset_numpy_{}_{}.npy".format(script_dir, settings_config.data_folder, name, name, aligned[0],one_hot[0]), allow_pickle=True)
    DraupnirUtils.folders(ntpath.basename(results_dir),script_dir)
    DraupnirUtils.folders(("{}/Tree_Alignment_Sampled/".format(ntpath.basename(results_dir))),script_dir)
    DraupnirUtils.folders(("{}/ReplacementPlots_Train/".format(ntpath.basename(results_dir))), script_dir)
    DraupnirUtils.folders(("{}/ReplacementPlots_Test/".format(ntpath.basename(results_dir))), script_dir)
    DraupnirUtils.folders(("{}/Train_Plots/".format(ntpath.basename(results_dir))), script_dir)
    DraupnirUtils.folders(("{}/Test_Plots/".format(ntpath.basename(results_dir))), script_dir)
    DraupnirUtils.folders(("{}/Test2_Plots/".format(ntpath.basename(results_dir))), script_dir)
    DraupnirUtils.folders(("{}/Test_argmax_Plots/".format(ntpath.basename(results_dir))), script_dir)
    DraupnirUtils.folders(("{}/Test2_argmax_Plots/".format(ntpath.basename(results_dir))), script_dir)
    DraupnirUtils.folders(("{}/Train_argmax_Plots/".format(ntpath.basename(results_dir))), script_dir)
    DraupnirUtils.folders(("{}/Draupnir_Checkpoints/".format(ntpath.basename(results_dir))), script_dir)
    if args.infer_angles:
        DraupnirUtils.folders(("{}/Train_Plots/Angles_plots_per_aa/".format(ntpath.basename(results_dir))), script_dir)
        #DraupnirUtils.folders(("{}/Train_argmax_Plots/Angles_plots_per_aa/".format(ntpath.basename(results_dir))), script_dir)
        DraupnirUtils.folders(("{}/Test_Plots/Angles_plots_per_aa/".format(ntpath.basename(results_dir))), script_dir)
        #DraupnirUtils.folders(("{}/Test_argmax_Plots/Angles_plots_per_aa/".format(ntpath.basename(results_dir))), script_dir)
        DraupnirUtils.folders(("{}/Test2_Plots/Angles_plots_per_aa/".format(ntpath.basename(results_dir))), script_dir)
        #DraupnirUtils.folders(("{}/Test2_argmax_Plots/Angles_plots_per_aa/".format(ntpath.basename(results_dir))), script_dir)
    dataset = DraupnirLoadUtils.remove_nan(dataset)
    DraupnirUtils.ramachandran_plot(dataset[:, 3:], "{}/TRAIN_OBSERVED_angles".format(results_dir + "/Train_Plots"), "Train Angles",one_hot_encoded=settings_config.one_hot_encoding)
    if build_config.alignment_file:
        alignment = AlignIO.read(build_config.alignment_file, "fasta")
        alignment_file = build_config.alignment_file
        alignment_array =np.array(alignment)
        gap_positions = np.where(alignment_array == "-")[1]
        np.save("{}/Alignment_gap_positions.npy".format(results_dir),gap_positions)
    else:
        alignment = AlignIO.read("{}/Mixed_info_Folder/{}.mafft".format(script_dir,name), "fasta")
        alignment_file = "{}/Mixed_info_Folder/{}.mafft".format(script_dir,name)
        align_array = np.array(alignment)
        gap_positions = np.where(align_array == "-")[1]
        np.save("{}/Alignment_gap_positions.npy".format(results_dir), gap_positions)

    aa_probs_updated = DraupnirLoadUtils.validate_aa_probs(alignment,build_config)

    percentID = DraupnirUtils.perc_identity_alignment(alignment)
    alignment_length = dataset.shape[1] - 3
    min_seq_len = int(np.min(dataset[:, 1, 0]))
    max_seq_len = int(np.max(dataset[:, 1, 0]))
    n_seq = dataset.shape[0]

    build_config = BuildConfig(alignment_file=alignment_file,
                               use_ancestral=build_config.use_ancestral,
                               n_test=build_config.n_test,
                               build_graph=build_config.build_graph,
                               aa_probs=aa_probs_updated,
                               triTSNE=False,
                               align_seq_len=alignment_length,
                               leaves_testing=build_config.leaves_testing,
                               batch_size = args.batch_size,
                               plate_subsample_size=args.plating_size,
                               script_dir=build_config.script_dir,
                               no_testing=build_config.no_testing)

    # Highlight: Correction of the batch size in case loaded dataset is splitted. n_test indicates the percege of sequences from the train to be used as test
    if build_config.n_test > 0:
        subtracted = int(n_seq*build_config.n_test/100)
    else:
        subtracted = 0


    batch_size = [DraupnirUtils.define_batch_size(n_seq-subtracted) if not args.batch_size else args.batch_size if args.batch_size > 1 else n_seq-subtracted][0]
    plate_size = [DraupnirUtils.define_batch_size(n_seq-subtracted) if not  args.plating_size and args.plating else args.plating_size][0]
    if not args.plating: assert plate_size == None, "Please set plating_size to None if you do not want to do plate subsampling"
    if args.plating: assert args.batch_size == 1, "We are plating, no batching, please set batch_size == 1"

    build_config = BuildConfig(alignment_file=alignment_file,
                               use_ancestral=build_config.use_ancestral,
                               n_test=build_config.n_test,
                               build_graph=build_config.build_graph,
                               aa_probs=aa_probs_updated,
                               triTSNE=False,
                               align_seq_len=alignment_length,
                               leaves_testing=build_config.leaves_testing,
                               batch_size=batch_size,
                               plate_subsample_size=plate_size,
                               script_dir=build_config.script_dir,
                               no_testing=build_config.no_testing)


    def hyperparameters():
        text_file = open("{}/Hyperparameters_{}_{}epochs.txt".format(results_dir, now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),args.num_epochs), "a")
        text_file.write(".....................Method: {} .....................\n".format(settings_config.model_design))
        text_file.write("Dataset name: {} \n".format(name))
        text_file.write("Num epochs: {} \n".format(args.num_epochs))
        text_file.write("Alignment %ID: {} \n".format(percentID))
        text_file.write("One Hot Encoding: {} \n".format(settings_config.one_hot_encoding))
        text_file.write("Aligned sequences: {} \n".format(settings_config.aligned_seq))
        text_file.write("Number of available sequences: {} \n".format(n_seq))
        text_file.write("Alignment length: {} \n".format(alignment_length))
        text_file.write("Max seq length: {} \n".format(max_seq_len))
        text_file.write("Min seq length: {} \n".format(min_seq_len))
        text_file.write("Learning Rate: {} \n".format(param_config["lr"]))
        text_file.write("Z dimension: {} \n".format(param_config["z_dim"]))
        text_file.write("GRU hidden size: {} \n".format(param_config["gru_hidden_dim"]))
        text_file.write("Kappa addition: {} \n".format(args.kappa_addition))
        text_file.write("Amino acid possibilities + gap: {} \n".format(build_config.aa_probs))
        text_file.write("Substitution matrix : {} \n".format(args.subs_matrix))
        text_file.write("Batch by clade : {} \n".format(args.batch_by_clade))
        text_file.write("Batch size (=1 means entire dataset): {} \n".format(batch_size))
        text_file.write("Plating (subsampling) : {} \n".format(args.plating))
        text_file.write("Plating size : {} \n".format(build_config.plate_subsample_size))
        text_file.write("Inferring angles : {} \n".format(args.infer_angles))
        text_file.write("Leaves testing (uses the full leaves latent space (NOT a subset)): {} \n".format(build_config.leaves_testing))
        text_file.write(str(param_config) + "\n")

    hyperparameters()
    patristic_matrix = pd.read_csv("{}/{}/{}/{}_patristic_distance_matrix.csv".format(script_dir,settings_config.data_folder,name,name), low_memory=False)
    patristic_matrix = patristic_matrix.rename(columns={'Unnamed: 0': 'rows'})
    patristic_matrix.set_index('rows',inplace=True)
    try:
        cladistic_matrix = pd.read_csv("{}/{}/{}/{}_cladistic_distance_matrix.csv".format(script_dir,settings_config.data_folder,name,name), index_col="rows",low_memory=False)
    except: #Highlight: For larger datasets , I do not calculate the cladistic matrix, because there is not a fast method. So no cladistic matrix and consequently , no patrocladistic matrix = evolutionary matrix
        cladistic_matrix = None

    ancestor_info = pd.read_csv("{}/{}/{}/{}_tree_levelorder_info.csv".format(script_dir,settings_config.data_folder,name,name,name), sep="\t",index_col=False,low_memory=False)
    ancestor_info["0"] = ancestor_info["0"].astype(str)
    ancestor_info.drop('Unnamed: 0', inplace=True, axis=1)
    nodes_names = ancestor_info["0"].tolist()
    tree_levelorder_names = np.asarray(nodes_names)
    if name.startswith("simulations"):# Highlight: Leaves start with A, internal nodes with I
        leave_nodes_indexes = [i for i, node in enumerate(nodes_names) if re.search('^A{1}[0-9]+(?![A-Z])+',str(node))]
        internal_nodes_indexes = [i for i, node in enumerate(nodes_names) if re.search('^(?!^A{1}[0-9]+(?![A-Z])+)', str(node))]
        internal_nodes_dict = dict((node, i) for i, node in enumerate(nodes_names) if re.search('^(?!^A{1}[0-9]+(?![A-Z])+)', str(node)))
        leaves_nodes_dict = dict((node,i) for i,node in enumerate(nodes_names) if re.search('^A{1}[0-9]+(?![A-Z])+', str(node)))

    else: # Highlight: Internal nodes start with A
        if name.startswith("benchmark_randall_original_naming"):
            internal_nodes_indexes = [node for i, node in enumerate(nodes_names) if re.search('^A{1}[0-9]+(?![A-Z])+', node)]
            leave_nodes_indexes = [node for i, node in enumerate(nodes_names) if re.search('^(?!^A{1}[0-9]+(?![A-Z])+)', node)]
            internal_nodes_dict = dict((node, i) for i, node in enumerate(nodes_names) if re.search('^A{1}[0-9]+(?![A-Z])+', str(node)))
            leaves_nodes_dict = dict((node, i) for i, node in enumerate(nodes_names) if re.search('^(?!^A{1}[0-9]+(?![A-Z])+)', node))

        else:#Highlight: For the datasets without given test nodes, who have baliphy or iqtree trees
            internal_nodes_indexes = [i for i, node in enumerate(nodes_names) if re.search('^A{1}[0-9]+(?![A-Z])+', node)]
            leave_nodes_indexes = [i for i, node in enumerate(nodes_names) if re.search('^(?!^A{1}[0-9]+(?![A-Z])+)', node)]
            internal_nodes_dict = dict((node, i) for i, node in enumerate(nodes_names) if re.search('^A{1}[0-9]+(?![A-Z])+', str(node)))
            leaves_nodes_dict = dict((node, i) for i, node in enumerate(nodes_names) if re.search('^(?!^A{1}[0-9]+(?![A-Z])+)', node))


    #Highlight: Load The clades and reassigning their names to the ones in tree levelorder
    clades_dict_leaves = pickle.load(open('{}/Mixed_info_Folder/{}_Clades_dict_leaves.p'.format(script_dir,name), "rb"))
    clades_dict_leaves = DraupnirLoadUtils.convert_clades_dict(name, clades_dict_leaves, leaves_nodes_dict, internal_nodes_dict,only_leaves=True)
    clades_dict_all = pickle.load(open('{}/Mixed_info_Folder/{}_Clades_dict_all.p'.format(script_dir, name), "rb"))
    clades_dict_all = DraupnirLoadUtils.convert_clades_dict(name, clades_dict_all, leaves_nodes_dict, internal_nodes_dict,only_leaves=False)
    # Highlight: Load the dictionary containing the closests leaves to the internal nodes, transform the names to their tree level order
    closest_leaves_dict = pickle.load(open('{}/Mixed_info_Folder/{}_Closest_leaves_dict.p'.format(script_dir, name), "rb"))
    closest_leaves_dict = DraupnirLoadUtils.convert_closest_leaves_dict(name, closest_leaves_dict, internal_nodes_dict, leaves_nodes_dict)
    #Highlight: Load the dictionary containing all the internal and leaves that descend from the each ancestor node
    descendants_dict = pickle.load(open('{}/Mixed_info_Folder/{}_Descendants_dict.p'.format(script_dir, name), "rb"))
    descendants_dict = DraupnirLoadUtils.convert_descendants(name,descendants_dict,internal_nodes_dict,leaves_nodes_dict)
    try:
        linked_nodes_dict = pickle.load(open('{}/Mixed_info_Folder/{}_Closest_children_dict.p'.format(script_dir,name),"rb"))
        linked_nodes_dict = DraupnirLoadUtils.convert_only_linked_children(name, linked_nodes_dict, internal_nodes_dict, leaves_nodes_dict)
    except:
        linked_nodes_dict = None
    ancestor_info_numbers = DraupnirLoadUtils.convert_ancestor_info(name,ancestor_info,tree_levelorder_names)
    Dataset,children_array = DraupnirLoadUtils.create_children_array(dataset,ancestor_info_numbers)
    sorted_distance_matrix = DraupnirLoadUtils.pairwise_distance_matrix(name,script_dir)

    leaves_names_list = pickle.load(open('{}/{}/{}/{}_Leafs_names_list.p'.format(script_dir,settings_config.data_folder,name,name),"rb"))

    #Highlight: Organize, conquer and divide
    dataset_train,\
    dataset_test, \
    evolutionary_matrix_train, \
    evolutionary_matrix_test, \
    patristic_matrix_train,\
    patristic_matrix_test,\
    patristic_matrix_full, \
    cladistic_matrix_train, \
    cladistic_matrix_test, \
    cladistic_matrix_full, \
    position_test, \
    leaves_names_test = DraupnirLoadUtils.processing(
        results_dir,
        Dataset,
        patristic_matrix,
        cladistic_matrix,
        sorted_distance_matrix,
        n_seq,
        build_config.n_test,
        now,
        name,
        build_config.aa_probs,
        leaves_names_list,
        nodes=tree_levelorder_names,
        ancestral=build_config.use_ancestral,
        one_hot_encoding=settings_config.one_hot_encoding)

    if dataset_test is not None:#Highlight: Dataset_test != None only when the test dataset is extracted from the train (testing leaves)
        DraupnirUtils.ramachandran_plot(dataset_test[:, 2:], "{}/TEST_OBSERVED_angles".format(results_dir+"/Test_Plots"),"Test Angles", one_hot_encoded=settings_config.one_hot_encoding)
        dataset_test = torch.from_numpy(dataset_test)
        patristic_matrix_test = torch.from_numpy(patristic_matrix_test)
        if cladistic_matrix_test is not None:
            cladistic_matrix_test = torch.from_numpy(cladistic_matrix_test)
    if cladistic_matrix_train is not None :
        cladistic_matrix_train = torch.from_numpy(cladistic_matrix_train)
        cladistic_matrix_full = torch.from_numpy(cladistic_matrix_full)
    #Highlight: Normalize the patristic distances/patrocladistic==evolutionary matrix
    normalize_patristic = False
    if normalize_patristic:
        print("Normalizing patristic matrix!")
        patristic_matrix_full[1:, 1:] = patristic_matrix_full[1:, 1:] / np.linalg.norm(patristic_matrix_full[1:, 1:])
        if evolutionary_matrix_train is not None:
            evolutionary_matrix_train[1:, 1:] = evolutionary_matrix_train[1:, 1:] / np.linalg.norm(evolutionary_matrix_train[1:, 1:])
        evolutionary_matrix_train = [torch.from_numpy(evolutionary_matrix_train) if evolutionary_matrix_train is not None else evolutionary_matrix_train][0]
    else:
        evolutionary_matrix_train = [torch.from_numpy(evolutionary_matrix_train) if evolutionary_matrix_train is not None else evolutionary_matrix_train][0]
    text_file = open("{}/Hyperparameters_{}_{}epochs.txt".format(results_dir, now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),args.num_epochs), "a")
    text_file.write("Normalize patristic: {}\n".format(str(normalize_patristic)))

    train_load =TrainLoad(dataset_train=torch.from_numpy(dataset_train),
                          evolutionary_matrix_train=evolutionary_matrix_train,
                          patristic_matrix_train=torch.from_numpy(patristic_matrix_train),
                          cladistic_matrix_train=cladistic_matrix_train)
    test_load =TestLoad(dataset_test=dataset_test,
             evolutionary_matrix_test=evolutionary_matrix_test,
             patristic_matrix_test=patristic_matrix_test,
             cladistic_matrix_test=cladistic_matrix_test,
             leaves_names_test=leaves_names_test,
             position_test=position_test,
             internal_nodes_indexes=internal_nodes_indexes)
    additional_load = AdditionalLoad(patristic_matrix_full=torch.from_numpy(patristic_matrix_full),
                    cladistic_matrix_full=cladistic_matrix_full,
                   children_array=children_array,
                   ancestor_info_numbers=ancestor_info_numbers,
                   tree_levelorder_names=tree_levelorder_names,
                   clades_dict_leaves=clades_dict_leaves,
                   closest_leaves_dict=closest_leaves_dict,
                   clades_dict_all=clades_dict_all,
                   linked_nodes_dict=linked_nodes_dict,
                   descendants_dict=descendants_dict,
                   alignment_length=alignment_length,
                   aa_frequencies=None,
                   correspondence_dict=None,
                   special_nodes_dict=None,
                   full_name=args.full_name)
    return train_load,test_load,additional_load, build_config
def save_checkpoint(Draupnir,save_directory, optimizer):
    """Saves the model and optimizer dict states to disk
    :param nn.module Draupnir: model
    :param str save_directory
    :param torch.optim optimizer"""
    save_directory = ("{}/Draupnir_Checkpoints/".format(ntpath.basename(save_directory)))
    optimizer.save(save_directory + "/Optimizer_state.p")
    torch.save(Draupnir.state_dict(), save_directory + "/Model_state_dict.p")
def load_checkpoint(model_dict_dir,optim_dir,optim,model):
    """Saves the model and optimizer dict states to disk
    :param nn.module guide: guide
    :param str save_directory"""
    if model_dict_dir is not None:
        print("Loading pretrained Model parameters...")
        model.load_state_dict(torch.load(model_dict_dir), strict=False)
    if optim_dir is not None:
        print("Loading pretrained Optimizer states...")
        optim.load(optim_dir)
def save_samples(dataset,patristic,samples_out,entropies,correspondence_dict,results_dir):
    """Saves a dictionary of tensors
    :param tensor dataset: [N_seqs, align_len, 30]
    :param tensor patristic: [N_seqs+1,N_seqs+1]
    :param namedttuple samples_out
    :param tensor entropies: entropies per sequence
    :param dict correspondence_dict: dictionary that contains the correspondence upon the node indexes (integers) and the tree nodes names (ordered in tree level order)
    :param str results_dir
    """
    info_dict = { "dataset": dataset,
                  "patristic":patristic,
                  "correspondence_dict":correspondence_dict,
                  "aa_predictions": samples_out.aa_sequences,
                  "latent_space": samples_out.latent_space,
                  "logits": samples_out.logits,
                  "entropies": entropies,
                  "phis" : samples_out.phis,
                  "psis": samples_out.psis,
                  "mean_phi":samples_out.mean_phi,
                  "mean_psi":samples_out.mean_psi,
                  "kappa_phi":samples_out.kappa_phi,
                  "kappa_psi": samples_out.kappa_psi}

    torch.save(info_dict, results_dir)
def visualize_latent_space(latent_space_train,patristic_matrix_train,additional_load,build_config,args,results_dir):
    """Joins the latent spaces of the test (internal) and train (leaves) to visualize them divided by clades
      :param tensor latent_space_train: tensor [n_leaves,z_dim]
      :param tensor latent_space_test: tensor [n_internal,z_dim]
      :param tensor patristic_matrix_train: tensor [n_leaves + 1 , z_dim]
      :param tensor patristic_matrix_test: tensor [n_internal + 1 , z_dim]
      :param namedtuple additional_load: contains information about the sequences
      :param namedtuple build_config: contains information about the training settings
      :param args: run arguments
      :param str results_dir
    """
    latent_space_train = torch.cat((patristic_matrix_train[1:, 0][:,None],latent_space_train),dim=1)
    # DraupnirPlots.plot_z(latent_space_full,
    #                                      additional_info.children_dict,
    #                                      args.num_epochs,
    #                                      results_dir + folder)
    DraupnirPlots.plot_pairwise_distances_only_leaves(latent_space_train,additional_load, args.num_epochs,
                                             results_dir,patristic_matrix_train)
    latent_space_train = latent_space_train.detach().cpu().numpy()
    DraupnirPlots.plot_latent_space_tsne_by_clade_leaves(latent_space_train,
                                         additional_load,
                                         args.num_epochs,
                                         results_dir,
                                         build_config.triTSNE)
    DraupnirPlots.plot_latent_space_umap_by_clade_leaves(latent_space_train,
                                                         additional_load,
                                                         args.num_epochs,
                                                         results_dir,
                                                         build_config.triTSNE)
    DraupnirPlots.plot_latent_space_pca_by_clade_leaves(latent_space_train,
                                    additional_load,
                                    args.num_epochs,
                                    results_dir)
def save_and_select_model(args,build_config, model_load, patristic_matrix_train,patristic_matrix_full,script_dir,results_dir):
    """Selects the Draupnir derivation model to use. It saves the model and guides code to a file for easy checking
    :param namedtuple args
    :param namedtuple build_config
    :param namedtuple model_load
    :param tensor patristic_matrix_train [n_leaves+1, n_leaves +1]
    :param tensor patristic_matrix_full [n_leaves + n_internal + 1, n_leaves + n_internal + 1]
    :param str script_dir
    :param str results_dir
    """
    #Highlight: Selecting the model
    Draupnir = DraupnirModels.DRAUPNIRModel_VAE(model_load)
    patristic_matrix_model = patristic_matrix_train
    plating_info = ["WITH PLATING" if args.plating else "WITHOUT plating"][0]
    print("Using model {} {}".format(Draupnir.get_class(),plating_info))
    text_file = open("{}/Hyperparameters_{}_{}epochs.txt".format(results_dir, now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),args.num_epochs), "a")
    text_file.write("Model Class:  {} \n".format(Draupnir.get_class()))
    text_file.close()
    #Highlight: Saving the model function to a separate file
    model_file = open("{}/ModelFunction.py".format(results_dir), "a+")
    draupnir_models_file = open("{}/Draupnir_VAEmodels.py".format(script_dir), "r+")
    model_text = draupnir_models_file.readlines()
    line_start = model_text.index("class {}(DRAUPNIRModelClass):\n".format(Draupnir.get_class()))
    line_stop= [index if "class" in line else len(model_text[line_start+1:]) for index,line in enumerate(model_text[line_start+1:])][0]
    model_text = model_text[line_start:]
    model_text = model_text[:line_stop] #'class DRAUPNIRModel2b(DRAUPNIRModelClass):\n'
    model_file.write("".join(model_text))
    model_file.close()
    return Draupnir, patristic_matrix_model
def draupnir_train(train_load,
                   test_load,
                   additional_load,
                   additional_info,
                   build_config,
                   settings_config,
                   params_config,
                   n_samples,
                   args,
                   device,
                   script_dir,
                   results_dir,
                   graph_coo=None,
                   clades_dict=None,
                   load_folder=None):
    """Trains Draupnir-Ordinary VAE by performing SVI inference
    :param namedtuple train_load
    :param namedtuple test_load
    :param namedtuple additional_load
    :param namedtuple additional_info
    :param namedtuple build_config
    :param namedtuple settings_config
    :param dict params_config
    :param int n_samples
    :param namedtuple args
    :param str device
    :param str script_dir
    :param str results_dir
    :param graph graph_coo: graph that embedds the tree into a COO graph that works with pytorch geometric
    :param dict clades_dict
    :param str load_folder: contains path to previously trained results that we can load"""
    align_seq_len = build_config.align_seq_len
    if not additional_load.correspondence_dict:
        correspondence_dict = dict(zip(list(range(len(additional_load.tree_levelorder_names))), additional_load.tree_levelorder_names))
    else:
        correspondence_dict = additional_load.correspondence_dict
    if args.use_cuda:
        dataset_test = test_load.dataset_test.cuda()
        blosum = additional_info.blosum.cuda()
        aa_frequencies = additional_load.aa_frequencies.cuda()
        dataset_train = train_load.dataset_train.cuda()
        patristic_matrix_train = train_load.patristic_matrix_train.cuda()
        patristic_matrix_test = test_load.patristic_matrix_test.cuda()
        patristic_matrix_full = additional_load.patristic_matrix_full.cuda()
        if train_load.cladistic_matrix_train is not None:
            cladistic_matrix_train = train_load.cladistic_matrix_train.cuda()
            cladistic_matrix_test = test_load.cladistic_matrix_test.cuda()
            cladistic_matrix_full = additional_load.cladistic_matrix_full.cuda()
        else:
            cladistic_matrix_train = cladistic_matrix_test = cladistic_matrix_full = None
        nodes_representations_array = additional_info.nodes_representations_array.cuda()
        dgl_graph = additional_info.dgl_graph
    else:
        dataset_test = test_load.dataset_test
        blosum = additional_info.blosum
        aa_frequencies = additional_load.aa_frequencies
        dataset_train = train_load.dataset_train
        patristic_matrix_train = train_load.patristic_matrix_train
        patristic_matrix_test = test_load.patristic_matrix_test
        patristic_matrix_full = additional_load.patristic_matrix_full
        cladistic_matrix_train = train_load.cladistic_matrix_train
        cladistic_matrix_test = test_load.cladistic_matrix_test
        cladistic_matrix_full = additional_load.cladistic_matrix_full
        nodes_representations_array = additional_info.nodes_representations_array
        dgl_graph = additional_info.dgl_graph


    # aa_probs = torch.unique(dataset_train[:, 2:, 0])

    blosum_max,blosum_weighted,variable_score = DraupnirUtils.process_blosum(blosum,aa_frequencies,align_seq_len,build_config.aa_probs)
    dataset_train_blosum = DraupnirUtils.blosum_embedding_encoder(blosum, aa_frequencies, align_seq_len,
                                                                  build_config.aa_probs, dataset_train,
                                                                  settings_config.one_hot_encoding)

    plt.plot(variable_score.cpu().detach().numpy())
    plt.savefig("{}/Variable_score.png".format(results_dir))

    model_load = ModelLoad(z_dim=int(params_config["z_dim"]),
                           align_seq_len=align_seq_len,
                           device=device,
                           args=args,
                           build_config=build_config,
                           leaves_nodes=dataset_train[:, 0, 1],
                           n_tree_levels=len(additional_info.tree_by_levels_dict),
                           gru_hidden_dim=int(params_config["gru_hidden_dim"]),
                           pretrained_params=None,
                           aa_frequencies=aa_frequencies,
                           blosum=blosum,
                           blosum_max=blosum_max,
                           blosum_weighted=blosum_weighted,
                           dataset_train_blosum=dataset_train_blosum,
                           # train dataset with blosum vectors instead of one-hot encodings
                           variable_score=variable_score,
                           internal_nodes=patristic_matrix_test[1:, 0],  # dataset_test[:,0,1]
                           graph_coo=graph_coo,
                           nodes_representations_array=nodes_representations_array,
                           dgl_graph=dgl_graph,
                           children_dict=additional_info.children_dict,
                           closest_leaves_dict=additional_load.closest_leaves_dict,
                           descendants_dict=additional_load.descendants_dict,
                           clades_dict_all=additional_load.clades_dict_all,
                           leaves_testing=build_config.leaves_testing,
                           plate_unordered=args.plate_unordered,
                           one_hot_encoding=settings_config.one_hot_encoding)

    Draupnir, patristic_matrix_model = save_and_select_model(args,build_config, model_load, patristic_matrix_train,patristic_matrix_full,script_dir,results_dir)

    #Highlight: Setting optimizer and guide
    adam_args = {"lr": params_config["lr"], "betas": (params_config["beta1"], params_config["beta2"]),"eps":params_config["eps"],"weight_decay":params_config["weight_decay"],"clip_norm":params_config["clip_norm"],"lrd":params_config["lrd"]}
    optim = pyro.optim.ClippedAdam(adam_args)
    #guide = AutoNormal(Draupnir.model)
    #guide = AutoDiagonalNormal(Draupnir.model)
    guide = AutoDelta(Draupnir.model)
    def load_tune_params(load_params):
        """Loading pretrained parameters and allowing to tune them"""
        if load_params:
            pyro.clear_param_store()
            #tune_folder = "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_SH3_pf00018_larger_than_30aa_2021_07_26_16h55min02s945627ms_15000epochs" #
            tune_folder = "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_SH3_pf00018_larger_than_30aa_2021_07_26_17h00min16s025722ms_15000epochs"
            print("Loading pretrained model dict from {}".format(tune_folder))
            optim_dir = None
            model_dir = "{}/Draupnir_Checkpoints/Model_state_dict.p".format(tune_folder)
            text_file = open("{}/Hyperparameters_{}_{}epochs.txt".format(results_dir, now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),args.num_epochs), "a")
            text_file.write("Load pretrained TUNED params Model: {}\n".format(model_dir))
            text_file.write("Load pretrained TUNED params Optim: {}\n".format(optim_dir))
            text_file.close()
            load_checkpoint(model_dict_dir=model_dir,optim_dir=optim_dir,optim=optim,model = Draupnir)
            #Draupnir.train(False)
    load_tune_params(False)

    elbo =Trace_ELBO()

    text_file = open("{}/Hyperparameters_{}_{}epochs.txt".format(results_dir, now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),args.num_epochs), "a")
    text_file.write("ELBO :  {} \n".format(str(elbo)))
    text_file.write("Guide :  {} \n".format(str(guide.__class__)))
    text_file.write("Optimizer :  {} \n".format(optim))

    svi = SVI(model=Draupnir.model, guide=guide, optim=optim, loss=elbo) #TODO: TraceMeanField_ELBO() http://docs.pyro.ai/en/0.3.0-release/inference_algos.html#pyro.infer.trace_mean_field_elbo.TraceMeanField_ELBO

    check_point_epoch = [50 if args.num_epochs < 100 else (args.num_epochs / 100)][0]

    #batch_size = [DraupnirUtils.Define_batch_size(dataset_train.shape[0]) if not args.batch_size else args.batch_size if args.batch_size > 1 else dataset_train.shape[0]][0]
    batching_method = ["batch_dim_0" if not args.batch_by_clade else "batch_by_clade"][0]
    train_loader = DraupnirLoadUtils.setup_data_loaders(dataset_train, patristic_matrix_train,clades_dict,blosum,build_config,args,method=batching_method, use_cuda=args.use_cuda)
    training_method= lambda f, svi, patristic_matrix_model, cladistic_matrix_full,train_loader,args: lambda svi, patristic_matrix_model, cladistic_matrix_full,train_loader,args: f(svi, patristic_matrix_model, cladistic_matrix_full,train_loader,args)

    training_function = training_method(DraupnirTrain.train, svi, patristic_matrix_model, cladistic_matrix_full,
                                        train_loader, args)

    ######################
    ####Training Loop#####
    ######################
    train_loss = []
    entropy = []
    gradient_norms = defaultdict(list)
    start_total = time.time()
    epoch = 0
    epoch_count=0
    added_epochs = 0
    while epoch < args.num_epochs:
        if check_point_epoch > 0 and epoch > 0 and epoch % check_point_epoch == 0:
            save_checkpoint(Draupnir,results_dir, optimizer=optim)
            DraupnirPlots.plot_ELBO(train_loss, results_dir)
            DraupnirPlots.plot_entropy(entropy, results_dir)
        start = time.time()
        total_epoch_loss_train = training_function(svi, patristic_matrix_model, cladistic_matrix_full,train_loader,args) #TODO: check that this is working 100%
        memory_usage_mib = torch.cuda.max_memory_allocated()*9.5367*1e-7 #convert byte to MiB
        stop = time.time()
        train_loss.append(float(total_epoch_loss_train)) #convert to float because otherwise it's kept in torch's history
        print("[epoch %03d]  average training loss: %.4f %.5g time/epoch %.2f MiB/epoch" % (epoch_count, total_epoch_loss_train, stop - start,memory_usage_mib))
        # Register hooks to monitor gradient norms.
        for name_i, value in pyro.get_param_store().named_parameters():
            value.register_hook(lambda g, name_i=name_i: gradient_norms[name_i].append(g.norm().item()))
        map_estimates = guide()
        sample_out_train = Draupnir.sample(map_estimates,
                                           1,
                                           dataset_train,
                                           patristic_matrix_full,
                                           cladistic_matrix_train,
                                           use_argmax=False,
                                           use_test=False)

        train_entropy_epoch = DraupnirModelsUtils.compute_sites_entropies(sample_out_train.logits.cpu().detach(),dataset_train.detach().cpu().long()[:,0,1])
        if epoch % args.test_frequency == 0:  # every n epochs --- sample
            pickle.dump(map_estimates, open('{}/Draupnir_Checkpoints/Map_estimates.p'.format(results_dir), 'wb'),protocol=pickle.HIGHEST_PROTOCOL)
            sample_out_train_argmax = Draupnir.sample(map_estimates,
                                                   1,
                                                   dataset_train,
                                                   patristic_matrix_full,
                                                   cladistic_matrix_full,
                                                   use_argmax=True,
                                                   use_test=False)

            save_samples(dataset_train,patristic_matrix_train,sample_out_train,train_entropy_epoch,correspondence_dict,"{}/train_info_dict.torch".format(results_dir + "/Train_Plots"))
            save_samples(dataset_train, patristic_matrix_train,sample_out_train_argmax, train_entropy_epoch, correspondence_dict,"{}/train_argmax_info_dict.torch".format(results_dir + "/Train_argmax_Plots"))

        del sample_out_train
        entropy.append(torch.mean(train_entropy_epoch[:,1]).item())
        if epoch == (args.num_epochs-1):
            DraupnirPlots.plot_ELBO(train_loss, results_dir)
            DraupnirPlots.plot_entropy(entropy, results_dir)

            save_checkpoint(Draupnir,results_dir, optimizer=optim)  # Saves the parameters gradients
            if len(train_loss) > 10 and args.activate_elbo_convergence:
                difference = sum(train_loss[-10:]) / 10 - total_epoch_loss_train
                convergence = [False if difference > 0.5 else True][0] # Highlight: this works , but what should be the treshold is yet to be determined
                if convergence:break
                else:
                    epoch -= int(args.num_epochs // 3)
                    added_epochs += int(args.num_epochs // 3)
            if len(train_loss) > 10 and args.activate_entropy_convergence:
                difference = sum(entropy[-10:]) / 10 - torch.mean(train_entropy_epoch[:,1]).item()
                convergence = [False if difference > 0.2 else True][0]  # Highlight: this works , but what should be the treshold is yet to be determined
                if convergence:
                    break
                else:
                    epoch -= int(args.num_epochs // 3)
                    added_epochs += int(args.num_epochs // 3)
        epoch += 1
        epoch_count +=1
        torch.cuda.empty_cache()
    end_total = time.time()
    print('Final timing: {}'.format(str(datetime.timedelta(seconds=end_total-start_total))))
    print("Added epochs : {}".format(added_epochs))
    text_file = open("{}/Hyperparameters_{}_{}epochs.txt".format(results_dir, now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),args.num_epochs), "a")
    text_file.write("Running time: {}\n".format(str(datetime.timedelta(seconds=end_total-start_total))))
    text_file.write("Total epochs (+added epochs): {}\n".format(args.num_epochs + added_epochs))

    pytorch_total_params = sum(val.numel() for param_name,val in pyro.get_param_store().named_parameters() if val.requires_grad and not param_name.startswith("decoder_attention"))

    text_file.write("Number of parameters: {} \n".format(pytorch_total_params))
    text_file.close()

    #DraupnirUtils.GradientsPlot(gradient_norms, args.num_epochs, results_dir) #Highlight: Very cpu intensive to compute
    print("Final Sampling....")

    map_estimates= guide()
    pickle.dump(map_estimates, open('{}/Draupnir_Checkpoints/Map_estimates.p'.format(results_dir), 'wb'),protocol=pickle.HIGHEST_PROTOCOL)

    sample_out_train = Draupnir.sample(map_estimates,
                                       n_samples,
                                       dataset_train,
                                       patristic_matrix_full,
                                       cladistic_matrix_train,
                                       use_argmax=False,use_test=False)


    sample_out_train_argmax = Draupnir.sample(map_estimates,
                                              n_samples,
                                              dataset_train,
                                              patristic_matrix_full,
                                              cladistic_matrix_train,
                                              use_argmax=True,use_test=False)


    #Highlight: Compute sequences Shannon entropies per site
    train_entropies = DraupnirModelsUtils.compute_sites_entropies(sample_out_train.logits.cpu(),
                                                                          dataset_train.cpu().long()[:,0,1])


    save_samples(dataset_train, patristic_matrix_train,sample_out_train, train_entropies,correspondence_dict,"{}/train_info_dict.torch".format(results_dir + "/Train_Plots"))
    save_samples(dataset_train, patristic_matrix_train,sample_out_train_argmax, train_entropies,correspondence_dict,"{}/train_argmax_info_dict.torch".format(results_dir + "/Train_argmax_Plots"))

    if load_folder: #TODO: Transform to loading model parameters and resampling
        print("Loading previously trained results")
        train_dict = torch.load("{}/Train_Plots/train_info_dict.torch".format(load_folder))
        def check_if_exists(a,b,key):
            "Deals with datasets that were trained before the current configuration of namedtuples"
            try:
                out = b[key]
            except:
                out = getattr(a, key)
            return out
        def one_or_another(a):
            try:
                out= a["predictions"]
            except:
                out = a["aa_predictions"]
            return out
        def load_dict_to_namedtuple(load_dict,sample_out):
            sample_out = SamplingOutput(aa_sequences=one_or_another(load_dict),# TODO: the old results have the name as predictions instead of aa_sequences
                                              latent_space=load_dict["latent_space"],
                                              logits=load_dict["logits"],
                                              phis=check_if_exists(sample_out, load_dict, key="phis"),
                                              psis=check_if_exists(sample_out, load_dict, key="psis"),
                                              mean_phi=check_if_exists(sample_out, load_dict, key="mean_phi"),
                                              mean_psi=check_if_exists(sample_out, load_dict, key="mean_psi"),
                                              kappa_phi=check_if_exists(sample_out, load_dict, key="kappa_phi"),
                                              kappa_psi=check_if_exists(sample_out, load_dict, key="kappa_psi"))
            return sample_out

        sample_out_train = load_dict_to_namedtuple(train_dict,sample_out_train)
        try:#some old predictions do not have train_argmax folder
            train_argmax_dict = torch.load("{}/Train_argmax_Plots/train_argmax_info_dict.torch".format(load_folder))
            sample_out_train_argmax = load_dict_to_namedtuple(train_argmax_dict,sample_out_train_argmax)
        except:
            print("Could not load train_argmax folders")
            pass

        train_entropies = train_dict["entropies"]
        #Highlight: Saving the pre-trained predictions! It overwrites the trained ones
        save_samples(dataset_train, patristic_matrix_train,sample_out_train, train_entropies,correspondence_dict,"{}/train_info_dict.torch".format(results_dir + "/Train_Plots"))
        save_samples(dataset_train, patristic_matrix_train,sample_out_train_argmax, train_entropies,correspondence_dict,"{}/train_argmax_info_dict.torch".format(results_dir + "/Train_argmax_Plots"))

    #Highlight: Concatenate leaves and internal latent space for plotting
    visualize_latent_space(sample_out_train.latent_space,patristic_matrix_train,additional_load,build_config,args,results_dir)

    start_plots = time.time()

    #Highlight: Plot samples
    preparing_plots(sample_out_train,
                    dataset_train,
                    dataset_train,
                    train_entropies,
                    results_dir + "/Train_Plots",
                    additional_load,
                    additional_info,
                    build_config,
                    n_samples,
                    dataset_train[:,0,1],
                    args,
                    replacement_plots=False,
                    plot_test=False,
                    no_testing=True,
                    overplapping_hist=False)
    #Highlight: Plot most likely sequence
    print("train argmax")
    preparing_plots(sample_out_train_argmax,
                    dataset_train,
                    dataset_train,
                    train_entropies,
                    results_dir + "/Train_argmax_Plots",
                    additional_load,
                    additional_info,
                    build_config,
                    n_samples,
                    dataset_train[:, 0, 1],
                    args,
                    replacement_plots=False,
                    plot_test=False,
                    no_testing=True,
                    overplapping_hist=False)
    stop_plots = time.time()
    print('Final plots timing: {}'.format(str(datetime.timedelta(seconds=stop_plots - start_plots))))
    print("##########################################################################################################")
def preparing_plots(samples_out,
                    dataset_true,
                    dataset_train,
                    entropies,
                    results_dir,
                    additional_load,
                    additional_info,
                    build_config,
                    n_samples,
                    test_ordered_nodes,
                    args,
                    replacement_plots=False,
                    overplapping_hist=False,
                    plot_test=True,
                    plot_angles=False,
                    no_testing=False):
    """
    Sets the plotting choices depending on the dataset choice and the training options
    :param namedtuple samples_out: named tuple containing the output samples from trained Draupnir
    :param tensor dataset_true: the true dataset, it can be the train or the test dataset
    :param tensor dataset_train: train(leaves) dataset [N_seqs, align_len +2 , 30]
    :param tensor entropies: site shannon entropies calculated from the logits
    :param str results_dir: path to output folder
    :param namedtuple additional_load
    :param namedtuple additional_info
    :param namedtuple build_config
    :param int n_samples: number of samples generated from the model
    :param tensor test_ordered_nodes: tensor with the order of the nodes to test (corresponding to the true dataset)
    :param namedtuple args
    :param bool replacement_plots: True --> Makes plot for amino acid replacement plots per sequence (expensive to plot)
    :param bool overplapping_hist: True --> Makes histogram that compares the percent identity of each ancestral sequence against each leaf sequence #TODO: remove
    :param bool plot_test: True --> Plotting performed on reconstructed ancestral sequences
    :param bool plot_angles: True --> Use with args.infer_angles, makes inference over backbone dihedral angles, use
    :param bool no_testing : Skip the testing of the ancestral nodes vs real ancestral nodes, because there are not available
    """
    if not additional_load.correspondence_dict:
        correspondence_dict = dict(zip(list(range(len(additional_load.tree_levelorder_names))), additional_load.tree_levelorder_names))
    else:
        correspondence_dict = additional_load.correspondence_dict
    if plot_test and args.name in ["Coral_Faviina","Coral_all"] or plot_test and args.name.endswith("_subtree"):
        #Highlight: select from the predictions only the sequences in the dataset_test. Remove gaps and align to the "observed"
        DraupnirPlots.clean_and_realign_train(args.name,
                                   dataset_true,
                                   dataset_train,
                                   samples_out.aa_sequences,  # test predictions
                                   test_ordered_nodes,
                                   n_samples,
                                   build_config.aa_probs,
                                   results_dir,
                                   additional_load,
                                   additional_info)
        DraupnirPlots.clean_and_realign_test(args.name,
                                   dataset_true,
                                   samples_out.aa_sequences,#test predictions
                                   test_ordered_nodes,
                                   n_samples,
                                   build_config.aa_probs,
                                   results_dir,
                                   additional_load,
                                   additional_info)

        DraupnirPlots.plot_entropies(args.name, entropies.detach().cpu().numpy(), results_dir, correspondence_dict)
    elif args.infer_angles and plot_angles:
        DraupnirPlots.plot_angles(samples_out,dataset_true,results_dir,additional_load,additional_info,n_samples,test_ordered_nodes)
        DraupnirPlots.plot_angles_per_aa(samples_out,dataset_true,results_dir,build_config,additional_load,additional_info,n_samples,test_ordered_nodes)
    else:
        DraupnirPlots.plot_heatmap_and_incorrect_aminoacids(args.name,
                                                                dataset_true,
                                                                samples_out.aa_sequences,
                                                                n_samples,
                                                                results_dir,
                                                                correspondence_dict,
                                                                build_config.aa_probs,
                                                                additional_load,
                                                                additional_info,
                                                                replacement_plots)
        DraupnirPlots.plot_entropies(args.name, entropies.detach().cpu().numpy(), results_dir, correspondence_dict)
        if overplapping_hist:
            DraupnirPlots.plot_overlapping_histogram(args.name,
                                                dataset_train,
                                                dataset_true,
                                                samples_out.aa_sequences,
                                                n_samples,
                                                results_dir,
                                                correspondence_dict,
                                                build_config.aa_probs)
def generate_config():
    config = {
        "lr": np.random.uniform(1e-2, 1e-5),
        "beta1": np.random.uniform(0.8, 0.9),
        "beta2": np.random.uniform(0.9, 0.99),
        "eps": np.random.choice([1e-10,1e-9, 1e-8, 1e-7, 1e-6]),
        "weight_decay": np.random.choice([0, 0.1, 0.2]),
        "clip_norm": np.random.choice([8, 9, 10,12,14]),
        "lrd": np.random.choice([1,2, 4, 6, 7]),
        "z_dim": np.random.choice([30, 40, 50, 60,70]),
        "gru_hidden_dim": np.random.choice([60, 70, 80, 90,100]),
    }
    return config
def config_build(args):
    if args.parameter_search:
        config = json.loads(args.config_dict)
    else:
        "Default hyperparameters (Clipped Adam optimizer), z dim and GRU"
        config = {
            "lr": 1e-3,
            "beta1": 0.9, #coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
            "beta2": 0.999,
            "eps": 1e-8,#term added to the denominator to improve numerical stability (default: 1e-8)
            "weight_decay": 0,#weight_decay: weight decay (L2 penalty) (default: 0)
            "clip_norm": 10,#clip_norm: magnitude of norm to which gradients are clipped (default: 10.0)
            "lrd": 1, #rate at which learning rate decays (default: 1.0)
            "z_dim": 30,
            "gru_hidden_dim": 60,
        }
    return config

def run_VAE(name,root_sequence_name,args,device,settings_config,build_config,script_dir):
    """Loads and pre-treats the data, executes Draupnir Ordinary VAE model for training
    :param str name
    :param str root_sequence_name
    :param namedtuple args
    :param device: torch device
    :param namedtuple settings_config
    :param namedtuple build_config
    :param str script_dir
    """
    results_dir = "{}/PLOTS_GP_VAE_{}_{}_{}epochs_{}".format(script_dir, name, now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),
                                                             args.num_epochs, args.select_guide)
    print("Loading datasets....")
    param_config = config_build(args)
    train_load, test_load, additional_load, build_config = load_data(name, settings_config, build_config, param_config,
                                                                     results_dir, script_dir, args)
    # align_seq_len = additional_load.alignment_length
    additional_info = DraupnirUtils.extra_processing(additional_load.ancestor_info_numbers,
                                                     additional_load.patristic_matrix_full, results_dir, args,
                                                     build_config)
    # name,root_sequence_name,train_load,test_load,additional_load,build_config,device,settings_config,script_dir
    train_load, test_load, additional_load = DraupnirLoadUtils.datasets_pretreatment(name, root_sequence_name,
                                                                                     train_load, test_load,
                                                                                     additional_load, build_config,
                                                                                     device, settings_config,
                                                                                     script_dir)
    torch.save(torch.get_rng_state(), "{}/rng_key.torch".format(results_dir))
    print("Starting Draupnir with Ordinary VAE ...")
    print("Dataset: {}".format(name))
    print("Number epochs: {}".format(args.num_epochs))
    print("Z/latent Size: {}".format(param_config["z_dim"]))
    print("GRU hidden size: {}".format(param_config["gru_hidden_dim"]))
    print("Number train sequences: {}".format(train_load.dataset_train.shape[0]))
    n_test = [test_load.dataset_test.shape[0] if test_load.dataset_test is not None else 0][0]
    print("Number test sequences: {}".format(n_test))
    print("Selected Substitution matrix : {}".format(args.subs_matrix))

    if not args.batch_by_clade:
        clades_dict = None
    else:
        clades_dict = additional_load.clades_dict_leaves
    graph_coo = None  # Highlight: use only with the GNN models (7)---> Otherwise is in additional_info
    # graph_coo = additional_info.graph_coo

    draupnir_train(train_load,
                       test_load,
                       additional_load,
                       additional_info,
                       build_config,
                       settings_config,
                       param_config,
                       args.n_samples,
                       args,
                       device,
                       script_dir,
                       results_dir,
                       graph_coo,
                       clades_dict)


