#!/usr/bin/env python3
"""
=======================
2022: Lys Sanz Moreta
Draupnir : Ancestral protein sequence reconstruction using a tree-structured Ornstein-Uhlenbeck variational autoencoder
=======================
"""
import os
import time
import warnings
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import sys
import re
from glob import glob
import subprocess
import ntpath
import pandas as pd
from collections import defaultdict
from Bio import AlignIO
import torch
import pyro
from pyro.infer import SVI
from pyro.infer.autoguide import  AutoDiagonalNormal,AutoDelta,AutoNormal
from pyro.infer import Trace_ELBO
import draupnir
draupnir_path = draupnir.__file__
import draupnir.utils as DraupnirUtils
import draupnir.models as DraupnirModels
import draupnir.guides as DraupnirGuides
import draupnir.plots as DraupnirPlots
import draupnir.train as DraupnirTrain
import draupnir.models_utils as DraupnirModelsUtils
import draupnir.load_utils as DraupnirLoadUtils
import datetime
import pickle
import json
import dill
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
BuildConfig = namedtuple('BuildConfig',['alignment_file','use_ancestral','n_test','build_graph',"aa_probs","triTSNE","align_seq_len",
                                        "leaves_testing","batch_size","plate_subsample_size","script_dir","no_testing"])
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

    :out namedtuple train_load: contains train related tensors. For example, dataset_train has shape [n_seqs, max_len + 2, 30], where in the second dimension
                                0  = [seq_len, position in tree, distance to root,ancestor, ..., 0]
                                1  = Git vector (30 dim) if available
                                2: = [(1 integer + 0*29) or (one hot encoded amino acid sequence (21 slots)+0*9)]"
    :out namedtuple test_load: similar to train load
    :out namedtuple additional_load
    :out namedtuple build_config
    """

    aligned = ["aligned" if settings_config.aligned_seq else "NOT_aligned"]
    one_hot = ["onehot" if settings_config.one_hot_encoding else "integers"]

    dataset = np.load("{}/{}_dataset_numpy_{}_{}.npy".format(settings_config.data_folder,name,aligned[0], one_hot[0]),allow_pickle=True)

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
        #DraupnirUtils.Folders(("{}/Train_argmax_Plots/Angles_plots_per_aa/".format(ntpath.basename(results_dir))), script_dir)
        DraupnirUtils.folders(("{}/Test_Plots/Angles_plots_per_aa/".format(ntpath.basename(results_dir))), script_dir)
        #DraupnirUtils.Folders(("{}/Test_argmax_Plots/Angles_plots_per_aa/".format(ntpath.basename(results_dir))), script_dir)
        DraupnirUtils.folders(("{}/Test2_Plots/Angles_plots_per_aa/".format(ntpath.basename(results_dir))), script_dir)
        #DraupnirUtils.Folders(("{}/Test2_argmax_Plots/Angles_plots_per_aa/".format(ntpath.basename(results_dir))), script_dir)
    json.dump(args.__dict__, open('{}/commandline_args.txt'.format(results_dir), 'w'), indent=2)
    dataset = DraupnirLoadUtils.remove_nan(dataset)
    DraupnirUtils.ramachandran_plot(dataset[:, 3:], "{}/TRAIN_OBSERVED_angles".format(results_dir + "/Train_Plots"), "Train Angles",one_hot_encoded=settings_config.one_hot_encoding)
    #Highlight: Read the alignment, find the alignment length and positions where there is any gap
    alignment = AlignIO.read(build_config.alignment_file, "fasta")
    alignment_array = np.array(alignment)
    gap_positions = np.where(alignment_array == "-")[1]
    np.save("{}/Alignment_gap_positions.npy".format(results_dir), gap_positions)
    # Highlight: count the majority character per site, this is useful for benchmarking
    sites_count = dict.fromkeys(np.unique(gap_positions))
    for site in np.unique(gap_positions):
        unique, counts = np.unique(alignment_array[:, site], return_counts=True)
        sites_count[site] = dict(zip(unique, counts))
    pickle.dump(sites_count, open("{}/Sites_count.p".format(results_dir), 'wb'),protocol=pickle.HIGHEST_PROTOCOL)
    #Highlight: Find if the preselected amount of amino acids (generally 20 + 1 gap) is correct or it needs a different value due to the presence of special amino acids
    aa_probs_updated = DraupnirLoadUtils.validate_aa_probs(alignment,build_config)
    percentID = DraupnirUtils.perc_identity_alignment(alignment)
    alignment_length = dataset.shape[1] - 3
    min_seq_len = int(np.min(dataset[:, 1, 0]))
    max_seq_len = int(np.max(dataset[:, 1, 0]))
    n_seq = dataset.shape[0]

    build_config = BuildConfig(alignment_file=build_config.alignment_file,
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

    build_config = BuildConfig(alignment_file=build_config.alignment_file,
                               use_ancestral=build_config.use_ancestral,
                               n_test=build_config.n_test,
                               build_graph=build_config.build_graph,
                               aa_probs=aa_probs_updated,
                               triTSNE=False,
                               align_seq_len=alignment_length,
                               leaves_testing=build_config.leaves_testing,
                               batch_size=batch_size,
                               plate_subsample_size=plate_size,
                               script_dir=script_dir,
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
        text_file.write("Plating unordered (not preserving the tree level order of the nodes) : {} \n".format(args.plate_unordered))
        text_file.write("Inferring angles : {} \n".format(args.infer_angles))
        text_file.write("Guide : {} \n".format(args.select_guide))
        text_file.write("Use learning rate scheduler : {} \n".format(args.use_scheduler))
        text_file.write("Leaves testing (uses the full leaves latent space (NOT a subset)): {} \n".format(build_config.leaves_testing))
        text_file.write(str(param_config) + "\n")

    hyperparameters()
    patristic_matrix = pd.read_csv("{}/{}_patristic_distance_matrix.csv".format(settings_config.data_folder,name,name), low_memory=False)
    patristic_matrix = patristic_matrix.rename(columns={'Unnamed: 0': 'rows'})
    patristic_matrix.set_index('rows',inplace=True)
    try:
        cladistic_matrix = pd.read_csv("{}/{}_cladistic_distance_matrix.csv".format(settings_config.data_folder,name), index_col="rows",low_memory=False)
    except: #Highlight: For larger datasets , I do not calculate the cladistic matrix, because there is not a fast method. So no cladistic matrix and consequently , no patrocladistic matrix = evolutionary matrix
        cladistic_matrix = None
    ancestor_info = pd.read_csv("{}/{}_tree_levelorder_info.csv".format(settings_config.data_folder,name), sep="\t",index_col=False,low_memory=False)
    ancestor_info["0"] = ancestor_info["0"].astype(str)
    ancestor_info.drop('Unnamed: 0', inplace=True, axis=1)
    nodes_names = ancestor_info["0"].tolist()
    tree_levelorder_names = np.asarray(nodes_names)
    if name.startswith("simulations"):# Highlight: Leaves start with A, internal nodes with I
        leave_nodes_indexes = [i for i, node in enumerate(nodes_names) if re.search('^A{1}[0-9]+(?![A-Z])+',str(node))]
        internal_nodes_indexes = [i for i, node in enumerate(nodes_names) if re.search('^(?!^A{1}[0-9]+(?![A-Z])+)', str(node))]
        internal_nodes_dict = dict((node, i) for i, node in enumerate(nodes_names) if re.search('^(?!^A{1}[0-9]+(?![A-Z])+)', str(node)))
        leaves_nodes_dict = dict((node,i) for i,node in enumerate(nodes_names) if re.search('^A{1}[0-9]+(?![A-Z])+', str(node)))

    else: # Highlight: when the internal nodes names start with A
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

    #TODO: dill and pickle module dependencies: https://oegedijk.github.io/blog/pickle/dill/python/2020/11/10/serializing-dill-references.html
    #Highlight: Load The clades and reassigning their names to the ones in tree levelorder
    clades_dict_leaves = pickle.load(open('{}/{}_Clades_dict_leaves.p'.format(settings_config.data_folder,name), "rb"))
    clades_dict_leaves = DraupnirLoadUtils.convert_clades_dict(name, clades_dict_leaves, leaves_nodes_dict, internal_nodes_dict,only_leaves=True)
    clades_dict_all = DraupnirLoadUtils.load_serialized(open('{}/{}_Clades_dict_all.p'.format(settings_config.data_folder,name), "rb"))
    clades_dict_all = DraupnirLoadUtils.convert_clades_dict(name, clades_dict_all, leaves_nodes_dict, internal_nodes_dict,only_leaves=False)
    # Highlight: Load the dictionary containing the closests leaves to the INTERNAL nodes, transform the names to their tree level order
    closest_leaves_dict = DraupnirLoadUtils.load_serialized(open('{}/{}_Closest_leaves_dict.p'.format(settings_config.data_folder,name), "rb"))
    closest_leaves_dict = DraupnirLoadUtils.convert_closest_leaves_dict(name, closest_leaves_dict, internal_nodes_dict, leaves_nodes_dict)
    #Highlight: Load the dictionary containing all the internal and leaves that descend from the each ancestor node
    descendants_dict = DraupnirLoadUtils.load_serialized(open('{}/{}_Descendants_dict.p'.format(settings_config.data_folder,name), "rb"))
    descendants_dict = DraupnirLoadUtils.convert_descendants(name,descendants_dict,internal_nodes_dict,leaves_nodes_dict)
    #Highlight: Load dictionary with the directly linked children nodes--> i only have it for one dataset
    try:
        linked_nodes_dict = DraupnirLoadUtils.load_serialized(open('{}/{}_Closest_children_dict.p'.format(settings_config.data_folder,name),"rb"))
        linked_nodes_dict = DraupnirLoadUtils.convert_only_linked_children(name, linked_nodes_dict, internal_nodes_dict, leaves_nodes_dict)
    except:
        linked_nodes_dict = None
    ancestor_info_numbers = DraupnirLoadUtils.convert_ancestor_info(name,ancestor_info,tree_levelorder_names)
    dataset,children_array = DraupnirLoadUtils.create_children_array(dataset,ancestor_info_numbers)
    sorted_distance_matrix = DraupnirLoadUtils.pairwise_distance_matrix(name,script_dir)

    leaves_names_list = pickle.load(open('{}/{}_Leafs_names_list.p'.format(settings_config.data_folder,name),"rb"))

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
        dataset,
        patristic_matrix,
        cladistic_matrix,
        sorted_distance_matrix,
        n_seq,
        build_config.n_test,
        now,
        name,
        build_config.aa_probs,
        leaves_names_list,
        one_hot_encoding=settings_config.one_hot_encoding,
        nodes=tree_levelorder_names,
        ancestral=build_config.use_ancestral)

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
                   full_name=settings_config.full_name)
    return train_load,test_load,additional_load, build_config
def save_checkpoint(Draupnir,save_directory, optimizer):
    """Saves the model and optimizer dict states to disk
    :param nn.module Draupnir: model
    :param str save_directory
    :param torch.optim optimizer"""
    save_directory = ("{}/Draupnir_Checkpoints/".format(save_directory))
    optimizer.save(save_directory + "/Optimizer_state.p")
    #keys = [key for key in Draupnir.state_dict() if "decoder_attention.rnn" not in key]
    torch.save(Draupnir.state_dict(), save_directory + "/Model_state_dict.p")
def save_checkpoint_guide(guide,save_directory):
    """Saves the model and optimizer dict states to disk
    :param nn.module guide: guide
    :param str save_directory"""
    save_directory = ("{}/Draupnir_Checkpoints/".format(save_directory))
    torch.save(guide.state_dict(), save_directory + "/Guide_state_dict.p")
def save_checkpoint_preloaded(state_dict,save_directory, optimizer_state):
    '''Saves the model and optimizer dict states to disk'''
    save_directory = ("{}/Draupnir_Checkpoints/".format(save_directory))
    torch.save(optimizer_state,save_directory + "/Optimizer_state.p")
    #keys = [key for key in Draupnir.state_dict() if "decoder_attention.rnn" not in key]
    torch.save(state_dict, save_directory + "/Model_state_dict.p")
def load_checkpoint(model_dict_dir,optim_dir,optim,model):
    """Loads the model and optimizer states from disk
    :param str model_dict_dir
    :param str optim_dir
    :param nn.module optim
    :param nn.module model"""
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
    :param namedtuple samples_out
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
def visualize_latent_space(latent_space_train,latent_space_test,patristic_matrix_train,patristic_matrix_test,additional_load,build_config,args,results_dir):
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
    #Highlight: Concatenate leaves and internal latent space for plotting
    latent_space_full = torch.cat((latent_space_train,latent_space_test),dim=0)
    #latent_space_indexes = torch.cat((dataset_train[:,0,1],dataset_test[:,0,1]),dim=0)
    latent_space_indexes = torch.cat((patristic_matrix_train[1:, 0], patristic_matrix_test[1:, 0]), dim=0)
    latent_space_full = torch.cat((latent_space_indexes[:,None],latent_space_full),dim=1)

    if additional_load.linked_nodes_dict is not None:
        if build_config.leaves_testing:
            #Highlight: Sorting indexes, not sure if necessary
            latent_space_idx_sorted, latent_space_idx = latent_space_full[:,0].sort()
            latent_space_full = latent_space_full[latent_space_idx]
            DraupnirPlots.plot_pairwise_distances_only_leaves(latent_space_full,additional_load,args.num_epochs, results_dir,additional_load.patristic_matrix_full)
        else:
            DraupnirPlots.plot_pairwise_distances(latent_space_full,additional_load, args.num_epochs,
                                             results_dir)
    latent_space_full = latent_space_full.detach().cpu().numpy()
    # DraupnirPlots.plot_z(latent_space_full,
    #                                      additional_info.children_dict,
    #                                      results_dir + folder)
    DraupnirPlots.plot_latent_space_tsne_by_clade(latent_space_full,
                                         additional_load,
                                         args.num_epochs,
                                         results_dir,
                                         build_config.triTSNE)
    DraupnirPlots.plot_latent_space_umap_by_clade(latent_space_full,
                                         additional_load,
                                         args.num_epochs,
                                         results_dir,
                                         build_config.triTSNE)
    DraupnirPlots.plot_latent_space_pca_by_clade(latent_space_full,
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
    #TODO: Warnings/Raise errors for not allowed combinations
    #todo: HOW TO MAKE THIS SIMPLER

    if args.batch_by_clade:# clade batching #TODO: remove??, does not seem better than normal batching
        Draupnir = DraupnirModels.DRAUPNIRModel_cladebatching(model_load)
        patristic_matrix_model =patristic_matrix_train
    elif args.plating :#plating without splitted weighted average of blosum scores
        assert args.batch_size == 1, "We are plating, no batching, please set batch_size == 1"
        Draupnir = DraupnirModels.DRAUPNIRModel_classic_plating(model_load) #plating in tree level order, no blosum splitting
        patristic_matrix_model = patristic_matrix_train
    elif build_config.leaves_testing and not args.infer_angles: #training and testing on leaves. In this case, the latent space is composed by both train-leaves and test-leaves patristic matrices, but only the train sequences are observed
        Draupnir = DraupnirModels.DRAUPNIRModel_leaftesting(model_load)
        patristic_matrix_model = patristic_matrix_full
    elif args.infer_angles:
        assert args.batch_size == 1, "Angles inference is not implemented with batching"
        Draupnir = DraupnirModels.DRAUPNIRModel_anglespredictions(model_load)
        if build_config.leaves_testing: #full inference of the leaves latent space
            patristic_matrix_model = patristic_matrix_train
        else: #partial inference of the leaves latent space
            patristic_matrix_model = patristic_matrix_full
    elif args.batch_size == 1:#Not batching, training on all leaves, testing on internal nodes & training on leaves but only using the latent space of the training leaves
        assert args.plating_size == None, "Please set to None the plate size. If you want to plate, use args.plate = True; if you want to batch, set batch_size to some number"
        if args.use_blosum:
            Draupnir = DraupnirModels.DRAUPNIRModel_classic(model_load)
        else:
            Draupnir = DraupnirModels.DRAUPNIRModel_classic_no_blosum(model_load)
        patristic_matrix_model = patristic_matrix_train
    else:  # batching
        print("Batching! (batch_size == None or batch_size != 1)")
        assert args.select_guide == "variational", "Batching does not support args.select_guide == delta_map"
        Draupnir = DraupnirModels.DRAUPNIRModel_batching(model_load)
        patristic_matrix_model = patristic_matrix_train

    plating_info = ["WITH PLATING" if args.plating else "WITHOUT plating"][0]
    print("Using model {} {}".format(Draupnir.get_class(),plating_info))
    text_file = open("{}/Hyperparameters_{}_{}epochs.txt".format(results_dir, now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),args.num_epochs), "a")
    text_file.write("Model Class:  {} \n".format(Draupnir.get_class()))
    text_file.close()
    #Highlight: Saving the model function to a separate file
    model_file = open("{}/ModelFunction.py".format(results_dir), "a+")

    draupnir_models_file = open("{}/models.py".format(os.path.dirname(draupnir_path)), "r+")
    model_text = draupnir_models_file.readlines()
    line_start = model_text.index("class {}(DRAUPNIRModelClass):\n".format(Draupnir.get_class()))
    #line_stop= [index if "class" in line else len(model_text[line_start+1:]) for index,line in enumerate(model_text[line_start+1:])][0]
    #model_text = model_text[line_start:]
    #model_text = model_text[:line_stop] #'class DRAUPNIRModel2b(DRAUPNIRModelClass):\n'
    model_file.write("".join(model_text))
    model_file.close()
    # Highlight: Saving the guide function to a separate file
    if args.select_guide.startswith("variational"):
        guide_file = open("{}/GuideFunction.py".format(results_dir), "a+")
        draupnir_guides_file = open("{}/guides.py".format(os.path.dirname(draupnir_path)), "r+")
        guide_text = draupnir_guides_file.readlines()
        guide_file.write("".join(guide_text))
    return Draupnir, patristic_matrix_model
def transform_to_integers(sample_out,build_config):
    """Transform the one-hot encoded sequences embedded in a namedtuple back to integers
    :param namedtuple sample_out: contains the tensors produced by Draupnir.sample
    :param namedtuple build_config"""
    sample_out = SamplingOutput(
        aa_sequences=DraupnirUtils.convert_to_integers(sample_out.aa_sequences.cpu(), build_config.aa_probs, axis=3),
        latent_space=sample_out.latent_space,
        logits=sample_out.logits,
        phis=sample_out.phis,
        psis=sample_out.phis,
        mean_phi=sample_out.mean_phi,
        mean_psi=sample_out.mean_psi,
        kappa_phi=sample_out.kappa_phi,
        kappa_psi=sample_out.kappa_psi)
    return sample_out
def select_quide(Draupnir,model_load,choice):
    """Select the guide type
    :param nn.module Draupnir
    :param namedtuple model_load
    :param str choice: guide name"""
    guide_types = {"delta_map":AutoDelta(Draupnir.model),
                   "diagonal_normal": AutoDiagonalNormal(Draupnir.model),
                   "normal":AutoNormal(Draupnir.model),
                   #"variational": DraupnirGuides.DRAUPNIRGUIDES(Draupnir.model,model_load,Draupnir)
                   }
    print("Using {} as guide".format(choice))
    #return guide_types[choice]
    if choice in guide_types.keys(): return guide_types[choice]
    else:
        guide = DraupnirGuides.DRAUPNIRGUIDES(Draupnir.model,model_load,Draupnir) #TODO: How to put inside of the Draupnir class??
        return guide
def calculate_percent_id(dataset_true,aa_sequences_predictions,align_lenght):
    """Fast version to calculate %ID among predictions and observed data, we are only using 1 sample, could use more but it's more memory expensive
    :param tensor dataset_true with shape [n_leaves,L+2,30]
    :param tensor aa_sequences_predictions"""

    #align_lenght = dataset_true[:, 2:, 0].shape[1]
    #aa_sequences_predictions = torch.cat((node_info, aa_sequences_predictions), dim=2)
    #aa_sequences_predictions = aa_sequences_predictions.permute(1, 0, 2) #[n_nodes,n_samples,L]
    equal_aminoacids = (aa_sequences_predictions == dataset_true[:, 2:,0]).float()  # is correct #[n_samples,n_nodes,L] #TODO: Review this is correct because it only works because n-sample = 1
    equal_aminoacids = (equal_aminoacids.sum(-1) / align_lenght)*100
    average_pid = equal_aminoacids.mean().cpu().numpy()
    std_pid = equal_aminoacids.std().cpu().numpy()
    return average_pid,std_pid
def plot_percent_id(average_pid_list,std_pid_list,results_dir):
    """Plots percent id
    :param list average_pid_list
    :param list std_pid_list
    :param str results_dir"""

    list_of_epochs = np.arange(0,len(average_pid_list))
    plt.plot(list_of_epochs,np.array(average_pid_list),color="seagreen")
    plt.fill_between(list_of_epochs, np.array(average_pid_list) - np.array(std_pid_list), np.array(average_pid_list) + np.array(std_pid_list),color="bisque",alpha=0.2)
    plt.xlabel("Epochs")
    plt.ylabel("Average percent %ID")
    plt.title("Percent ID performance")
    plt.savefig("{}/Percent_ID.png".format(results_dir))
    plt.clf()
    plt.close()
def draupnir_sample(train_load,
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
                    clades_dict=None):
    """Sample new sequences from a pretrained model
    :param namedtuple train_load: see utils.create_dataset for information on the input data format
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
    :param dict clades_dict"""
    align_seq_len = build_config.align_seq_len
    if not additional_load.correspondence_dict:
        correspondence_dict = dict(zip(list(range(len(additional_load.tree_levelorder_names))), additional_load.tree_levelorder_names))
    else:
        correspondence_dict = additional_load.correspondence_dict
    device = torch.device("cuda" if args.use_cuda else "cpu") #TODO: Check that this works
    blosum = additional_info.blosum.to(device)
    aa_frequencies = additional_load.aa_frequencies.to(device)
    dataset_train = train_load.dataset_train.to(device)
    patristic_matrix_train = train_load.patristic_matrix_train.to(device)
    patristic_matrix_full = additional_load.patristic_matrix_full.to(device)
    patristic_matrix_test = test_load.patristic_matrix_test.to(device)
    dataset_test = test_load.dataset_test.to(device)
    if train_load.cladistic_matrix_train is not None:
        cladistic_matrix_train = train_load.cladistic_matrix_train.to(device)
        cladistic_matrix_test = \
        [test_load.cladistic_matrix_test.to(device) if test_load.cladistic_matrix_test is not None else None][0]
        cladistic_matrix_full = additional_load.cladistic_matrix_full.to(device)
    else:
        cladistic_matrix_train = cladistic_matrix_test = cladistic_matrix_full = None
    nodes_representations_array = additional_info.nodes_representations_array.to(device)
    dgl_graph = additional_info.dgl_graph


    blosum_max, blosum_weighted, variable_score = DraupnirUtils.process_blosum(blosum, aa_frequencies, align_seq_len,
                                                                               build_config.aa_probs)
    dataset_train_blosum = DraupnirUtils.blosum_embedding_encoder(blosum, aa_frequencies, align_seq_len,
                                                                  build_config.aa_probs, dataset_train,
                                                                  settings_config.one_hot_encoding)

    # Highlight: plot the amount of change per position in the alignment
    plt.plot(variable_score.cpu().detach().numpy())
    plt.savefig("{}/Variable_score.png".format(results_dir))
    plt.close()
    plt.clf()


    print("WARNING: Fixing the parameters from pretrained ones to sample!!!")
    load_pretrained_folder = args.load_pretrained_path
    model_dict_dir = "{}/Draupnir_Checkpoints/Model_state_dict.p".format(load_pretrained_folder)
    guide_dict_dir = "{}/Draupnir_Checkpoints/Guide_state_dict.p".format(load_pretrained_folder)

    text_file = open("{}/Hyperparameters_{}_{}epochs.txt".format(results_dir, now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),args.num_epochs), "a")
    text_file.write("Load pretrained FIXED params from: {}\n".format(load_pretrained_folder))
    text_file.close()
    pretrained_params_dict_model = torch.load(model_dict_dir)
    pretrained_params_dict_guide = torch.load(guide_dict_dir)


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

    Draupnir, patristic_matrix_model = save_and_select_model(args, build_config, model_load, patristic_matrix_train,patristic_matrix_full,script_dir,results_dir)

    hyperparameter_file = glob('{}/Hyperparameters*'.format(load_pretrained_folder))[0]
    select_guide = [line.split(":")[1].strip("\n") for line in open(hyperparameter_file,"r+").readlines() if line.startswith("Guide :")][0]
    select_guide = "".join(select_guide.split())
    guide = select_quide(Draupnir,model_load,select_guide)

    with torch.no_grad():
        for name, parameter in guide.named_parameters():
            parameter.copy_(pretrained_params_dict_guide[name])
        for name, parameter in Draupnir.named_parameters():
            parameter.copy_(pretrained_params_dict_model[name])

    #map_estimates = guide(dataset_train,patristic_matrix_train,cladistic_matrix_train,batch_blosum=None)

    print("Generating new samples!")
    if select_guide == "variational":
        #map_estimates_dict = defaultdict()
        print("Variational approach: Re-sampling from the guide")
        #map_estimates_dict = dill.load(open('{}/Draupnir_Checkpoints/Map_estimates.p'.format(args.load_pretrained_path), "rb"))
        map_estimates_dict = defaultdict()
        samples_names = ["sample_{}".format(i) for i in range(n_samples)]
        # Highlight: Train storage
        aa_sequences_train_samples = torch.zeros((n_samples, dataset_train.shape[0], dataset_train.shape[1] - 2)).detach()
        latent_space_train_samples = torch.zeros((n_samples, dataset_train.shape[0], int(params_config["z_dim"]))).detach()
        logits_train_samples = torch.zeros((n_samples, dataset_train.shape[0], dataset_train.shape[1] - 2, build_config.aa_probs)).detach()
        # Highlight: Test storage
        aa_sequences_test_samples = torch.zeros((n_samples, patristic_matrix_test[1:].shape[0], dataset_train.shape[1] - 2)).detach()
        latent_space_test_samples = torch.zeros((n_samples, patristic_matrix_test[1:].shape[0], int(params_config["z_dim"]))).detach()
        logits_test_samples = torch.zeros((n_samples, patristic_matrix_test[1:].shape[0], dataset_train.shape[1] - 2, build_config.aa_probs)).detach()
        for sample_idx, sample in enumerate(samples_names):
            map_estimates = guide(dataset_train, patristic_matrix_train, cladistic_matrix_train, dataset_train_blosum,batch_blosum=None)  # only saving 1 sample

            map_estimates_dict[sample] = {val:key.detach() for val,key in map_estimates.items()}
            # Highlight: Sample one train sequence
            train_sample = Draupnir.sample(map_estimates,
                                           1,
                                           dataset_train,
                                           patristic_matrix_full,
                                           cladistic_matrix_train,
                                           use_argmax=False,
                                           use_test=False,
                                           use_test2=False)
            aa_sequences_train_samples[sample_idx] = train_sample.aa_sequences.detach()
            latent_space_train_samples[sample_idx] = train_sample.latent_space.detach()
            logits_train_samples[sample_idx] = train_sample.logits.detach()
            del train_sample
            # Highlight: Sample one test sequence
            test_sample = Draupnir.sample(map_estimates,
                                          1,
                                          dataset_test,
                                          patristic_matrix_full,
                                          cladistic_matrix_full,
                                          use_argmax=False,
                                          use_test=True,
                                          use_test2=False)
            aa_sequences_test_samples[sample_idx] = test_sample.aa_sequences.detach()
            latent_space_test_samples[sample_idx] = test_sample.latent_space.detach()
            logits_test_samples[sample_idx] = test_sample.logits.detach()
            del test_sample
            del map_estimates
        dill.dump(map_estimates_dict, open('{}/Draupnir_Checkpoints/Map_estimates.p'.format(results_dir), 'wb'))
        sample_out_train = SamplingOutput(aa_sequences=aa_sequences_train_samples,
                                          latent_space=latent_space_train_samples,
                                          logits=logits_train_samples,
                                          phis=None,
                                          psis=None,
                                          mean_phi=None,
                                          mean_psi=None,
                                          kappa_phi=None,
                                          kappa_psi=None)
        sample_out_test = SamplingOutput(aa_sequences=aa_sequences_test_samples,
                                         latent_space=latent_space_test_samples,
                                         logits=logits_test_samples,
                                         phis=None,
                                         psis=None,
                                         mean_phi=None,
                                         mean_psi=None,
                                         kappa_phi=None,
                                         kappa_psi=None)
        warnings.warn("In variational method Test folder results = Test2 folder results")
        sample_out_test2 = sample_out_test
        # Highlight: compute majority vote
        sample_out_train_argmax = SamplingOutput(
            aa_sequences=torch.mode(sample_out_train.aa_sequences, dim=0)[0].unsqueeze(0),  # I think is correct
            latent_space=sample_out_train.latent_space[0],  # TODO:Average?
            logits=sample_out_train.logits[0],
            phis=None,
            psis=None,
            mean_phi=None,
            mean_psi=None,
            kappa_phi=None,
            kappa_psi=None)
        sample_out_test_argmax = SamplingOutput(
            aa_sequences=torch.mode(sample_out_test.aa_sequences, dim=0)[0].unsqueeze(0),
            latent_space=sample_out_test.latent_space[0],
            logits=sample_out_test.logits[0],
            phis=None,
            psis=None,
            mean_phi=None,
            mean_psi=None,
            kappa_phi=None,
            kappa_psi=None)
        sample_out_test_argmax2 = sample_out_test_argmax
        # # Highlight: Compute sequences Shannon entropies per site
        train_entropies = DraupnirModelsUtils.compute_sites_entropies(sample_out_train_argmax.logits.cpu(),dataset_train.cpu().long()[:, 0, 1])
        test_entropies = DraupnirModelsUtils.compute_sites_entropies(sample_out_test_argmax.logits.cpu(),patristic_matrix_test.cpu().long()[1:, 0])
        test_entropies2 = DraupnirModelsUtils.compute_sites_entropies(sample_out_test_argmax.logits.cpu(),patristic_matrix_test.cpu().long()[1:, 0])
        # Highlight : save the samples
        save_samples(dataset_test, patristic_matrix_test, sample_out_test, test_entropies, correspondence_dict,"{}/test_info_dict.torch".format(results_dir + "/Test_Plots"))
        save_samples(dataset_test, patristic_matrix_test, sample_out_test_argmax, test_entropies,correspondence_dict,"{}/test_argmax_info_dict.torch".format(results_dir + "/Test_argmax_Plots"))
        save_samples(dataset_test, patristic_matrix_test, sample_out_test2, test_entropies2, correspondence_dict,"{}/test_info_dict2.torch".format(results_dir + "/Test2_Plots"))
        save_samples(dataset_test, patristic_matrix_test, sample_out_test_argmax2, test_entropies2, correspondence_dict,"{}/test2_argmax_info_dict.torch".format(results_dir + "/Test2_argmax_Plots"))
        save_samples(dataset_train, patristic_matrix_train, sample_out_train, train_entropies, correspondence_dict,"{}/train_info_dict.torch".format(results_dir + "/Train_Plots"))
        save_samples(dataset_train, patristic_matrix_train, sample_out_train_argmax, train_entropies,correspondence_dict,"{}/train_argmax_info_dict.torch".format(results_dir + "/Train_argmax_Plots"))
    elif select_guide == "delta_map":
        samples_names = ["sample_{}".format(i) for i in range(n_samples)]
        print("Loading map estimates")
        map_estimates = pickle.load(open('{}/Draupnir_Checkpoints/Map_estimates.p'.format(args.load_pretrained_path), "rb")) # the params estimates are the same, can be loaded
        # Highlight: Train storage
        # aa_sequences_train_samples = torch.zeros((n_samples, dataset_train.shape[0], dataset_train.shape[1] - 2)).detach()
        # latent_space_train_samples = torch.zeros((n_samples, dataset_train.shape[0], int(config["z_dim"]))).detach()
        # logits_train_samples = torch.zeros((n_samples, dataset_train.shape[0], dataset_train.shape[1] - 2, build_config.aa_probs)).detach()
        # Highlight: Test storage: Marginal
        aa_sequences_test_samples = torch.zeros((n_samples, patristic_matrix_test[1:].shape[0], dataset_train.shape[1] - 2)).detach()
        latent_space_test_samples = torch.zeros((n_samples, patristic_matrix_test[1:].shape[0], int(params_config["z_dim"]))).detach()
        logits_test_samples = torch.zeros((n_samples, patristic_matrix_test[1:].shape[0], dataset_train.shape[1] - 2, build_config.aa_probs)).detach()
        for sample_idx, sample in enumerate(samples_names):
            # Highlight: Sample one test sequence (from Marginal)
            test_sample = Draupnir.sample(map_estimates,
                                          1,
                                          dataset_test,
                                          patristic_matrix_full,
                                          cladistic_matrix_full,
                                          use_argmax=False,
                                          use_test=True,
                                          use_test2=False)
            aa_sequences_test_samples[sample_idx] = test_sample.aa_sequences.detach()
            latent_space_test_samples[sample_idx] = test_sample.latent_space.detach()
            logits_test_samples[sample_idx] = test_sample.logits.detach()
            del test_sample

        sample_out_train = Draupnir.sample(map_estimates,
                                           n_samples,
                                           dataset_train,
                                           patristic_matrix_full,
                                           cladistic_matrix_train,
                                           use_argmax=False,# <----ATTENTION, not using most likely sequence, cause not conditional sampling
                                           use_test=False,
                                           use_test2=False)

        sample_out_test = SamplingOutput(aa_sequences=aa_sequences_test_samples,
                                         latent_space=latent_space_test_samples,
                                         logits=logits_test_samples,
                                         phis=None,
                                         psis=None,
                                         mean_phi=None,
                                         mean_psi=None,
                                         kappa_phi=None,
                                         kappa_psi=None)
        # Highlight: Sample n_samples sequence (for MAP)
        sample_out_test2 = Draupnir.sample(map_estimates,
                                           n_samples,
                                           dataset_test,
                                           patristic_matrix_full,
                                           cladistic_matrix_full,
                                           use_argmax=False,
                                           use_test=False,
                                           use_test2=True)
        # Highlight: compute majority vote for "most likely sequence"
        sample_out_train_argmax = SamplingOutput(
            aa_sequences=torch.mode(sample_out_train.aa_sequences, dim=0)[0].unsqueeze(0),
            latent_space=sample_out_train.latent_space,
            logits=sample_out_train.logits,
            phis=None,
            psis=None,
            mean_phi=None,
            mean_psi=None,
            kappa_phi=None,
            kappa_psi=None)
        # Highlight: compute majority vote for "most likely sequence"
        sample_out_test_argmax = SamplingOutput(
            aa_sequences=torch.mode(sample_out_test.aa_sequences, dim=0)[0].unsqueeze(0),
            latent_space=sample_out_test.latent_space[0],
            logits=sample_out_test.logits[0],
            phis=None,
            psis=None,
            mean_phi=None,
            mean_psi=None,
            kappa_phi=None,
            kappa_psi=None)

        # Highlight = Sample MAP sequences
        sample_out_test_argmax2 = Draupnir.sample(map_estimates,
                                                  n_samples,
                                                  dataset_test,
                                                  patristic_matrix_full,
                                                  cladistic_matrix_full,
                                                  use_argmax=True,  # Attention!
                                                  use_test2=True,
                                                  use_test=False)


        # # Highlight: Compute sequences Shannon entropies per site
        train_entropies = DraupnirModelsUtils.compute_sites_entropies(sample_out_train_argmax.logits.cpu(),
                                                                      dataset_train.cpu().long()[:, 0, 1])
        test_entropies = DraupnirModelsUtils.compute_sites_entropies(sample_out_test_argmax.logits.cpu(),
                                                                     patristic_matrix_test.cpu().long()[1:, 0])
        test_entropies2 = DraupnirModelsUtils.compute_sites_entropies(sample_out_test_argmax2.logits.cpu(),
                                                                      patristic_matrix_test.cpu().long()[1:, 0])
        # Highlight : save the samples
        save_samples(dataset_test, patristic_matrix_test, sample_out_test, test_entropies, correspondence_dict,
                     "{}/test_info_dict.torch".format(results_dir + "/Test_Plots"))
        save_samples(dataset_test, patristic_matrix_test, sample_out_test_argmax, test_entropies,
                     correspondence_dict,
                     "{}/test_argmax_info_dict.torch".format(results_dir + "/Test_argmax_Plots"))
        save_samples(dataset_test, patristic_matrix_test, sample_out_test2, test_entropies2, correspondence_dict,
                     "{}/test_info_dict2.torch".format(results_dir + "/Test2_Plots"))
        save_samples(dataset_test, patristic_matrix_test, sample_out_test_argmax2, test_entropies2,
                     correspondence_dict,
                     "{}/test2_argmax_info_dict.torch".format(results_dir + "/Test2_argmax_Plots"))
        save_samples(dataset_train, patristic_matrix_train, sample_out_train, train_entropies, correspondence_dict,
                     "{}/train_info_dict.torch".format(results_dir + "/Train_Plots"))
        save_samples(dataset_train, patristic_matrix_train, sample_out_train_argmax, train_entropies,
                     correspondence_dict,
                     "{}/train_argmax_info_dict.torch".format(results_dir + "/Train_argmax_Plots"))

    visualize_latent_space(sample_out_train_argmax.latent_space,
                           sample_out_test_argmax.latent_space,
                           patristic_matrix_train,
                           patristic_matrix_test,
                           additional_load,
                           build_config,
                           args,
                           results_dir)
    visualize_latent_space(sample_out_train_argmax.latent_space,
                           sample_out_test_argmax2.latent_space,
                           patristic_matrix_train,
                           patristic_matrix_test,
                           additional_load,
                           build_config,
                           args,
                           "{}/Test2_Plots".format(results_dir))

    if settings_config.one_hot_encoding: #TODO: has not been checked, simply change the model to have OneHotCategorical
        print("Transforming one-hot back to integers")
        sample_out_train = transform_to_integers(sample_out_train,build_config)
        # sample_out_train_argmax = convert_to_integers(sample_out_train_argmax) #argmax sets directly the aa to the highest logit
        sample_out_test = transform_to_integers(sample_out_test,build_config)
        # sample_out_test_argmax = convert_to_integers(sample_out_test_argmax)
        sample_out_test2 = transform_to_integers(sample_out_test2,build_config)
        # sample_out_test_argmax2 = convert_to_integers(sample_out_test_argmax2)
        dataset_train = DraupnirUtils.convert_to_integers(dataset_train.cpu(), build_config.aa_probs, axis=2)
        if build_config.leaves_testing:  # TODO: Check that this works, when one hot encoding is fixed
            dataset_test = DraupnirUtils.convert_to_integers(dataset_test.cpu(), build_config.aa_probs,
                                                           axis=2)  # no need to do it with the test of the simulations, never was one hot encoded. Only for when we are testing leaves

    send_to_plot(n_samples,
                     dataset_train,
                     dataset_test,
                     patristic_matrix_test,
                     train_entropies,
                     test_entropies,
                     test_entropies2,
                     sample_out_train,
                     sample_out_train_argmax,
                     sample_out_test, sample_out_test_argmax,
                     sample_out_test2, sample_out_test_argmax2,
                     additional_load, additional_info, build_config, args, results_dir)


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
                   clades_dict=None):
    """Trains Draupnir-OU by performing SVI inference with non-batched training
    :param namedtuple train_load : contains several tensors and other data structures related to the training leaves. See see utils.create_dataset for information on the input data format
    :param namedtuple test_load : contains several tensors and other data structures related to the test internal nodes
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
    :param dict clades_dict"""
    align_seq_len = build_config.align_seq_len
    if not additional_load.correspondence_dict:
        correspondence_dict = dict(zip(list(range(len(additional_load.tree_levelorder_names))), additional_load.tree_levelorder_names))
    else:
        correspondence_dict = additional_load.correspondence_dict
    #device = [torch.device("cuda") if args.use_cuda else torch.device("cpu")][0]
    device = torch.device("cuda" if args.use_cuda else "cpu") #TODO: Check that this works
    blosum = additional_info.blosum.to(device)
    aa_frequencies = additional_load.aa_frequencies.to(device)
    dataset_train = train_load.dataset_train.to(device)
    patristic_matrix_train = train_load.patristic_matrix_train.to(device)
    patristic_matrix_full = additional_load.patristic_matrix_full.to(device)
    patristic_matrix_test = test_load.patristic_matrix_test.to(device)
    dataset_test = test_load.dataset_test.to(device)
    if train_load.cladistic_matrix_train is not None:
        cladistic_matrix_train = train_load.cladistic_matrix_train.to(device)
        cladistic_matrix_test = \
        [test_load.cladistic_matrix_test.to(device) if test_load.cladistic_matrix_test is not None else None][0]
        cladistic_matrix_full = additional_load.cladistic_matrix_full.to(device)
    else:
        cladistic_matrix_train = cladistic_matrix_test = cladistic_matrix_full = None
    nodes_representations_array = additional_info.nodes_representations_array.to(device)
    dgl_graph = additional_info.dgl_graph

    blosum_max,blosum_weighted,variable_score = DraupnirUtils.process_blosum(blosum,aa_frequencies,align_seq_len,build_config.aa_probs)
    dataset_train_blosum = DraupnirUtils.blosum_embedding_encoder(blosum,aa_frequencies,align_seq_len,build_config.aa_probs,dataset_train,settings_config.one_hot_encoding)
    #Highlight: plot the amount of change per position in the alignment
    plt.plot(variable_score.cpu().detach().numpy())
    plt.savefig("{}/Variable_score.png".format(results_dir))
    plt.close()
    plt.clf()


    model_load = ModelLoad(z_dim=int(params_config["z_dim"]),
                           align_seq_len=align_seq_len,
                           device=device,
                           args=args,
                           build_config = build_config,
                           leaves_nodes = dataset_train[:,0,1],
                           n_tree_levels=len(additional_info.tree_by_levels_dict),
                           gru_hidden_dim=int(params_config["gru_hidden_dim"]),
                           pretrained_params=None,
                           aa_frequencies=aa_frequencies,
                           blosum =blosum,
                           blosum_max=blosum_max,
                           blosum_weighted=blosum_weighted,
                           dataset_train_blosum = dataset_train_blosum, #train dataset with blosum vectors instead of one-hot encodings
                           variable_score=variable_score,
                           internal_nodes= patristic_matrix_test[1:,0], #dataset_test[:,0,1]
                           graph_coo=graph_coo,
                           nodes_representations_array = nodes_representations_array,
                           dgl_graph=dgl_graph,
                           children_dict= additional_info.children_dict,
                           closest_leaves_dict=additional_load.closest_leaves_dict,
                           descendants_dict = additional_load.descendants_dict,
                           clades_dict_all = additional_load.clades_dict_all,
                           leaves_testing = build_config.leaves_testing,
                           plate_unordered = args.plate_unordered,
                           one_hot_encoding= settings_config.one_hot_encoding)

    Draupnir, patristic_matrix_model = save_and_select_model(args,build_config, model_load, patristic_matrix_train,patristic_matrix_full,script_dir,results_dir)

    guide = select_quide(Draupnir, model_load, args.select_guide)
    elbo =Trace_ELBO()

    text_file = open("{}/Hyperparameters_{}_{}epochs.txt".format(results_dir, now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),args.num_epochs), "a")
    text_file.write("ELBO :  {} \n".format(str(elbo)))
    text_file.write("Guide :  {} \n".format(str(guide.__class__)))

    #Highlight: Select optimizer/scheduler
    if args.use_scheduler:
        print("Using a learning rate scheduler on top of the optimizer!")
        adam_args = {"lr": params_config["lr"], "betas": (params_config["beta1"], params_config["beta2"]), "eps": params_config["eps"],
                     "weight_decay": params_config["weight_decay"]}
        optim = torch.optim.Adam #Highlight: For the scheduler we need to use TORCH.optim not PYRO.optim, and there is no clipped adam in torch
        #Highlight: "Reduce LR on plateau: Scheduler: Reduce learning rate when a metric has stopped improving."
        optim = pyro.optim.ReduceLROnPlateau({'optimizer': optim, 'optim_args': adam_args})

    else:
        clippedadam_args = {"lr": params_config["lr"], "betas": (params_config["beta1"], params_config["beta2"]), "eps": params_config["eps"],
                     "weight_decay": params_config["weight_decay"], "clip_norm": params_config["clip_norm"], "lrd": params_config["lrd"]}
        optim = pyro.optim.ClippedAdam(clippedadam_args)
    def load_tune_params(load_params):
        """Loading pretrained parameters and allowing to tune them"""
        if load_params:
            pyro.clear_param_store()
            tune_folder = "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_PF00400_2021_11_17_22h59min34s252388ms_30000epochs"#delta map
            #tune_folder = ""
            print("WARNING: Loading pretrained model dict from {} !!!!".format(tune_folder))
            optim_dir = None
            model_dir = "{}/Draupnir_Checkpoints/Model_state_dict.p".format(tune_folder)
            text_file = open("{}/Hyperparameters_{}_{}epochs.txt".format(results_dir, now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),args.num_epochs), "a")
            text_file.write("Load pretrained TUNED params Model: {}\n".format(model_dir))
            text_file.write("Load pretrained TUNED params Optim: {}\n".format(optim_dir))
            text_file.close()
            load_checkpoint(model_dict_dir=model_dir,optim_dir=optim_dir,optim=optim,model = Draupnir)
            #Draupnir.train(False)
    load_tune_params(False)

    svi = SVI(Draupnir.model, guide,optim,elbo)
    text_file.write("Optimizer :  {} \n".format(optim))

    check_point_epoch = [50 if args.num_epochs < 100 else (args.num_epochs / 100)][0]

    batching_method = ["batch_dim_0" if not args.batch_by_clade else "batch_by_clade"][0]
    train_loader = DraupnirLoadUtils.setup_data_loaders(dataset_train, patristic_matrix_train,clades_dict,blosum,build_config,args,method=batching_method, use_cuda=args.use_cuda)

    training_method= lambda f, svi, patristic_matrix_model, cladistic_matrix_full,dataset_train_blosum,train_loader,args: lambda svi, patristic_matrix_model, cladistic_matrix_full,train_loader,args: f(svi, patristic_matrix_model, cladistic_matrix_full,dataset_train_blosum,train_loader,args)

    if args.batch_by_clade and clades_dict:
        training_function = training_method(DraupnirTrain.train_batch_clade,svi, patristic_matrix_model, cladistic_matrix_full,dataset_train_blosum,train_loader,args)
    elif args.batch_size == 1:#no batching or plating
        training_function = training_method(DraupnirTrain.train, svi, patristic_matrix_model,cladistic_matrix_full,dataset_train_blosum, train_loader, args)

    else:#batching
        training_function = training_method(DraupnirTrain.train_batch, svi, patristic_matrix_model,
                                            cladistic_matrix_full,dataset_train_blosum, train_loader, args)

    ######################
    ####Training Loop#####
    ######################
    train_loss = []
    entropy = []
    average_pid_list = []
    std_pid_list = []
    #gradient_norms = defaultdict(list)
    start_total = time.time()
    epoch = 0
    epoch_count=0
    added_epochs = 0
    output_file = open("{}/output.log".format(results_dir),"w")
    while epoch < args.num_epochs:
        if check_point_epoch > 0 and epoch > 0 and epoch % check_point_epoch == 0:
            DraupnirPlots.plot_ELBO(train_loss, results_dir)
            DraupnirPlots.plot_entropy(entropy, results_dir)
            plot_percent_id(average_pid_list, std_pid_list, results_dir)
        start = time.time()
        total_epoch_loss_train = training_function(svi, patristic_matrix_model, cladistic_matrix_full,train_loader,args)
        memory_usage_mib = torch.cuda.max_memory_allocated()*9.5367*1e-7 #convert byte to MiB
        stop = time.time()
        train_loss.append(float(total_epoch_loss_train)) #convert to float because otherwise it's kept in torch's history
        print("[epoch %03d]  average training loss: %.4f %.5g time/epoch %.2f MiB/epoch" % (epoch_count, total_epoch_loss_train, stop - start,memory_usage_mib))
        print("[epoch %03d]  average training loss: %.4f %.5g time/epoch %.2f MiB/epoch" % (epoch_count, total_epoch_loss_train, stop - start,memory_usage_mib),file=output_file)
        print("Current total time : {}".format(str(datetime.timedelta(seconds=stop-start_total))),file=output_file)
        # # Register hooks to monitor gradient norms.
        # for name_i, value in pyro.get_param_store().named_parameters():
        #     value.register_hook(lambda g, name_i=name_i: gradient_norms[name_i].append(g.norm().item()))
        #map_estimates = guide(dataset_train,patristic_matrix_train,cladistic_matrix_train,batch_blosum=None) #only saving 1 sample
        map_estimates = guide(dataset_train,patristic_matrix_train,cladistic_matrix_train,dataset_train_blosum,batch_blosum=None) #only saving 1 sample

        map_estimates = {val: key.detach() for val, key in map_estimates.items()}
        sample_out_train = Draupnir.sample(map_estimates,
                                           1,
                                           dataset_train,
                                           patristic_matrix_full,
                                           cladistic_matrix_train,
                                           use_argmax=True,
                                           use_test=False,
                                           use_test2=False)
        save_checkpoint(Draupnir, results_dir, optimizer=optim)  # Saves the parameters gradients
        save_checkpoint_guide(guide,results_dir)
        #Highlight: Plot entropies
        train_entropy_epoch = DraupnirModelsUtils.compute_sites_entropies(sample_out_train.logits.detach(),dataset_train.detach().long()[:,0,1])
        #Highlight: Plot percent id prediction performance
        average_pid, std_pid = calculate_percent_id(dataset_train.detach(), sample_out_train.aa_sequences.detach(),model_load.align_seq_len)
        average_pid_list.append(average_pid)
        std_pid_list.append(std_pid)
        if epoch % args.test_frequency == 0:  # every n epochs --- sample
            dill.dump(map_estimates, open('{}/Draupnir_Checkpoints/Map_estimates.p'.format(results_dir), 'wb'),protocol=pickle.HIGHEST_PROTOCOL)
            sample_out_test = Draupnir.sample(map_estimates,
                                              n_samples,
                                              dataset_test,
                                              patristic_matrix_full,
                                              cladistic_matrix_full,
                                              use_argmax=True,
                                              use_test=True,
                                              use_test2=False)
            sample_out_test_argmax = Draupnir.sample(map_estimates,
                                                n_samples,
                                                dataset_test,
                                                patristic_matrix_full,
                                                cladistic_matrix_full,
                                                use_argmax=True,
                                                use_test=True,
                                                use_test2=False)
            sample_out_train_argmax = Draupnir.sample(map_estimates,
                                                   n_samples,
                                                   dataset_train,
                                                   patristic_matrix_full,
                                                   cladistic_matrix_full,
                                                   use_argmax=True,
                                                   use_test=False,
                                                   use_test2=False)
            sample_out_test2 = Draupnir.sample(map_estimates,
                                               n_samples,
                                               dataset_test,
                                               patristic_matrix_full,
                                               cladistic_matrix_full,
                                               use_argmax=True,
                                               use_test=True,
                                               use_test2=False)
            sample_out_test_argmax2 = Draupnir.sample(map_estimates,
                                                      n_samples,
                                                      dataset_test,
                                                      patristic_matrix_full,
                                                      cladistic_matrix_full,
                                                      use_argmax=True,
                                                      use_test=False,
                                                      use_test2=True)

            test_entropies = DraupnirModelsUtils.compute_sites_entropies(sample_out_test.logits.detach(),patristic_matrix_test.detach().long()[1:, 0])
            test_entropies2 = DraupnirModelsUtils.compute_sites_entropies(sample_out_test2.logits.detach(),patristic_matrix_test.detach().long()[1:, 0])

            save_samples(dataset_test, patristic_matrix_test,sample_out_test, test_entropies,correspondence_dict,"{}/test_info_dict.torch".format(results_dir + "/Test_Plots"))
            save_samples(dataset_test, patristic_matrix_test,sample_out_test_argmax, test_entropies,correspondence_dict,"{}/test_argmax_info_dict.torch".format(results_dir + "/Test_argmax_Plots"))
            save_samples(dataset_train,patristic_matrix_train,sample_out_train,train_entropy_epoch,correspondence_dict,"{}/train_info_dict.torch".format(results_dir + "/Train_Plots"))
            save_samples(dataset_train, patristic_matrix_train,sample_out_train_argmax, train_entropy_epoch, correspondence_dict,"{}/train_argmax_info_dict.torch".format(results_dir + "/Train_argmax_Plots"))
            save_samples(dataset_test, patristic_matrix_test, sample_out_test2, test_entropies2, correspondence_dict,"{}/test_info_dict2.torch".format(results_dir + "/Test2_Plots"))
            save_samples(dataset_test, patristic_matrix_test, sample_out_test_argmax2, test_entropies2,correspondence_dict,"{}/test2_argmax_info_dict.torch".format(results_dir + "/Test2_argmax_Plots"))
            #Highlight: Freeing memory
            del sample_out_train_argmax
            del sample_out_test
            del sample_out_test_argmax
            del sample_out_test2
            del sample_out_test_argmax2

        del sample_out_train
        entropy.append(torch.mean(train_entropy_epoch[:,1]).item())
        if epoch == (args.num_epochs-1):
            DraupnirPlots.plot_ELBO(train_loss, results_dir)
            DraupnirPlots.plot_entropy(entropy, results_dir)
            plot_percent_id(average_pid_list, std_pid_list, results_dir)
            save_checkpoint(Draupnir,results_dir, optimizer=optim)  # Saves the parameters gradients
            save_checkpoint_guide(guide, results_dir)  # Saves the parameters gradients
            if len(train_loss) > 10 and args.activate_elbo_convergence:
                difference = sum(train_loss[-10:]) / 10 - total_epoch_loss_train
                convergence = [False if difference > 0.5 else True][0] # Highlight: this works, but what should be the treshold is yet to be determined
                if convergence:break
                else:
                    epoch -= int(args.num_epochs // 3)
                    added_epochs += int(args.num_epochs // 3)
            if len(train_loss) > 10 and args.activate_entropy_convergence:
                difference = sum(entropy[-10:]) / 10 - torch.mean(train_entropy_epoch[:,1]).item()
                convergence = [False if difference > 0.2 else True][0]  # Highlight: this works, but what should be the threshold is yet to be determined
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
    if args.select_guide.startswith("variational"):
        pytorch_total_params = sum([val.numel() for param_name,val in pyro.get_param_store().named_parameters() if val.requires_grad and not param_name.startswith("DRAUPNIRGUIDES.draupnir")])
    else: #TODO: Investigate again, but seems correct
        pytorch_total_params = sum(val.numel() for param_name,val in pyro.get_param_store().named_parameters() if val.requires_grad and not param_name.startswith("decoder_attention"))

    text_file.write("Number of parameters: {} \n".format(pytorch_total_params))
    text_file.close()
    #DraupnirUtils.GradientsPlot(gradient_norms, args.num_epochs, results_dir) #Highlight: Very cpu intensive to compute
    print("Final Sampling....")
    if args.select_guide == "variational":
        map_estimates_dict = defaultdict()
        samples_names = ["sample_{}".format(i) for i in range(n_samples)]
        #Highlight: Train storage
        aa_sequences_train_samples = torch.zeros((n_samples,dataset_train.shape[0],dataset_train.shape[1]-2)).detach()
        latent_space_train_samples = torch.zeros((n_samples,dataset_train.shape[0],int(params_config["z_dim"]))).detach()
        logits_train_samples = torch.zeros((n_samples,dataset_train.shape[0],dataset_train.shape[1]-2,build_config.aa_probs)).detach()
        #Highlight: Test storage
        aa_sequences_test_samples = torch.zeros((n_samples, patristic_matrix_test[1:].shape[0], dataset_train.shape[1] - 2)).detach()
        latent_space_test_samples = torch.zeros((n_samples, patristic_matrix_test[1:].shape[0], int(params_config["z_dim"]))).detach()
        logits_test_samples = torch.zeros((n_samples, patristic_matrix_test[1:].shape[0], dataset_train.shape[1] - 2, build_config.aa_probs)).detach()
        for sample_idx,sample in enumerate(samples_names):
            map_estimates = guide(dataset_train, patristic_matrix_train, cladistic_matrix_train, dataset_train_blosum,batch_blosum=None)  # only saving 1 sample

            map_estimates_dict[sample] = {val:key.detach() for val,key in map_estimates.items()}
            #Highlight: Sample one train sequence
            train_sample = Draupnir.sample(map_estimates,
                                                      1,
                                                      dataset_train,
                                                      patristic_matrix_full,
                                                      cladistic_matrix_train,
                                                      use_argmax=False,
                                                      use_test=False,
                                                      use_test2=False)

            aa_sequences_train_samples[sample_idx] = train_sample.aa_sequences.detach()
            latent_space_train_samples[sample_idx] = train_sample.latent_space.detach()
            logits_train_samples[sample_idx] = train_sample.logits.detach()
            del train_sample
            # Highlight: Sample one test sequence
            test_sample = Draupnir.sample(map_estimates,
                                          1,
                                          dataset_test,
                                          patristic_matrix_full,
                                          cladistic_matrix_full,
                                          use_argmax=False,
                                          use_test=True,
                                          use_test2=False)
            aa_sequences_test_samples[sample_idx] = test_sample.aa_sequences.detach()
            latent_space_test_samples[sample_idx] = test_sample.latent_space.detach()
            logits_test_samples[sample_idx] = test_sample.logits.detach()
            del test_sample
            del map_estimates
            torch.cuda.empty_cache()

        dill.dump(map_estimates_dict, open('{}/Draupnir_Checkpoints/Map_estimates.p'.format(results_dir), 'wb'))
        sample_out_train = SamplingOutput(aa_sequences=aa_sequences_train_samples,
                                      latent_space=latent_space_train_samples,
                                      logits=logits_train_samples,
                                      phis=None,
                                      psis=None,
                                      mean_phi=None,
                                      mean_psi=None,
                                      kappa_phi=None,
                                      kappa_psi=None)
        sample_out_test = SamplingOutput(aa_sequences=aa_sequences_test_samples,
                                            latent_space=latent_space_test_samples,
                                            logits=logits_test_samples,
                                            phis=None,
                                            psis=None,
                                            mean_phi=None,
                                            mean_psi=None,
                                            kappa_phi=None,
                                            kappa_psi=None)
        warnings.warn("In variational method Test folder results = Test2 folder results = Variational results")
        sample_out_test2 = sample_out_test
        #Highlight: compute majority vote/ Argmax
        sample_out_train_argmax = SamplingOutput(aa_sequences=torch.mode(sample_out_train.aa_sequences,dim=0)[0].unsqueeze(0), #I think is correct
                                          latent_space=sample_out_train.latent_space[0], #TODO:Average?
                                          logits=sample_out_train.logits[0],
                                          phis=None,
                                          psis=None,
                                          mean_phi=None,
                                          mean_psi=None,
                                          kappa_phi=None,
                                          kappa_psi=None)
        sample_out_test_argmax = SamplingOutput(aa_sequences=torch.mode(sample_out_test.aa_sequences,dim=0)[0].unsqueeze(0),
                                         latent_space=sample_out_test.latent_space[0],
                                         logits=sample_out_test.logits[0],
                                         phis=None,
                                         psis=None,
                                         mean_phi=None,
                                         mean_psi=None,
                                         kappa_phi=None,
                                         kappa_psi=None)
        sample_out_test_argmax2 = sample_out_test_argmax
        # # Highlight: Compute sequences Shannon entropies per site
        train_entropies = DraupnirModelsUtils.compute_sites_entropies(sample_out_train_argmax.logits.cpu(),dataset_train.cpu().long()[:, 0, 1])
        test_entropies = DraupnirModelsUtils.compute_sites_entropies(sample_out_test_argmax.logits.cpu(),patristic_matrix_test.cpu().long()[1:, 0])
        test_entropies2 = DraupnirModelsUtils.compute_sites_entropies(sample_out_test_argmax.logits.cpu(),patristic_matrix_test.cpu().long()[1:, 0])
        #Highlight : save the samples
        save_samples(dataset_test, patristic_matrix_test, sample_out_test, test_entropies, correspondence_dict,"{}/test_info_dict.torch".format(results_dir + "/Test_Plots"))
        save_samples(dataset_test, patristic_matrix_test, sample_out_test_argmax, test_entropies,correspondence_dict,"{}/test_argmax_info_dict.torch".format(results_dir + "/Test_argmax_Plots"))
        save_samples(dataset_test, patristic_matrix_test, sample_out_test2, test_entropies2, correspondence_dict,"{}/test_info_dict2.torch".format(results_dir + "/Test2_Plots"))
        save_samples(dataset_test, patristic_matrix_test, sample_out_test_argmax2, test_entropies2,correspondence_dict,"{}/test2_argmax_info_dict.torch".format(results_dir + "/Test2_argmax_Plots"))
        save_samples(dataset_train, patristic_matrix_train, sample_out_train, train_entropies, correspondence_dict,"{}/train_info_dict.torch".format(results_dir + "/Train_Plots"))
        save_samples(dataset_train, patristic_matrix_train, sample_out_train_argmax, train_entropies,correspondence_dict,"{}/train_argmax_info_dict.torch".format(results_dir + "/Train_argmax_Plots"))
    elif args.select_guide == "delta_map":
        samples_names = ["sample_{}".format(i) for i in range(n_samples)]
        #map_estimates = guide(dataset_train, patristic_matrix_train, cladistic_matrix_train, batch_blosum=None)
        map_estimates = guide(dataset_train,patristic_matrix_train,cladistic_matrix_train,dataset_train_blosum,batch_blosum=None) #only saving 1 sample

        pickle.dump(map_estimates, open('{}/Draupnir_Checkpoints/Map_estimates.p'.format(results_dir), 'wb'),protocol=pickle.HIGHEST_PROTOCOL)
        # Highlight: Test storage: Marginal
        aa_sequences_test_samples = torch.zeros((n_samples, patristic_matrix_test[1:].shape[0], dataset_train.shape[1] - 2))
        latent_space_test_samples = torch.zeros((n_samples, patristic_matrix_test[1:].shape[0], int(params_config["z_dim"])))
        logits_test_samples = torch.zeros((n_samples, patristic_matrix_test[1:].shape[0], dataset_train.shape[1] - 2, build_config.aa_probs))
        for sample_idx, sample in enumerate(samples_names):
            # Highlight: Sample one test sequence (for Marginal)
            test_sample = Draupnir.sample(map_estimates,
                                          1, #n_samples
                                          dataset_test,
                                          patristic_matrix_full,
                                          cladistic_matrix_full,
                                          use_argmax=False,
                                          use_test=True,
                                          use_test2=False)
            aa_sequences_test_samples[sample_idx] = test_sample.aa_sequences.detach()
            latent_space_test_samples[sample_idx] = test_sample.latent_space.detach()
            logits_test_samples[sample_idx] = test_sample.logits.detach()
            del test_sample

        sample_out_train = Draupnir.sample(map_estimates,
                                           n_samples,
                                           dataset_train,
                                           patristic_matrix_full,
                                           cladistic_matrix_train,
                                           use_argmax=False, #<----ATTENTION, not using most likely sequence, cause not conditional sampling
                                           use_test=False,
                                           use_test2=False)
        sample_out_test = SamplingOutput(aa_sequences=aa_sequences_test_samples,
                                         latent_space=latent_space_test_samples,
                                         logits=logits_test_samples,
                                         phis=None,
                                         psis=None,
                                         mean_phi=None,
                                         mean_psi=None,
                                         kappa_phi=None,
                                         kappa_psi=None)
        warnings.warn("With delta_map guide (Test folder results = Marginal) != (Test2 folder results = MAP)")
        #Highlight = Sample MAP sequences
        sample_out_test2 = Draupnir.sample(map_estimates,
                                          n_samples,
                                          dataset_test,
                                          patristic_matrix_full,
                                          cladistic_matrix_full,
                                          use_argmax=False,
                                          use_test2=True,
                                          use_test=False)
        #Highlight: Get the most likely sequence for the train
        sample_out_train_argmax = Draupnir.sample(map_estimates,
                                          n_samples,
                                          dataset_train,
                                          patristic_matrix_full,
                                          cladistic_matrix_train,
                                          use_argmax=True,
                                          use_test=False,
                                          use_test2=False)
        # Highlight: compute majority vote to get the "most likely sequence"
        sample_out_test_argmax = SamplingOutput(aa_sequences=torch.mode(sample_out_test.aa_sequences,dim=0)[0].unsqueeze(0),
                                                latent_space=sample_out_test.latent_space[0],
                                                logits=sample_out_test.logits[0],
                                                phis=None,
                                                psis=None,
                                                mean_phi=None,
                                                mean_psi=None,
                                                kappa_phi=None,
                                                kappa_psi=None)
        #Highlight = Sample MAP sequences
        sample_out_test_argmax2 = Draupnir.sample(map_estimates,
                                          n_samples,
                                          dataset_test,
                                          patristic_matrix_full,
                                          cladistic_matrix_full,
                                          use_argmax=True, #Attention!
                                          use_test2=True, #Attention!
                                          use_test=False)

        # # Highlight: Compute sequences Shannon entropies per site
        train_entropies = DraupnirModelsUtils.compute_sites_entropies(sample_out_train_argmax.logits.cpu(),dataset_train.cpu().long()[:, 0, 1])
        test_entropies = DraupnirModelsUtils.compute_sites_entropies(sample_out_test_argmax.logits.cpu(),patristic_matrix_test.cpu().long()[1:, 0])
        test_entropies2 = DraupnirModelsUtils.compute_sites_entropies(sample_out_test_argmax.logits.cpu(),patristic_matrix_test.cpu().long()[1:, 0])
        # Highlight : save the samples
        save_samples(dataset_test, patristic_matrix_test, sample_out_test, test_entropies, correspondence_dict,"{}/test_info_dict.torch".format(results_dir + "/Test_Plots"))
        save_samples(dataset_test, patristic_matrix_test, sample_out_test_argmax, test_entropies,correspondence_dict,"{}/test_argmax_info_dict.torch".format(results_dir + "/Test_argmax_Plots"))
        save_samples(dataset_test, patristic_matrix_test, sample_out_test2, test_entropies2, correspondence_dict,"{}/test_info_dict2.torch".format(results_dir + "/Test2_Plots"))
        save_samples(dataset_test, patristic_matrix_test, sample_out_test_argmax2, test_entropies2,correspondence_dict,"{}/test2_argmax_info_dict.torch".format(results_dir + "/Test2_argmax_Plots"))
        save_samples(dataset_train, patristic_matrix_train, sample_out_train, train_entropies, correspondence_dict,"{}/train_info_dict.torch".format(results_dir + "/Train_Plots"))
        save_samples(dataset_train, patristic_matrix_train, sample_out_train_argmax, train_entropies,correspondence_dict,"{}/train_argmax_info_dict.torch".format(results_dir + "/Train_argmax_Plots"))

    #Highlight: Concatenate leaves and internal latent space for plotting
    visualize_latent_space(sample_out_train_argmax.latent_space,
                           sample_out_test_argmax.latent_space,
                           patristic_matrix_train,
                           patristic_matrix_test,
                           additional_load,
                           build_config,
                           args,
                           results_dir)
    visualize_latent_space(sample_out_train_argmax.latent_space,
                           sample_out_test_argmax2.latent_space,
                           patristic_matrix_train,
                           patristic_matrix_test,
                           additional_load,
                           build_config,
                           args,
                           "{}/Test2_Plots".format(results_dir))

    if settings_config.one_hot_encoding:
        print("Transforming one-hot back to integers")
        sample_out_train_argmax = transform_to_integers(sample_out_train_argmax,build_config) #argmax sets directly the aa to the highest logit
        sample_out_test_argmax = transform_to_integers(sample_out_test_argmax,build_config)
        sample_out_test_argmax2 = transform_to_integers(sample_out_test_argmax2,build_config)
        dataset_train = DraupnirUtils.convert_to_integers(dataset_train.cpu(),build_config.aa_probs,axis=2)
        if build_config.leaves_testing: #TODO: Check that this works
            dataset_test = DraupnirUtils.convert_to_integers(dataset_test.cpu(),build_config.aa_probs,axis=2) #no need to do it with the test of the simulations, never was one hot encoded. Only for testing leaves

    send_to_plot(n_samples,
                     dataset_train,
                     dataset_test,
                     patristic_matrix_test,
                     train_entropies,
                     test_entropies,
                     test_entropies2,
                     sample_out_train,
                     sample_out_train_argmax,
                     sample_out_test, sample_out_test_argmax,
                     sample_out_test2, sample_out_test_argmax2,
                     additional_load, additional_info, build_config, args, results_dir)
def draupnir_train_batching(train_load,
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
                   clades_dict=None):
    """Trains Draupnir-OU by performing SVI inference with conditionally independent batched training
    :param namedtuple train_load : contains several tensors and other data structures related to the training leaves. See see utils.create_dataset for information on the input data format
    :param namedtuple test_load : contains several tensors and other data structures related to the test internal nodes
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
    :param dict clades_dict"""
    align_seq_len = build_config.align_seq_len
    if not additional_load.correspondence_dict:
        correspondence_dict = dict(
            zip(list(range(len(additional_load.tree_levelorder_names))), additional_load.tree_levelorder_names))
    else:
        correspondence_dict = additional_load.correspondence_dict
    device = torch.device("cuda" if args.use_cuda else "cpu") #TODO: Check that this works
    blosum = additional_info.blosum.to(device)
    aa_frequencies = additional_load.aa_frequencies.to(device)
    dataset_train = train_load.dataset_train.to(device)
    patristic_matrix_train = train_load.patristic_matrix_train.to(device)
    patristic_matrix_full = additional_load.patristic_matrix_full.to(device)
    patristic_matrix_test = test_load.patristic_matrix_test.to(device)
    dataset_test = test_load.dataset_test.to(device)
    if train_load.cladistic_matrix_train is not None:
        cladistic_matrix_train = train_load.cladistic_matrix_train.to(device)
        cladistic_matrix_test = \
        [test_load.cladistic_matrix_test.to(device) if test_load.cladistic_matrix_test is not None else None][0]
        cladistic_matrix_full = additional_load.cladistic_matrix_full.to(device)
    else:
        cladistic_matrix_train = cladistic_matrix_test = cladistic_matrix_full = None
    nodes_representations_array = additional_info.nodes_representations_array.to(device)
    dgl_graph = additional_info.dgl_graph

    # aa_prob = torch.unique(dataset_train[:, 2:, 0])

    blosum_max, blosum_weighted, variable_score = DraupnirUtils.process_blosum(blosum, aa_frequencies, align_seq_len,
                                                                               build_config.aa_probs)
    dataset_train_blosum = DraupnirUtils.blosum_embedding_encoder(blosum, aa_frequencies, align_seq_len,
                                                                  build_config.aa_probs, dataset_train,
                                                                  settings_config.one_hot_encoding)

    # Highlight: plot the amount of change per position in the alignment
    plt.plot(variable_score.cpu().detach().numpy())
    plt.savefig("{}/Variable_score.png".format(results_dir))
    plt.close()
    plt.clf()


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


    guide = select_quide(Draupnir, model_load, args.select_guide)
    elbo = Trace_ELBO()

    text_file = open("{}/Hyperparameters_{}_{}epochs.txt".format(results_dir, now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),
                                                                 args.num_epochs), "a")
    text_file.write("ELBO :  {} \n".format(str(elbo)))
    text_file.write("Guide :  {} \n".format(str(guide.__class__)))

    # Highlight: Select optimizer/scheduler
    if args.use_scheduler:
        print("Using a learning rate scheduler on top of the optimizer!")
        adam_args = {"lr": params_config["lr"], "betas": (params_config["beta1"], params_config["beta2"]), "eps": params_config["eps"],
                     "weight_decay": params_config["weight_decay"]}
        optim = torch.optim.Adam  # Highlight: For the scheduler we need to use TORCH.optim not PYRO.optim, there is no clipped adam in torch
        # Highlight: "Reduce LR on plateau: Scheduler: Reduce learning rate when a metric has stopped improving."
        optim = pyro.optim.ReduceLROnPlateau({'optimizer': optim, 'optim_args': adam_args})

    else:
        clippedadam_args = {"lr": params_config["lr"], "betas": (params_config["beta1"], params_config["beta2"]), "eps": params_config["eps"],
                            "weight_decay": params_config["weight_decay"], "clip_norm": params_config["clip_norm"],
                            "lrd": params_config["lrd"]}
        optim = pyro.optim.ClippedAdam(clippedadam_args)

    def load_tune_params(load_params):
        """Loading pretrained parameters and allowing to tune them"""
        if load_params:
            pyro.clear_param_store()
            tune_folder = "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_PF00400_2021_11_17_22h59min34s252388ms_30000epochs"  # delta map
            # tune_folder = ""
            print("WARNING: Loading pretrained model dict from {} !!!!".format(tune_folder))
            optim_dir = None
            model_dir = "{}/Draupnir_Checkpoints/Model_state_dict.p".format(tune_folder)
            text_file = open(
                "{}/Hyperparameters_{}_{}epochs.txt".format(results_dir, now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),
                                                            args.num_epochs), "a")
            text_file.write("Load pretrained TUNED params Model: {}\n".format(model_dir))
            text_file.write("Load pretrained TUNED params Optim: {}\n".format(optim_dir))
            text_file.close()
            load_checkpoint(model_dict_dir=model_dir, optim_dir=optim_dir, optim=optim, model=Draupnir)
            # Draupnir.train(False)

    load_tune_params(False)

    svi = SVI(Draupnir.model, guide, optim,elbo)  # TODO: TraceMeanField_ELBO() http://docs.pyro.ai/en/0.3.0-release/inference_algos.html#pyro.infer.trace_mean_field_elbo.TraceMeanField_ELBO
    text_file.write("Optimizer :  {} \n".format(optim))

    check_point_epoch = [50 if args.num_epochs < 100 else (args.num_epochs / 100)][0]

    batching_method = ["batch_dim_0" if not args.batch_by_clade else "batch_by_clade"][0]
    train_loader = DraupnirLoadUtils.setup_data_loaders(dataset_train, patristic_matrix_train,clades_dict,blosum,build_config,args,method=batching_method, use_cuda=args.use_cuda)


    training_method= lambda f, svi, patristic_matrix_model, cladistic_matrix_full,dataset_train_blosum,train_loader,args: lambda svi, patristic_matrix_model, cladistic_matrix_full,train_loader,args: f(svi, patristic_matrix_model, cladistic_matrix_full,dataset_train_blosum,train_loader,args)

    if args.batch_by_clade and clades_dict:
        training_function = training_method(DraupnirTrain.train_batch_clade,svi, patristic_matrix_model, cladistic_matrix_full,dataset_train_blosum,train_loader,args)
    elif args.batch_size == 1:#no batching or plating
        training_function = training_method(DraupnirTrain.train, svi, patristic_matrix_model,cladistic_matrix_full,dataset_train_blosum, train_loader, args)

    else:#batching
        training_function = training_method(DraupnirTrain.train_batch, svi, patristic_matrix_model,
                                            cladistic_matrix_full,dataset_train_blosum, train_loader, args)

    ######################
    ####Training Loop#####
    ######################
    n_train_seqs = dataset_train.shape[0]
    blocks_train = DraupnirModelsUtils.intervals(n_train_seqs // build_config.batch_size, n_train_seqs)
    train_loss = []
    entropy = []
    start_total = time.time()
    epoch = 0
    epoch_count = 0
    added_epochs = 0
    output_file = open("{}/output.log".format(results_dir), "w")
    while epoch < args.num_epochs:
        if check_point_epoch > 0 and epoch > 0 and epoch % check_point_epoch == 0:
            DraupnirPlots.plot_ELBO(train_loss, results_dir)
            DraupnirPlots.plot_entropy(entropy, results_dir)
        start = time.time()
        total_epoch_loss_train = training_function(svi, patristic_matrix_model, cladistic_matrix_full, train_loader,args)

        memory_usage_mib = torch.cuda.max_memory_allocated() * 9.5367 * 1e-7  # convert byte to MiB
        stop = time.time()
        train_loss.append(float(total_epoch_loss_train))  # convert to float because otherwise it's kept in torch's history
        print("[epoch %03d]  average training loss: %.4f %.5g time/epoch %.2f MiB/epoch" % (
        epoch_count, total_epoch_loss_train, stop - start, memory_usage_mib))
        print("[epoch %03d]  average training loss: %.4f %.5g time/epoch %.2f MiB/epoch" % (
        epoch_count, total_epoch_loss_train, stop - start, memory_usage_mib), file=output_file)
        print("Current total time : {}".format(str(datetime.timedelta(seconds=stop - start_total))), file=output_file)

        map_estimates = guide(dataset_train, patristic_matrix_train, cladistic_matrix_train, dataset_train_blosum,batch_blosum=None)  # only saving 1 sample
        map_estimates = {val: key.detach() for val, key in map_estimates.items()}
        sample_out_train = Draupnir.sample_batched(map_estimates,
                                           n_samples,
                                           dataset_train,
                                           patristic_matrix_full,
                                           patristic_matrix_test,
                                           batch_idx=blocks_train[0],#only perform testing on one of the batches
                                           use_argmax=True,
                                           use_test=False,
                                           use_test2=False)
        save_checkpoint(Draupnir,results_dir, optimizer=optim)  # Saves the parameters gradients
        save_checkpoint_guide(guide, results_dir)
        train_entropy_epoch = DraupnirModelsUtils.compute_sites_entropies(sample_out_train.logits.cpu(),
                                                                          dataset_train.cpu().long()[int(blocks_train[0][0]):int(blocks_train[0][1]), 0, 1])
        # percent_id_df, _, _ = extract_percent_id(dataset_train, sample_out_train.aa_sequences, n_samples_dict[folder], results_dir,correspondence_dict)
        if epoch % args.test_frequency == 0:  # every n epochs --- sample
            dill.dump(map_estimates, open('{}/Draupnir_Checkpoints/Map_estimates.p'.format(results_dir), 'wb'),
                      protocol=pickle.HIGHEST_PROTOCOL)
            sample_out_test = Draupnir.sample_batched(map_estimates,
                                              n_samples,
                                              dataset_test,
                                              patristic_matrix_full,
                                              patristic_matrix_test,
                                              batch_idx = blocks_train[0], #we only use the first batch, so we can use the same trick as with the train and only use the "first batch block"
                                              use_argmax=True,
                                              use_test=True,
                                              use_test2=False)
            sample_out_test_argmax = Draupnir.sample_batched(map_estimates,
                                                     n_samples,
                                                     dataset_test,
                                                     patristic_matrix_full,
                                                     patristic_matrix_test,
                                                     batch_idx=blocks_train[0],
                                                     use_argmax=True,
                                                     use_test=True,
                                                     use_test2=False)
            sample_out_train_argmax = Draupnir.sample_batched(map_estimates,
                                                      n_samples,
                                                      dataset_train,
                                                      patristic_matrix_full,
                                                      patristic_matrix_test,
                                                      batch_idx=blocks_train[0],
                                                      use_argmax=True,
                                                      use_test=False,
                                                      use_test2=False)
            sample_out_test2 = Draupnir.sample_batched(map_estimates,
                                               n_samples,
                                               dataset_test,
                                               patristic_matrix_full,
                                               patristic_matrix_test,
                                               batch_idx=blocks_train[0],
                                               use_argmax=True,
                                               use_test=True,
                                               use_test2=False)
            sample_out_test_argmax2 = Draupnir.sample_batched(map_estimates,
                                                      n_samples,
                                                      dataset_test,
                                                      patristic_matrix_full,
                                                      patristic_matrix_test,
                                                      batch_idx=blocks_train[0],
                                                      use_argmax=True,
                                                      use_test=False,
                                                      use_test2=True)

            test_entropies = DraupnirModelsUtils.compute_sites_entropies(sample_out_test.logits.cpu(),
                                                                         patristic_matrix_test.cpu().long()[1:(int(blocks_train[0][1])+1), 0]) #slice the test nodes, use only the first "batch_size" nodes while training
            test_entropies2 = DraupnirModelsUtils.compute_sites_entropies(sample_out_test2.logits.cpu(),
                                                                          patristic_matrix_test.cpu().long()[1:(int(blocks_train[0][1])+1), 0])
            save_samples(dataset_test, patristic_matrix_test, sample_out_test, test_entropies, correspondence_dict,"{}/test_info_dict.torch".format(results_dir + "/Test_Plots"))
            save_samples(dataset_test, patristic_matrix_test, sample_out_test_argmax, test_entropies,correspondence_dict,"{}/test_argmax_info_dict.torch".format(results_dir + "/Test_argmax_Plots"))
            save_samples(dataset_train, patristic_matrix_train, sample_out_train, train_entropy_epoch,
                         correspondence_dict, "{}/train_info_dict.torch".format(results_dir + "/Train_Plots"))
            save_samples(dataset_train, patristic_matrix_train, sample_out_train_argmax, train_entropy_epoch,
                         correspondence_dict,
                         "{}/train_argmax_info_dict.torch".format(results_dir + "/Train_argmax_Plots"))
            save_samples(dataset_test, patristic_matrix_test, sample_out_test2, test_entropies2, correspondence_dict,
                         "{}/test_info_dict2.torch".format(results_dir + "/Test2_Plots"))
            save_samples(dataset_test, patristic_matrix_test, sample_out_test_argmax2, test_entropies2,
                         correspondence_dict,
                         "{}/test2_argmax_info_dict.torch".format(results_dir + "/Test2_argmax_Plots"))
            # Highlight: Freeing memory
            del sample_out_train_argmax
            del sample_out_test
            del sample_out_test_argmax

        del sample_out_train
        entropy.append(torch.mean(train_entropy_epoch[:, 1]).item())
        if epoch == (args.num_epochs - 1):
            DraupnirPlots.plot_ELBO(train_loss, results_dir)
            DraupnirPlots.plot_entropy(entropy, results_dir)
            save_checkpoint(Draupnir, results_dir, optimizer=optim)  # Saves the parameters gradients
            save_checkpoint_guide(guide, results_dir)  # Saves the parameters gradients
            if len(train_loss) > 10 and args.activate_elbo_convergence:
                difference = sum(train_loss[-10:]) / 10 - total_epoch_loss_train
                convergence = [False if difference > 0.5 else True][
                    0]  # Highlight: this works , but what should be the treshold is yet to be determined
                if convergence:
                    break
                else:
                    epoch -= int(args.num_epochs // 3)
                    added_epochs += int(args.num_epochs // 3)
            if len(train_loss) > 10 and args.activate_entropy_convergence:
                difference = sum(entropy[-10:]) / 10 - torch.mean(train_entropy_epoch[:, 1]).item()
                convergence = [False if difference > 0.2 else True][
                    0]  # Highlight: this works , but what should be the threshold is yet to be determined
                if convergence:
                    break
                else:
                    epoch -= int(args.num_epochs // 3)
                    added_epochs += int(args.num_epochs // 3)
        epoch += 1
        epoch_count += 1
        torch.cuda.empty_cache()
    end_total = time.time()
    print('Final timing: {}'.format(str(datetime.timedelta(seconds=end_total - start_total))))
    print("Added epochs : {}".format(added_epochs))
    text_file = open("{}/Hyperparameters_{}_{}epochs.txt".format(results_dir, now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),
                                                                 args.num_epochs), "a")
    text_file.write("Running time: {}\n".format(str(datetime.timedelta(seconds=end_total - start_total))))
    text_file.write("Total epochs (+added epochs): {}\n".format(args.num_epochs + added_epochs))

    if args.select_guide.startswith("variational"):
        pytorch_total_params = sum([val.numel() for param_name, val in pyro.get_param_store().named_parameters() if
                                    val.requires_grad and not param_name.startswith("DRAUPNIRGUIDES.draupnir")])
    else:  # TODO: Investigate again, but seems correct
        pytorch_total_params = sum(val.numel() for param_name, val in pyro.get_param_store().named_parameters() if
                                   val.requires_grad and not param_name.startswith("decoder_attention"))

    text_file.write("Number of parameters: {} \n".format(pytorch_total_params))
    text_file.close()
    print("Final Sampling...using variational guide")
    map_estimates_dict = defaultdict()
    samples_names = ["sample_{}".format(i) for i in range(n_samples)]
    # Highlight: Train storage
    aa_sequences_train_samples = torch.zeros((n_samples, dataset_train.shape[0], dataset_train.shape[1] - 2)).detach()
    latent_space_train_samples = torch.zeros((n_samples, dataset_train.shape[0], int(params_config["z_dim"]))).detach()
    logits_train_samples = torch.zeros((n_samples, dataset_train.shape[0], dataset_train.shape[1] - 2, build_config.aa_probs)).detach()
    # Highlight: Test storage
    aa_sequences_test_samples = torch.zeros((n_samples, patristic_matrix_test[1:].shape[0], dataset_train.shape[1] - 2)).detach()
    latent_space_test_samples = torch.zeros((n_samples, patristic_matrix_test[1:].shape[0], int(params_config["z_dim"]))).detach()
    logits_test_samples = torch.zeros((n_samples, patristic_matrix_test[1:].shape[0], dataset_train.shape[1] - 2,
                                      build_config.aa_probs)).detach()
    n_train_seqs = dataset_train.shape[0]
    n_test_seqs = dataset_test.shape[0]
    print("Sampling is also divided in batches")
    #blocks_train = DraupnirModelsUtils.intervals(n_train_seqs // build_config.batch_size, n_train_seqs)
    assert n_test_seqs - build_config.batch_size + 1 >= blocks_train[-1][0], "Please select a smaller batch size" #TODO: review
    blocks_test = blocks_train.copy()
    blocks_test[-1] = (blocks_test[-1][0],None) #this trick works by re-using blocks train, but this approach is more flexible

    for sample_idx, sample in enumerate(samples_names):
        # print("sample idx {}".format(sample_idx))
        map_estimates = guide(dataset_train, patristic_matrix_train, cladistic_matrix_train,dataset_train_blosum, batch_blosum=None)
        map_estimates_dict[sample] = {val: key.detach() for val, key in map_estimates.items()}
        for batch_idx,batch_idx_test in zip(blocks_train,blocks_test):
            batch_train_sample = Draupnir.sample_batched(map_estimates,
                                                 1,
                                                 dataset_train,
                                                 patristic_matrix_full,
                                                 cladistic_matrix_train, #substitute with something else
                                                 batch_idx=batch_idx,
                                                 use_argmax=False,
                                                 use_test=False,
                                                 use_test2=False)
            aa_sequences_train_samples[sample_idx,int(batch_idx[0]):int(batch_idx[1])] = batch_train_sample.aa_sequences.detach()
            latent_space_train_samples[sample_idx,int(batch_idx[0]):int(batch_idx[1])] = batch_train_sample.latent_space.detach()
            logits_train_samples[sample_idx,int(batch_idx[0]):int(batch_idx[1])] = batch_train_sample.logits.detach()
            del batch_train_sample
            test_sample = Draupnir.sample_batched(map_estimates,
                                                  1,
                                                  dataset_test,
                                                  patristic_matrix_full,
                                                  patristic_matrix_test,
                                                  batch_idx= batch_idx_test,
                                                  use_argmax=False,
                                                  use_test=True,
                                                  use_test2=False)
            if batch_idx[1] is None:#last batch
                aa_sequences_test_samples[sample_idx, int(batch_idx[0]):] = test_sample.aa_sequences.detach()
                latent_space_test_samples[sample_idx, int(batch_idx[0]):] = test_sample.latent_space.detach()
                logits_test_samples[sample_idx, int(batch_idx[0]):] = test_sample.logits.detach()

            else:#last batch
                aa_sequences_test_samples[sample_idx,int(batch_idx[0]):int(batch_idx[1])] = test_sample.aa_sequences.detach()
                latent_space_test_samples[sample_idx,int(batch_idx[0]):int(batch_idx[1])] = test_sample.latent_space.detach()
                logits_test_samples[sample_idx,int(batch_idx[0]):int(batch_idx[1])] = test_sample.logits.detach()
            del test_sample
        del map_estimates
        torch.cuda.empty_cache()

    dill.dump(map_estimates_dict, open('{}/Draupnir_Checkpoints/Map_estimates.p'.format(results_dir), 'wb'))
    sample_out_train = SamplingOutput(aa_sequences=aa_sequences_train_samples,
                                      latent_space=latent_space_train_samples,
                                      logits=logits_train_samples,
                                      phis=None,
                                      psis=None,
                                      mean_phi=None,
                                      mean_psi=None,
                                      kappa_phi=None,
                                      kappa_psi=None)
    sample_out_test = SamplingOutput(aa_sequences=aa_sequences_test_samples,
                                     latent_space=latent_space_test_samples,
                                     logits=logits_test_samples,
                                     phis=None,
                                     psis=None,
                                     mean_phi=None,
                                     mean_psi=None,
                                     kappa_phi=None,
                                     kappa_psi=None)
    warnings.warn("In variational method Test folder results = Test2 folder results = Variational results")
    sample_out_test2 = sample_out_test
    # Highlight: compute majority vote/ Argmax
    sample_out_train_argmax = SamplingOutput(
        aa_sequences=torch.mode(sample_out_train.aa_sequences, dim=0)[0].unsqueeze(0),  # I think is correct
        latent_space=sample_out_train.latent_space[0],  # TODO:Average?
        logits=sample_out_train.logits[0],
        phis=None,
        psis=None,
        mean_phi=None,
        mean_psi=None,
        kappa_phi=None,
        kappa_psi=None)
    sample_out_test_argmax = SamplingOutput(
        aa_sequences=torch.mode(sample_out_test.aa_sequences, dim=0)[0].unsqueeze(0),
        latent_space=sample_out_test.latent_space[0],
        logits=sample_out_test.logits[0],
        phis=None,
        psis=None,
        mean_phi=None,
        mean_psi=None,
        kappa_phi=None,
        kappa_psi=None)
    sample_out_test_argmax2 = sample_out_test_argmax
    # # Highlight: Compute sequences Shannon entropies per site
    train_entropies = DraupnirModelsUtils.compute_sites_entropies(sample_out_train_argmax.logits.cpu(),
                                                                  dataset_train.cpu().long()[:, 0, 1])
    test_entropies = DraupnirModelsUtils.compute_sites_entropies(sample_out_test_argmax.logits.cpu(),
                                                                 patristic_matrix_test.cpu().long()[1:, 0])
    test_entropies2 = DraupnirModelsUtils.compute_sites_entropies(sample_out_test_argmax.logits.cpu(),
                                                                  patristic_matrix_test.cpu().long()[1:, 0])
    # Highlight : save the samples
    save_samples(dataset_test, patristic_matrix_test, sample_out_test, test_entropies, correspondence_dict,
                 "{}/test_info_dict.torch".format(results_dir + "/Test_Plots"))
    save_samples(dataset_test, patristic_matrix_test, sample_out_test_argmax, test_entropies,
                 correspondence_dict,
                 "{}/test_argmax_info_dict.torch".format(results_dir + "/Test_argmax_Plots"))
    save_samples(dataset_test, patristic_matrix_test, sample_out_test2, test_entropies2,
                 correspondence_dict, "{}/test_info_dict2.torch".format(results_dir + "/Test2_Plots"))
    save_samples(dataset_test, patristic_matrix_test, sample_out_test_argmax2, test_entropies2,
                 correspondence_dict,
                 "{}/test2_argmax_info_dict.torch".format(results_dir + "/Test2_argmax_Plots"))
    save_samples(dataset_train, patristic_matrix_train, sample_out_train, train_entropies,
                 correspondence_dict, "{}/train_info_dict.torch".format(results_dir + "/Train_Plots"))
    save_samples(dataset_train, patristic_matrix_train, sample_out_train_argmax, train_entropies,
                 correspondence_dict,
                 "{}/train_argmax_info_dict.torch".format(results_dir + "/Train_argmax_Plots"))

    #Highlight: Concatenate leaves and internal latent space for plotting
    visualize_latent_space(sample_out_train_argmax.latent_space,
                           sample_out_test_argmax.latent_space,
                           patristic_matrix_train,
                           patristic_matrix_test,
                           additional_load,
                           build_config,
                           args,
                           results_dir)
    visualize_latent_space(sample_out_train_argmax.latent_space,
                           sample_out_test_argmax2.latent_space,
                           patristic_matrix_train,
                           patristic_matrix_test,
                           additional_load,
                           build_config,
                           args,
                           "{}/Test2_Plots".format(results_dir))

    if settings_config.one_hot_encoding:
        print("Transforming one-hot back to integers")
        sample_out_train_argmax = transform_to_integers(sample_out_train_argmax,build_config)  # argmax sets directly the aa to the highest logit
        sample_out_test_argmax = transform_to_integers(sample_out_test_argmax,build_config)
        sample_out_test_argmax2 = transform_to_integers(sample_out_test_argmax2,build_config)
        dataset_train = DraupnirUtils.convert_to_integers(dataset_train.cpu(), build_config.aa_prob, axis=2)
        if build_config.leaves_testing:  # TODO: Check that this works
            dataset_test = DraupnirUtils.convert_to_integers(dataset_test.cpu(), build_config.aa_prob,
                                                           axis=2)  # no need to do it with the test of the simulations, never was one hot encoded. Only for testing leaves

    send_to_plot(n_samples,
                     dataset_train,
                     dataset_test,
                     patristic_matrix_test,
                     train_entropies,
                     test_entropies,
                     test_entropies2,
                     sample_out_train,
                     sample_out_train_argmax,
                     sample_out_test, sample_out_test_argmax,
                     sample_out_test2, sample_out_test_argmax2,
                     additional_load, additional_info, build_config, args, results_dir)

def draupnir_train_batch_by_clade(train_load,
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
                   clades_dict=None):
    """Trains Draupnir-OU by performing SVI inference with batched training, where each batch is a clade in the tree #TODO: scale ELBO by batch size
    :param namedtuple train_load : contains several tensors and other data structures related to the training leaves. See see utils.create_dataset for information on the input data format
    :param namedtuple test_load : contains several tensors and other data structures related to the test internal nodes
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
    :param dict clades_dict"""
    print("Batching by clade")
    align_seq_len = build_config.align_seq_len
    if not additional_load.correspondence_dict:
        correspondence_dict = dict(
            zip(list(range(len(additional_load.tree_levelorder_names))), additional_load.tree_levelorder_names))
    else:
        correspondence_dict = additional_load.correspondence_dict
    device = torch.device("cuda" if args.use_cuda else "cpu") #TODO: Check that this works
    blosum = additional_info.blosum.to(device)
    aa_frequencies = additional_load.aa_frequencies.to(device)
    dataset_train = train_load.dataset_train.to(device)
    patristic_matrix_train = train_load.patristic_matrix_train.to(device)
    patristic_matrix_full = additional_load.patristic_matrix_full.to(device)
    patristic_matrix_test = test_load.patristic_matrix_test.to(device)
    dataset_test = test_load.dataset_test.to(device)
    if train_load.cladistic_matrix_train is not None:
        cladistic_matrix_train = train_load.cladistic_matrix_train.to(device)
        cladistic_matrix_test = \
        [test_load.cladistic_matrix_test.to(device) if test_load.cladistic_matrix_test is not None else None][0]
        cladistic_matrix_full = additional_load.cladistic_matrix_full.to(device)
    else:
        cladistic_matrix_train = cladistic_matrix_test = cladistic_matrix_full = None
    nodes_representations_array = additional_info.nodes_representations_array.to(device)
    dgl_graph = additional_info.dgl_graph

    # aa_prob = torch.unique(dataset_train[:, 2:, 0])

    blosum_max, blosum_weighted, variable_score = DraupnirUtils.process_blosum(blosum, aa_frequencies, align_seq_len,
                                                                               build_config.aa_probs)
    dataset_train_blosum = DraupnirUtils.blosum_embedding_encoder(blosum, aa_frequencies, align_seq_len,
                                                                  build_config.aa_probs, dataset_train,
                                                                  settings_config.one_hot_encoding)

    # Highlight: plot the amount of change per position in the alignment
    plt.plot(variable_score.cpu().detach().numpy())
    plt.savefig("{}/Variable_score.png".format(results_dir))
    plt.close()
    plt.clf()


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


    guide = select_quide(Draupnir, model_load, args.select_guide)
    elbo = Trace_ELBO()

    text_file = open("{}/Hyperparameters_{}_{}epochs.txt".format(results_dir, now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),
                                                                 args.num_epochs), "a")
    text_file.write("ELBO :  {} \n".format(str(elbo)))
    text_file.write("Guide :  {} \n".format(str(guide.__class__)))

    # Highlight: Select optimizer/scheduler
    if args.use_scheduler:
        print("Using a learning rate scheduler on top of the optimizer!")
        adam_args = {"lr": params_config["lr"], "betas": (params_config["beta1"], params_config["beta2"]), "eps": params_config["eps"],
                     "weight_decay": params_config["weight_decay"]}
        optim = torch.optim.Adam  # Highlight: For the scheduler we need to use TORCH.optim not PYRO.optim, there is no clipped adam in torch
        # Highlight: "Reduce LR on plateau: Scheduler: Reduce learning rate when a metric has stopped improving."
        optim = pyro.optim.ReduceLROnPlateau({'optimizer': optim, 'optim_args': adam_args})

    else:
        clippedadam_args = {"lr": params_config["lr"], "betas": (params_config["beta1"], params_config["beta2"]), "eps": params_config["eps"],
                            "weight_decay": params_config["weight_decay"], "clip_norm": params_config["clip_norm"],
                            "lrd": params_config["lrd"]}
        optim = pyro.optim.ClippedAdam(clippedadam_args)

    def load_tune_params(load_params):
        """Loading pretrained parameters and allowing to tune them. TODO: not using it"""
        if load_params:
            pyro.clear_param_store()
            tune_folder = "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_PF00400_2021_11_17_22h59min34s252388ms_30000epochs"  # delta map
            # tune_folder = ""
            print("WARNING: Loading pretrained model dict from {} !!!!".format(tune_folder))
            optim_dir = None
            model_dir = "{}/Draupnir_Checkpoints/Model_state_dict.p".format(tune_folder)
            text_file = open(
                "{}/Hyperparameters_{}_{}epochs.txt".format(results_dir, now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),
                                                            args.num_epochs), "a")
            text_file.write("Load pretrained TUNED params Model: {}\n".format(model_dir))
            text_file.write("Load pretrained TUNED params Optim: {}\n".format(optim_dir))
            text_file.close()
            load_checkpoint(model_dict_dir=model_dir, optim_dir=optim_dir, optim=optim, model=Draupnir)
            # Draupnir.train(False)

    load_tune_params(False)

    svi = SVI(Draupnir.model, guide, optim,elbo)  # TODO: TraceMeanField_ELBO() http://docs.pyro.ai/en/0.3.0-release/inference_algos.html#pyro.infer.trace_mean_field_elbo.TraceMeanField_ELBO
    text_file.write("Optimizer :  {} \n".format(optim))

    check_point_epoch = [50 if args.num_epochs < 100 else (args.num_epochs / 100)][0]

    batching_method = ["batch_dim_0" if not args.batch_by_clade else "batch_by_clade"][0]
    train_loader = DraupnirLoadUtils.setup_data_loaders(dataset_train, patristic_matrix_train, clades_dict, blosum,
                                                        build_config, args, method=batching_method,
                                                        use_cuda=args.use_cuda)

    training_method = lambda f, svi, patristic_matrix_model, cladistic_matrix_full, dataset_train_blosum, train_loader,args: lambda svi, patristic_matrix_model, cladistic_matrix_full, train_loader, args: f(svi,
                                                                                                                    patristic_matrix_model,
                                                                                                                    cladistic_matrix_full,
                                                                                                                    dataset_train_blosum,
                                                                                                                    train_loader,
                                                                                                                    args)

    if args.batch_by_clade and clades_dict:
        training_function = training_method(DraupnirTrain.train_batch_clade, svi, patristic_matrix_model,
                                            cladistic_matrix_full, dataset_train_blosum, train_loader, args)
    elif args.batch_size == 1:  # no batching or plating
        training_function = training_method(DraupnirTrain.train, svi, patristic_matrix_model, cladistic_matrix_full,
                                            dataset_train_blosum, train_loader, args)
    else:  # batching
        training_function = training_method(DraupnirTrain.train_batch, svi, patristic_matrix_model,
                                            cladistic_matrix_full, dataset_train_blosum, train_loader, args)

    ######################
    ####Training Loop#####
    ######################
    #n_train_seqs = dataset_train.shape[0]
    #blocks_train = DraupnirModelsUtils.intervals(n_train_seqs // build_config.batch_size, n_train_seqs) #TODO: Fix to have the clades or just take a slice?
    blocks_train = [[value] if isinstance(value,int) else value for value in clades_dict.values()]
    blocks_test = [[value["internal"]] if isinstance(value["internal"],int) else value["internal"] for key, value in additional_load.clades_dict_all.items()]
    train_loss = []
    entropy = []
    start_total = time.time()
    epoch = 0
    epoch_count = 0
    added_epochs = 0
    output_file = open("{}/output.log".format(results_dir), "w")
    while epoch < args.num_epochs:
        if check_point_epoch > 0 and epoch > 0 and epoch % check_point_epoch == 0:
            DraupnirPlots.plot_ELBO(train_loss, results_dir)
            DraupnirPlots.plot_entropy(entropy, results_dir)
        start = time.time()
        total_epoch_loss_train = training_function(svi, patristic_matrix_model, cladistic_matrix_full, train_loader,
                                                   args)
        memory_usage_mib = torch.cuda.max_memory_allocated() * 9.5367 * 1e-7  # convert byte to MiB
        stop = time.time()
        train_loss.append(float(total_epoch_loss_train))  # convert to float because otherwise it's kept in torch's history
        print("[epoch %03d]  average training loss: %.4f %.5g time/epoch %.2f MiB/epoch" % (
        epoch_count, total_epoch_loss_train, stop - start, memory_usage_mib))
        print("[epoch %03d]  average training loss: %.4f %.5g time/epoch %.2f MiB/epoch" % (
        epoch_count, total_epoch_loss_train, stop - start, memory_usage_mib), file=output_file)
        print("Current total time : {}".format(str(datetime.timedelta(seconds=stop - start_total))), file=output_file)

        map_estimates = guide(dataset_train, patristic_matrix_train, cladistic_matrix_train, dataset_train_blosum,
                              batch_blosum=None)  # only saving 1 sample
        map_estimates = {val: key.detach() for val, key in map_estimates.items()}
        sample_out_train = Draupnir.sample_batched(map_estimates,
                                           n_samples,
                                           dataset_train,
                                           patristic_matrix_full,
                                           patristic_matrix_test,
                                           batch_idx=blocks_train[0],#only perform testing on one of the batches
                                           use_argmax=True,
                                           use_test=False,
                                           use_test2=False)

        save_checkpoint(Draupnir, results_dir, optimizer=optim)  # Saves the parameters gradients
        save_checkpoint_guide(guide, results_dir)
        indexes = torch.eq(dataset_train[:,0,1], torch.Tensor(blocks_train[0]))
        train_entropy_epoch = DraupnirModelsUtils.compute_sites_entropies(sample_out_train.logits.cpu(),
                                                                          dataset_train.cpu().long()[indexes, 0, 1])
        # percent_id_df, _, _ = extract_percent_id(dataset_train, sample_out_train.aa_sequences, n_samples_dict[folder], results_dir,correspondence_dict)
        if epoch % args.test_frequency == 0:  # every n epochs --- sample
            dill.dump(map_estimates, open('{}/Draupnir_Checkpoints/Map_estimates.p'.format(results_dir), 'wb'),
                      protocol=pickle.HIGHEST_PROTOCOL)
            sample_out_test = Draupnir.sample_batched(map_estimates,
                                              n_samples,
                                              dataset_test,
                                              patristic_matrix_full,
                                              patristic_matrix_test,
                                              batch_idx = blocks_test[0], #we only use the first batch, so we can use the same trick as with the train and only use the "first batch block"
                                              use_argmax=True,
                                              use_test=True,
                                              use_test2=False)
            sample_out_test_argmax = Draupnir.sample_batched(map_estimates,
                                                     n_samples,
                                                     dataset_test,
                                                     patristic_matrix_full,
                                                     patristic_matrix_test,
                                                     batch_idx=blocks_test[0],
                                                     use_argmax=True,
                                                     use_test=True,
                                                     use_test2=False)
            sample_out_train_argmax = Draupnir.sample_batched(map_estimates,
                                                      n_samples,
                                                      dataset_train,
                                                      patristic_matrix_full,
                                                      patristic_matrix_test,
                                                      batch_idx=blocks_train[0],
                                                      use_argmax=True,
                                                      use_test=False,
                                                      use_test2=False)
            sample_out_test2 = Draupnir.sample_batched(map_estimates,
                                               n_samples,
                                               dataset_test,
                                               patristic_matrix_full,
                                               patristic_matrix_test,
                                               batch_idx=blocks_test[0],
                                               use_argmax=True,
                                               use_test=True,
                                               use_test2=False)
            sample_out_test_argmax2 = Draupnir.sample_batched(map_estimates,
                                                      n_samples,
                                                      dataset_test,
                                                      patristic_matrix_full,
                                                      patristic_matrix_test,
                                                      batch_idx=blocks_test[0],
                                                      use_argmax=True,
                                                      use_test=False,
                                                      use_test2=True)

            indexes_test =  (patristic_matrix_test[:, 0][..., None] == torch.tensor(blocks_test[0])).any(-1)
            #indexes_test[0] = False

            test_entropies = DraupnirModelsUtils.compute_sites_entropies(sample_out_test.logits.cpu(),
                                                                         patristic_matrix_test.cpu().long()[indexes_test, 0]) #slice the test nodes, use only the first "batch_size" nodes while training
            test_entropies2 = DraupnirModelsUtils.compute_sites_entropies(sample_out_test2.logits.cpu(),
                                                                          patristic_matrix_test.cpu().long()[indexes_test,0])
            save_samples(dataset_test, patristic_matrix_test, sample_out_test, test_entropies, correspondence_dict,"{}/test_info_dict.torch".format(results_dir + "/Test_Plots"))
            save_samples(dataset_test, patristic_matrix_test, sample_out_test_argmax, test_entropies,correspondence_dict,"{}/test_argmax_info_dict.torch".format(results_dir + "/Test_argmax_Plots"))
            save_samples(dataset_train, patristic_matrix_train, sample_out_train, train_entropy_epoch,
                         correspondence_dict, "{}/train_info_dict.torch".format(results_dir + "/Train_Plots"))
            save_samples(dataset_train, patristic_matrix_train, sample_out_train_argmax, train_entropy_epoch,
                         correspondence_dict,
                         "{}/train_argmax_info_dict.torch".format(results_dir + "/Train_argmax_Plots"))
            save_samples(dataset_test, patristic_matrix_test, sample_out_test2, test_entropies2, correspondence_dict,
                         "{}/test_info_dict2.torch".format(results_dir + "/Test2_Plots"))
            save_samples(dataset_test, patristic_matrix_test, sample_out_test_argmax2, test_entropies2,
                         correspondence_dict,
                         "{}/test2_argmax_info_dict.torch".format(results_dir + "/Test2_argmax_Plots"))
            # Highlight: Freeing memory
            del sample_out_train_argmax
            del sample_out_test
            del sample_out_test_argmax

        del sample_out_train
        entropy.append(torch.mean(train_entropy_epoch[:, 1]).item())
        if epoch == (args.num_epochs - 1):
            DraupnirPlots.plot_ELBO(train_loss, results_dir)
            DraupnirPlots.plot_entropy(entropy, results_dir)
            save_checkpoint(Draupnir, results_dir, optimizer=optim)  # Saves the parameters gradients
            save_checkpoint_guide(guide, results_dir)  # Saves the parameters gradients
            if len(train_loss) > 10 and args.activate_elbo_convergence:
                difference = sum(train_loss[-10:]) / 10 - total_epoch_loss_train
                convergence = [False if difference > 0.5 else True][0]  # Highlight: this works , but what should be the treshold is yet to be determined
                if convergence:
                    break
                else:
                    epoch -= int(args.num_epochs // 3)
                    added_epochs += int(args.num_epochs // 3)
            if len(train_loss) > 10 and args.activate_entropy_convergence:
                difference = sum(entropy[-10:]) / 10 - torch.mean(train_entropy_epoch[:, 1]).item()
                convergence = [False if difference > 0.2 else True][0]  # Highlight: this works , but what should be the threshold is yet to be determined
                if convergence:
                    break
                else:
                    epoch -= int(args.num_epochs // 3)
                    added_epochs += int(args.num_epochs // 3)
        epoch += 1
        epoch_count += 1
        torch.cuda.empty_cache()
    end_total = time.time()
    print('Final timing: {}'.format(str(datetime.timedelta(seconds=end_total - start_total))))
    print("Added epochs : {}".format(added_epochs))
    text_file = open("{}/Hyperparameters_{}_{}epochs.txt".format(results_dir, now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),
                                                                 args.num_epochs), "a")
    text_file.write("Running time: {}\n".format(str(datetime.timedelta(seconds=end_total - start_total))))
    text_file.write("Total epochs (+added epochs): {}\n".format(args.num_epochs + added_epochs))

    if args.select_guide.startswith("variational"):
        pytorch_total_params = sum([val.numel() for param_name, val in pyro.get_param_store().named_parameters() if
                                    val.requires_grad and not param_name.startswith("DRAUPNIRGUIDES.draupnir")])
    else:  # TODO: Investigate again, but seems correct
        pytorch_total_params = sum(val.numel() for param_name, val in pyro.get_param_store().named_parameters() if
                                   val.requires_grad and not param_name.startswith("decoder_attention"))

    text_file.write("Number of parameters: {} \n".format(pytorch_total_params))
    text_file.close()
    # DraupnirUtils.GradientsPlot(gradient_norms, args.num_epochs, results_dir) #Highlight: Very cpu intensive to compute
    print("Final Sampling....Only variational version available")
    warnings.warn("In variational method Test folder results = Test2 folder results = Variational results")

    map_estimates_dict = defaultdict()
    samples_names = ["sample_{}".format(i) for i in range(n_samples)]
    # Highlight: Train storage
    aa_sequences_train_samples = torch.zeros((n_samples, dataset_train.shape[0], dataset_train.shape[1] - 2)).detach()
    latent_space_train_samples = torch.zeros((n_samples, dataset_train.shape[0], int(params_config["z_dim"]))).detach()
    logits_train_samples = torch.zeros((n_samples, dataset_train.shape[0], dataset_train.shape[1] - 2, build_config.aa_probs)).detach()
    # Highlight: Test storage
    aa_sequences_test_samples = torch.zeros((n_samples, patristic_matrix_test[1:].shape[0], dataset_train.shape[1] - 2)).detach()
    latent_space_test_samples = torch.zeros((n_samples, patristic_matrix_test[1:].shape[0], int(params_config["z_dim"]))).detach()
    logits_test_samples = torch.zeros((n_samples, patristic_matrix_test[1:].shape[0], dataset_train.shape[1] - 2,
                                      build_config.aa_probs)).detach()
    print("Sampling is also divided in batches by clade")

    for sample_idx, sample in enumerate(samples_names):
        map_estimates = guide(dataset_train, patristic_matrix_train, cladistic_matrix_train,dataset_train_blosum, batch_blosum=None)
        map_estimates_dict[sample] = {val: key.detach() for val, key in map_estimates.items()}
        for batch_idx,batch_idx_test in zip(blocks_train,blocks_test): #still valid because the internal nodes and leaves are organize at the same clades
            batch_train_sample = Draupnir.sample_batched(map_estimates,
                                                 1,
                                                 dataset_train,
                                                 patristic_matrix_full,
                                                 cladistic_matrix_train, #substitute with something else
                                                 batch_idx=batch_idx,
                                                 use_argmax=False,
                                                 use_test=False,
                                                 use_test2=False)
            batch_idx = (dataset_train[:, 0,1][..., None] == torch.tensor(batch_idx)).any(-1)
            aa_sequences_train_samples[sample_idx,batch_idx] = batch_train_sample.aa_sequences.double().detach()
            latent_space_train_samples[sample_idx,batch_idx] = batch_train_sample.latent_space.detach()
            logits_train_samples[sample_idx,batch_idx] = batch_train_sample.logits.detach()
            del batch_train_sample
            test_sample = Draupnir.sample_batched(map_estimates,
                                                  1,
                                                  dataset_test,
                                                  patristic_matrix_full,
                                                  patristic_matrix_test,
                                                  batch_idx= batch_idx_test,
                                                  use_argmax=False,
                                                  use_test=True,
                                                  use_test2=False)
            batch_idx_test = (patristic_matrix_test[1:, 0][..., None] == torch.tensor(batch_idx_test)).any(-1)
            aa_sequences_test_samples[sample_idx,batch_idx_test] = test_sample.aa_sequences.double().detach()
            latent_space_test_samples[sample_idx,batch_idx_test] = test_sample.latent_space.detach()
            logits_test_samples[sample_idx,batch_idx_test] = test_sample.logits.detach()
            del test_sample
        del map_estimates
        torch.cuda.empty_cache()

    dill.dump(map_estimates_dict, open('{}/Draupnir_Checkpoints/Map_estimates.p'.format(results_dir), 'wb'))
    sample_out_train = SamplingOutput(aa_sequences=aa_sequences_train_samples,
                                      latent_space=latent_space_train_samples,
                                      logits=logits_train_samples,
                                      phis=None,
                                      psis=None,
                                      mean_phi=None,
                                      mean_psi=None,
                                      kappa_phi=None,
                                      kappa_psi=None)
    sample_out_test = SamplingOutput(aa_sequences=aa_sequences_test_samples,
                                     latent_space=latent_space_test_samples,
                                     logits=logits_test_samples,
                                     phis=None,
                                     psis=None,
                                     mean_phi=None,
                                     mean_psi=None,
                                     kappa_phi=None,
                                     kappa_psi=None)
    sample_out_test2 = sample_out_test
    # Highlight: compute majority vote/ Argmax
    sample_out_train_argmax = SamplingOutput(
        aa_sequences=torch.mode(sample_out_train.aa_sequences, dim=0)[0].unsqueeze(0),  # I think is correct
        latent_space=sample_out_train.latent_space[0],
        logits=sample_out_train.logits[0],
        phis=None,
        psis=None,
        mean_phi=None,
        mean_psi=None,
        kappa_phi=None,
        kappa_psi=None)
    sample_out_test_argmax = SamplingOutput(
        aa_sequences=torch.mode(sample_out_test.aa_sequences, dim=0)[0].unsqueeze(0),
        latent_space=sample_out_test.latent_space[0],
        logits=sample_out_test.logits[0],
        phis=None,
        psis=None,
        mean_phi=None,
        mean_psi=None,
        kappa_phi=None,
        kappa_psi=None)
    sample_out_test_argmax2 = sample_out_test_argmax
    # # Highlight: Compute sequences Shannon entropies per site
    train_entropies = DraupnirModelsUtils.compute_sites_entropies(sample_out_train_argmax.logits.cpu(),
                                                                  dataset_train.cpu().long()[:, 0, 1])
    test_entropies = DraupnirModelsUtils.compute_sites_entropies(sample_out_test_argmax.logits.cpu(),
                                                                 patristic_matrix_test.cpu().long()[1:, 0])
    test_entropies2 = DraupnirModelsUtils.compute_sites_entropies(sample_out_test_argmax.logits.cpu(),
                                                                  patristic_matrix_test.cpu().long()[1:, 0])
    # Highlight : save the samples
    save_samples(dataset_test, patristic_matrix_test, sample_out_test, test_entropies, correspondence_dict,
                 "{}/test_info_dict.torch".format(results_dir + "/Test_Plots"))
    save_samples(dataset_test, patristic_matrix_test, sample_out_test_argmax, test_entropies,
                 correspondence_dict,
                 "{}/test_argmax_info_dict.torch".format(results_dir + "/Test_argmax_Plots"))
    save_samples(dataset_test, patristic_matrix_test, sample_out_test2, test_entropies2,
                 correspondence_dict, "{}/test_info_dict2.torch".format(results_dir + "/Test2_Plots"))
    save_samples(dataset_test, patristic_matrix_test, sample_out_test_argmax2, test_entropies2,
                 correspondence_dict,
                 "{}/test2_argmax_info_dict.torch".format(results_dir + "/Test2_argmax_Plots"))
    save_samples(dataset_train, patristic_matrix_train, sample_out_train, train_entropies,
                 correspondence_dict, "{}/train_info_dict.torch".format(results_dir + "/Train_Plots"))
    save_samples(dataset_train, patristic_matrix_train, sample_out_train_argmax, train_entropies,
                 correspondence_dict,
                 "{}/train_argmax_info_dict.torch".format(results_dir + "/Train_argmax_Plots"))


    # Highlight: Concatenate leaves and internal latent space for plotting
    visualize_latent_space(sample_out_train_argmax.latent_space,
                           sample_out_test_argmax.latent_space,
                           patristic_matrix_train,
                           patristic_matrix_test,
                           additional_load,
                           build_config,
                           args,
                           results_dir)
    visualize_latent_space(sample_out_train_argmax.latent_space,
                           sample_out_test_argmax2.latent_space,
                           patristic_matrix_train,
                           patristic_matrix_test,
                           additional_load,
                           build_config,
                           args,
                           "{}/Test2_Plots".format(results_dir))

    if settings_config.one_hot_encoding:
        print("Transforming one-hot back to integers")
        sample_out_train_argmax = transform_to_integers(sample_out_train_argmax,build_config)  # argmax sets directly the aa to the highest logit
        sample_out_test_argmax = transform_to_integers(sample_out_test_argmax,build_config)
        sample_out_test_argmax2 = transform_to_integers(sample_out_test_argmax2,build_config)
        dataset_train = DraupnirUtils.convert_to_integers(dataset_train.cpu(), build_config.aa_prob, axis=2)
        if build_config.leaves_testing:  # TODO: Check that this works
            dataset_test = DraupnirUtils.convert_to_integers(dataset_test.cpu(), build_config.aa_prob,axis=2)  # no need to do it with the test of the simulations, never was one hot encoded. Only for testing leaves

    send_to_plot(n_samples,
                 dataset_train,
                 dataset_test,
                 patristic_matrix_test,
                 train_entropies,
                 test_entropies,
                 test_entropies2,
                 sample_out_train,
                 sample_out_train_argmax,
                 sample_out_test, sample_out_test_argmax,
                 sample_out_test2, sample_out_test_argmax2,
                 additional_load, additional_info, build_config, args, results_dir)
def send_to_plot(n_samples,
                 dataset_train,
                 dataset_test,
                 patristic_matrix_test,
                 train_entropies,
                 test_entropies,
                 test_entropies2,
                 sample_out_train,
                 sample_out_train_argmax,
                 sample_out_test,sample_out_test_argmax,
                 sample_out_test2,sample_out_test_argmax2,
                 additional_load,additional_info,build_config,args,results_dir):
    """Starts the plotting engine
    :param int n_samples: number of sequences samples
    :param torch-tensor dataset_train  : [n_seq_train, align_len + 2, 30]
    :param torch-tensor dataset_test : [n_seq_test, align_len + 2, 30]
    :param torch-tensor patristic_matrix_test: [n_seq_test + 1, n_seq_test + 1] ; row 0 and column 0 are the node's names
    :param torch-tensor train_entropies
    :param torch-tensor test_entropies
    :param torch-tensor test_entropies2
    :param namedtuple sample_out_train : model sampling output for the train/leaf sampled sequences
    :param namedtuple sample_out_train_argmax : model sampling output for the train/leaf MAP sequences
    :param namedtuple sample_out_test: model sampling output for the test/internal sequences
    :param namedtuple sample_out_test_argmax: model sampling output for the test/internal Most Voted Sequence or MAP sequences
    :param namedtuple sample_out_test2: model sampling output for the test/internal sampled sequences
    :param namedtuple sample_out_test_argmax2: model sampling output for the test/internal Most Voted Sequence or MAP sequences
    :param namedtuple additional_load
    :param namedtuple additional_info
    :param namedtuple build_config
    :param str results_dir"""
    start_plots = time.time()
    # aa_sequences_predictions_test = dataset_test[:,2:,0].repeat(n_samples,1,1)
    # aa_sequences_predictions_train = dataset_train[:, 2:, 0].repeat(n_samples, 1, 1)
    #
    # sample_out_train = SamplingOutput(
    #     aa_sequences=aa_sequences_predictions_train,
    #     latent_space=sample_out_train.latent_space,
    #     logits=sample_out_train.logits,
    #     phis=None,
    #     psis=None,
    #     mean_phi=None,
    #     mean_psi=None,
    #     kappa_phi=None,
    #     kappa_psi=None)
    # sample_out_test = SamplingOutput(
    #     aa_sequences=aa_sequences_predictions_test,
    #     latent_space=sample_out_test.latent_space,
    #     logits=sample_out_test.logits,
    #     phis=None,
    #     psis=None,
    #     mean_phi=None,
    #     mean_psi=None,
    #     kappa_phi=None,
    #     kappa_psi=None)

    if n_samples != sample_out_test.aa_sequences.shape[0]:
        n_samples = sample_out_test.aa_sequences.shape[0]

    if args.infer_angles:  # TODO: not correct anymore, gotta fix
        preparing_plots(sample_out_train,
                        dataset_train,
                        dataset_train,
                        train_entropies,
                        results_dir + "/Train_Plots",
                        additional_load,
                        additional_info,
                        build_config,
                        n_samples,
                        dataset_train[:, 0, 1],
                        args,
                        replacement_plots=False,
                        plot_test=False,
                        plot_angles=True,
                        no_testing=True)
        preparing_plots(sample_out_test,
                        dataset_test,
                        dataset_train,
                        test_entropies,
                        results_dir + "/Test_Plots",
                        additional_load,
                        additional_info,
                        build_config,
                        n_samples,
                        patristic_matrix_test[1:, 0],
                        args,
                        replacement_plots=False,
                        overplapping_hist=False,
                        plot_angles=True,
                        no_testing=build_config.no_testing)
        preparing_plots(sample_out_test2,
                        dataset_test,
                        dataset_train,
                        test_entropies2,
                        results_dir + "/Test2_Plots",
                        additional_load,
                        additional_info,
                        build_config,
                        n_samples,
                        patristic_matrix_test[1:, 0],
                        args,
                        replacement_plots=False,
                        overplapping_hist=False,
                        plot_angles=True,
                        no_testing=build_config.no_testing)

    # Highlight: Plot samples
    print("train")
    preparing_plots(sample_out_train,
                    dataset_train,
                    dataset_train,
                    train_entropies,
                    results_dir + "/Train_Plots",
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
    print("test")
    preparing_plots(sample_out_test,
                    dataset_test,
                    dataset_train,
                    test_entropies,
                    results_dir + "/Test_Plots",
                    additional_load,
                    additional_info,
                    build_config,
                    n_samples,
                    patristic_matrix_test[1:, 0],
                    args,
                    replacement_plots=False,
                    overplapping_hist=False,
                    no_testing=build_config.no_testing)
    print("test2")
    preparing_plots(sample_out_test2,
                    dataset_test,
                    dataset_train,
                    test_entropies2,
                    results_dir + "/Test2_Plots",
                    additional_load,
                    additional_info,
                    build_config,
                    n_samples,
                    patristic_matrix_test[1:, 0],
                    args,
                    replacement_plots=False,
                    overplapping_hist=False,
                    no_testing=build_config.no_testing)

    if n_samples != sample_out_test_argmax.aa_sequences.shape[0]:  # most likely sequences ---> most voted sequence now?
        n_samples = sample_out_test_argmax.aa_sequences.shape[0]

    # Highlight: Plot most likely sequence
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
    print("test argmax")
    preparing_plots(sample_out_test_argmax,
                    dataset_test,
                    dataset_train,
                    test_entropies,
                    results_dir + "/Test_argmax_Plots",
                    additional_load,
                    additional_info,
                    build_config,
                    n_samples,
                    patristic_matrix_test[1:, 0],
                    args,
                    replacement_plots=False,
                    overplapping_hist=False,
                    no_testing=build_config.no_testing)
    print("test_argmax 2")
    preparing_plots(sample_out_test_argmax2,
                    dataset_test,
                    dataset_train,
                    test_entropies2,
                    results_dir + "/Test2_argmax_Plots",
                    additional_load,
                    additional_info,
                    build_config,
                    n_samples,
                    patristic_matrix_test[1:, 0],
                    args,
                    replacement_plots=False,
                    overplapping_hist=False,
                    no_testing=build_config.no_testing)

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
    name= args.dataset_name
    if not additional_load.correspondence_dict:
        correspondence_dict = dict(zip(list(range(len(additional_load.tree_levelorder_names))), additional_load.tree_levelorder_names))
    else:
        correspondence_dict = additional_load.correspondence_dict
    if plot_test and no_testing:
        print("Print no ancestral sequence testing!")
        DraupnirPlots.save_ancestors_predictions(name, dataset_true, samples_out.aa_sequences, n_samples, results_dir,
                                   correspondence_dict, build_config.aa_probs)
    elif plot_test and name in ["Douglas_SRC","Coral_Faviina","Coral_all"] or plot_test and name.endswith("_subtree"):
        #Highlight: select from the predictions only the sequences in the dataset_test. Remove gaps and align to the "observed"
        DraupnirPlots.save_ancestors_predictions_coral(name, test_ordered_nodes, samples_out.aa_sequences, n_samples, results_dir,
                                                 correspondence_dict, build_config.aa_probs)
        DraupnirPlots.clean_and_realign_train(name,
                                   dataset_true, #in this case it will always be the dataset_test with the ancestral sequences , because plot_test
                                   dataset_train,
                                   samples_out.aa_sequences,  # test predictions
                                   test_ordered_nodes,
                                   n_samples,
                                   build_config.aa_probs,
                                   results_dir,
                                   additional_load,
                                   additional_info)
        DraupnirPlots.clean_and_realign_test(name,
                                   dataset_true, #in this case it will always be the dataset_test with the ancestral sequences , because plot_test
                                   samples_out.aa_sequences,#test predictions
                                   test_ordered_nodes,
                                   n_samples,
                                   build_config.aa_probs,
                                   results_dir,
                                   additional_load,
                                   additional_info)



        DraupnirPlots.plot_entropies(name, entropies.detach().cpu().numpy(), results_dir, correspondence_dict)
    elif args.infer_angles and plot_angles:
        DraupnirPlots.plot_angles(samples_out,dataset_true,results_dir,additional_load,additional_info,n_samples,test_ordered_nodes)
        DraupnirPlots.plot_angles_per_aa(samples_out,dataset_true,results_dir,build_config,additional_load,additional_info,n_samples,test_ordered_nodes)
    else:
        DraupnirPlots.plot_heatmap_and_incorrect_aminoacids(name,
                                                                dataset_true,
                                                                samples_out.aa_sequences,
                                                                n_samples,
                                                                results_dir,
                                                                correspondence_dict,
                                                                build_config.aa_probs,
                                                                additional_load,
                                                                additional_info,
                                                                replacement_plots)
        DraupnirPlots.plot_entropies(name, entropies.detach().cpu().numpy(), results_dir, correspondence_dict)
        if overplapping_hist: #TODO: Fix or remove?
            DraupnirPlots.plot_overlapping_histogram(name,
                                                dataset_train,
                                                dataset_true,
                                                samples_out.aa_sequences,
                                                n_samples,
                                                results_dir,
                                                correspondence_dict,
                                                build_config.aa_probs)
def generate_config():
    """Generate random parameter values"""
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
    """Select a default configuration dictionary. It can load a string dictionary from the command line (using json) or use the default parameters
    :param namedtuple args"""
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
            "gru_hidden_dim": 60, #60
        }
    return config
def manual_random_search(): #TODO: This probably does not work
    """Performs random grid search in the hyperparameter space"""
    #sys.stdout = open('Random_Search_results.txt', 'w')
    # if click.confirm('Do you want to delete previous files? Otherwise results will be appended', default=True):
    #     print("Deleting previous run ...")
    #     os.remove("{}/Random_Search_results.txt".format(script_dir))

    global config
    n_runs = 20
    for i in range(n_runs):
        config = generate_config()
        print(config)
        proc= subprocess.Popen(args=[sys.executable,"Draupnir_example.py","--parameter-search","True","--config-dict",str(config).replace("'", '"')],stdout=open('Random_Search_results.txt', 'a')) #stdout=open(os.devnull, 'wb'),stderr=open(os.devnull, 'wb')
        proc.communicate()
def run(name,root_sequence_name,args,device,settings_config,build_config,script_dir):
    """Loads and pre-treats the data for inference, executes Draupnir model for training or for sampling.
    :param str name
    :param str root_sequence_name
    :param namedtuple args
    :param device: torch device
    :param namedtuple settings_config
    :param namedtuple build_config
    :param str script_dir
    """
    results_dir = "{}/PLOTS_Draupnir_{}_{}_{}epochs_{}".format(script_dir, name, now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),args.num_epochs, args.select_guide)

    print("Loading datasets....")
    param_config = config_build(args)
    train_load,test_load,additional_load,build_config = load_data(name,settings_config,build_config,param_config,results_dir,script_dir,args)
    additional_info=DraupnirUtils.extra_processing(additional_load.ancestor_info_numbers, additional_load.patristic_matrix_full,results_dir,args,build_config)
    train_load,test_load,additional_load= DraupnirLoadUtils.datasets_pretreatment(name,root_sequence_name,train_load,test_load,additional_load,build_config,device,settings_config,script_dir)
    torch.save(torch.get_rng_state(),"{}/rng_key.torch".format(results_dir))
    if args.one_hot_encoded:
        raise ValueError("Please set one_hot_encoding to False")
    print("Starting Draupnir ...")
    print("Dataset: {}".format(name))
    print("Number epochs: {}".format(args.num_epochs))
    print("Z/latent Size: {}".format(param_config["z_dim"]))
    print("GRU hidden size: {}".format(param_config["gru_hidden_dim"]))
    print("Number train sequences: {}".format(train_load.dataset_train.shape[0]))
    n_test = [test_load.dataset_test.shape[0] if test_load.dataset_test is not None else 0][0]
    print("Number test sequences: {}".format(n_test))
    print("Selected Substitution matrix : {}".format(args.subs_matrix))

    if not args.batch_by_clade:
        clades_dict=None
    else:
        clades_dict = additional_load.clades_dict_leaves
    graph_coo = None #Highlight: use only with the GNN models (7)---> Otherwise it is found in additional_info
    #graph_coo = additional_info.graph_coo
    if args.generate_samples: #TODO: generate samples by batch for large data sets
        print("Generating samples not training!")
        draupnir_sample(train_load,
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
    elif args.batch_size == None or args.batch_size > 1:
        print("Batching, splits the OU stochastic process, no guarantee on latent space with tree structure")
        if args.batch_by_clade:
            draupnir_train_batch_by_clade(train_load,
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
        else:
            draupnir_train_batching(train_load,
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
    #TODO: draupnir_tuning
    else:
        print("Training Draupnir with the entire tree at once, not batching")
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





