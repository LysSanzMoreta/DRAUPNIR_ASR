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
import Draupnir_utils as DraupnirUtils
import Draupnir_VAEmodels as DraupnirModels
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
SettingsConfig = namedtuple("SettingsConfig",["one_hot_encoding", "model_design","aligned_seq","uniprot"])
ModelLoad = namedtuple("ModelLoad",["z_dim","max_seq_len","device","args","build_config","leaves_nodes","n_tree_levels","gru_hidden_dim","pretrained_params","aa_frequencies","blosum",
                                    "blosum_max","blosum_weighted","variable_score","internal_nodes","graph_coo","nodes_representations_array","dgl_graph","children_dict",
                                    "closest_leaves_dict","descendants_dict","clades_dict_all","leaves_testing"])
BuildConfig = namedtuple('BuildConfig',['alignment_file','use_ancestral','n_test','build_graph',"aa_prob","triTSNE","max_seq_len","leaves_testing","batch_size","plate_subsample_size","script_dir"])

SamplingOutput = namedtuple("SamplingOutput",["aa_sequences","latent_space","logits","phis","psis","mean_phi","mean_psi","kappa_phi","kappa_psi"])

def load_data(settings_config,build_config):
    """return
    Train and Test Datasets arrays: [n_seqs, max_len + 2, 30]
        For dimension 2:
            0 column = [seq_len, position in tree, distance to root,ancestor, ..., 0]
            1 column = Git vector (30 dim) if available
            Rest of columns = [integer or one hot encoded amino acid sequence]"""

    aligned = ["aligned" if settings_config.aligned_seq else "NOT_aligned"]
    one_hot = ["OneHotEncoded" if settings_config.one_hot_encoding else "Integers"]
    uniprot = ["_UNIPROT" if settings_config.uniprot else ""]

    dataset = np.load("{}/Datasets_Folder/Dataset_numpy_{}_{}_{}{}.npy".format(script_dir,aligned[0], one_hot[0], name, uniprot[0]),allow_pickle=True)
    DraupnirUtils.folders(ntpath.basename(RESULTS_DIR),script_dir)
    DraupnirUtils.folders(("{}/Tree_Alignment_Sampled/".format(ntpath.basename(RESULTS_DIR))),script_dir)
    DraupnirUtils.folders(("{}/ReplacementPlots_Train/".format(ntpath.basename(RESULTS_DIR))), script_dir)
    DraupnirUtils.folders(("{}/ReplacementPlots_Test/".format(ntpath.basename(RESULTS_DIR))), script_dir)
    DraupnirUtils.folders(("{}/Train_Plots/".format(ntpath.basename(RESULTS_DIR))), script_dir)
    DraupnirUtils.folders(("{}/Test_Plots/".format(ntpath.basename(RESULTS_DIR))), script_dir)
    DraupnirUtils.folders(("{}/Test2_Plots/".format(ntpath.basename(RESULTS_DIR))), script_dir)
    DraupnirUtils.folders(("{}/Test_argmax_Plots/".format(ntpath.basename(RESULTS_DIR))), script_dir)
    DraupnirUtils.folders(("{}/Test2_argmax_Plots/".format(ntpath.basename(RESULTS_DIR))), script_dir)
    DraupnirUtils.folders(("{}/Train_argmax_Plots/".format(ntpath.basename(RESULTS_DIR))), script_dir)
    DraupnirUtils.folders(("{}/Draupnir_Checkpoints/".format(ntpath.basename(RESULTS_DIR))), script_dir)
    if args.infer_angles:
        DraupnirUtils.folders(("{}/Train_Plots/Angles_plots_per_aa/".format(ntpath.basename(RESULTS_DIR))), script_dir)
        #DraupnirUtils.folders(("{}/Train_argmax_Plots/Angles_plots_per_aa/".format(ntpath.basename(RESULTS_DIR))), script_dir)
        DraupnirUtils.folders(("{}/Test_Plots/Angles_plots_per_aa/".format(ntpath.basename(RESULTS_DIR))), script_dir)
        #DraupnirUtils.folders(("{}/Test_argmax_Plots/Angles_plots_per_aa/".format(ntpath.basename(RESULTS_DIR))), script_dir)
        DraupnirUtils.folders(("{}/Test2_Plots/Angles_plots_per_aa/".format(ntpath.basename(RESULTS_DIR))), script_dir)
        #DraupnirUtils.folders(("{}/Test2_argmax_Plots/Angles_plots_per_aa/".format(ntpath.basename(RESULTS_DIR))), script_dir)
    dataset = DraupnirLoadUtils.remove_nan(dataset)
    DraupnirUtils.Ramachandran_plot(dataset[:, 3:], "{}/TRAIN_OBSERVED_angles".format(RESULTS_DIR + "/Train_Plots"), "Train Angles",one_hot_encoded=settings_config.one_hot_encoding)
    if build_config.alignment_file:
        alignment = AlignIO.read(build_config.alignment_file, "fasta")
        alignment_file = build_config.alignment_file
        alignment_array =np.array(alignment)
        gap_positions = np.where(alignment_array == "-")[1]
        np.save("{}/Alignment_gap_positions.npy".format(RESULTS_DIR),gap_positions)
    else:
        alignment = AlignIO.read("{}/Mixed_info_Folder/{}.mafft".format(script_dir,name), "fasta")
        alignment_file = "{}/Mixed_info_Folder/{}.mafft".format(script_dir,name)
        align_array = np.array(alignment)
        gap_positions = np.where(align_array == "-")[1]
        np.save("{}/Alignment_gap_positions.npy".format(RESULTS_DIR), gap_positions)

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
                               aa_prob=aa_probs_updated,
                               triTSNE=False,
                               max_seq_len=alignment_length,
                               leaves_testing=build_config.leaves_testing,
                               batch_size = args.batch_size,
                               plate_subsample_size=args.plating_size,
                               script_dir=build_config.script_dir)

    # Highlight: Correction of the batch size in case loaded dataset is splitted. n_test indicates the percege of sequences from the train to be used as test
    if build_config.n_test > 0:
        subtracted = int(n_seq*build_config.n_test/100)
    else:
        subtracted = 0


    batch_size = [DraupnirUtils.Define_batch_size(n_seq-subtracted) if not args.batch_size else args.batch_size if args.batch_size > 1 else n_seq-subtracted][0]
    plate_size = [DraupnirUtils.Define_batch_size(n_seq-subtracted) if not  args.plating_size and args.plating else args.plating_size][0]
    if not args.plating: assert plate_size == None, "Please set plating_size to None if you do not want to do plate subsampling"
    if args.plating: assert args.batch_size == 1, "We are plating, no batching, please set batch_size == 1"

    build_config = BuildConfig(alignment_file=alignment_file,
                               use_ancestral=build_config.use_ancestral,
                               n_test=build_config.n_test,
                               build_graph=build_config.build_graph,
                               aa_prob=aa_probs_updated,
                               triTSNE=False,
                               max_seq_len=alignment_length,
                               leaves_testing=build_config.leaves_testing,
                               batch_size=batch_size,
                               plate_subsample_size=plate_size,
                               script_dir=build_config.script_dir)


    def hyperparameters():
        text_file = open("{}/Hyperparameters_{}_{}epochs.txt".format(RESULTS_DIR, now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),args.num_epochs), "a")
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
        text_file.write("Learning Rate: {} \n".format(config["lr"]))
        text_file.write("Z dimension: {} \n".format(config["z_dim"]))
        text_file.write("GRU hidden size: {} \n".format(config["gru_hidden_dim"]))
        text_file.write("Kappa addition: {} \n".format(args.kappa_addition))
        text_file.write("Amino acid possibilities + gap: {} \n".format(build_config.aa_prob))
        text_file.write("Substitution matrix : {} \n".format(args.subs_matrix))
        text_file.write("Batch by clade : {} \n".format(args.batch_by_clade))
        text_file.write("Batch size (=1 means entire dataset): {} \n".format(batch_size))
        text_file.write("Plating (subsampling) : {} \n".format(args.plating))
        text_file.write("Plating size : {} \n".format(build_config.plate_subsample_size))
        text_file.write("Inferring angles : {} \n".format(args.infer_angles))
        text_file.write("Leaves testing (uses the full leaves latent space (NOT a subset)): {} \n".format(build_config.leaves_testing))
        text_file.write(str(config) + "\n")

    hyperparameters()
    patristic_matrix = pd.read_csv("{}/Datasets_Folder/Patristic_distance_matrix_{}.csv".format(script_dir, name), low_memory=False)
    patristic_matrix = patristic_matrix.rename(columns={'Unnamed: 0': 'rows'})
    patristic_matrix.set_index('rows',inplace=True)
    try:
        cladistic_matrix = pd.read_csv("{}/Datasets_Folder/Cladistic_distance_matrix_{}.csv".format(script_dir,name), index_col="rows",low_memory=False)
    except: #Highlight: For larger datasets , I do not calculate the cladistic matrix, because there is not a fast method. So no cladistic matrix and consequently , no patrocladistic matrix = evolutionary matrix
        cladistic_matrix = None

    ancestor_info = pd.read_csv("{}/Datasets_Folder/Tree_LevelOrderInfo_{}_{}.csv".format(script_dir,one_hot[0], name), sep="\t",index_col=False,low_memory=False)
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

    leaves_names_list = pickle.load(open('{}/Mixed_info_Folder/leafs_names_list_{}.p'.format(script_dir,name),"rb"))

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
        RESULTS_DIR,
        Dataset,
        patristic_matrix,
        cladistic_matrix,
        sorted_distance_matrix,
        n_seq,
        build_config.n_test,
        now,
        name,
        build_config.aa_prob,
        leaves_names_list,
        nodes=tree_levelorder_names,
        ancestral=build_config.use_ancestral,
        one_hot_encoding=settings_config.one_hot_encoding)

    if dataset_test is not None:#Highlight: Dataset_test != None only when the test dataset is extracted from the train (testing leaves)
        DraupnirUtils.Ramachandran_plot(dataset_test[:, 2:], "{}/TEST_OBSERVED_angles".format(RESULTS_DIR+"/Test_Plots"),"Test Angles", one_hot_encoded=settings_config.one_hot_encoding)
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
    text_file = open("{}/Hyperparameters_{}_{}epochs.txt".format(RESULTS_DIR, now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),args.num_epochs), "a")
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
                   full_name=full_name)
    return train_load,test_load,additional_load, build_config
def save_checkpoint(Draupnir,save_directory, optimizer):
    '''Saves the model and optimizer dict states to disk'''
    save_directory = ("{}/Draupnir_Checkpoints/".format(ntpath.basename(save_directory)))
    optimizer.save(save_directory + "/Optimizer_state.p")
    torch.save(Draupnir.state_dict(), save_directory + "/Model_state_dict.p")
def load_checkpoint(model_dict_dir,optim_dir,optim,model):
    '''Loads the model and optimizer states from disk'''
    if model_dict_dir is not None:
        print("Loading pretrained Model parameters...")
        model.load_state_dict(torch.load(model_dict_dir), strict=False)
    if optim_dir is not None:
        print("Loading pretrained Optimizer states...")
        optim.load(optim_dir)
def datasets_pretreatment(train_load,test_load,additional_load,build_config,device,name,simulation_folder,root_sequence_name,dataset_number,settings_config):
    """Loads "external" test datasets, depending on the dataset"""
    #TODO: Move to DraupnirLoadutils?
    #Highlight: Loading for special test datasets
    if name.startswith("simulation"):
        dataset_test,internal_names_test,max_len_test = DraupnirUtils.SimulationTest_Load(settings_config.aligned_seq,
                                                                                        max_seq_len,
                                                                                        additional_load.tree_levelorder_names,
                                                                                        simulation_folder,
                                                                                        root_sequence_name,
                                                                                        dataset_number,
                                                                                        build_config.aa_prob,
                                                                                        script_dir)

        test_nodes_observed = dataset_test[:, 0, 1].tolist()
        test_nodes = torch.tensor(test_nodes_observed, device="cpu")
        patristic_matrix_full = additional_load.patristic_matrix_full
        cladistic_matrix_full = additional_load.cladistic_matrix_full
        vals, idx = torch.sort(test_nodes)
        test_nodes = test_nodes[idx]
        dataset_test = dataset_test[idx]
        test_indx_patristic = (patristic_matrix_full[:, 0][..., None] == test_nodes).any(-1)
        test_indx_patristic[0] = True  # To re-add the node names
        patristic_matrix_test = patristic_matrix_full[test_indx_patristic]
        patristic_matrix_test = patristic_matrix_test[:, test_indx_patristic]
        if cladistic_matrix_full is not None:
            cladistic_matrix_test = cladistic_matrix_full[test_indx_patristic]
            cladistic_matrix_test = cladistic_matrix_test[:, test_indx_patristic]
        else:
            cladistic_matrix_test = None

        correspondence_dict = special_nodes_dict=None


    elif name.startswith("benchmark"):
        dataset_test, internal_names_test = DraupnirUtils.BenchmarkTest_Load(name,test_load.dataset_test)
        test_nodes_observed =  dataset_test[:, 0, 1].tolist()
        special_nodes_dict=None
        if name == "benchmark_randall":  # we need to get the corresponding nodes in order to extract the right patristic distances
            patristic_matrix_train, \
            patristic_matrix_test, \
            cladistic_matrix_train, \
            cladistic_matrix_test, \
            dataset_test, \
            dataset_train,\
            correspondence_dict = DraupnirLoadUtils.pretreatment_Benchmark(dataset_test,
                                                                             train_load.dataset_train,
                                                                             additional_load.patristic_matrix_full,
                                                                             additional_load.cladistic_matrix_full,
                                                                             test_nodes_observed,
                                                                             device,
                                                                             inferred=True)
        if name == "benchmark_randall_original":
            patristic_matrix_train, \
            patristic_matrix_test, \
            cladistic_matrix_train, \
            cladistic_matrix_test, \
            dataset_test, \
            dataset_train,\
            correspondence_dict = DraupnirLoadUtils.pretreatment_Benchmark(dataset_test,
                                                                             train_load.dataset_train,
                                                                             additional_load.patristic_matrix_full,
                                                                             additional_load.cladistic_matrix_full,
                                                                             test_nodes_observed,
                                                                             device, inferred=False)
        if name == "benchmark_randall_original_naming":
            patristic_matrix_train, \
            patristic_matrix_test, \
            cladistic_matrix_train, \
            cladistic_matrix_test, \
            dataset_test,\
            dataset_train,\
            correspondence_dict = DraupnirLoadUtils.pretreatment_Benchmark(dataset_test,
                                                                             train_load.dataset_train,
                                                                             additional_load.patristic_matrix_full,
                                                                             additional_load.cladistic_matrix_full,
                                                                             test_nodes_observed,
                                                                             device, inferred=False,
                                                                             original_naming=True)

    elif name in ["Douglas_SRC","ANC_A1_subtree","ANC_A2_subtree","ANC_AS_subtree","ANC_S1_subtree"]:

        dataset_test, internal_names_test , max_lenght_internal_aligned,special_nodes_dict = DraupnirUtils.SRCKinasesDatasetTest(name,
                                                                                                                                 ancestral_file="{}/Douglas_SRC_Dataset/Ancs.fasta".format(script_dir),
                                                                                                                                 script_dir=script_dir,
                                                                                                                                 tree_level_order_names=additional_load.tree_levelorder_names)

        patristic_matrix_full = additional_load.patristic_matrix_full
        cladistic_matrix_full = additional_load.cladistic_matrix_full
        train_nodes = train_load.dataset_train[:, 0, 1]
        train_indx_patristic = (patristic_matrix_full[:, 0][..., None] == train_nodes).any(-1)
        # train_indx_patristic[0] = True  # To re-add the node names ---> not necessary because we do the opposite, we keep the False ones
        patristic_matrix_test = patristic_matrix_full[~train_indx_patristic]
        patristic_matrix_test = patristic_matrix_test[:, ~train_indx_patristic]
        vals, idx = torch.sort(patristic_matrix_test[:,0])  # unnecessary but leave in case we predict all fav and all coral at the same time
        patristic_matrix_test = patristic_matrix_test[idx] #sort just in case
        if cladistic_matrix_full is not None:
            cladistic_matrix_test = cladistic_matrix_full[~train_indx_patristic]
            cladistic_matrix_test = cladistic_matrix_test[:, ~train_indx_patristic]
            cladistic_matrix_test = cladistic_matrix_test[idx] #sort
        else:
            cladistic_matrix_test = None
        correspondence_dict=None


    elif name in ["Coral_Faviina","Coral_all","Cnidarian"]:
        dataset_test, \
        internal_names_test , \
        max_lenght_internal_aligned,\
        special_nodes_dict =DraupnirUtils.CFPTest(name = name,
                                                         ancestral_file="{}/GPFCoralDataset/Ancestral_Sequences.fasta".format(script_dir),
                                                         tree_level_order_names =additional_load.tree_levelorder_names,
                                                         aa_probs=build_config.aa_prob)

        dataset_test = torch.from_numpy(dataset_test)
        #test_nodes_observed = dataset_test[:, 0, 1].tolist()
        #Highlight: here we need to do the opposite to the other datasets. The test patristic distances will be those that are not the train
        test_nodes = torch.tensor(internal_names_test, device="cpu")
        patristic_matrix_full = additional_load.patristic_matrix_full
        cladistic_matrix_full = additional_load.cladistic_matrix_full
        vals, idx = torch.sort(test_nodes) #unnecessary but leave in case we predict all fav and all coral at the same time
        test_nodes = test_nodes[idx]
        dataset_test = dataset_test[idx]
        train_nodes = train_load.dataset_train[:,0,1]
        train_indx_patristic = (patristic_matrix_full[:, 0][..., None] == train_nodes).any(-1)
        #train_indx_patristic[0] = True  # To re-add the node names ---> not necessary because we do the opposite, we keep the False ones
        patristic_matrix_test = patristic_matrix_full[~train_indx_patristic]
        patristic_matrix_test = patristic_matrix_test[:, ~train_indx_patristic]

        cladistic_matrix_test = cladistic_matrix_full[~train_indx_patristic]
        cladistic_matrix_test = cladistic_matrix_test[:, ~train_indx_patristic]
        vals, idx = torch.sort(patristic_matrix_test[:,0])  # unnecessary but leave in case we predict all fav and all coral at the same time
        patristic_matrix_test = patristic_matrix_test[idx]
        cladistic_matrix_test = cladistic_matrix_test[idx]
        correspondence_dict = None
    else: #leave testing, the training datset has been pre-splitted
        print("Leave testing: Only sorting the dataframes")
        # Highlight: the patristic matrix full has nodes n_leaves + n_internal, where n_internal = n_leaves-1!!!!!!!!!
        correspondence_dict = special_nodes_dict= None
        patristic_matrix_test = test_load.patristic_matrix_test
        cladistic_matrix_test = test_load.cladistic_matrix_test
        dataset_test = test_load.dataset_test
        test_nodes = dataset_test[:,0,1]
        vals, idx = torch.sort(test_nodes)
        dataset_test = dataset_test[idx]
        matrix_sorted, matrix_sorted_idx = torch.sort(patristic_matrix_test[:, 0])
        patristic_matrix_test = patristic_matrix_test[matrix_sorted_idx]  # sorted rows
        patristic_matrix_test = patristic_matrix_test[:, matrix_sorted_idx]  # sorted columns
        if cladistic_matrix_test is not None:
            cladistic_matrix_test = cladistic_matrix_test[matrix_sorted_idx]  # sorted rows
            cladistic_matrix_test = cladistic_matrix_test[:, matrix_sorted_idx]  # sorted columns

    dataset_train,\
    patristic_matrix_full,\
    patristic_matrix_train,\
    cladistic_matrix_full,\
    cladistic_matrix_train,\
    aa_frequencies = DraupnirLoadUtils.pretreatment(train_load.dataset_train, additional_load.patristic_matrix_full,additional_load.cladistic_matrix_full, build_config)


    test_load = TestLoad(dataset_test=dataset_test,
                         evolutionary_matrix_test=test_load.evolutionary_matrix_test,
                         patristic_matrix_test=patristic_matrix_test,
                         cladistic_matrix_test=cladistic_matrix_test,
                         leaves_names_test=test_load.leaves_names_test,
                         position_test=test_load.position_test,
                         internal_nodes_indexes=test_load.internal_nodes_indexes)
    train_load = TrainLoad(dataset_train=dataset_train,
                           evolutionary_matrix_train=train_load.evolutionary_matrix_train,
                           patristic_matrix_train=patristic_matrix_train,
                           cladistic_matrix_train=cladistic_matrix_train)
    additional_load = AdditionalLoad(patristic_matrix_full=patristic_matrix_full,
                                     cladistic_matrix_full=cladistic_matrix_full,
                                     children_array=additional_load.children_array,
                                     ancestor_info_numbers=additional_load.ancestor_info_numbers,
                                     tree_levelorder_names=additional_load.tree_levelorder_names,
                                     clades_dict_leaves =additional_load.clades_dict_leaves,
                                     closest_leaves_dict=additional_load.closest_leaves_dict,
                                     clades_dict_all=additional_load.clades_dict_all,
                                     linked_nodes_dict=additional_load.linked_nodes_dict,
                                     descendants_dict= additional_load.descendants_dict,
                                     alignment_length=additional_load.alignment_length,
                                     aa_frequencies=aa_frequencies,
                                     correspondence_dict = correspondence_dict,
                                     special_nodes_dict=special_nodes_dict,
                                     full_name=additional_load.full_name)

    return train_load,test_load,additional_load
def save_samples(dataset,patristic,samples_out,entropies,correspondence_dict,results_dir):
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
def visualize_latent_space(latent_space_train,patristic_matrix_train,additional_load, additional_info,folder):

    latent_space_train = torch.cat((patristic_matrix_train[1:, 0][:,None],latent_space_train),dim=1)
    # DraupnirPlots.plot_z(latent_space_full,
    #                                      additional_info.children_dict,
    #                                      args.num_epochs,
    #                                      RESULTS_DIR + folder)
    DraupnirPlots.plot_pairwise_distances_only_leaves(latent_space_train,additional_load, args.num_epochs,
                                             RESULTS_DIR + folder,patristic_matrix_train)
    latent_space_train = latent_space_train.detach().cpu().numpy()
    DraupnirPlots.plot_latent_space_tsne_by_clade_leaves(latent_space_train,
                                         additional_load,
                                         args.num_epochs,
                                         RESULTS_DIR + folder,
                                         build_config.triTSNE)
    DraupnirPlots.plot_latent_space_umap_by_clade_leaves(latent_space_train,
                                                         additional_load,
                                                         args.num_epochs,
                                                         RESULTS_DIR + folder,
                                                         build_config.triTSNE)
    DraupnirPlots.plot_latent_space_pca_by_clade_leaves(latent_space_train,
                                    additional_load,
                                    args.num_epochs,
                                    RESULTS_DIR + folder)
def save_and_select_model(args,build_config, model_load, patristic_matrix_train,patristic_matrix_full):
    #Highlight: Selecting the model
    Draupnir = DraupnirModels.DRAUPNIRModel_VAE(model_load)
    patristic_matrix_model = patristic_matrix_train
    plating_info = ["WITH PLATING" if args.plating else "WITHOUT plating"][0]
    print("Using model {} {}".format(Draupnir.get_class(),plating_info))
    text_file = open("{}/Hyperparameters_{}_{}epochs.txt".format(RESULTS_DIR, now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),args.num_epochs), "a")
    text_file.write("Model Class:  {} \n".format(Draupnir.get_class()))
    text_file.close()
    #Highlight: Saving the model function to a separate file
    model_file = open("{}/ModelFunction.py".format(RESULTS_DIR), "a+")
    draupnir_models_file = open("{}/Draupnir_VAEmodels.py".format(script_dir), "r+")
    model_text = draupnir_models_file.readlines()
    line_start = model_text.index("class {}(DRAUPNIRModelClass):\n".format(Draupnir.get_class()))
    line_stop= [index if "class" in line else len(model_text[line_start+1:]) for index,line in enumerate(model_text[line_start+1:])][0]
    model_text = model_text[line_start:]
    model_text = model_text[:line_stop] #'class DRAUPNIRModel2b(DRAUPNIRModelClass):\n'
    model_file.write("".join(model_text))
    model_file.close()
    return Draupnir, patristic_matrix_model
def draupnir_train(train_load,test_load,additional_load,additional_info,build_config,settings_config,n_samples,graph_coo=None,clades_dict=None):

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


    # aa_prob = torch.unique(dataset_train[:, 2:, 0])

    blosum_max,blosum_weighted,variable_score = DraupnirUtils.process_blosum(blosum,aa_frequencies,max_seq_len,build_config.aa_prob)
    plt.plot(variable_score.cpu().detach().numpy())
    plt.savefig("{}/Variable_score.png".format(RESULTS_DIR))
    def load_fixed_params(load=False):
        if load:
            print("Fixing the parameters from pretrained ones")
            load_folder = "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_simulations_src_sh3_1_2021_06_30_12h32min41s886995ms_21000epochs"
            model_dict_dir = "{}/Draupnir_Checkpoints/Model_state_dict.p".format(load_folder)
            text_file = open("{}/Hyperparameters_{}_{}epochs.txt".format(RESULTS_DIR, now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),args.num_epochs),"a")
            text_file.write("Load pretrained FIXED params from: {}\n".format(model_dict_dir))
            text_file.close()
            pretrained_params_dict = torch.load(model_dict_dir)
        else:
            pretrained_params_dict = None
        return pretrained_params_dict
    pretrained_params_dict=load_fixed_params(False)

    model_load = ModelLoad(z_dim=int(config["z_dim"]),
                           max_seq_len=max_seq_len,
                           device=device,
                           args=args,
                           build_config = build_config,
                           leaves_nodes = dataset_train[:,0,1],
                           n_tree_levels=len(additional_info.tree_by_levels_dict),
                           gru_hidden_dim=int(config["gru_hidden_dim"]),
                           pretrained_params=pretrained_params_dict,
                           aa_frequencies=aa_frequencies,
                           blosum =blosum,
                           blosum_max=blosum_max,
                           blosum_weighted=blosum_weighted,
                           variable_score=variable_score,
                           internal_nodes= patristic_matrix_test[1:,0], #dataset_test[:,0,1]
                           graph_coo=graph_coo,
                           nodes_representations_array = nodes_representations_array,
                           dgl_graph=dgl_graph,
                           children_dict= additional_info.children_dict,
                           closest_leaves_dict=additional_load.closest_leaves_dict,
                           descendants_dict = additional_load.descendants_dict,
                           clades_dict_all = additional_load.clades_dict_all,
                           leaves_testing = build_config.leaves_testing)

    Draupnir, patristic_matrix_model = save_and_select_model(args,build_config, model_load, patristic_matrix_train,patristic_matrix_full)

    #Highlight: Setting optimizer and guide
    adam_args = {"lr": config["lr"], "betas": (config["beta1"], config["beta2"]),"eps":config["eps"],"weight_decay":config["weight_decay"],"clip_norm":config["clip_norm"],"lrd":config["lrd"]}
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
            text_file = open("{}/Hyperparameters_{}_{}epochs.txt".format(RESULTS_DIR, now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),args.num_epochs), "a")
            text_file.write("Load pretrained TUNED params Model: {}\n".format(model_dir))
            text_file.write("Load pretrained TUNED params Optim: {}\n".format(optim_dir))
            text_file.close()
            load_checkpoint(model_dict_dir=model_dir,optim_dir=optim_dir,optim=optim,model = Draupnir)
            #Draupnir.train(False)
    load_tune_params(False)

    elbo =Trace_ELBO()

    text_file = open("{}/Hyperparameters_{}_{}epochs.txt".format(RESULTS_DIR, now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),args.num_epochs), "a")
    text_file.write("ELBO :  {} \n".format(str(elbo)))
    text_file.write("Guide :  {} \n".format(str(guide.__class__)))
    text_file.write("Optimizer :  {} \n".format(optim))

    svi = SVI(model=Draupnir.model, guide=guide, optim=optim, loss=elbo) #TODO: TraceMeanField_ELBO() http://docs.pyro.ai/en/0.3.0-release/inference_algos.html#pyro.infer.trace_mean_field_elbo.TraceMeanField_ELBO

    check_point_epoch = [50 if args.num_epochs < 100 else (args.num_epochs / 100)][0]

    #batch_size = [DraupnirUtils.Define_batch_size(dataset_train.shape[0]) if not args.batch_size else args.batch_size if args.batch_size > 1 else dataset_train.shape[0]][0]
    batching_method = ["batch_dim_0" if not args.batch_by_clade else "batch_by_clade"][0]
    train_loader = DraupnirUtils.setup_data_loaders(dataset_train, patristic_matrix_train,clades_dict,blosum,build_config,args,method=batching_method, use_cuda=args.use_cuda)
    training_method= lambda f, svi, patristic_matrix_model, cladistic_matrix_full,train_loader,args: lambda svi, patristic_matrix_model, cladistic_matrix_full,train_loader,args: f(svi, patristic_matrix_model, cladistic_matrix_full,train_loader,args)
    if args.batch_by_clade and clades_dict:
        training_function = training_method(DraupnirTrain.train_batch_clade,svi, patristic_matrix_model, cladistic_matrix_full,train_loader,args)
    elif args.plating: #todo: in experimental phase
        training_function = training_method(DraupnirTrain.train_plating, svi, patristic_matrix_model, cladistic_matrix_full,train_loader, args)
    elif args.batch_size == 1:#no batching / plating
        training_function = training_method(DraupnirTrain.train, svi, patristic_matrix_model,cladistic_matrix_full, train_loader, args)
    else:#batching
        training_function = training_method(DraupnirTrain.train_batch, svi, patristic_matrix_model,
                                            cladistic_matrix_full, train_loader, args)

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
            save_checkpoint(Draupnir,RESULTS_DIR, optimizer=optim)
            DraupnirUtils.Plot_ELBO(train_loss, RESULTS_DIR, test_frequency=1)
            DraupnirUtils.Plot_Entropy(entropy, RESULTS_DIR, test_frequency=1)
        start = time.time()
        # if not clades_dict:
        #     total_epoch_loss_train = DraupnirTrain.train(svi, patristic_matrix_model, cladistic_matrix_full,train_loader,args)
        # else:
        #     total_epoch_loss_train = DraupnirTrain.train_clade(svi, patristic_matrix_model, cladistic_matrix_full,train_loader,args)
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
                                           n_samples,
                                           dataset_train,
                                           patristic_matrix_full,
                                           cladistic_matrix_train,
                                           use_argmax=False,
                                           use_test=False)

        train_entropy_epoch = DraupnirModelsUtils.compute_sites_entropies(sample_out_train.logits.cpu(),dataset_train.cpu().long()[:,0,1])
        if epoch % args.test_frequency == 0:  # every n epochs --- sample
            pickle.dump(map_estimates, open('{}/Draupnir_Checkpoints/Map_estimates.p'.format(RESULTS_DIR), 'wb'),protocol=pickle.HIGHEST_PROTOCOL)
            sample_out_train_argmax = Draupnir.sample(map_estimates,
                                                   n_samples,
                                                   dataset_train,
                                                   patristic_matrix_full,
                                                   cladistic_matrix_full,
                                                   use_argmax=True,
                                                   use_test=False)

            save_samples(dataset_train,patristic_matrix_train,sample_out_train,train_entropy_epoch,correspondence_dict,"{}/train_info_dict.torch".format(RESULTS_DIR + "/Train_Plots"))
            save_samples(dataset_train, patristic_matrix_train,sample_out_train_argmax, train_entropy_epoch, correspondence_dict,"{}/train_argmax_info_dict.torch".format(RESULTS_DIR + "/Train_argmax_Plots"))

        del sample_out_train
        entropy.append(torch.mean(train_entropy_epoch[:,1]).item())
        if epoch == (args.num_epochs-1):
            DraupnirUtils.Plot_ELBO(train_loss, RESULTS_DIR, test_frequency=1)
            DraupnirUtils.Plot_Entropy(entropy, RESULTS_DIR, test_frequency=1)

            save_checkpoint(Draupnir,RESULTS_DIR, optimizer=optim)  # Saves the parameters gradients
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
    text_file = open("{}/Hyperparameters_{}_{}epochs.txt".format(RESULTS_DIR, now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),args.num_epochs), "a")
    text_file.write("Running time: {}\n".format(str(datetime.timedelta(seconds=end_total-start_total))))
    text_file.write("Total epochs (+added epochs): {}\n".format(args.num_epochs + added_epochs))
    if Draupnir.use_attention:
        pytorch_total_params = sum(val.numel() for param_name, val in pyro.get_param_store().named_parameters() if
                                   val.requires_grad and not param_name.startswith("decoder$$$"))
    else: #TODO: Investigate again, but seems correct
        pytorch_total_params = sum(val.numel() for param_name,val in pyro.get_param_store().named_parameters() if val.requires_grad and not param_name.startswith("decoder_attention"))

    text_file.write("Number of parameters: {} \n".format(pytorch_total_params))
    text_file.close()

    #DraupnirUtils.GradientsPlot(gradient_norms, args.num_epochs, RESULTS_DIR) #Highlight: Very cpu intensive to compute
    print("Final Sampling....")
    load_map = False
    if load_map:
        map_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_SH3_pf00018_larger_than_30aa_2021_08_26_15h51min34s041768ms_15000epochs"
        map_estimates = pickle.load(open('{}/Draupnir_Checkpoints/Map_estimates.p'.format(map_dir), "rb"))
    else:
        map_estimates= guide()
        pickle.dump(map_estimates, open('{}/Draupnir_Checkpoints/Map_estimates.p'.format(RESULTS_DIR), 'wb'),protocol=pickle.HIGHEST_PROTOCOL)

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


    save_samples(dataset_train, patristic_matrix_train,sample_out_train, train_entropies,correspondence_dict,"{}/train_info_dict.torch".format(RESULTS_DIR + "/Train_Plots"))
    save_samples(dataset_train, patristic_matrix_train,sample_out_train_argmax, train_entropies,correspondence_dict,"{}/train_argmax_info_dict.torch".format(RESULTS_DIR + "/Train_argmax_Plots"))

    load_predictions = True
    if load_predictions:
        print("Loading previously trained parameters")
        #load_folder = "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_VAE_simulations_src_sh3_1_2021_09_06_15h43min55s967103ms_21000epochs"
        load_folder = "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_simulations_src_sh3_1_2021_06_30_12h32min41s886995ms_21000epochs"
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
        save_samples(dataset_train, patristic_matrix_train,sample_out_train, train_entropies,correspondence_dict,"{}/train_info_dict.torch".format(RESULTS_DIR + "/Train_Plots"))
        save_samples(dataset_train, patristic_matrix_train,sample_out_train_argmax, train_entropies,correspondence_dict,"{}/train_argmax_info_dict.torch".format(RESULTS_DIR + "/Train_argmax_Plots"))

    #Highlight: Concatenate leaves and internal latent space for plotting
    visualize_latent_space(sample_out_train.latent_space,patristic_matrix_train,additional_load,additional_info,folder="")
    exit()
    start_plots = time.time()

    #Highlight: Plot samples
    preparing_plots(sample_out_train, dataset_train, dataset_train, train_entropies,RESULTS_DIR + "/Train_Plots",additional_load,additional_info,n_samples,dataset_train[:,0,1],replacement_plots=False,plot_test=False)

    #Highlight: Plot most likely sequence
    preparing_plots(sample_out_train_argmax, dataset_train, dataset_train, train_entropies,RESULTS_DIR + "/Train_argmax_Plots",additional_load,additional_info,n_samples,dataset_train[:,0,1],replacement_plots=False,plot_test=False)

    stop_plots = time.time()
    print('Final plots timing: {}'.format(str(datetime.timedelta(seconds=stop_plots - start_plots))))
    print("##########################################################################################################")
def preparing_plots(samples_out,dataset_test,dataset_train,entropies,results_dir,additional_load,additional_info,n_samples,test_ordered_nodes,replacement_plots=False,overplapping_hist=False,plot_test=True,plot_angles=False):
    """samples_out: named tuple containing the output from Draupnir
    dataset_test: the true dataset, it can be the train or the test dataset
    entropies
    results_dir
    additional_load: Named tuple
    n_samples: number of samples from the model
    test_ordered_nodes:
    """
    if not additional_load.correspondence_dict:
        correspondence_dict = dict(zip(list(range(len(additional_load.tree_levelorder_names))), additional_load.tree_levelorder_names))
    else:
        correspondence_dict = additional_load.correspondence_dict
    if plot_test and name in ["Douglas_SRC","Coral_Faviina","Coral_all"] or plot_test and name.endswith("_subtree"):
        #Highlight: select from the predictions only the sequences in the dataset_test. Remove gaps and align to the "observed"
        DraupnirPlots.CleanRealign_Train(name,
                                   dataset_test,
                                   dataset_train,
                                   samples_out.aa_sequences,  # test predictions
                                   test_ordered_nodes,
                                   n_samples,
                                   build_config.aa_prob,
                                   results_dir,
                                   additional_load,
                                   additional_info)
        DraupnirPlots.CleanRealign(name,
                                   dataset_test,
                                   samples_out.aa_sequences,#test predictions
                                   test_ordered_nodes,
                                   n_samples,
                                   build_config.aa_prob,
                                   results_dir,
                                   additional_load,
                                   additional_info)

        DraupnirPlots.plot_entropies(name, entropies.detach().cpu().numpy(), results_dir, correspondence_dict)
    elif args.infer_angles and plot_angles:
        DraupnirPlots.plotting_angles(samples_out,dataset_test,results_dir,additional_load,additional_info,n_samples,test_ordered_nodes)
        DraupnirPlots.plotting_angles_per_aa(samples_out,dataset_test,results_dir,build_config,additional_load,additional_info,n_samples,test_ordered_nodes)
    else:
        DraupnirPlots.Plotting_Heatmap_and_Incorrect_Aminoacids(name,
                                                                dataset_test,
                                                                samples_out.aa_sequences,
                                                                n_samples,
                                                                results_dir,
                                                                correspondence_dict,
                                                                build_config.aa_prob,
                                                                additional_load,
                                                                additional_info,
                                                                replacement_plots)
        DraupnirPlots.plot_entropies(name, entropies.detach().cpu().numpy(), results_dir, correspondence_dict)
        if overplapping_hist:
            DraupnirPlots.Plot_Overlapping_Hist(name,
                                                dataset_train,
                                                dataset_test,
                                                samples_out.aa_sequences,
                                                n_samples,
                                                results_dir,
                                                correspondence_dict,
                                                build_config.aa_prob)
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
def config_build(parameter_search):
    if parameter_search:
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
def Manual_Random_Search():
    #sys.stdout = open('Random_Search_results.txt', 'w')
    # if click.confirm('Do you want to delete previous files? Otherwise results will be appended', default=True):
    #     print("Deleting previous run ...")
    #     os.remove("{}/Random_Search_results.txt".format(script_dir))

    global config
    n_runs = 20
    for i in range(n_runs):
        config = generate_config()
        print(config)
        proc= subprocess.Popen(args=[sys.executable,"Draupnir.py","--parameter-search","False","--config-dict",str(config).replace("'", '"')],stdout=open('Random_Search_results.txt', 'a')) #stdout=open(os.devnull, 'wb'),stderr=open(os.devnull, 'wb')
        proc.communicate()
def main(config):

    global build_config,name,RESULTS_DIR,max_seq_len,full_name
    #dataset_number,simulation_folder,root_sequence_name = 1,"BLactamase","BetaLactamase_seq" #32 leaves
    dataset_number,simulation_folder,root_sequence_name = 1, "SRC_simulations","SRC_SH3" #100 leaves
    #dataset_number,simulation_folder,root_sequence_name = 2, "SRC_simulations","SRC_SH3" #800 leaves
    #dataset_number, simulation_folder, root_sequence_name = 3, "SRC_simulations", "SRC_SH3" #200 leaves
    datasets = {0:"benchmark_randall",#the tree is inferred
                1:"benchmark_randall_original", #uses the original tree but changes the naming of the nodes (because the original tree was not rooted)
                2:"benchmark_randall_original_naming", #uses the original tree and it's original node naming
                3:"SH3_pf00018_larger_than_30aa", #SRC kinases domain SH3 ---> Leaves and angles testing
                4:"simulations_blactamase_{}".format(dataset_number),#EvolveAGene4 Betalactamase simulation
                5:"simulations_src_sh3_{}".format(dataset_number),#EvolveAGene4 SRC SH3 domain simulation
                6:"Douglas_SRC", #Douglas's Full SRC Kinases #Highlight: The tree is not similar to the one in the paper, therefore the sequences where splitted in subtrees according to the ancetral sequences in the paper
                7:"ANC_A1_subtree", #highlight: 3D structure not available #TODO: only working at dragon server
                8:"ANC_A2_subtree", #highlight: 3D structure not available
                9:"ANC_AS_subtree",
                10:"ANC_S1_subtree", #highlight: 3D structure not available
                11:"Coral_Faviina",#Faviina clade from coral sequences
                12:"Coral_all", # All Coral sequences (includes Faviina clade and additional sequences)
                13:"Cnidarian", # All Coral sequences plus other fluorescent cnidarians #Highlight: The tree is too different to certainly locate the all-coral / all-fav ancestors
                16:"PKinase_PF07714"} # Protein Kinases, medium dataset, 123 seq,

    datasets_full_names = {"benchmark_randall":"Randall's Coral fluorescent proteins (CFP) benchmark dataset",  # the tree is inferred
                "benchmark_randall_original":"Randall's Coral fluorescent proteins (CFP) benchmark dataset",
                "benchmark_randall_original_naming":"Randall's Coral fluorescent proteins (CFP) benchmark dataset",  # uses the original tree and it's original node naming
                "SH3_pf00018_larger_than_30aa":"PF00018 Pfam family of Protein Tyrosine Kinases SH3 domains",  # SRC kinases domain SH3 ---> Leaves and angles testing
                "simulations_blactamase_1":"32 leaves Simulation Beta-Lactamase",  # EvolveAGene4 Betalactamase simulation
                "simulations_src_sh3_1":"100 leaves Simulation SRC-Kinase SH3 domain",  # EvolveAGene4 SRC SH3 domain simulation
                "simulations_src_sh3_2": "800 leaves Simulation SRC-Kinase SH3 domain",
                "simulations_src_sh3_3": "200 leaves Simulation SRC-Kinase SH3 domain",
                "Douglas_SRC":"Protein Tyrosin Kinases.",
                "ANC_A1_subtree":"Protein Tyrosin Kinases ANC-A1 clade",
                "ANC_A2_subtree":"Protein Tyrosin Kinases ANC-A2 clade",
                "ANC_AS_subtree":"Protein Tyrosin Kinases ANC-AS clade",
                "ANC_S1_subtree":"Protein Tyrosin Kinases ANC-S1 clade",
                "Coral_Faviina":"Coral fluorescent proteins (CFP) Faviina clade",  # Faviina clade from coral sequences
                "Coral_all":"Coral fluorescent proteins (CFP) clade",  # All Coral sequences (includes Faviina clade and additional sequences)
                "Cnidarian":"Cnidarian fluorescent proteins (CFP) clade",# All Coral sequences plus other fluorescent cnidarians #Highlight: The tree is too different to certainly locate the all-coral / all-fav ancestors
                "PKinase_PF07714":"PF07714 Pfam family of Protein Tyrosin Kinases"}  # Protein Kinases, medium dataset, 123 seq,

    name = datasets[5]
    full_name = datasets_full_names[name]
    if name.startswith("simulations"):
        short_name = re.split(r"_[0-9]+", name.split("simulations_")[1])[0]
        if simulation_folder.split("_")[0].lower() not in name.split("_")[1]:
            raise ValueError("Wrong simulation folder: {} vs name: {}".format(simulation_folder, name))

    RESULTS_DIR = "{}/PLOTS_VAE_{}_{}_{}epochs".format(script_dir,name, now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),args.num_epochs)

    build_config= DraupnirDatasets.Build_Dataset(name,
                                                 script_dir,
                                                 build=False,#Activate build in order to create the dataset again, not recommended if the dataset already exists
                                                 dataset_number=dataset_number,
                                                 simulation_folder=simulation_folder,
                                                 root_sequence_name=root_sequence_name)

    settings_config = SettingsConfig(one_hot_encoding=False,
                             model_design="VAE_only",
                             aligned_seq=True,
                             uniprot=False)

    print("Loading datasets....")
    train_load,test_load,additional_load,build_config = load_data(settings_config,build_config)
    max_seq_len = additional_load.alignment_length
    additional_info=DraupnirUtils.extra_processing(additional_load.ancestor_info_numbers, additional_load.patristic_matrix_full,RESULTS_DIR,args,build_config) #TODO: Fix build graph in server
    train_load,test_load,additional_load= datasets_pretreatment(train_load,test_load,additional_load,build_config,
                                          device,name,simulation_folder,root_sequence_name,dataset_number,settings_config)
    torch.save(torch.get_rng_state(),"{}/rng_key.torch".format(RESULTS_DIR))
    print("Training....")
    print("Dataset: {}".format(name))
    print("Number epochs: {}".format(args.num_epochs))
    print("Z/latent Size: {}".format(config["z_dim"]))
    print("GRU hidden size: {}".format(config["gru_hidden_dim"]))
    print("Number train sequences: {}".format(train_load.dataset_train.shape[0]))
    print("Number test sequences: {}".format(test_load.dataset_test.shape[0]))
    print("Selected Substitution matrix : {}".format(args.subs_matrix))

    if not args.batch_by_clade:
        clades_dict=None
    else:
        clades_dict = additional_load.clades_dict_leaves
    graph_coo = None #Highlight: use only with the GNN models (7)---> Otherwise is in additional_info
    #graph_coo = additional_info.graph_coo
    draupnir_train(train_load,test_load,additional_load,additional_info,build_config,settings_config,args.n_samples,graph_coo,clades_dict)

    return RESULTS_DIR

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Draupnir args")
    parser.add_argument('-n', '--num-epochs', default=1, type=int, help='number of training epochs')
    parser.add_argument('-batch-size', default=1, type=int,help='set batch size. '
                                                                'If set to 1 to NOT batch (batch_size = 1 = 1 batch = 1 entire dataset). '
                                                                'If set to None it automatically suggests a batch size. '
                                                                'If batch_by_clade=True: 1 batch= 1 clade (given by clades_dict).'
                                                                'Else set the batchsize to the given number')
    parser.add_argument('-batch-by-clade', default=False, type=bool, help='Use the leaves divided by their corresponding clades into batches. Do not use with leaf-testing')
    parser.add_argument('-infer-angles', default=False, type=bool,help='Additional Inference of angles')
    parser.add_argument('-plating', default=False, type=bool, help='Plating/Subsampling the mapping of the sequences (ONLY, not the latent space). Remember to set plating size, otherwise it is done automatically')
    parser.add_argument('-plating_size', default=None, type=bool,help='Set plating size '
                                                                    'If set to None it automatically suggests a plate size, if args.plating is TRUE!. Otherwise it remains as None and no plating occurs '
                                                                    'Else set the plate size to the given number')
    #TODO: for splitted blosum/ random or not indexes
    parser.add_argument('-use-SRU', default=False, type=bool, help='Use SRU mapping instead of GRU')
    parser.add_argument('-aa-prob', default=21, type=int, help='20 amino acids,1 gap probabilities')
    parser.add_argument('-n_samples', default=50, type=int, help='Number of samples')
    parser.add_argument('-kappa-addition', default=5, type=int, help='lower bound on angles')
    parser.add_argument('-subs_matrix', default="BLOSUM62", type=str, help='blosum matrix to create blosum embeddings, choose one from /home/lys/anaconda3/pkgs/biopython-1.76-py37h516909a_0/lib/python3.7/site-packages/Bio/Align/substitution_matrices/data')
    parser.add_argument('-embedding-dim', default=50, type=int, help='Blosum embedding dim')
    parser.add_argument('-position-embedding-dim', default=30, type=int, help='Tree position embedding dim')
    parser.add_argument('-max-indel-size', default=5, type=int, help='maximum insertion deletion size (not used)')
    parser.add_argument('-use-cuda', default=True, type=bool, help='Use GPU')
    parser.add_argument('-activate-elbo-convergence', default=False, type=bool, help='extends the running time until a convergence criteria in the elbo loss is met')
    parser.add_argument('-activate-entropy-convergence', default=False, type=bool, help='extends the running time until a convergence criteria in the sequence entropy is met')
    parser.add_argument('-test-frequency', default=100, type=int, help='sampling frequency during training')
    parser.add_argument('-d', '--config-dict', default=None,type=str)
    parser.add_argument('--parameter-search', default=True, type=str) #TODO: Change to something that makes more sense
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

    #Highlight: Activate Random parameter search when saying parameter_search = True
    parameter_search = False
    if parameter_search and ast.literal_eval(str(args.parameter_search)):
        Manual_Random_Search()
    else:
        config = config_build(parameter_search)
        main(config)


