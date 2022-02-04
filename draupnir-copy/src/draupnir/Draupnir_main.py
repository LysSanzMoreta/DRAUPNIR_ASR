#!/usr/bin/env python3
"""
2021: aleatoryscience
Lys Sanz Moreta
Draupnir : GP prior VAE for Ancestral Sequence Resurrection
=======================
"""
import argparse
import time
#import easydict
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
from pyro.infer import SVI, config_enumerate,infer_discrete
from pyro.infer.autoguide import AutoMultivariateNormal, AutoDiagonalNormal,AutoDelta,AutoNormal
from pyro.infer import Trace_ELBO, JitTrace_ELBO,TraceMeanField_ELBO,JitTraceMeanField_ELBO,TraceEnum_ELBO
sys.path.append("./draupnir/draupnir")
import Draupnir_utils as DraupnirUtils
import Draupnir_models as DraupnirModels
import Draupnir_guides as DraupnirGuides
import Draupnir_plots as DraupnirPlots
import Draupnir_train as DraupnirTrain
import Draupnir_models_utils as DraupnirModelsUtils
import Draupnir_load_utils as DraupnirLoadUtils
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
SettingsConfig = namedtuple("SettingsConfig",["one_hot_encoding", "model_design","aligned_seq","data_folder"])
ModelLoad = namedtuple("ModelLoad",["z_dim","max_seq_len","device","args","build_config","leaves_nodes","n_tree_levels","gru_hidden_dim","pretrained_params","aa_frequencies","blosum",
                                    "blosum_max","blosum_weighted","dataset_train_blosum","variable_score","internal_nodes","graph_coo","nodes_representations_array","dgl_graph","children_dict",
                                    "closest_leaves_dict","descendants_dict","clades_dict_all","leaves_testing","plate_unordered","one_hot_encoding"])
BuildConfig = namedtuple('BuildConfig',['alignment_file','use_ancestral','n_test','build_graph',"aa_prob","triTSNE","max_seq_len","leaves_testing","batch_size","plate_subsample_size","script_dir","no_testing"])

SamplingOutput = namedtuple("SamplingOutput",["aa_sequences","latent_space","logits","phis","psis","mean_phi","mean_psi","kappa_phi","kappa_psi"])

def load_data(name,settings_config,build_config,param_config,results_dir,script_dir,args):
    """return
    Train and Test Datasets arrays: [n_seqs, max_len + 2, 30]
        For dimension 2:
            0 column = [seq_len, position in tree, distance to root,ancestor, ..., 0]
            1 column = Git vector (30 dim) if available
            Rest of columns = [integer or one hot encoded amino acid sequence]"""

    aligned = ["aligned" if settings_config.aligned_seq else "NOT_aligned"]
    one_hot = ["OneHotEncoded" if settings_config.one_hot_encoding else "integers"]
    #TODO: Fix the dataset load issue

    dataset = np.load("{}/{}/{}/{}_dataset_numpy_{}_{}.npy".format(script_dir,settings_config.data_folder,name,name,aligned[0], one_hot[0]),allow_pickle=True)

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
    dataset = DraupnirLoadUtils.remove_nan(dataset)
    DraupnirUtils.Ramachandran_plot(dataset[:, 3:], "{}/TRAIN_OBSERVED_angles".format(results_dir + "/Train_Plots"), "Train Angles",one_hot_encoded=settings_config.one_hot_encoding)

    if build_config.alignment_file:
        alignment_file = build_config.alignment_file
        alignment = AlignIO.read(build_config.alignment_file, "fasta")
    else:
        alignment_file = "{}/Mixed_info_Folder/{}.mafft".format(script_dir, name)
        alignment = AlignIO.read(alignment_file, "fasta")
    alignment_array = np.array(alignment)
    gap_positions = np.where(alignment_array == "-")[1]
    np.save("{}/Alignment_gap_positions.npy".format(results_dir), gap_positions)
    # Highlight: count the majority character per site, this is useful for benchmarking
    sites_count = dict.fromkeys(np.unique(gap_positions))
    for site in np.unique(gap_positions):
        unique, counts = np.unique(alignment_array[:, site], return_counts=True)
        sites_count[site] = dict(zip(unique, counts))
    pickle.dump(sites_count, open("{}/Sites_count.p".format(results_dir), 'wb'),protocol=pickle.HIGHEST_PROTOCOL)

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
                               script_dir=build_config.script_dir,
                               no_testing=build_config.no_testing)

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
        text_file.write("Amino acid possibilities + gap: {} \n".format(build_config.aa_prob))
        text_file.write("Substitution matrix : {} \n".format(args.subs_matrix))
        text_file.write("Batch by clade : {} \n".format(args.batch_by_clade))
        text_file.write("Batch size (=1 means entire dataset): {} \n".format(batch_size))
        text_file.write("Plating (subsampling) : {} \n".format(args.plating))
        text_file.write("Plating size : {} \n".format(build_config.plate_subsample_size))
        text_file.write("Plating unorderd (not preserving the tree level order of the nodes) : {} \n".format(args.plate_unordered))
        text_file.write("Inferring angles : {} \n".format(args.infer_angles))
        text_file.write("Guide : {} \n".format(args.select_guide))
        text_file.write("Use learning rate scheduler : {} \n".format(args.use_scheduler))
        text_file.write("Leaves testing (uses the full leaves latent space (NOT a subset)): {} \n".format(build_config.leaves_testing))
        text_file.write(str(config) + "\n")

    hyperparameters()
    patristic_matrix = pd.read_csv("{}/{}/{}/{}_patristic_distance_matrix.csv".format(script_dir,settings_config.data_folder,name,name), low_memory=False)
    patristic_matrix = patristic_matrix.rename(columns={'Unnamed: 0': 'rows'})
    patristic_matrix.set_index('rows',inplace=True)
    try:
        cladistic_matrix = pd.read_csv("{}/{}/{}/{}_cladistic_distance_matrix.csv".format(script_dir,settings_config.data_folder,name,name), index_col="rows",low_memory=False)
    except: #Highlight: For larger datasets , I do not calculate the cladistic matrix, because there is not a fast method. So no cladistic matrix and consequently , no patrocladistic matrix = evolutionary matrix
        cladistic_matrix = None

    ancestor_info = pd.read_csv("{}/Datasets_Folder/{}_tree_levelorder_info.csv".format(script_dir,settings_config.data_folder,name,name), sep="\t",index_col=False,low_memory=False)
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
    # Highlight: Load the dictionary containing the closests leaves to the INTERNAL nodes, transform the names to their tree level order
    closest_leaves_dict = pickle.load(open('{}/Mixed_info_Folder/{}_Closest_leaves_dict.p'.format(script_dir, name), "rb"))
    closest_leaves_dict = DraupnirLoadUtils.convert_closest_leaves_dict(name, closest_leaves_dict, internal_nodes_dict, leaves_nodes_dict)
    #Highlight: Load the dictionary containing all the internal and leaves that descend from the each ancestor node
    descendants_dict = pickle.load(open('{}/Mixed_info_Folder/{}_Descendants_dict.p'.format(script_dir, name), "rb"))
    descendants_dict = DraupnirLoadUtils.convert_descendants(name,descendants_dict,internal_nodes_dict,leaves_nodes_dict)
    #Highlight: Load dictionary with the directly linked children nodes--> i only have it for one dataset
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
        results_dir,
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
        one_hot_encoding=settings_config.one_hot_encoding,
        nodes=tree_levelorder_names,
        ancestral=build_config.use_ancestral)

    if dataset_test is not None:#Highlight: Dataset_test != None only when the test dataset is extracted from the train (testing leaves)
        DraupnirUtils.Ramachandran_plot(dataset_test[:, 2:], "{}/TEST_OBSERVED_angles".format(results_dir+"/Test_Plots"),"Test Angles", one_hot_encoded=settings_config.one_hot_encoding)
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
                   full_name=full_name)
    return train_load,test_load,additional_load, build_config
def save_checkpoint(Draupnir,save_directory, optimizer):
    '''Saves the model and optimizer dict states to disk'''
    save_directory = ("{}/Draupnir_Checkpoints/".format(save_directory))
    optimizer.save(save_directory + "/Optimizer_state.p")
    #keys = [key for key in Draupnir.state_dict() if "decoder_attention.rnn" not in key]
    torch.save(Draupnir.state_dict(), save_directory + "/Model_state_dict.p")
def save_checkpoint_guide(guide,save_directory):
    '''Saves the model and optimizer dict states to disk'''
    save_directory = ("{}/Draupnir_Checkpoints/".format(save_directory))
    torch.save(guide.state_dict(), save_directory + "/Guide_state_dict.p")
def save_checkpoint_preloaded(state_dict,save_directory, optimizer_state):
    '''Saves the model and optimizer dict states to disk'''
    save_directory = ("{}/Draupnir_Checkpoints/".format(save_directory))
    torch.save(optimizer_state,save_directory + "/Optimizer_state.p")
    #keys = [key for key in Draupnir.state_dict() if "decoder_attention.rnn" not in key]
    torch.save(state_dict, save_directory + "/Model_state_dict.p")
def load_checkpoint(model_dict_dir,optim_dir,optim,model):
    '''Loads the model and optimizer states from disk'''
    if model_dict_dir is not None:
        print("Loading pretrained Model parameters...")
        model.load_state_dict(torch.load(model_dict_dir), strict=False)
    if optim_dir is not None:
        print("Loading pretrained Optimizer states...")
        optim.load(optim_dir)
def datasets_pretreatment(name,root_sequence_name,train_load,test_load,additional_load,build_config,device,settings_config,script_dir):
    """ Constructs and corrects the test and train datasets, parsing sequences when necessary
    :param str name: dataset_name
    :param str root_sequence_name: for the default simulated datasets, we need an additional name string to retrieve the ancestral sequences
    Loads "external" test datasets, depending on the dataset"""
    #TODO: Move to DraupnirLoadutils?
    #Highlight: Loading for special test datasets
    if name.startswith("simulation"):
        dataset_test,internal_names_test,max_len_test = DraupnirUtils.load_simulations_ancestral_sequences(name,
                                                                                        settings_config.aligned_seq,
                                                                                        additional_load.max_seq_len,#TODO: Hopefully this is always correct
                                                                                        additional_load.tree_levelorder_names,
                                                                                        root_sequence_name,
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
        dataset_test, internal_names_test = DraupnirUtils.load_randalls_benchmark_ancestral_sequences(script_dir) #TODO: this directory is not correct
        test_nodes_observed =  dataset_test[:, 0, 1].tolist()
        special_nodes_dict=None
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



    elif name in ["Coral_Faviina","Coral_all"]:
        dataset_test, \
        internal_names_test , \
        max_lenght_internal_aligned,\
        special_nodes_dict =DraupnirUtils.CFPTest(name = name,
                                                         ancestral_file="{}/datasets/default/{}/Ancestral_Sequences.fasta".format(script_dir,name),
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

    else: #leave testing, the training dataset has been pre-splitted
        correspondence_dict = special_nodes_dict = None
        if not build_config.no_testing:
            print("Leaf testing: The test dataset is composed by a portion of the leaves")
            # Highlight: the patristic matrix full has nodes n_leaves + n_internal, where n_internal = n_leaves-1!!!!!!!!!
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
        else:
            print("No testing, there is not a test dataset, we will just predict the ancestors without checking their accuracy due to abscence of test data")

            cladistic_matrix_full = additional_load.cladistic_matrix_full

            patristic_matrix_full = additional_load.patristic_matrix_full
            train_nodes = train_load.dataset_train[:, 0, 1]
            train_indx_patristic = (patristic_matrix_full[:, 0][..., None] == train_nodes).any(-1)
            # train_indx_patristic[0] = True  # To re-add the node names ---> not necessary because we do the opposite, we keep the False ones
            patristic_matrix_test = patristic_matrix_full[~train_indx_patristic]
            patristic_matrix_test = patristic_matrix_test[:, ~train_indx_patristic]
            if cladistic_matrix_full is not None:
                cladistic_matrix_test = cladistic_matrix_full[~train_indx_patristic]
                cladistic_matrix_test = cladistic_matrix_test[:, ~train_indx_patristic]
            else:
                cladistic_matrix_test = None

            matrix_sorted, matrix_sorted_idx = torch.sort(patristic_matrix_test[:, 0])
            patristic_matrix_test = patristic_matrix_test[matrix_sorted_idx]  # sorted rows
            patristic_matrix_test = patristic_matrix_test[:, matrix_sorted_idx]  # sorted columns
            #Highlight: Fake, empty dataset, just with the internal nodes "names"
            print("Creating empty test dataset ONLY with the internal nodes names (no sequences) ")
            dataset_test = torch.zeros((patristic_matrix_test.shape[0] - 1, train_load.dataset_train.shape[1], 30))
            dataset_test[:, 0, 1] = patristic_matrix_test[1:, 0]

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
                                     linked_nodes_dict = additional_load.linked_nodes_dict,
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
def visualize_latent_space(latent_space_train,latent_space_test,patristic_matrix_train,patristic_matrix_test,additional_load, additional_info,args,folder):
    #Highlight: Concatenate leaves and internal latent space for plotting
    latent_space_full = torch.cat((latent_space_train,latent_space_test),dim=0)
    #latent_space_indexes = torch.cat((dataset_train[:,0,1],dataset_test[:,0,1]),dim=0)

    latent_space_indexes = torch.cat((patristic_matrix_train[1:, 0], patristic_matrix_test[1:, 0]), dim=0)

    latent_space_full = torch.cat((latent_space_indexes[:,None],latent_space_full),dim=1)
    if additional_load.linked_nodes_dict is not None:
        DraupnirPlots.plot_pairwise_distances(latent_space_full,additional_load, args.num_epochs,
                                             RESULTS_DIR + folder)
    latent_space_full = latent_space_full.detach().cpu().numpy()
    # DraupnirPlots.plot_z(latent_space_full,
    #                                      additional_info.children_dict,
    #                                      args.num_epochs,
    #                                      RESULTS_DIR + folder)
    DraupnirPlots.plot_latent_space_tsne_by_clade(latent_space_full,
                                         additional_load,
                                         args.num_epochs,
                                         RESULTS_DIR + folder,
                                         build_config.triTSNE)
    DraupnirPlots.plot_latent_space_umap_by_clade(latent_space_full,
                                         additional_load,
                                         args.num_epochs,
                                         RESULTS_DIR + folder,
                                         build_config.triTSNE)
    DraupnirPlots.plot_latent_space_pca_by_clade(latent_space_full,
                                    additional_load,
                                    args.num_epochs,
                                    RESULTS_DIR + folder)
def save_and_select_model(args,build_config, model_load, patristic_matrix_train,patristic_matrix_full,script_dir):
    #Highlight: Selecting the model
    #TODO: Warnings/Raise errors for not allowed combinations
    #todo: HOW TO MAKE THIS SIMPLER
    # if args.select_guide == "variational":
    #     if args.use_blosum:
    #         Draupnir = DraupnirModels.DRAUPNIRModel_classic_VAE(model_load)
    #     patristic_matrix_model = patristic_matrix_train
    if args.batch_by_clade:# and not build_config.leaves_testing: #clade batching #TODO: Remove
        Draupnir = DraupnirModels.DRAUPNIRModel_cladebatching(model_load)
        patristic_matrix_model =patristic_matrix_train
    elif args.plating :#plating with splitted blosum matrix
        assert args.batch_size == 1, "We are plating, no batching, please set batch_size == 1"
        if args.use_SRU:
            Draupnir = DraupnirModels.DRAUPNIRModel_classic_SRU(model_load)  # plating in order
        else:
            Draupnir = DraupnirModels.DRAUPNIRModel_classic_plating(model_load) #plating in tree level order, no blosum splitting
        patristic_matrix_model = patristic_matrix_train
    elif args.use_SRU and not args.infer_angles: #Use SRU instead of GRU
        if args.use_blosum:
            Draupnir = DraupnirModels.DRAUPNIRModel_classic_SRU(model_load)
        else:
            Draupnir = DraupnirModels.DRAUPNIRModel_classic_no_blosum_SRU(model_load)
        patristic_matrix_model = patristic_matrix_train
    elif args.use_Transformer and not args.infer_angles: #Use Transformer instead of GRU
        if args.use_blosum:
            Draupnir = DraupnirModels.DRAUPNIRModel_classic_Transformer(model_load)
        else:
            Draupnir = DraupnirModels.DRAUPNIRModel_classic_no_blosum_Transformer(model_load)
        patristic_matrix_model = patristic_matrix_train
    elif args.use_MLP and not args.infer_angles: #Use SRU instead of GRU
        if args.use_blosum:
            Draupnir = DraupnirModels.DRAUPNIRModel_MLP(model_load)
        else:
            Draupnir = DraupnirModels.DRAUPNIRModel_classic_no_blosum(model_load)
        patristic_matrix_model = patristic_matrix_train
    elif build_config.leaves_testing and not args.infer_angles: #training and testing on leaves. In this case, the latent space is composed by both train-leaves and test-leaves patristic matrices, but only the train sequences are observed
        Draupnir = DraupnirModels.DRAUPNIRModel_leaftesting(model_load)
        patristic_matrix_model = patristic_matrix_full
    elif args.infer_angles:
        if args.use_SRU:
            Draupnir = DraupnirModels.DRAUPNIRModel_anglespredictions_SRU(model_load)
        else:
            Draupnir = DraupnirModels.DRAUPNIRModel_anglespredictions(model_load)
        if build_config.leaves_testing:
            patristic_matrix_model = patristic_matrix_full
        else: #partial inteference of the latent
            patristic_matrix_model = patristic_matrix_train
    elif args.batch_size == 1:#Not batching, training on all leaves, testing on internal nodes & training on leaves but only using the latent space of the training leaves
        assert args.plating_size == None, "Please set to None the plate size. If you want to plate, use args.plate = True"
        if args.use_blosum:
            Draupnir = DraupnirModels.DRAUPNIRModel_classic(model_load)
        else:
            Draupnir = DraupnirModels.DRAUPNIRModel_classic_no_blosum(model_load)
        patristic_matrix_model = patristic_matrix_train
    else: #batching #TODO: Remove?
        Draupnir = DraupnirModels.DRAUPNIRModel_batching(model_load)
        patristic_matrix_model = patristic_matrix_train

    plating_info = ["WITH PLATING" if args.plating else "WITHOUT plating"][0]
    print("Using model {} {}".format(Draupnir.get_class(),plating_info))
    text_file = open("{}/Hyperparameters_{}_{}epochs.txt".format(RESULTS_DIR, now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),args.num_epochs), "a")
    text_file.write("Model Class:  {} \n".format(Draupnir.get_class()))
    text_file.close()
    #Highlight: Saving the model function to a separate file
    model_file = open("{}/ModelFunction.py".format(RESULTS_DIR), "a+")
    draupnir_models_file = open("{}/Draupnir_models.py".format(script_dir), "r+")
    model_text = draupnir_models_file.readlines()
    line_start = model_text.index("class {}(DRAUPNIRModelClass):\n".format(Draupnir.get_class()))
    #line_stop= [index if "class" in line else len(model_text[line_start+1:]) for index,line in enumerate(model_text[line_start+1:])][0]
    #model_text = model_text[line_start:]
    #model_text = model_text[:line_stop] #'class DRAUPNIRModel2b(DRAUPNIRModelClass):\n'
    model_file.write("".join(model_text))
    model_file.close()
    # Highlight: Saving the guide function to a separate file
    if args.select_guide.startswith("variational"):
        guide_file = open("{}/GuideFunction.py".format(RESULTS_DIR), "a+")
        draupnir_guides_file = open("{}/Draupnir_guides.py".format(script_dir), "r+")
        guide_text = draupnir_guides_file.readlines()
        guide_file.write("".join(guide_text))
    return Draupnir, patristic_matrix_model
def convert_to_integers(sample_out):
    sample_out = SamplingOutput(
        aa_sequences=DraupnirUtils.convert_to_integers(sample_out.aa_sequences.cpu(), build_config.aa_prob, axis=3),
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
# def histogram_density(logits,folder): #TODO: remove
#     print("Plotting histograms of logits and probabilities")
#     #Highlight: Plot only gaps
#     logitsgaps = logits[:, :, 0]
#     logitsgaps = logitsgaps.detach().cpu().numpy().flatten() #logits.astype(np.uint8)
#     probsgaps = np.exp(logitsgaps)/(np.exp(logitsgaps)+1)
#     plt.clf()
#     sns.histplot(probsgaps.astype(np.float16),bins=30) #np.random.rand((10)
#     plt.title("Gaps probabilities")
#     plt.savefig("{}/{}/Histogram_density_probabilities_GAPS.png".format(RESULTS_DIR, folder))
#     plt.clf()
#     sns.histplot(logitsgaps,bins=30) #np.random.rand((10)
#     plt.title("Gaps logits")
#     plt.savefig("{}/{}/Histogram_density_logits_GAPS.png".format(RESULTS_DIR, folder))
#     plt.clf()
#     # Highlight: Remove the probabilities of the gaps
#     logitsnogaps = logits[:, :, 1:]
#     logitsnogaps = logitsnogaps.detach().cpu().numpy().flatten() #logits.astype(np.uint8)
#     probsnogaps = np.exp(logitsnogaps)/(np.exp(logitsnogaps)+1)
#     plt.clf()
#     sns.histplot(probsnogaps.astype(np.float16),bins=30) #np.random.rand((10)
#     plt.title("No gaps probabilities")
#     plt.savefig("{}/{}/Histogram_density_probabilities_NOGAPS.png".format(RESULTS_DIR, folder))
#     plt.clf()
#     sns.histplot(logitsnogaps,bins=30) #np.random.rand((10)
#     plt.title("No gaps logits")
#     plt.savefig("{}/{}/Histogram_density_logits_NOGAPS.png".format(RESULTS_DIR, folder))
#     plt.clf()
#     #Highlight: plot all
#     logits = logits.detach().cpu().numpy().flatten() #logits.astype(np.uint8)
#     probs = np.exp(logits)/(np.exp(logits)+1)
#     plt.clf()
#     sns.histplot(probs.astype(np.float16),bins=30) #np.random.rand((10)
#     plt.title("all probabilities")
#     plt.savefig("{}/{}/Histogram_density_probabilities_ALL.png".format(RESULTS_DIR, folder))
#     plt.clf()
#     sns.histplot(logits,bins=30) #np.random.rand((10)
#     plt.title("No gaps logits")
#     plt.savefig("{}/{}/Histogram_density_logits_ALL.png".format(RESULTS_DIR, folder))
#     plt.clf()
def extract_percent_id(dataset,aa_sequences_predictions,n_samples,results_directory,correspondence_dict):
    "Fast version to calculate %ID among predictions and observed data"
    len_info = dataset[:, 0, 0].repeat(n_samples).unsqueeze(-1).reshape(n_samples, len(dataset), 1)
    node_info = dataset[:, 0, 1].repeat(n_samples).unsqueeze(-1).reshape(n_samples, len(dataset), 1)
    distance_info = dataset[:, 0, 2].repeat(n_samples).unsqueeze(-1).reshape(n_samples, len(dataset), 1)
    node_names = ["{}//{}".format(correspondence_dict[index], index) for index in dataset[:, 0, 1].tolist()]
    aa_sequences_predictions = torch.cat((len_info, node_info, distance_info, aa_sequences_predictions), dim=2)
    "Fast version to calculate %ID among predictions and observed data"
    align_lenght = dataset[:, 2:, 0].shape[1]
    # node_names = ["{}//{}".format(correspondence_dict[index], index) for index in Dataset_test[:, 0, 1].tolist()]
    samples_names = ["sample_{}".format(index) for index in range(n_samples)]
    equal_aminoacids = (aa_sequences_predictions[:, :, 3:] == dataset[:, 2:,0]).float()  # is correct #[n_samples,n_nodes,L]
    # Highlight: Incorrectly predicted sites
    incorrectly_predicted_sites = (~equal_aminoacids.bool()).float().sum(-1)
    incorrectly_predicted_sites_per_sample = np.concatenate([node_info.cpu().detach().numpy(), incorrectly_predicted_sites.cpu().detach().numpy()[:, :, np.newaxis]],axis=-1)
    np.save("{}/Incorrectly_Predicted_Sites_Fast".format(results_directory), incorrectly_predicted_sites_per_sample)
    incorrectly_predicted_sites_df = pd.DataFrame(incorrectly_predicted_sites.T.cpu().detach().numpy(),
                                                  index=node_names)
    incorrectly_predicted_sites_df.columns = samples_names
    incorrectly_predicted_sites_df["Average"] = incorrectly_predicted_sites_df.mean(1).values.tolist()
    incorrectly_predicted_sites_df["Std"] = incorrectly_predicted_sites_df.std(1).values.tolist()
    incorrectly_predicted_sites_df.to_csv("{}/Incorrectly_predicted_sites_df.csv".format(results_directory),sep="\t")
    # Highlight: PERCENT ID
    equal_aminoacids = equal_aminoacids.sum(-1) / align_lenght  # equal_aminoacids.sum(-1)
    percent_id_df = pd.DataFrame(equal_aminoacids.T.cpu().detach().numpy() * 100,
                                 index=node_names)  # [n_nodes, n_samples]
    percent_id_df.columns = samples_names
    percent_id_df["Average"] = percent_id_df.mean(1).values.tolist()
    percent_id_df["Std"] = percent_id_df.std(1).values.tolist()
    percent_id_df.to_csv("{}/PercentID_df.csv".format(results_directory), sep="\t")
    return percent_id_df, incorrectly_predicted_sites_df, align_lenght

def draupnir_sample(train_load,
                    test_load,
                    additional_load,
                    additional_info,
                    build_config,
                    settings_config,
                    n_samples,
                    args, #Highlight: added
                    device, #Highlight: added
                    script_dir, #Highlight: added
                    results_dir,#Highlight: added
                    graph_coo=None,
                    clades_dict=None):
    "Sample new sequences from a pretrained model"
    max_seq_len = additional_info.max_seq_len
    if not additional_load.correspondence_dict:
        correspondence_dict = dict(
            zip(list(range(len(additional_load.tree_levelorder_names))), additional_load.tree_levelorder_names))
    else:
        correspondence_dict = additional_load.correspondence_dict
    if args.use_cuda:
        blosum = additional_info.blosum.cuda()
        aa_frequencies = additional_load.aa_frequencies.cuda()
        dataset_train = train_load.dataset_train.cuda()
        patristic_matrix_train = train_load.patristic_matrix_train.cuda()
        patristic_matrix_full = additional_load.patristic_matrix_full.cuda()
        patristic_matrix_test = test_load.patristic_matrix_test.cuda()
        dataset_test = test_load.dataset_test.cuda()
        if train_load.cladistic_matrix_train is not None:
            cladistic_matrix_train = train_load.cladistic_matrix_train.cuda()
            cladistic_matrix_test = \
            [test_load.cladistic_matrix_test.cuda() if test_load.cladistic_matrix_test is not None else None][0]
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


    blosum_max, blosum_weighted, variable_score = DraupnirUtils.process_blosum(blosum, aa_frequencies, max_seq_len,
                                                                               build_config.aa_prob)
    dataset_train_blosum = DraupnirUtils.blosum_embedding_encoder(blosum, aa_frequencies, max_seq_len,
                                                                  build_config.aa_prob, dataset_train,
                                                                  settings_config.one_hot_encoding)

    plt.plot(variable_score.cpu().detach().numpy())
    plt.savefig("{}/Variable_score.png".format(results_dir))


    print("WARNING: Fixing the parameters from pretrained ones to sample!!!")
    load_pretrained_folder = args.load_pretrained_path
    model_dict_dir = "{}/Draupnir_Checkpoints/Model_state_dict.p".format(load_pretrained_folder)
    guide_dict_dir = "{}/Draupnir_Checkpoints/Guide_state_dict.p".format(load_pretrained_folder)

    text_file = open("{}/Hyperparameters_{}_{}epochs.txt".format(results_dir, now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),args.num_epochs), "a")
    text_file.write("Load pretrained FIXED params from: {}\n".format(load_pretrained_folder))
    text_file.close()
    pretrained_params_dict_model = torch.load(model_dict_dir)
    pretrained_params_dict_guide = torch.load(guide_dict_dir)


    model_load = ModelLoad(z_dim=int(config["z_dim"]),
                           max_seq_len=max_seq_len,
                           device=device,
                           args=args,
                           build_config=build_config,
                           leaves_nodes=dataset_train[:, 0, 1],
                           n_tree_levels=len(additional_info.tree_by_levels_dict),
                           gru_hidden_dim=int(config["gru_hidden_dim"]),
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

    Draupnir, patristic_matrix_model = save_and_select_model(args, build_config, model_load, patristic_matrix_train,
                                                             patristic_matrix_full,script_dir)


    hyperparameter_file = glob('{}/Hyperparameters*'.format(load_pretrained_folder))[0]
    select_guide = [line.split(":")[1].strip("\n") for line in open(hyperparameter_file,"r+").readlines() if line.startswith("Guide :")][0]
    select_guide = "".join(select_guide.split())
    guide = select_quide(Draupnir,model_load,select_guide)

    with torch.no_grad():
        for name, parameter in guide.named_parameters():
            parameter.copy_(pretrained_params_dict_guide[name])
        for name, parameter in Draupnir.named_parameters():
            parameter.copy_(pretrained_params_dict_model[name])

    #map_estimates = guide(dataset_train,patristic_matrix_train,cladistic_matrix_train,clade_blosum=None)

    print("Generating new samples!")
    if select_guide == "variational":
        #map_estimates_dict = defaultdict()
        print("Variational approach: Re-sampling from the guide")
        #map_estimates_dict = dill.load(open('{}/Draupnir_Checkpoints/Map_estimates.p'.format(args.load_pretrained_path), "rb"))
        map_estimates_dict = defaultdict()
        samples_names = ["sample_{}".format(i) for i in range(n_samples)]
        # Highlight: Train storage
        aa_sequences_train_samples = torch.zeros((n_samples, dataset_train.shape[0], dataset_train.shape[1] - 2)).detach()
        latent_space_train_samples = torch.zeros((n_samples, dataset_train.shape[0], int(config["z_dim"]))).detach()
        logits_train_samples = torch.zeros((n_samples, dataset_train.shape[0], dataset_train.shape[1] - 2, build_config.aa_prob)).detach()
        # Highlight: Test storage
        aa_sequences_test_samples = torch.zeros((n_samples, patristic_matrix_test[1:].shape[0], dataset_train.shape[1] - 2)).detach()
        latent_space_test_samples = torch.zeros((n_samples, patristic_matrix_test[1:].shape[0], int(config["z_dim"]))).detach()
        logits_test_samples = torch.zeros((n_samples, patristic_matrix_test[1:].shape[0], dataset_train.shape[1] - 2, build_config.aa_prob)).detach()
        for sample_idx, sample in enumerate(samples_names):
            map_estimates = guide(dataset_train,patristic_matrix_train,cladistic_matrix_train,clade_blosum=None) #TODO: Correct?
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
        # logits_train_samples = torch.zeros((n_samples, dataset_train.shape[0], dataset_train.shape[1] - 2, build_config.aa_prob)).detach()
        # Highlight: Test storage: Marginal
        aa_sequences_test_samples = torch.zeros((n_samples, patristic_matrix_test[1:].shape[0], dataset_train.shape[1] - 2)).detach()
        latent_space_test_samples = torch.zeros((n_samples, patristic_matrix_test[1:].shape[0], int(config["z_dim"]))).detach()
        logits_test_samples = torch.zeros((n_samples, patristic_matrix_test[1:].shape[0], dataset_train.shape[1] - 2, build_config.aa_prob)).detach()
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

    # plot_probs=False
    # if plot_probs:
    #     histogram_density(sample_out_test_argmax.logits, folder="Test_Plots")
    #     histogram_density(sample_out_test_argmax2.logits, folder="Test2_Plots")
    visualize_latent_space(sample_out_train_argmax.latent_space,
                           sample_out_test_argmax.latent_space,
                           patristic_matrix_train,
                           patristic_matrix_test,
                           additional_load,
                           additional_info,
                           args,
                           folder="")
    visualize_latent_space(sample_out_train_argmax.latent_space,
                           sample_out_test_argmax2.latent_space,
                           patristic_matrix_train,
                           patristic_matrix_test,
                           additional_load,
                           additional_load,
                           args,
                           folder="/Test2_Plots")

    if settings_config.one_hot_encoding:
        print("Transforming one-hot back to integers")
        sample_out_train = convert_to_integers(sample_out_train)
        # sample_out_train_argmax = convert_to_integers(sample_out_train_argmax) #argmax sets directly the aa to the highest logit
        sample_out_test = convert_to_integers(sample_out_test)
        # sample_out_test_argmax = convert_to_integers(sample_out_test_argmax)
        sample_out_test2 = convert_to_integers(sample_out_test2)
        # sample_out_test_argmax2 = convert_to_integers(sample_out_test_argmax2)
        dataset_train = DraupnirUtils.ConvertToIntegers(dataset_train.cpu(), build_config.aa_prob, axis=2)
        if build_config.leaves_testing:  # TODO: Check that this works
            dataset_test = DraupnirUtils.ConvertToIntegers(dataset_test.cpu(), build_config.aa_prob,
                                                           axis=2)  # no need to do it with the test of the simulations, never was one hot encoded. Only for testing leaves

    start_plots = time.time()
    # aa_sequences_predictions_test = dataset_test[:,2:,0].repeat(50,1,1)
    # aa_sequences_predictions_train = dataset_train[:, 2:, 0].repeat(50, 1, 1)
    if n_samples != sample_out_test.aa_sequences.shape[0]:
        n_samples = sample_out_test.aa_sequences.shape[0]
    if args.infer_angles:
        preparing_plots(sample_out_train, dataset_train, dataset_train, train_entropies,
                        results_dir + "/Train_Plots", additional_load, additional_info, n_samples,
                        dataset_train[:, 0, 1], replacement_plots=False, plot_test=False, plot_angles=True,
                        no_testing=True)
        preparing_plots(sample_out_test, dataset_test, dataset_train, test_entropies,
                        results_dir + "/Test_Plots", additional_load, additional_info, n_samples,
                        patristic_matrix_test[1:, 0], replacement_plots=False, overplapping_hist=False,
                        plot_angles=True, no_testing=build_config.no_testing)
        preparing_plots(sample_out_test2, dataset_test, dataset_train, test_entropies2,
                        results_dir + "/Test2_Plots", additional_load, additional_info, n_samples,
                        patristic_matrix_test[1:, 0], replacement_plots=False, overplapping_hist=False,
                        plot_angles=True, no_testing=build_config.no_testing)

    # Highlight: Plot samples
    print("train")
    preparing_plots(sample_out_train, dataset_train, dataset_train, train_entropies, results_dir + "/Train_Plots",
                    additional_load, additional_info, n_samples, dataset_train[:, 0, 1], replacement_plots=False,
                    plot_test=False, no_testing=True, overplapping_hist=False)
    print("test")
    preparing_plots(sample_out_test, dataset_test, dataset_train, test_entropies, results_dir + "/Test_Plots",
                    additional_load, additional_info, n_samples, patristic_matrix_test[1:, 0], replacement_plots=False,
                    overplapping_hist=False, no_testing=build_config.no_testing)
    print("test2")
    preparing_plots(sample_out_test2, dataset_test, dataset_train, test_entropies2, results_dir + "/Test2_Plots",
                    additional_load, additional_info, n_samples, patristic_matrix_test[1:, 0], replacement_plots=False,
                    overplapping_hist=False, no_testing=build_config.no_testing)

    if n_samples != sample_out_test_argmax.aa_sequences.shape[0]:  # most likely sequences
        n_samples = sample_out_test_argmax.aa_sequences.shape[0]

    # Highlight: Plot most likely sequence
    print("train argmax")
    preparing_plots(sample_out_train_argmax, dataset_train, dataset_train, train_entropies,
                    results_dir + "/Train_argmax_Plots", additional_load, additional_info, n_samples,
                    dataset_train[:, 0, 1], replacement_plots=False, plot_test=False, no_testing=True,
                    overplapping_hist=False)
    print("test argmax")
    preparing_plots(sample_out_test_argmax, dataset_test, dataset_train, test_entropies,
                    results_dir + "/Test_argmax_Plots", additional_load, additional_info, n_samples,
                    patristic_matrix_test[1:, 0], replacement_plots=False, overplapping_hist=False,
                    no_testing=build_config.no_testing)
    print("test_argmax 2")
    preparing_plots(sample_out_test_argmax2, dataset_test, dataset_train, test_entropies2,
                    results_dir + "/Test2_argmax_Plots", additional_load, additional_info, n_samples,
                    patristic_matrix_test[1:, 0], replacement_plots=False, overplapping_hist=False,
                    no_testing=build_config.no_testing)

    stop_plots = time.time()
    print('Final plots timing: {}'.format(str(datetime.timedelta(seconds=stop_plots - start_plots))))
    print("##########################################################################################################")

def draupnir_train(train_load,test_load,additional_load,additional_info,build_config,settings_config,n_samples,
                   args, #Highlight: Added
                   device, #Highlight: Added
                   script_dir,  #Highlight: Added
                   results_dir,
                   graph_coo=None,clades_dict=None):
    max_seq_len = additional_info.max_seq_len
    if not additional_load.correspondence_dict:
        correspondence_dict = dict(zip(list(range(len(additional_load.tree_levelorder_names))), additional_load.tree_levelorder_names))
    else:
        correspondence_dict = additional_load.correspondence_dict
    if args.use_cuda:
        blosum = additional_info.blosum.cuda()
        aa_frequencies = additional_load.aa_frequencies.cuda()
        dataset_train = train_load.dataset_train.cuda()
        patristic_matrix_train = train_load.patristic_matrix_train.cuda()
        patristic_matrix_full = additional_load.patristic_matrix_full.cuda()
        patristic_matrix_test = test_load.patristic_matrix_test.cuda()
        dataset_test = test_load.dataset_test.cuda()
        if train_load.cladistic_matrix_train is not None:
            cladistic_matrix_train = train_load.cladistic_matrix_train.cuda()
            cladistic_matrix_test = [test_load.cladistic_matrix_test.cuda() if test_load.cladistic_matrix_test is not None else None][0]
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
    dataset_train_blosum = DraupnirUtils.blosum_embedding_encoder(blosum,aa_frequencies,max_seq_len,build_config.aa_prob,dataset_train,settings_config.one_hot_encoding)

    plt.plot(variable_score.cpu().detach().numpy())
    plt.savefig("{}/Variable_score.png".format(results_dir))


    model_load = ModelLoad(z_dim=int(config["z_dim"]),
                           max_seq_len=max_seq_len,
                           device=device,
                           args=args,
                           build_config = build_config,
                           leaves_nodes = dataset_train[:,0,1],
                           n_tree_levels=len(additional_info.tree_by_levels_dict),
                           gru_hidden_dim=int(config["gru_hidden_dim"]),
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

    Draupnir, patristic_matrix_model = save_and_select_model(args,build_config, model_load, patristic_matrix_train,patristic_matrix_full,script_dir)

    guide = select_quide(Draupnir, model_load, args.select_guide)
    elbo =Trace_ELBO()

    text_file = open("{}/Hyperparameters_{}_{}epochs.txt".format(results_dir, now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),args.num_epochs), "a")
    text_file.write("ELBO :  {} \n".format(str(elbo)))
    text_file.write("Guide :  {} \n".format(str(guide.__class__)))

    #Highlight: Select optimizer/scheduler
    if args.use_scheduler:
        print("Using a learning rate scheduler on top of the optimizer!")
        adam_args = {"lr": config["lr"], "betas": (config["beta1"], config["beta2"]), "eps": config["eps"],
                     "weight_decay": config["weight_decay"]}
        optim = torch.optim.Adam #Highlight: For the scheduler we need to use TORCH.optim not PYRO.optim, and there is no clipped adam in torch
        #Highlight: "Reduce LR on plateau: Scheduler: Reduce learning rate when a metric has stopped improving."
        optim = pyro.optim.ReduceLROnPlateau({'optimizer': optim, 'optim_args': adam_args})

    else:
        clippedadam_args = {"lr": config["lr"], "betas": (config["beta1"], config["beta2"]), "eps": config["eps"],
                     "weight_decay": config["weight_decay"], "clip_norm": config["clip_norm"], "lrd": config["lrd"]}
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

    svi = SVI(Draupnir.model, guide,optim,elbo) #TODO: TraceMeanField_ELBO() http://docs.pyro.ai/en/0.3.0-release/inference_algos.html#pyro.infer.trace_mean_field_elbo.TraceMeanField_ELBO
    text_file.write("Optimizer :  {} \n".format(optim))

    check_point_epoch = [50 if args.num_epochs < 100 else (args.num_epochs / 100)][0]

    batching_method = ["batch_dim_0" if not args.batch_by_clade else "batch_by_clade"][0]
    train_loader = DraupnirUtils.setup_data_loaders(dataset_train, patristic_matrix_train,clades_dict,blosum,build_config,args,method=batching_method, use_cuda=args.use_cuda)
    training_method= lambda f, svi, patristic_matrix_model, cladistic_matrix_full,train_loader,args: lambda svi, patristic_matrix_model, cladistic_matrix_full,train_loader,args: f(svi, patristic_matrix_model, cladistic_matrix_full,train_loader,args)
    if args.batch_by_clade and clades_dict:
        training_function = training_method(DraupnirTrain.train_batch_clade,svi, patristic_matrix_model, cladistic_matrix_full,train_loader,args)
    elif args.batch_size == 1:#no batching or plating
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
    output_file = open("{}/output.log".format(results_dir),"w")
    while epoch < args.num_epochs:
        if check_point_epoch > 0 and epoch > 0 and epoch % check_point_epoch == 0:
            DraupnirUtils.Plot_ELBO(train_loss, results_dir, test_frequency=1)
            DraupnirUtils.Plot_Entropy(entropy, results_dir, test_frequency=1)
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
        map_estimates = guide(dataset_train,patristic_matrix_train,cladistic_matrix_train,clade_blosum=None) #only saving 1 sample
        map_estimates = {val: key.detach() for val, key in map_estimates.items()}
        sample_out_train = Draupnir.sample(map_estimates,
                                           n_samples,
                                           dataset_train,
                                           patristic_matrix_full,
                                           cladistic_matrix_train,
                                           use_argmax=True,
                                           use_test=False,
                                           use_test2=False)
        save_checkpoint(Draupnir, results_dir, optimizer=optim)  # Saves the parameters gradients
        save_checkpoint_guide(guide,results_dir)
        train_entropy_epoch = DraupnirModelsUtils.compute_sites_entropies(sample_out_train.logits.cpu(),dataset_train.cpu().long()[:,0,1])
        #percent_id_df, _, _ = extract_percent_id(dataset_train, sample_out_train.aa_sequences, n_samples_dict[folder], results_dir,correspondence_dict)
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

            test_entropies = DraupnirModelsUtils.compute_sites_entropies(sample_out_test.logits.cpu(),patristic_matrix_test.cpu().long()[1:, 0])
            test_entropies2 = DraupnirModelsUtils.compute_sites_entropies(sample_out_test2.logits.cpu(),patristic_matrix_test.cpu().long()[1:, 0])
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

        del sample_out_train
        entropy.append(torch.mean(train_entropy_epoch[:,1]).item())
        if epoch == (args.num_epochs-1):
            DraupnirUtils.Plot_ELBO(train_loss, results_dir, test_frequency=1)
            DraupnirUtils.Plot_Entropy(entropy, results_dir, test_frequency=1)
            save_checkpoint(Draupnir,results_dir, optimizer=optim)  # Saves the parameters gradients
            save_checkpoint_guide(guide, results_dir)  # Saves the parameters gradients
            if len(train_loss) > 10 and args.activate_elbo_convergence:
                difference = sum(train_loss[-10:]) / 10 - total_epoch_loss_train
                convergence = [False if difference > 0.5 else True][0] # Highlight: this works , but what should be the treshold is yet to be determined
                if convergence:break
                else:
                    epoch -= int(args.num_epochs // 3)
                    added_epochs += int(args.num_epochs // 3)
            if len(train_loss) > 10 and args.activate_entropy_convergence:
                difference = sum(entropy[-10:]) / 10 - torch.mean(train_entropy_epoch[:,1]).item()
                convergence = [False if difference > 0.2 else True][0]  # Highlight: this works , but what should be the threshold is yet to be determined
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
    if Draupnir.use_attention:
        pytorch_total_params = sum(val.numel() for param_name, val in pyro.get_param_store().named_parameters() if val.requires_grad and not param_name.startswith("decoder$$$"))
    elif args.select_guide.startswith("variational"):
        pytorch_total_params = sum([val.numel() for param_name,val in pyro.get_param_store().named_parameters() if val.requires_grad and not param_name.startswith("DRAUPNIRGUIDES.draupnir")])
    else: #TODO: Investigate again, but seems correct
        pytorch_total_params = sum(val.numel() for param_name,val in pyro.get_param_store().named_parameters() if val.requires_grad and not param_name.startswith("decoder_attention"))

    text_file.write("Number of parameters: {} \n".format(pytorch_total_params))
    text_file.close()
    #DraupnirUtils.GradientsPlot(gradient_norms, args.num_epochs, results_dir) #Highlight: Very cpu intensive to compute
    print("Final Sampling....")
    if args.num_epochs > 0:
        if args.select_guide == "variational":
            map_estimates_dict = defaultdict()
            samples_names = ["sample_{}".format(i) for i in range(n_samples)]
            #Highlight: Train storage
            aa_sequences_train_samples = torch.zeros((n_samples,dataset_train.shape[0],dataset_train.shape[1]-2)).detach()
            latent_space_train_samples = torch.zeros((n_samples,dataset_train.shape[0],int(config["z_dim"]))).detach()
            logits_train_samples = torch.zeros((n_samples,dataset_train.shape[0],dataset_train.shape[1]-2,build_config.aa_prob)).detach()
            #Highlight: Test storage
            aa_sequences_test_samples = torch.zeros((n_samples, patristic_matrix_test[1:].shape[0], dataset_train.shape[1] - 2)).detach()
            latent_space_test_samples = torch.zeros((n_samples, patristic_matrix_test[1:].shape[0], int(config["z_dim"]))).detach()
            logits_test_samples = torch.zeros((n_samples, patristic_matrix_test[1:].shape[0], dataset_train.shape[1] - 2, build_config.aa_prob)).detach()
            for sample_idx,sample in enumerate(samples_names):
                #print("sample idx {}".format(sample_idx))
                map_estimates = guide(dataset_train, patristic_matrix_train, cladistic_matrix_train, clade_blosum=None)
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
            map_estimates = guide(dataset_train, patristic_matrix_train, cladistic_matrix_train, clade_blosum=None)
            pickle.dump(map_estimates, open('{}/Draupnir_Checkpoints/Map_estimates.p'.format(results_dir), 'wb'),protocol=pickle.HIGHEST_PROTOCOL)
            # Highlight: Test storage: Marginal
            aa_sequences_test_samples = torch.zeros((n_samples, patristic_matrix_test[1:].shape[0], dataset_train.shape[1] - 2))
            latent_space_test_samples = torch.zeros((n_samples, patristic_matrix_test[1:].shape[0], int(config["z_dim"])))
            logits_test_samples = torch.zeros((n_samples, patristic_matrix_test[1:].shape[0], dataset_train.shape[1] - 2, build_config.aa_prob))
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
    else: #no training, dummy data
        print("WARNING No training (you have set the number of epochs to 0)!!!!!!! Then, I am creating a meaningless dummy result to avoid code errors. I hope you remembered to load previously trained results from a folder to override it....")
        time.sleep(2)
        #TODO: Check that this works with one hot encoding
        sample_out_train =  sample_out_train_argmax = SamplingOutput(aa_sequences=torch.zeros(dataset_train[:,0,1].shape),
                                      latent_space=torch.ones((dataset_train.shape[0],model_load.z_dim)),
                                      logits=torch.rand((dataset_train.shape[0],model_load.max_seq_len,build_config.aa_prob)),
                                      phis=None,
                                      psis=None,
                                      mean_phi=None,
                                      mean_psi=None,
                                      kappa_phi=None,
                                      kappa_psi=None)
        sample_out_test =  sample_out_test2  = SamplingOutput(aa_sequences=torch.zeros((n_samples,patristic_matrix_test.shape[0]-1,model_load.max_seq_len)),
                                      latent_space=torch.ones((patristic_matrix_test.shape[0]-1,model_load.z_dim)),
                                      logits=torch.rand((patristic_matrix_test.shape[0]-1,model_load.max_seq_len,build_config.aa_prob)),
                                      phis=None,
                                      psis=None,
                                      mean_phi=None,
                                      mean_psi=None,
                                      kappa_phi=None,
                                      kappa_psi=None)
        sample_out_test_argmax =  sample_out_test_argmax2  = SamplingOutput(aa_sequences=torch.zeros((1,patristic_matrix_test.shape[0]-1,model_load.max_seq_len)),
                                      latent_space=torch.ones((patristic_matrix_test.shape[0]-1,model_load.z_dim)),
                                      logits=torch.rand((patristic_matrix_test.shape[0]-1,model_load.max_seq_len,build_config.aa_prob)),
                                      phis=None,
                                      psis=None,
                                      mean_phi=None,
                                      mean_psi=None,
                                      kappa_phi=None,
                                      kappa_psi=None)
    if args.load_trained_predictions:
        print("¡¡¡Loading previously trained results!!!")
        load_folder = args.load_trained_predictions_path
        text_file = open("{}/Hyperparameters_{}_{}epochs.txt".format(results_dir, now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),args.num_epochs), "a")
        text_file.write("Load predictions from TUNED params Model: {}\n".format(load_folder))
        text_file.close()
        Draupnir_state_dict = torch.load("{}/Draupnir_Checkpoints/Model_state_dict.p".format(load_folder))
        Draupnir_optimizer_state = torch.load("{}/Draupnir_Checkpoints/Optimizer_state.p".format(load_folder))
        save_checkpoint_preloaded(Draupnir_state_dict, results_dir, Draupnir_optimizer_state)
        test_dict = torch.load("{}/Test_Plots/test_info_dict.torch".format(load_folder))
        test_argmax_dict = torch.load("{}/Test_argmax_Plots/test_argmax_info_dict.torch".format(load_folder))
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
        def tryexcept(a,key):
            try:
                out = a[key]
            except:
                out = None
            return out
        def load_dict_to_namedtuple(load_dict,sample_out):
            if args.num_epochs > 0:
                sample_out = SamplingOutput(aa_sequences=one_or_another(load_dict),# TODO: the old results have the name as predictions instead of aa_sequences
                                                  latent_space=load_dict["latent_space"],
                                                  logits=load_dict["logits"],
                                                  phis=check_if_exists(sample_out, load_dict, key="phis"),
                                                  psis=check_if_exists(sample_out, load_dict, key="psis"),
                                                  mean_phi=check_if_exists(sample_out, load_dict, key="mean_phi"),
                                                  mean_psi=check_if_exists(sample_out, load_dict, key="mean_psi"),
                                                  kappa_phi=check_if_exists(sample_out, load_dict, key="kappa_phi"),
                                                  kappa_psi=check_if_exists(sample_out, load_dict, key="kappa_psi"))
            else:
                sample_out = SamplingOutput(aa_sequences=one_or_another(load_dict),# TODO: the old results have the name as predictions instead of aa_sequences
                                                  latent_space=load_dict["latent_space"],
                                                  logits=load_dict["logits"],
                                                  phis=tryexcept(load_dict,"phis"), #TODO: try , excep None
                                                  psis=tryexcept(load_dict,"psis"),
                                                  mean_phi=tryexcept(load_dict,"mean_phi"),
                                                  mean_psi=tryexcept(load_dict,"mean_psi"),
                                                  kappa_phi=tryexcept(load_dict,"kappa_phi"),
                                                  kappa_psi=tryexcept(load_dict,"kappa_psi"))

            return sample_out

        sample_out_train = load_dict_to_namedtuple(train_dict,sample_out_train)
        sample_out_test = load_dict_to_namedtuple(test_dict,sample_out_test)
        sample_out_test_argmax = load_dict_to_namedtuple(test_argmax_dict,sample_out_test_argmax)
        try:#some old predictions do not have train_argmax folder
            train_argmax_dict = torch.load("{}/Train_argmax_Plots/train_argmax_info_dict.torch".format(load_folder))
            sample_out_train_argmax = load_dict_to_namedtuple(train_argmax_dict,sample_out_train_argmax)
        except:
            print("Could not load train_argmax folders")
            pass
        train_entropies = train_dict["entropies"]
        test_entropies = test_dict["entropies"]
        try:  # some old predictions do not have test2 folders
            test_argmax_dict2 = torch.load("{}/Test2_argmax_Plots/test2_argmax_info_dict.torch".format(load_folder))
            test_dict2 = torch.load("{}/Test2_Plots/test_info_dict2.torch".format(load_folder))
            sample_out_test_argmax2 = load_dict_to_namedtuple(test_argmax_dict2, sample_out_test_argmax2)
            sample_out_test2 = load_dict_to_namedtuple(test_dict2,sample_out_test2)
            test_entropies2 = test_dict2["entropies"]
        except:
            print("Could not load test2 folders!!!---> the results are not from a trained run")
            test_entropies2 = test_entropies
            pass


        #Highlight: Saving the pre-trained predictions! It overwrites the trained ones
        save_samples(dataset_test, patristic_matrix_test,sample_out_test, test_entropies,correspondence_dict,"{}/test_info_dict.torch".format(results_dir + "/Test_Plots"))
        save_samples(dataset_test,patristic_matrix_test, sample_out_test_argmax,test_entropies, correspondence_dict,"{}/test_argmax_info_dict.torch".format(results_dir + "/Test_argmax_Plots"))
        save_samples(dataset_test, patristic_matrix_test,sample_out_test2, test_entropies2,correspondence_dict,"{}/test_info_dict2.torch".format(results_dir + "/Test2_Plots"))
        save_samples(dataset_test,patristic_matrix_test,sample_out_test_argmax2,test_entropies2, correspondence_dict,"{}/test2_argmax_info_dict.torch".format(results_dir + "/Test2_argmax_Plots"))
        save_samples(dataset_train, patristic_matrix_train,sample_out_train, train_entropies,correspondence_dict,"{}/train_info_dict.torch".format(results_dir + "/Train_Plots"))
        save_samples(dataset_train, patristic_matrix_train,sample_out_train_argmax, train_entropies,correspondence_dict,"{}/train_argmax_info_dict.torch".format(results_dir + "/Train_argmax_Plots"))
    else:
        if args.num_epochs == 0: raise ValueError("You forgot to load previously trained predictions !")
        else: pass


    #Highlight: Concatenate leaves and internal latent space for plotting
    visualize_latent_space(sample_out_train_argmax.latent_space,sample_out_test_argmax.latent_space,patristic_matrix_train,patristic_matrix_test,additional_load,additional_info,folder="")
    visualize_latent_space(sample_out_train_argmax.latent_space,sample_out_test_argmax2.latent_space,patristic_matrix_train,patristic_matrix_test,additional_load,additional_load,folder="/Test2_Plots")

    if settings_config.one_hot_encoding:
        print("Transforming one-hot back to integers")
        sample_out_train_argmax = convert_to_integers(sample_out_train_argmax) #argmax sets directly the aa to the highest logit
        sample_out_test_argmax = convert_to_integers(sample_out_test_argmax)
        sample_out_test_argmax2 = convert_to_integers(sample_out_test_argmax2)
        dataset_train = DraupnirUtils.convert_to_integers(dataset_train.cpu(),build_config.aa_prob,axis=2)
        if build_config.leaves_testing: #TODO: Check that this works
            dataset_test = DraupnirUtils.convert_to_integers(dataset_test.cpu(),build_config.aa_prob,axis=2) #no need to do it with the test of the simulations, never was one hot encoded. Only for testing leaves

    start_plots = time.time()
    #aa_sequences_predictions_test = dataset_test[:,2:,0].repeat(50,1,1)
    #aa_sequences_predictions_train = dataset_train[:, 2:, 0].repeat(50, 1, 1)
    if n_samples != sample_out_test.aa_sequences.shape[0]:
        n_samples = sample_out_test.aa_sequences.shape[0]
    #

    if args.infer_angles: #TODO: not correct anymore
        preparing_plots(name,
                        sample_out_train,
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
        preparing_plots(name,
                        sample_out_test,
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
        preparing_plots(name,
                        sample_out_test2,
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

    #Highlight: Plot samples
    print("train")
    preparing_plots(name,
                    sample_out_train,
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
    print("test")
    preparing_plots(name,
                    sample_out_test,
                    dataset_test,
                    dataset_train,
                    test_entropies,
                    results_dir + "/Test_Plots",
                    additional_load,
                    additional_info,
                    build_config,
                    n_samples,
                    patristic_matrix_test[1:,0],
                    args,
                    replacement_plots=False,
                    overplapping_hist=False,
                    no_testing=build_config.no_testing)
    print("test2")
    preparing_plots(name,
                    sample_out_test2,
                    dataset_test,
                    dataset_train,
                    test_entropies2,
                    results_dir + "/Test2_Plots",
                    additional_load,
                    additional_info,
                    build_config,
                    n_samples,
                    patristic_matrix_test[1:,0],
                    args,
                    replacement_plots=False,overplapping_hist=False,no_testing=build_config.no_testing)

    if n_samples != sample_out_test_argmax.aa_sequences.shape[0]:#most likely sequences ---> most voted sequence now?
        n_samples = sample_out_test_argmax.aa_sequences.shape[0]

    #Highlight: Plot most likely sequence
    print("train argmax")
    preparing_plots(name,
                    sample_out_train_argmax,
                    dataset_train,
                    dataset_train,
                    train_entropies,
                    results_dir + "/Train_argmax_Plots",
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
    print("test argmax")
    preparing_plots(name,
                    sample_out_test_argmax,
                    dataset_test,
                    dataset_train,
                    test_entropies,
                    results_dir + "/Test_argmax_Plots",
                    additional_load,
                    additional_info,
                    build_config,
                    n_samples,
                    patristic_matrix_test[1:,0],
                    args,
                    replacement_plots=False,
                    overplapping_hist=False,
                    no_testing=build_config.no_testing)
    print("test_argmax 2")
    preparing_plots(name,
                    sample_out_test_argmax2,
                    dataset_test,
                    dataset_train,
                    test_entropies2,
                    results_dir + "/Test2_argmax_Plots",
                    additional_load,
                    additional_info,
                    build_config,
                    n_samples,
                    patristic_matrix_test[1:,0],
                    args,
                    replacement_plots=False,
                    overplapping_hist=False,
                    no_testing=build_config.no_testing)

    stop_plots = time.time()
    print('Final plots timing: {}'.format(str(datetime.timedelta(seconds=stop_plots - start_plots))))
    print("##########################################################################################################")
def preparing_plots(name,
                    samples_out,
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
    :param namedtuple samples_out: named tuple containing the output from Draupnir
    dataset_true: the true dataset, it can be the train or the test dataset
    entropies
    results_dir
    additional_load: Named tuple
    n_samples: number of samples from the model
    test_ordered_nodes:
    no_testing : There is nota test dataset with the true sequences
    """
    if not additional_load.correspondence_dict:
        correspondence_dict = dict(zip(list(range(len(additional_load.tree_levelorder_names))), additional_load.tree_levelorder_names))
    else:
        correspondence_dict = additional_load.correspondence_dict
    if plot_test and no_testing:
        print("Print no testing!")
        DraupnirPlots.save_ancestors_predictions(name, dataset_true, samples_out.aa_sequences, n_samples, results_dir,
                                   correspondence_dict, build_config.aa_prob)
    elif plot_test and name in ["Douglas_SRC","Coral_Faviina","Coral_all"] or plot_test and name.endswith("_subtree"):
        #Highlight: select from the predictions only the sequences in the dataset_test. Remove gaps and align to the "observed"
        DraupnirPlots.save_ancestors_predictions_coral(name, test_ordered_nodes, samples_out.aa_sequences, n_samples, results_dir,
                                                 correspondence_dict, build_config.aa_prob)
        DraupnirPlots.CleanRealign_Train(name,
                                   dataset_true,
                                   dataset_train,
                                   samples_out.aa_sequences,  # test predictions
                                   test_ordered_nodes,
                                   n_samples,
                                   build_config.aa_prob,
                                   results_dir,
                                   additional_load,
                                   additional_info)
        DraupnirPlots.CleanRealign(name,
                                   dataset_true,
                                   samples_out.aa_sequences,#test predictions
                                   test_ordered_nodes,
                                   n_samples,
                                   build_config.aa_prob,
                                   results_dir,
                                   additional_load,
                                   additional_info)



        DraupnirPlots.plot_entropies(name, entropies.detach().cpu().numpy(), results_dir, correspondence_dict)
    elif args.infer_angles and plot_angles:
        DraupnirPlots.plotting_angles(samples_out,dataset_true,results_dir,additional_load,additional_info,n_samples,test_ordered_nodes)
        DraupnirPlots.plotting_angles_per_aa(samples_out,dataset_true,results_dir,build_config,additional_load,additional_info,n_samples,test_ordered_nodes)
    else:
        DraupnirPlots.plotting_heatmap_and_incorrect_aminoacids(name,
                                                                dataset_true,
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
                                                dataset_true,
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
            "gru_hidden_dim": 60, #60
        }
    return config
def manual_random_search():
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
def draupnir_main(name,root_sequence_name,args,device,settings_config,build_config,script_dir):

    #global params_config,build_config,name,results_dir,max_seq_len,full_name
    results_dir = "{}/PLOTS_GP_VAE_{}_{}_{}epochs_{}".format(script_dir, name, now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),
                                                             args.num_epochs, args.select_guide)
    print("Loading datasets....")
    param_config = config_build(args)
    train_load,test_load,additional_load,build_config = load_data(name,settings_config,build_config,param_config,results_dir,script_dir,args)
    exit()
    #max_seq_len = additional_load.alignment_length
    additional_info=DraupnirUtils.extra_processing(additional_load.ancestor_info_numbers, additional_load.patristic_matrix_full,results_dir,args,build_config)
    train_load,test_load,additional_load= datasets_pretreatment(name,root_sequence_name,train_load,test_load,additional_load,build_config,device,name,settings_config)
    torch.save(torch.get_rng_state(),"{}/rng_key.torch".format(results_dir))
    print("Starting Draupnir ...")
    print("Dataset: {}".format(name))
    print("Number epochs: {}".format(args.num_epochs))
    print("Z/latent Size: {}".format(config["z_dim"]))
    print("GRU hidden size: {}".format(config["gru_hidden_dim"]))
    print("Number train sequences: {}".format(train_load.dataset_train.shape[0]))
    n_test = [test_load.dataset_test.shape[0] if test_load.dataset_test is not None else 0][0]
    print("Number test sequences: {}".format(n_test))
    print("Selected Substitution matrix : {}".format(args.subs_matrix))

    if not args.batch_by_clade:
        clades_dict=None
    else:
        clades_dict = additional_load.clades_dict_leaves
    graph_coo = None #Highlight: use only with the GNN models (7)---> Otherwise is in additional_info
    #graph_coo = additional_info.graph_coo
    if args.generate_samples:
        print("Generating samples not training!")
        draupnir_sample(train_load,
                            test_load,
                            additional_load,
                            additional_info,
                            build_config,
                            settings_config,
                            args.n_samples,
                            args,
                            device,
                            script_dir,
                            results_dir,
                            graph_coo,
                            clades_dict)
    else:
        draupnir_train(train_load,
                       test_load,
                       additional_load,
                       additional_info,
                       build_config,
                       settings_config,
                       args.n_samples,
                       args,
                       device,
                       script_dir,
                       results_dir,
                       graph_coo,
                       clades_dict)





