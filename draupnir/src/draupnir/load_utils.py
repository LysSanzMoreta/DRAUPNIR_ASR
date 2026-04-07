"""
=======================
2022: Lys Sanz Moreta
Draupnir : Ancestral protein sequence reconstruction using a tree-structured Ornstein-Uhlenbeck variational autoencoder
=======================
"""
import itertools
from collections import defaultdict
import os
import numpy as np
import pandas as pd
import torch
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO
import draupnir.utils as DraupnirUtils
import draupnir.models_utils as DraupnirModelUtils
import draupnir.datasets as DraupnirDatasets
from collections import namedtuple
from torch.utils.data import Dataset, DataLoader
from dill import Unpickler
from typing import Union



SamplingOutput = namedtuple("SamplingOutput",["aa_sequences","latent_space","logits","phis","psis","mean_phi","mean_psi","kappa_phi","kappa_psi"])
TrainLoad = namedtuple('TrainLoad', ['dataset_train', 'evolutionary_matrix_train', 'patristic_matrix_train','cladistic_matrix_train','embeddings_train'])
TestLoad = namedtuple('TestLoad',
                      ['dataset_test', 'evolutionary_matrix_test', 'patristic_matrix_test','cladistic_matrix_test',"leaves_names_test",
                       "position_test", "internal_nodes_indexes"])
AdditionalLoad = namedtuple("AdditionalLoad",
                            ["patristic_matrix_full", "cladistic_matrix_full","children_array", "ancestor_info_numbers", "alignment_length",
                             "tree_levelorder_names", "clades_dict_leaves", "closest_leaves_dict","clades_dict_all","linked_nodes_dict","descendants_dict","aa_frequencies_train","aa_frequencies_test",
                             "correspondence_dict","special_nodes_dict","full_name"])

def load_tree_levelorder_names(data_folder,name):
    ancestor_info = pd.read_csv("{}/{}_tree_levelorder_info.csv".format(data_folder,name), sep="\t",index_col=False,low_memory=False)
    ancestor_info["0"] = ancestor_info["0"].astype(str)
    ancestor_info.drop('Unnamed: 0', inplace=True, axis=1)
    nodes_names = ancestor_info["0"].tolist()
    tree_levelorder_names = np.asarray(nodes_names)
    return tree_levelorder_names,nodes_names, ancestor_info
def convert_clades_dict(name,clades_dict,leave_nodes_dict,internal_nodes_dict, only_leaves):
    """Transforms the names of the nodes to their tree transversal level order number
    :param str name
    :param dict clades_dict: dictionary containing the tree organized by clades, with the naming stablished in the newick tree (a string)
    :param dict leave_nodes_dict: dictionary that contains the name of the leaf and the tre and its correspondant index in tree level order
    :param dict internal_nodes_dict: dictionary that contains the name of the internal node and the tre and its correspondant index in tree level order
    :param bool only_leaves: True --> Translate only the leaves to tree level order index
    """
    if only_leaves:
        values_list = []
        if name == "benchmark_randall_original_naming": #TODO: needs to be checked
            for key,values in clades_dict.items():
                #for val in values: #matrix.index.str.replace("A","").astype("int")
                if len(values) > 1:
                    vals_list = []
                    for val in values:
                        vals_list.append(int(val.replace("A","")))
                    values_list.append(sorted(vals_list))
                else:
                    values_list.append(int(values[0].replace("A","")))

        else:
            for key, values in clades_dict.items():
                if len(values) > 1: #should not be necessary anymore
                    vals_list = []
                    for val in values:
                        vals_list.append(leave_nodes_dict[val])
                    values_list.append(sorted(vals_list))
                else:
                    values_list.append(leave_nodes_dict[values[0]])
        clades_dict = dict(zip(clades_dict.keys(), values_list))
        return clades_dict
    else:
        clades_dict_renamed = defaultdict(lambda: defaultdict())

        if name == "benchmark_randall_original_naming":
            for key, values in clades_dict.items():
                internal_list = []
                leaves_list = []
                for internal_node in clades_dict[key]["internal"]:
                    internal_list.append(int(internal_node.replace("A","")))
                for leave_node in clades_dict[key]["leaves"]:
                    leaves_list.append(int(leave_node.replace("A",""))) #should not be necessary to replace, the leaves have numbers as names
                clades_dict_renamed[key]["internal"] = sorted(internal_list)
                clades_dict_renamed[key]["leaves"] = sorted(leaves_list)
        else:
            for key, values in clades_dict.items():
                internal_list = []
                leaves_list = []
                for internal_node in clades_dict[key]["internal"]:
                    internal_list.append(internal_nodes_dict[internal_node])
                for leave_node in clades_dict[key]["leaves"]:
                    leaves_list.append(leave_nodes_dict[leave_node])
                clades_dict_renamed[key]["internal"] = sorted(internal_list)
                clades_dict_renamed[key]["leaves"] = sorted(leaves_list)

        return clades_dict_renamed
def convert_closest_leaves_dict(name,closest_leaves_dict,internal_nodes_dict,leave_nodes_dict):
    """Transforms the names of the nodes to their tree transversal level order number
    :param str name
    :param dict closest_leaves_dict: dictionary containing the internal nodes and the immediate closest leaves
    :param dict internal_nodes_dict: dictionary that contains the name of the internal node and the tre and its correspondant index in tree level order
    :param dict leave_nodes_dict: dictionary that contains the name of the leaf and the tre and its correspondant index in tree level order
    """
    keys_list = []
    values_list = []
    if name == "benchmark_randall_original_naming":
        for key, values in closest_leaves_dict.items():
            keys_list.append(int(key.replace("A","")))
            if len(values) > 1:
                vals_list = []
                for val in values:
                    vals_list.append(int(val))
                values_list.append(vals_list)
            else:
                values_list.append(int(values[0]))
    else:
        for key, values in closest_leaves_dict.items():
            keys_list.append(internal_nodes_dict[key])
            if len(values) > 1:
                vals_list = []
                for val in values:
                    vals_list.append(leave_nodes_dict[val])
                values_list.append(vals_list)
            else:
                values_list.append(leave_nodes_dict[values[0]])
    closest_leaves_dict = dict(zip(keys_list, values_list))
    return closest_leaves_dict
def convert_only_linked_children(name,linked_nodes_dict,internal_nodes_dict,leaves_nodes_dict):
    """Transform the nodes names to tree transversal level order
    :param str name
    :param dict linked_nodes_dict: dictionary containing the 2 children nodes directly linked to a node (not all the children from that node) {node:children}
    :param dict internal_nodes_dict: dictionary that contains the name of the internal node and the tre and its correspondant index in tree level order
    :param dict leave_nodes_dict: dictionary that contains the name of the leaf and the tre and its correspondant index in tree level order"""
    #merge the internal nodes and leaves dict
    all_nodes_dict = {**internal_nodes_dict, **leaves_nodes_dict}
    if name == "benchmark_randall_original_naming": #leaves keep their numeration , internal nodes have A in front
        linked_nodes_renamed_dict = defaultdict()
        for key, val in linked_nodes_dict.items():
            if val:
                linked_nodes_renamed_dict[int(key.replace("A",""))] = [int(children.replace("A","")) for children in val]
            else:
                linked_nodes_renamed_dict[int(key.replace("A",""))] = []

    else:
        linked_nodes_renamed_dict = defaultdict()
        for key, val in linked_nodes_dict.items():
            if val:
                linked_nodes_renamed_dict[all_nodes_dict[key]] = [all_nodes_dict[children] for children in val]
            else:
                linked_nodes_renamed_dict[all_nodes_dict[key]] = []
    return linked_nodes_renamed_dict
# def convert_nearest_leaf(name, nearest_leaf_dict,leaves_nodes_dict):
#     nearest_leaf_renamed_dict = defaultdict()
#     if name == "benchmark_randall_original_naming": #leaves keep their numeration , internal nodes have A in front
#         for key, val in nearest_leaf_dict.items():
#             nearest_leaf_renamed_dict[int(key.replace("A",""))] = int(val.replace("A",""))
#     else:
#         for key, val in nearest_leaf_dict.items():
#
#             nearest_leaf_renamed_dict[leaves_nodes_dict[key]] = leaves_nodes_dict[val]
#     return nearest_leaf_renamed_dict
def convert_descendants(name, descendants_dict,internal_nodes_dict,leave_nodes_dict):
    """Transform the nodes names to tree transversal level order
    :param str name
    :param dict descendants_dict: dictionary containing a node and all of its descendants (internal and leaves){node:descendants}
    :param dict internal_nodes_dict: dictionary that contains the name of the internal node and the tre and its correspondant index in tree level order
    :param dict leave_nodes_dict: dictionary that contains the name of the leaf and the tre and its correspondant index in tree level order"""
    if name == "benchmark_randall_original_naming": #TODO: check that is correct
        descendants_dict_renamed = defaultdict(lambda: defaultdict())
        for key, values in descendants_dict.items():
            new_node_name = int(key.replace("A", ""))
            internal_list = []
            leaves_list = []
            for internal_node in descendants_dict[key]["internal"]:
                internal_list.append(int(internal_node.replace("A", "")))
            for leave_node in descendants_dict[key]["leaves"]:
                leaves_list.append(int(leave_node.replace("A", "")))
            descendants_dict_renamed[new_node_name]["internal"] = sorted(internal_list)
            descendants_dict_renamed[new_node_name]["leaves"] = sorted(leaves_list)
    else:
        descendants_dict_renamed = defaultdict(lambda : defaultdict())
        for key, values in descendants_dict.items():
            internal_list = []
            leaves_list = []
            for internal_node in descendants_dict[key]["internal"]:
                internal_list.append(internal_nodes_dict[internal_node])
            for leave_node in descendants_dict[key]["leaves"]:
                leaves_list.append(leave_nodes_dict[leave_node])
            new_node_name = internal_nodes_dict[key]
            descendants_dict_renamed[new_node_name]["internal"] = sorted(internal_list)
            descendants_dict_renamed[new_node_name]["leaves"] = sorted(leaves_list)

    return descendants_dict_renamed
def create_children_array(dataset,ancestor_info_numbers):
    """Group nodes by common ancestors and update dataset with new information about immediate ancestors of the leaves
     :param dataset: leaves dataset [n_leaves, L, 30]
     :param ancestor_info_numbers: array containing  [ancestor number,root distance, ancestors,...]
     :out dataset: it is updated in the 1st row, 3rd position of every node with the immediate ancestor
     :out children_array: contains the node and then the list of descendants"""
    c = ancestor_info_numbers[:, [0, 2]] #column 0 contains the node of interest and the second one the immediate ancestor (column 1 is distance to root)
    unique_nodes = np.unique(c[:, 1].astype(str), return_index=True)  # returns ordered nodes and indexes # Highlight: cast the values as string type, otherwise np.unique messes up with the string types
    v = np.split(c[:, 0], unique_nodes[1])[1:]
    length = max(map(len, v))
    children_array = np.array([xi.tolist() + [None] * (length - len(xi)) for xi in v])
    # unique_nodes = np.array(pd.DataFrame(c[:, 1]).drop_duplicates().values).squeeze(-1) #Highlight: Replaces np.unique(c[:, 1]
    children_array = np.concatenate((unique_nodes[0].astype(float)[:, np.newaxis], children_array), axis=1)
    # Add ancestors to Dataset, to the 1 row, in position 3 ---> Skip row 0 (name)
    dataset[:, 1, 3] = [c[np.isin(c[:, 0], node)][0][1] for node in dataset[:, 1, 1]]
    return dataset,children_array
def convert_ancestor_info(name,ancestor_info,tree_levelorder_names):
    """Transforms the nodes names to their tree lever order index in the ancestors dataframe
    :param str name: project data name
    :param pandas-array ancestor_info: dataframe that contains on each row a leaf node and all of its ancestors
    :param numpy-array tree_levelorder_names: array with the nodes and their tree level order names"""
    # Highlight: Assign nodes to their tree level order index
    ancestor_info = ancestor_info.to_numpy()
    ancestor_info_numbers = ancestor_info  # keep a copy
    if name == "benchmark_randall_original_naming":
        for index_row, row in enumerate(ancestor_info):
            for index_element, element in enumerate(row):
                if not isinstance(element, int) and not isinstance(element, float) and not pd.isnull(element) and element != "nan":
                    ancestor_info_numbers[index_row, index_element] = int(element.replace("A",""))
                elif element == "nan" or pd.isnull(element):
                    ancestor_info_numbers[index_row, index_element] = np.nan
    else:
        for index_row, row in enumerate(ancestor_info):
            for index_element, element in enumerate(row):
                if not isinstance(element, int) and not isinstance(element, float) and not pd.isnull(element) and element != "nan":
                    ancestor_info_numbers[index_row, index_element] = np.where(tree_levelorder_names == element)[0][0]
                elif element == "nan" or pd.isnull(element):
                    ancestor_info_numbers[index_row, index_element] = np.nan
    return ancestor_info_numbers
def validate_aa_probs(alignment,build_config):
    """Validating that the pre-selected amino acid probabilities are correct.
    If the data set contains more character types than the selected ones, it updates the aa_probs. It also detects DNA
    :param alignment; biopython alignment"""

    align_array = np.array([record.seq for record in alignment])
    different_elements = "".join(np.unique(align_array).tolist())
    aa_probs_updated = DraupnirUtils.validate_sequence_alphabet(different_elements.lower())
    return aa_probs_updated

def sort_distance_matrix(distance_matrix):
    if distance_matrix is not None:
        # Find the smallest distance among the sequences
        min_row, min_column = distance_matrix[distance_matrix.gt(0)].stack().idxmin()  # distance_matrix.loc[[min_row], [min_column]]
        sorted_distance_matrix = distance_matrix[distance_matrix.gt(0)].stack().sort_values().to_frame()
        sorted_distance_matrix.reset_index(level=0, inplace=True)
        sorted_distance_matrix.columns = ["Sequence_0", "Distance"]
        sorted_distance_matrix['Sequence_1'] = sorted_distance_matrix.index
        sorted_distance_matrix = sorted_distance_matrix[["Sequence_1", "Sequence_0", "Distance"]]
        sorted_distance_matrix.reset_index(inplace=True)
        sorted_distance_matrix.drop(["index"], inplace=True, axis=1)
        # sorted_distance_matrix.drop_duplicates(subset=['Distance'])
        sorted_distance_matrix = sorted_distance_matrix[~sorted_distance_matrix[['Sequence_1', 'Sequence_0']].apply(frozenset,axis=1).duplicated()]  # Remove repeated combinations of sequences
        sorted_distance_matrix = sorted_distance_matrix.reset_index(drop=True)
    else: #This has been fixed already
        sorted_distance_matrix = None
    return sorted_distance_matrix

def load_pairwise_distance_matrix(name,script_dir,sort=False):
    """Reads any of the available pairwise distances file and sorts them in pairs in ascending order
    :param str name
    :param str script_dir"""


    train_pairwise_distance_matrix_file = "{}/{}_pairwise_distance_matrix_TRAIN.csv".format(script_dir,name) #file computed while creating the dataset
    test_pairwise_distance_matrix_file = "{}/{}_pairwise_distance_matrix_TEST.csv".format(script_dir,name) #file computed while creating the dataset

    if os.path.exists(train_pairwise_distance_matrix_file) and os.path.exists(test_pairwise_distance_matrix_file):
        train_pairwise_distance_matrix = pd.read_csv(train_pairwise_distance_matrix_file, index_col=0)
        test_pairwise_distance_matrix = pd.read_csv(test_pairwise_distance_matrix_file, index_col=0)
        #todo: i have not calculated the distances between the train and the test, since, in principle we do not need them (they should not be used)

        distance_matrix = pd.concat([train_pairwise_distance_matrix,test_pairwise_distance_matrix],axis=0)
        #distance_matrix.fillna(0)

    else:
        distance_matrix = None

    if sort:
        sorted_distance_matrix = sort_distance_matrix(distance_matrix)
    else:
        sorted_distance_matrix = None

    return distance_matrix,sorted_distance_matrix

def remove_nan2(dataset): #TODO: remove if its not been used anywhere
    """Detect and remove nan angle pair, where either phi or psi are nan, due mostly to nh3 or coo terminals.
    We assign the aminoacid where there are any nan values to gap
    :param numpy-array dataset: [N_leaves, align_len,30]"""
    aa_angles = dataset[:,3:,0:3].astype(float) #[n_seq,len_seq,[phi,psi]]
    #print(angles[np.isnan(angles)])
    indexes_nan = np.argwhere(np.isnan(aa_angles))#.squeeze(0) #n_matches, dim0,dim1,dim2 ---> we only want dim0 and dim1
    #nan_angles = angles[indexes_nan[:,0],indexes_nan[:,1],indexes_nan[:,2]] #all angles that are nan
    aa_angles[indexes_nan[:, 0], indexes_nan[:, 1]] = 0. #update all the angles to 0 and asign the aa  to 0 (gap), will it fail when more than one match?
    #simply update the angles inside dataset
    dataset[:, 3:, 0:3] = aa_angles
    return dataset
def remove_nan(dataset:Union[np.ndarray |None]):
    """Detect and remove nan angle pair, where either phi or psi are nan, due mostly to nh3 or coo terminals.
    To solve it, we assign the aminoacid where there are any nan values to gap
    :param numpy-array dataset: [N_leaves, align_len,30]"""
    if dataset is not None:
        aa_angles = dataset[:,3:,0:3].astype(float) #[n_seq,len_seq,[phi,psi]]
        aa_angles = np.apply_along_axis(lambda r: np.zeros_like(r) if np.isnan(r).any() else r, 2, aa_angles)
        dataset[:, 3:, 0:3] = aa_angles
        return dataset
    else:
        return dataset
def processing_helper(distance_matrix,ancestral,leaves_names_test,nodes,name):
    distance_matrix = DraupnirUtils.symmetrize_and_clean(distance_matrix, keep_ancestral=ancestral)

    distance_matrix_test = distance_matrix.loc[distance_matrix.index.isin(leaves_names_test)]
    distance_matrix_test = distance_matrix_test.loc[:, distance_matrix.index.isin(leaves_names_test)]

    distance_matrix_train = distance_matrix.loc[~distance_matrix.index.isin(leaves_names_test)]
    distance_matrix_train = distance_matrix_train.loc[:, ~distance_matrix.index.isin(leaves_names_test)]

    position_test = distance_matrix_test.index.tolist()
    position_test = [np.where(distance_matrix.index == name)[0][0] for name in position_test]

    distance_matrix_train = DraupnirUtils.rename_axis(distance_matrix_train, nodes, name_file=name)
    distance_matrix_test = DraupnirUtils.rename_axis(distance_matrix_test, nodes, name_file=name)
    
    return dict(d = distance_matrix,
                d_train = distance_matrix_train,
                d_test = distance_matrix_test,
                position_test = position_test)
def train_data_processing(args:namedtuple,
               settings_config,
               results_dir:str,
               dataset:np.ndarray,
               patristic_matrix,
               cladistic_matrix,
               pairwise_distance_matrix,
               n_seq_train,
               n_seq_test,
               now,
               name:str,
               aa_probs:Union[int | float],
               leaves_names_list:list,
               one_hot_encoding: bool,
               embeddings : Union[np.ndarray | None],
               sequence_representations: Union[np.ndarray | None],
               nodes:list=[],
               ancestral:bool =True,
               ):
    """Divides the dataset into train and test, also the evolutionary/patristic matrices.
    :param str results_dir
    :param pandas-array dataset
    :param patristic_matrix: contains all the patristic distances or branch lengths between the nodes in the tree
    :param cladistic_matrix: contains the number of nodes that separate the nodes in the tree
    :param sorted_distance_matrix: contains the pairwise distances between the nodes in the tree
    :param n_seq_train: not used, supposedly for selecting the number of train sequences to use, using all
    :param n_seq_test: used with leaf testing, selects a percentage of leaves, randomly, to act as the test sequences
    :param str-time now: inherit the execution time of the main script to open the Hyperparameters file
    :param name: data project name
    :param aa_probs: amino acid probabilities
    :param leaves_nodes: list of the names of the leaves nodes
    :param one_hot_encoding: bool
    :param nodes: tree level order names of the nodes
    :param ancestral : Keeps(true) or discard (false) the ancestral nodes from the dataset. We use False when we are performing leaf testing"""


    if n_seq_test == 0:
        dataset_train = dataset
        embeddings_train = embeddings
        sequence_representations_train = sequence_representations

        # Write the train sequences to a fasta file
        with open("{}/{}_training.fasta".format(results_dir, name), "w") as output_handle,open("{}/{}_training_aligned.fasta".format(results_dir, name), "w") as output_handle2:
            records = []
            records_aligned = []
            for sequence in dataset_train:
                if one_hot_encoding:
                    sequence_to_translate = np.argmax(sequence[3:,0:21],axis=1)
                else:
                    sequence_to_translate = sequence[3:,0]
                train_sequence = DraupnirUtils.convert_to_letters(sequence_to_translate, aa_probs) #TODO: vectorize
                record = SeqRecord(Seq(''.join(train_sequence).replace("-", "")),
                                   annotations={"molecule_type": "protein"},
                                   id=str(sequence[0, 0]), description="") #TODO: There is an error here for the one-hot encoded version!!!
                records.append(record)
                record_aligned = SeqRecord(Seq(''.join(train_sequence)),
                                   annotations={"molecule_type": "protein"},
                                   id=str(sequence[0, 0]),
                                   description="")  # TODO: There is an error here for the one-hot encoded version!!! we are not using it anyways
                records_aligned.append(record_aligned)

            SeqIO.write(records, output_handle, "fasta")
            SeqIO.write(records_aligned, output_handle2, "fasta")
        dataset_train = dataset_train[:, 1:].astype('float64') # Eliminate the str name part in the row, so that not everything needs to be changed
        embeddings_train = embeddings_train[:,1:].astype("float64") if embeddings_train is not None else None


        dataset_test = None
        embeddings_test = None


        patristic_matrix_train = DraupnirUtils.symmetrize_and_clean(patristic_matrix,keep_ancestral=ancestral)  # Highlight: if ancestral is true, patristic_matrix_train = patristic_matrix_full

        patristic_matrix_train = DraupnirUtils.rename_axis(patristic_matrix_train, nodes, name_file=name)

        if pairwise_distance_matrix is not None:

            pairwise_matrix_train =  DraupnirUtils.symmetrize_and_clean(pairwise_distance_matrix,keep_ancestral=ancestral)
            pairwise_matrix_train = DraupnirUtils.rename_axis(pairwise_matrix_train, nodes, name_file=name)
            pairwise_matrix_train = DraupnirUtils.pandas_to_numpy(pairwise_matrix_train)

        else:
            pairwise_matrix_train = None

        if cladistic_matrix is not None:
            cladistic_matrix_train = DraupnirUtils.symmetrize_and_clean(cladistic_matrix, keep_ancestral=ancestral)
            cladistic_matrix_train = DraupnirUtils.rename_axis(cladistic_matrix_train, nodes, name_file=name)
            evolutionary_matrix_train = DraupnirUtils.sum_matrices(patristic_matrix_train, cladistic_matrix_train)
            evolutionary_matrix_train = np.array(evolutionary_matrix_train)
            cladistic_matrix_train = DraupnirUtils.pandas_to_numpy(cladistic_matrix_train)
        else:
            print("No cladistic matrix available. Evolutionary matrix = Patristic matrix")
            evolutionary_matrix_train = DraupnirUtils.pandas_to_numpy(patristic_matrix_train)
            cladistic_matrix_train = None

        patristic_matrix_train = DraupnirUtils.pandas_to_numpy(patristic_matrix_train)
        patristic_matrix_test = None
        cladistic_matrix_test = None
        pairwise_matrix_test = None

        evolutionary_matrix_test = None
        position_test = [None, None]
        leaves_names_test = None

    else: #pick some of the train sequences as test
        n_train = dataset.shape[0]
        n_test = int(dataset.shape[0] * n_seq_test/ 100)
        #find the leaves in the 25%-75% area, in the middle, not the extremes
        n_25 = int((n_train*25)/100)
        n_75 = int((n_train*75)/100)
        test_indexes = np.random.choice(np.arange(n_25,n_75),n_test,replace=False)
        leaves_names_test = [leaves_names_list[index] for index in test_indexes]

        # rank = 100  # Higher rank, more pairwise distance
        # sequence_0 = sorted_distance_matrix.loc[rank].Sequence_0
        # sequence_1 = sorted_distance_matrix.loc[rank].Sequence_1
        #
        print("Using {} leaves for testing".format(len(leaves_names_test)))
        file_hyperparams = open("{}/Hyperparameters_{}_{}.txt".format(results_dir, now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"), results_dir.split("_")[-1]), "a")
        file_hyperparams.write('Test sequences names: {} \n'.format(leaves_names_test))
        file_hyperparams.close()


        test_idx = np.isin(dataset[:, 0, 0], leaves_names_test)
        dataset_test = dataset[test_idx]
        dataset_train = dataset[np.logical_not(test_idx)]

        embeddings_train = embeddings[~test_idx] if embeddings is not None else None
        embeddings_test = embeddings[test_idx] if embeddings is not None else None

        sequence_representations_train = sequence_representations[~test_idx]

        # Write the train sequences to a fasta file to assess the conversion was correct
        # with open("{}/{}_training.fasta".format(results_dir, name), "w") as output_handle:
        #     records = []
        #     for sequence in dataset_train: #todo: this can be done with vectorization
        #         train_sequence = DraupnirUtils.convert_to_letters(sequence[2:, 0],aa_probs)
        #         record = SeqRecord(Seq(''.join(train_sequence).replace("-", "")),
        #                            annotations={"molecule_type": "protein"},
        #                            id=sequence[0, 0], description="")
        #         records.append(record)
        #     SeqIO.write(records, output_handle, "fasta")


        # patristic_matrix = DraupnirUtils.symmetrize_and_clean(patristic_matrix, keep_ancestral=ancestral)
        #
        # patristic_matrix_test = patristic_matrix.loc[patristic_matrix.index.isin(leaves_names_test)]
        # patristic_matrix_test = patristic_matrix_test.loc[:,patristic_matrix.index.isin(leaves_names_test)]
        #
        # patristic_matrix_train = patristic_matrix.loc[~patristic_matrix.index.isin(leaves_names_test)]
        # patristic_matrix_train = patristic_matrix_train.loc[:,~patristic_matrix.index.isin(leaves_names_test)]
        #
        # position_test = patristic_matrix_test.index.tolist()
        # position_test = [np.where(patristic_matrix.index == name)[0][0] for name in position_test]
        #
        # patristic_matrix_train = DraupnirUtils.rename_axis(patristic_matrix_train, nodes, name_file=name)
        # patristic_matrix_test = DraupnirUtils.rename_axis(patristic_matrix_test, nodes, name_file=name)

        out_patristic = processing_helper(patristic_matrix,ancestral,leaves_names_test,nodes,name)
        patristic_matrix,patristic_matrix_train,patristic_matrix_test,position_test = out_patristic["d"],out_patristic["d_train"], out_patristic["d_test"],out_patristic["position_test"]

        if pairwise_distance_matrix is not None:
            out_pairwise = processing_helper(pairwise_distance_matrix,ancestral,leaves_names_test,nodes,name)
            pairwise_matrix, pairwise_matrix_train, pairwise_matrix_test, position_test = out_pairwise["d"], \
            out_pairwise["d_train"], out_pairwise["d_test"], out_pairwise["position_test"]
        else:
            pairwise_matrix,pairwise_matrix_train,pairwise_matrix_test = None

        if cladistic_matrix is not None:
            cladistic_matrix= DraupnirUtils.symmetrize_and_clean(cladistic_matrix,keep_ancestral=ancestral) #check
            cladistic_matrix_train = cladistic_matrix.loc[~cladistic_matrix.index.isin(leaves_names_test)]
            cladistic_matrix_train = cladistic_matrix_train.loc[:, ~cladistic_matrix.index.isin(leaves_names_test)]

            cladistic_matrix_test = cladistic_matrix.loc[cladistic_matrix.index.isin(leaves_names_test)]
            cladistic_matrix_test = cladistic_matrix_test.loc[:,cladistic_matrix.index.isin(leaves_names_test)]

            cladistic_matrix_train = DraupnirUtils.rename_axis(cladistic_matrix_train, nodes, name_file=name)
            cladistic_matrix_test = DraupnirUtils.rename_axis(cladistic_matrix_test, nodes, name_file=name)

            evolutionary_matrix_train = DraupnirUtils.sum_matrices(patristic_matrix_train, cladistic_matrix_train)
            evolutionary_matrix_train = np.array(evolutionary_matrix_train)
            evolutionary_matrix_test = DraupnirUtils.sum_matrices(patristic_matrix_test, cladistic_matrix_test)
            evolutionary_matrix_test = np.array(evolutionary_matrix_test)

            cladistic_matrix_train = DraupnirUtils.pandas_to_numpy(cladistic_matrix_train)
            cladistic_matrix_test = DraupnirUtils.pandas_to_numpy(cladistic_matrix_test)
        else:
            print("No cladistic matrix available. Evolutionary matrix = Patristic matrix")
            evolutionary_matrix_train = DraupnirUtils.pandas_to_numpy(patristic_matrix_train)
            evolutionary_matrix_test = DraupnirUtils.pandas_to_numpy(patristic_matrix_test)
            cladistic_matrix_train = None
            cladistic_matrix_test = None

        patristic_matrix_train = DraupnirUtils.pandas_to_numpy(patristic_matrix_train)
        patristic_matrix_test = DraupnirUtils.pandas_to_numpy(patristic_matrix_test)

        # Eliminate the name part in the row, so that not everything needs to be changed
        dataset_test = dataset_test[:, 1:].astype('float64')
        dataset_train = dataset_train[:, 1:].astype('float64')


    #train_mask = np.where(~Dataset_train[:, 3:].any(axis=2), 0, 1)
    patristic_matrix_full = DraupnirUtils.symmetrize(patristic_matrix)
    patristic_matrix_full = DraupnirUtils.rename_axis(patristic_matrix_full, nodes, name_file=name)
    patristic_matrix_full = DraupnirUtils.pandas_to_numpy(patristic_matrix_full)

    if pairwise_distance_matrix is not None:
        pairwise_matrix_full = DraupnirUtils.symmetrize(pairwise_distance_matrix)
        pairwise_matrix_full = DraupnirUtils.rename_axis(pairwise_matrix_full, nodes, name_file=name)
        pairwise_matrix_full = DraupnirUtils.pandas_to_numpy(pairwise_matrix_full)
    else:
        patristic_matrix_full = pairwise_distance_matrix

    if cladistic_matrix is not None:
        cladistic_matrix_full = DraupnirUtils.symmetrize(cladistic_matrix)
        cladistic_matrix_full = DraupnirUtils.rename_axis(cladistic_matrix_full, nodes, name_file=name)
        cladistic_matrix_full = DraupnirUtils.pandas_to_numpy(cladistic_matrix_full)
    else:
        cladistic_matrix_full = cladistic_matrix


    if dataset_test is not None:#Highlight: Dataset_test != None only when the test dataset is extracted from the train (part of the leaves are test )
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
    text_file = open("{}/Hyperparameters_{}_{}epochs.txt".format(results_dir, now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),args.num_epochs), "a")
    text_file.write("Normalize patristic: {}\n".format(str(normalize_patristic)))
    if normalize_patristic:
        print("Normalizing patristic matrix!")
        patristic_matrix_full[1:, 1:] = patristic_matrix_full[1:, 1:] / np.linalg.norm(patristic_matrix_full[1:, 1:])
        if evolutionary_matrix_train is not None:
            evolutionary_matrix_train[1:, 1:] = evolutionary_matrix_train[1:, 1:] / np.linalg.norm(evolutionary_matrix_train[1:, 1:])
        evolutionary_matrix_train = [torch.from_numpy(evolutionary_matrix_train) if evolutionary_matrix_train is not None else evolutionary_matrix_train][0]
    else:
        evolutionary_matrix_train = [torch.from_numpy(evolutionary_matrix_train) if evolutionary_matrix_train is not None else evolutionary_matrix_train][0]

    results_dict = {"dataset_train":dataset_train,
                   "dataset_test":dataset_test,
                   "evolutionary_matrix_train":evolutionary_matrix_train,
                   "evolutionary_matrix_test":evolutionary_matrix_test,
                   "patristic_matrix_train":patristic_matrix_train,
                   "patristic_matrix_test":patristic_matrix_test,
                   "patristic_matrix_full":patristic_matrix_full,
                    "pairwise_matrix_train": pairwise_matrix_train,
                    "pairwise_matrix_test": pairwise_matrix_test,
                    "pairwise_matrix_full": pairwise_matrix_full,
                   "cladistic_matrix_train":cladistic_matrix_train,
                   "cladistic_matrix_test":cladistic_matrix_test,
                   "cladistic_matrix_full":cladistic_matrix_full,
                   "position_test":position_test,
                   "leaves_names_test":leaves_names_test,
                    "embeddings_train":embeddings_train,
                    "embeddings_test":embeddings_test,
                    "sequences_representations_train": sequence_representations_train,
                    }


    #return dataset_train,dataset_test,evolutionary_matrix_train,evolutionary_matrix_test,patristic_matrix_train,patristic_matrix_test,patristic_matrix_full, cladistic_matrix_train,cladistic_matrix_test,cladistic_matrix_full,position_test,leaves_names_test
    return results_dict

def matrix_sort(dataset_train,matrix,trim=False):
    """Sorts the input matrix by the nodes indexes in ascending order"""
    # Highlight: Sort by descent the patristic distances by node id
    if matrix is not None:
        matrix_sorted, matrix_sorted_idx = torch.sort(matrix[:, 0])
        matrix_sorted = matrix[matrix_sorted_idx]  # sorted rows
        matrix_sorted = matrix_sorted[:, matrix_sorted_idx]  # sorted columns
        if trim: #remove the ancestors info
            # Highlight: Find only the observed/train/leaves nodes indexes on the patristic matrix
            obs_indx = (matrix_sorted[:, 0][..., None] == dataset_train[:, 0, 1]).any(-1)
            obs_indx[0] = True  # To re-add the node names
            matrix_sorted = matrix_sorted[obs_indx]
            matrix_sorted = matrix_sorted[:, obs_indx]
        return matrix_sorted


def test_pretreatment_helper(train_load,additional_load,build_config,settings_config):
    """ALigns the order of the nodes in the dataset and the patristic and cladistic matrices. Calculates the amino acid frequencies.
    :param tensor dataset_train
    :param tensor patristic_matrix_full
    :param tensor cladistic_matrix_full
    :param namedtuple build_config
    """
    patristic_matrix_full, cladistic_matrix_full, pairwise_matrix_full = additional_load.patristic_matrix_full,additional_load.cladistic_matrix_full,additional_load.pairwise_matrix_full
    dataset_train = train_load.dataset_train
    # Highlight: alternative aa_freqs = DraupnirUtils.calculate_aa_frequencies_torch(dataset_train[:,2:,0],freq_bins=build_config.aa_prob)
    aa_frequencies_train = DraupnirUtils.calculate_aa_frequencies(dataset_train[:,2:,0].cpu().detach().numpy(),freq_bins=build_config.aa_probs)
    aa_frequencies_train = torch.from_numpy(aa_frequencies_train)
    aa_properties = DraupnirUtils.aa_properties(build_config.aa_probs,settings_config.data_folder)

    patristic_matrix_train = matrix_sort(dataset_train,patristic_matrix_full,trim=True)
    patristic_matrix_full = matrix_sort(dataset_train,patristic_matrix_full)

    cladistic_matrix_train = matrix_sort(dataset_train,cladistic_matrix_full,trim=True) #the if None check is inside
    cladistic_matrix_full = matrix_sort(dataset_train,cladistic_matrix_full)

    pairwise_matrix_train = matrix_sort(dataset_train,pairwise_matrix_full,trim=True) #the if None check is inside
    pairwise_matrix_full = matrix_sort(dataset_train,pairwise_matrix_full)

    # Highlight: Sort by descent the family data by node id, so that the order of the patristic distances and the sequences are matching
    dataset_train_sorted_vals, dataset_train_sorted_idx = torch.sort(dataset_train[:, 0, 1])
    dataset_train_sorted = dataset_train[dataset_train_sorted_idx]
    embeddings_train_sorted = train_load.embeddings_train[dataset_train_sorted_idx] if train_load.embeddings_train is not None else None
    sequences_representations_train_sorted = train_load.sequences_representations_train[dataset_train_sorted_idx] if train_load.sequences_representations_train is not None else None

    assert np.array_equal(embeddings_train_sorted[:,0,1],dataset_train_sorted[:,0,1]), "sorting was not done correctly for the embeddings"
    assert np.array_equal(sequences_representations_train_sorted[:,0],dataset_train_sorted[:,0,1]), "sorting was not done correctly for the sequence representations"

    results_dict = {"dataset_train_sorted": dataset_train_sorted,
                    "embeddings_train_sorted":embeddings_train_sorted,
                    "sequences_representations_train_sorted":sequences_representations_train_sorted,
                    "patristic_matrix_full": patristic_matrix_full,
                    "patristic_matrix_train": patristic_matrix_train,
                    "cladistic_matrix_full": cladistic_matrix_full,
                    "cladistic_matrix_train":cladistic_matrix_train,
                    "pairwise_matrix_full": pairwise_matrix_full,
                    "pairwise_matrix_train": pairwise_matrix_train,
                    "aa_frequencies_train": aa_frequencies_train
                    }


    #return dataset_train_sorted, embeddings_train_sorted,patristic_matrix_full,patristic_matrix_train,cladistic_matrix_full,cladistic_matrix_train,aa_frequencies
    return results_dict

def pretreatment_benchmark_randall(dataset_test,
                                   dataset_train,
                                   patristic_matrix_full,
                                   cladistic_matrix_full,
                                   test_nodes_observed,
                                   device,
                                   inferred=True,
                                   original_naming=True):
    if inferred:
        test_nodes_observed_correspondence = [21, 30, 32, 31, 22, 33, 34, 35, 28, 23, 36, 29, 27, 24, 26,25]  # numbers in the benchmark dataset paper/original names
        test_nodes_inferred_list = [19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35]  # iqtree correspondence/see Tree_Pictures/return_becnhmark.png
    else:
        if original_naming:
            test_nodes_observed_correspondence = [21,30,37,32,31,34,35,36,33,28,29,22,23,27,24,26,25]
            test_nodes_inferred_list = [21,30,37,32,31,34,35,36,33,28,29,22,23,27,24,26,25]
            assert test_nodes_observed_correspondence == test_nodes_inferred_list
            train_nodes_observed_correspondence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
            train_nodes_inferred_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
            assert train_nodes_observed_correspondence == train_nodes_inferred_list
            test_nodes_observed_correspondence += train_nodes_observed_correspondence
            test_nodes_inferred_list += train_nodes_inferred_list
        else:
            test_nodes_observed_correspondence =  [37,22,30,23,28,31,32,24,27,29,33,34,25,26,35,36] #original names of test/internal/ancestral
            test_nodes_inferred_list = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,36]  # new names for ancestral nodes (it's the observed tree but ete3 changes the names of the ancestral)..see Tree_Pictures/return_becnhmark_original
    correspondence_dict = dict(zip(test_nodes_observed_correspondence, test_nodes_inferred_list))
    test_nodes_inferred_list_correspondence = [correspondence_dict[val] for val in test_nodes_observed]

    # Highlight: Keep only the ancestral nodes, because , some of the ancestral nodes have the same number as the leaves, so we cannot just do ==
    n_obs = dataset_train.shape[0]
    patristic_matrix_test = patristic_matrix_full[:n_obs-1] #removed n_obs-1 ---> could mess up the problem with the inferred tree
    patristic_matrix_test = patristic_matrix_test[:, :n_obs-1]
    # if cladistic_matrix_full is not None:
    #     cladistic_matrix_test = cladistic_matrix_full[:n_obs - 1]  # removed n_obs-1 ---> could mess up the problem with the inferred tree
    #     cladistic_matrix_test = cladistic_matrix_test[:, :n_obs - 1]

    test_nodes = torch.tensor(test_nodes_inferred_list_correspondence).cpu()
    #Highlight: replace also in the original dataset with the correspondent node names
    dataset_test[:,0,1] = test_nodes
    vals, idx = torch.sort(test_nodes)
    test_nodes = test_nodes[idx]
    dataset_test = dataset_test[idx] #also order! same as the patristic!
    test_indx_patristic = (patristic_matrix_test[:, 0][..., None] == test_nodes).any(-1)
    test_indx_patristic[0] = True  # To re-add the node names
    patristic_matrix_test = patristic_matrix_test[test_indx_patristic]
    patristic_matrix_test = patristic_matrix_test[:, test_indx_patristic]
    # if cladistic_matrix is not None: #TODO: I do not think is necessary, the benchmark has cladistic
    #     cladistic_matrix_test = cladistic_matrix_full[:n_obs - 1]  # removed n_obs-1 ---> could mess up the problem with the inferred tree
    #     cladistic_matrix_test = cladistic_matrix_test[:, :n_obs - 1]
    #     cladistic_matrix_test = cladistic_matrix_test[test_indx_patristic]
    #     cladistic_matrix_test = cladistic_matrix_test[:, test_indx_patristic]
    #Highlight:  Sort by descent the patristic distances by node id ( for the train sequences is done in Draupnir.preprocessing)
    vals_test_sorted, idx_test_sorted = torch.sort(patristic_matrix_test[:, 0])
    patristic_matrix_test= patristic_matrix_test[idx_test_sorted]  # sorted rows
    patristic_matrix_test = patristic_matrix_test[:, idx_test_sorted]  # sorted columns
    assert patristic_matrix_test[1:, 0].tolist() == dataset_test[:, 0, 1].tolist()
    if cladistic_matrix_full is not None:
        cladistic_matrix_test = cladistic_matrix_full[:n_obs - 1]  # removed n_obs-1 ---> could mess up the problem with the inferred tree
        cladistic_matrix_test = cladistic_matrix_test[:, :n_obs - 1]
        cladistic_matrix_test = cladistic_matrix_test[test_indx_patristic]
        cladistic_matrix_test = cladistic_matrix_test[:, test_indx_patristic]
        vals_test_sorted, idx_test_sorted = torch.sort(cladistic_matrix_test[:, 0])
        cladistic_matrix_test= cladistic_matrix_test[idx_test_sorted]  # sorted rows
        cladistic_matrix_test = cladistic_matrix_test[:, idx_test_sorted]  # sorted columns
    #Highlight : Training matrix

    obs_node_names = dataset_train[:,0,1]
    train_indx_patristic = (patristic_matrix_full[:, 0][..., None] == obs_node_names).any(-1)
    train_indx_patristic[0] = True

    # Highlight: Skip ancestral number 19 (repeated!!!)
    if not original_naming:
        train_indx_patristic[1] = False
    patristic_matrix_train = patristic_matrix_full[train_indx_patristic]
    patristic_matrix_train = patristic_matrix_train[:, train_indx_patristic]
    vals,idx = torch.sort(patristic_matrix_train[:,0]) #sort just in case
    patristic_matrix_train = patristic_matrix_train[idx]
    if cladistic_matrix_full is not None:
        cladistic_matrix_train = cladistic_matrix_full[train_indx_patristic]
        cladistic_matrix_train = cladistic_matrix_train[:, train_indx_patristic]
        cladistic_matrix_train = cladistic_matrix_train[idx]
    assert dataset_train[:,0,1].tolist() == patristic_matrix_train[1:,0].tolist()
    #Highlight: Need to invert the dict mapping for later
    correspondence_dict = {v: k for k, v in correspondence_dict.items()}
    return patristic_matrix_train,patristic_matrix_test,cladistic_matrix_train,cladistic_matrix_test,dataset_test,dataset_train,correspondence_dict

def test_datasets_pretreatment(name,root_sequence_name,train_load,test_load,additional_load,build_config,args,settings_config,script_dir):
    """ Loads the ancestral sequences depending on the data set, when available. Corrects and sorts all matrices so that they are ordered accordingly.
    :param str name: dataset_name
    :param str root_sequence_name: for the default simulated datasets, we need an additional name string to retrieve the ancestral sequences
    :param namedtuple train_load: namedtuple with all tensors concerning the leaves
    :param namedtuple test_load
    :param namedtuple additional_load
    :param namedtuple build_config
    :param device: torch cpu or torch gpu
    :param namedtuple settings_config
    :param str script_dir
    """
    device = args.device
    #Highlight: Loading for special test datasets
    if name.startswith("simulation"):
        # dataset_test,internal_names_test,max_len_test = DraupnirDatasets.load_simulations_ancestral_sequences(name,
        #                                                                                 settings_config,
        #                                                                                 build_config.align_seq_len,
        #                                                                                 additional_load.tree_levelorder_names,
        #                                                                                 root_sequence_name,
        #                                                                                 build_config.aa_probs,
        #                                                                                 script_dir)

        test_data = np.load("{}/{}_dataset_numpy_aligned_integers_TEST.npz".format(settings_config.data_folder, name),allow_pickle=True)
        dataset_test, internal_names_test, max_len_test = test_data["array"],test_data["internal_names_test"],test_data["max_lenght_internal_aligned"]

        dataset_test = dataset_test[:,1:] #same as train, skip node names
        dataset_test = torch.from_numpy(dataset_test.astype(np.float64))

        test_nodes_observed = dataset_test[:, 0, 1].tolist()
        test_nodes = torch.tensor(test_nodes_observed, device="cpu")
        patristic_matrix_full = additional_load.patristic_matrix_full
        cladistic_matrix_full = additional_load.cladistic_matrix_full
        pairwise_matrix_full = additional_load.pairwise_matrix_full

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

        if pairwise_matrix_full is not None:
            pairwise_matrix_test = pairwise_matrix_full[test_indx_patristic]
            pairwise_matrix_test = pairwise_matrix_test[:, test_indx_patristic]
        else:
            pairwise_matrix_test = None

        correspondence_dict = special_nodes_dict = None


    elif name.startswith("benchmark"):
        dataset_test, internal_names_test = DraupnirDatasets.load_randalls_benchmark_ancestral_sequences(settings_config)
        test_nodes_observed =  dataset_test[:, 0, 1].tolist()
        special_nodes_dict=None
        patristic_matrix_train, \
        patristic_matrix_test, \
        cladistic_matrix_train, \
        cladistic_matrix_test, \
        dataset_test,\
        dataset_train,\
        correspondence_dict = pretreatment_benchmark_randall(dataset_test,
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
        special_nodes_dict =DraupnirDatasets.load_coral_fluorescent_proteins_ancestral_sequences(name = name,
                                                         ancestral_file="{}/Ancestral_Sequences.fasta".format(settings_config.data_folder),
                                                         tree_levelorder_names =additional_load.tree_levelorder_names,
                                                         aa_probs=build_config.aa_probs)


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

    # dataset_train,\
    # embeddings_train, \
    # patristic_matrix_full,\
    # patristic_matrix_train,\
    # cladistic_matrix_full,\
    # cladistic_matrix_train,\
    # aa_frequencies_train = pretreatment(train_load, additional_load.patristic_matrix_full,additional_load.cladistic_matrix_full, build_config,settings_config)

    results_dict = test_pretreatment_helper(train_load, additional_load, build_config,settings_config)

    aa_frequencies_test = DraupnirUtils.calculate_aa_frequencies(dataset_test[:,2:,0].cpu().detach().numpy(),freq_bins=build_config.aa_probs)
    aa_frequencies_test = torch.from_numpy(aa_frequencies_test)



    test_load = test_load._replace(dataset_test=dataset_test,
                         evolutionary_matrix_test=test_load.evolutionary_matrix_test,
                         patristic_matrix_test=patristic_matrix_test,
                         cladistic_matrix_test=cladistic_matrix_test,
                         pairwise_matrix_test=pairwise_matrix_test,
                         leaves_names_test=test_load.leaves_names_test,
                         position_test=test_load.position_test,
                         internal_nodes_indexes=test_load.internal_nodes_indexes)


    train_load = train_load._replace(dataset_train=results_dict["dataset_train_sorted"],
                           embeddings_train = results_dict["embeddings_train_sorted"],
                           sequences_representations_train = results_dict["sequences_representations_train_sorted"],
                           evolutionary_matrix_train=train_load.evolutionary_matrix_train, #contains both leafs and internal nodes, do not remember why
                           patristic_matrix_train=results_dict["patristic_matrix_train"],
                           cladistic_matrix_train=results_dict["cladistic_matrix_train"],
                           pairwise_matrix_train=results_dict["pairwise_matrix_train"] #i need to replace all of each of them, otherwise it rejects the replace
                                     )




    additional_load = additional_load._replace(patristic_matrix_full=results_dict["patristic_matrix_full"],
                                     cladistic_matrix_full=results_dict["cladistic_matrix_full"],
                                     children_array=additional_load.children_array,
                                     ancestor_info_numbers=additional_load.ancestor_info_numbers,
                                     tree_levelorder_names=additional_load.tree_levelorder_names,
                                     clades_dict_leaves =additional_load.clades_dict_leaves,
                                     closest_leaves_dict=additional_load.closest_leaves_dict,
                                     clades_dict_all=additional_load.clades_dict_all,
                                     linked_nodes_dict = additional_load.linked_nodes_dict,
                                     descendants_dict= additional_load.descendants_dict,
                                     alignment_length=additional_load.alignment_length,
                                     aa_frequencies_train=results_dict["aa_frequencies_train"],
                                     aa_frequencies_test=aa_frequencies_test,
                                     correspondence_dict = correspondence_dict,
                                     special_nodes_dict=special_nodes_dict,
                                     full_name=additional_load.full_name)



    return train_load,test_load,additional_load
def check_if_exists(a, b, key):
    "Deals with datasets that were trained before the current configuration of namedtuples"
    try:
        out = b[key]
    except:
        out = getattr(a, key)
    return out
def one_or_another(a):
    try:
        out = a["predictions"]
    except:
        out = a["aa_predictions"]
    return out
def tryexcept(a, key):
    try:
        out = a[key]
    except:
        out = None
    return out
def load_dict_to_namedtuple(load_dict):

    sample_out = SamplingOutput(aa_sequences=one_or_another(load_dict),
                                # TODO: the old results have the name as predictions instead of aa_predictions
                                latent_space=load_dict["latent_space"],
                                logits=load_dict["logits"],
                                phis=tryexcept(load_dict, "phis"),  # TODO: try , except None
                                psis=tryexcept(load_dict, "psis"),
                                mean_phi=tryexcept(load_dict, "mean_phi"),
                                mean_psi=tryexcept(load_dict, "mean_psi"),
                                kappa_phi=tryexcept(load_dict, "kappa_phi"),
                                kappa_psi=tryexcept(load_dict, "kappa_psi"))

    return sample_out

class CladesDataset(Dataset):
    def __init__(self,clades_names,clades_data,clades_patristic,clades_blosum_weighted,clades_data_blosum,clades_embeddings, clades_seq_representation):
        self.clades_names = clades_names
        self.clades_data = clades_data
        self.clades_patristic = clades_patristic
        self.clades_blosum_weighted = clades_blosum_weighted
        self.clades_data_blosum = clades_data_blosum
        self.clades_embeddings = clades_embeddings
        self.clades_seq_representation = clades_seq_representation


    def __getitem__(self, index): #sets a[i]
        clade_name = self.clades_names[index]
        clade_data = self.clades_data[index]
        clade_patristic = self.clades_patristic[index]
        clade_blosum_weighted = self.clades_blosum_weighted[index]
        clade_data_blosum = self.clades_data_blosum[index]
        clade_embedding = self.clades_embeddings[index]
        clade_seq_representation = self.clades_seq_representation[index]

        return {'clade_name': clade_name,
                'clade_data': clade_data,
                'clade_patristic': clade_patristic ,
                'clade_blosum_weighted':clade_blosum_weighted,
                'clade_data_blosum':clade_data_blosum,
                'clade_embedding':clade_embedding,
                'clade_seq_representation':clade_seq_representation,
                }

    def __len__(self):
        return len(self.clades_names)

class SplittedDataset(Dataset):
    def __init__(self, batches_names,
                 batches_data,
                 batches_patristic,
                 batches_blosum_weighted,
                 batches_data_blosums,
                 batches_embeddings,
                 batches_sequence_representations):
        self.batches_names = batches_names
        self.batches_data = batches_data
        self.batches_patristic = batches_patristic
        self.batches_blosum_weighted = batches_blosum_weighted
        self.batches_data_blosum = batches_data_blosums
        self.batches_embeddings = batches_embeddings
        self.batches_sequence_representations = batches_sequence_representations

    def __getitem__(self, index):  # sets a[i]
        batch_name = self.batches_names[index]
        batch_data = self.batches_data[index]
        batch_patristic = self.batches_patristic[index]
        batch_blosum_weighted = self.batches_blosum_weighted[index]
        batch_data_blosum = self.batches_data_blosum[index]
        batch_embedding = self.batches_embeddings[index]
        batch_sequence_representation = self.batches_sequence_representations[index]
        return {'batch_name': batch_name,
                'batch_data_int': batch_data,
                'batch_patristic': batch_patristic,
                'batch_blosum_weighted': batch_blosum_weighted,
                'batch_data_blosum':batch_data_blosum,
                'batch_embedding':batch_embedding,
                'batch_sequence_representation':batch_sequence_representation,
                }
    def __len__(self):
        return len(self.batches_names)

class CustomDataset(Dataset):
    def __init__(self, data_array_blosum,data_array_int,data_array_onehot,data_array_embedding,data_array_seq_representation):
        self.batch_data_blosum = data_array_blosum
        self.batch_data_int = data_array_int
        self.batch_data_onehot = data_array_onehot
        self.data_array_embedding = data_array_embedding
        self.data_array_seq_representation = data_array_seq_representation


    def __getitem__(self, index):  # sets a[i]
        batch_data_blosum = self.batch_data_blosum[index]
        batch_data_int = self.batch_data_int[index] if self.batch_data_int is not None else None
        batch_data_onehot = self.batch_data_onehot[index] if self.batch_data_onehot is not None else None
        batch_data_embedding = self.data_array_embedding[index] if self.data_array_embedding is not None else None
        batch_seq_representation = self.data_array_seq_representation[index] if self.data_array_seq_representation is not None else None
        return {'blosum': batch_data_blosum,
                'int':batch_data_int,
                'onehot':batch_data_onehot,
                'embedding':batch_data_embedding,
                'sequences_representations':batch_seq_representation,
                }
    def __len__(self):
        return len(self.batch_data_blosum)


def setup_data_loaders(datasets,patristic_matrix_train,clades_dict,blosum,build_config,args,method="batch_dim_0", use_cuda=True):
    """Loads the data set into the model. There are 3 modalities of data loading:
    a) No batching, load the entire data set (batch_dim_0, batch_size=1)
    b) batching, split evenly the data set, the batch size is automatically calculated if None is given (batch_dim_0, batch_size>1 or None)
    c) Clade batching, loads the data divided by the tree's clades: the latent space is not splitted, full inference (args.batch_by_clade=True: 1 batch= 1 clade )
    :param dataset
    :param patristic_matrix_train
    :param clades_dict: dictionary containing the tree nodes divided by clades
    :param blosum: BLOSUM matrix
    :param namedtuple build_config
    :param namedtuple args
    :param method: batching method
    NOTES: If a clade_dict is present it will Load each clade one at the time. Otherwise a predefined batch size is used"""
    # torch.manual_seed(0)    # For same random split of train/test set every time the code runs!
    #kwargs = {'num_workers': 0, 'pin_memory': use_cuda}  # pin-memory has to do with transferring CPU tensors to GPU
    kwargs = {'num_workers': 0, 'pin_memory': use_cuda}  # pin-memory has to do with transferring CPU tensors to GPU
    patristic_matrix_train = patristic_matrix_train.detach().cpu()  # otherwise it cannot be used with the train loader
    n_seqs = datasets["int"].shape[0]
    if method == "batch_dim_0":
        if args.batch_size == 1 : #only 1 batch // plating
            #train_loader = DataLoader(dataset.cpu(),batch_size=build_config.batch_size,shuffle=False,**kwargs)
            train_loader = CustomDataset(datasets["blosum"].cpu(),datasets["int"].cpu(),datasets["onehot"].cpu(),datasets["embedding"].cpu(), datasets["sequences_representations"].cpu())
            train_loader = DataLoader(train_loader,batch_size=build_config.batch_size,shuffle=False,**kwargs)
            if use_cuda:
                for dataset in train_loader:
                    for x in dataset.values():
                        x.to('cuda', non_blocking=True)
                #train_loader = [x.to('cuda', non_blocking=True) for x in train_loader]
        else: #split the dataset for batching
            
            blocks = DraupnirModelUtils.intervals(n_seqs // build_config.batch_size, n_seqs)

            batch_labels = ["batch_{}".format(i) for i in range(len(blocks))]
            batch_embeddings = []
            batch_sequences_representations = []
            batch_datasets = []
            batch_patristics = []
            batch_aa_freqs = []
            batch_blosums_max = []
            batch_blosums_weighted = []
            batch_data_blosums = []
            for block_idx in blocks:
                batch_data = datasets["int"][int(block_idx[0]):int(block_idx[1])]
                batch_datasets.append(batch_data.cpu())
                batch_nodes = batch_data[:, 0, 1]
                patristic_indexes = (patristic_matrix_train[:, 0][..., None] == batch_nodes.cpu()).any(-1)
                patristic_indexes[0] = True  # To re-add the node names
                batch_patristic = patristic_matrix_train[patristic_indexes]
                batch_patristic = batch_patristic[:, patristic_indexes]
                batch_patristics.append(batch_patristic)
                batch_embedding = datasets["embedding"][int(block_idx[0]):int(block_idx[1])]
                batch_embeddings.append(batch_embedding.cpu())
                batch_seq_representation = datasets["sequences_representations"][int(block_idx[0]):int(block_idx[1])]
                batch_sequences_representations.append(batch_seq_representation.cpu())


                batch_aa_frequencies = DraupnirUtils.calculate_aa_frequencies(batch_data[:, 2:, 0].cpu().numpy(),
                                                                build_config.aa_probs)
                batch_aa_freqs.append(batch_aa_frequencies)
                batch_blosum_max, batch_blosum_weighted, batch_variable_score = DraupnirUtils.process_blosum(blosum.cpu(),
                                                                                               torch.from_numpy(batch_aa_frequencies),
                                                                                               build_config.align_seq_len,
                                                                                               build_config.aa_probs)
                batch_blosums_max.append(batch_blosum_max)
                batch_blosums_weighted.append(batch_blosum_weighted)
                batch_data_blosum = DraupnirUtils.blosum_encoding(blosum, batch_aa_frequencies, build_config.align_seq_len,build_config.aa_probs, batch_data, False)
                batch_data_blosums.append(batch_data_blosum.cpu())

            Splitted_Datasets = SplittedDataset(batch_labels,
                                                batch_datasets,
                                                batch_patristics,
                                                batch_blosums_weighted,
                                                batch_data_blosums,
                                                batch_embeddings,
                                                batch_sequences_representations)

            train_loader = DataLoader(Splitted_Datasets, **kwargs)
            # for batch_number, dataset in enumerate(train_loader): #TODO: not necessary, since in the end I opted to transfer everything in the training loop
            #     #train_loader = [ x.to('cuda', non_blocking=True) if isinstance(x,torch.Tensor) else x for x in dataset.values()]
            #     for batch_label, batch_dataset, batch_patristic, batch_blosum_weighted, batch_data_blosum, batch_embedding, batch_seq_representation in zip(
            #             dataset["batch_name"], dataset["batch_data_int"], dataset["batch_patristic"],
            #             dataset["batch_blosum_weighted"], dataset["batch_data_blosum"], dataset["batch_embedding"], dataset["batch_sequence_representation"]):
            #
            #         batch_dataset = batch_dataset.to('cuda', non_blocking=True)
            #         batch_patristic = batch_patristic.to('cuda', non_blocking=True)
            #         batch_blosum_weighted= batch_blosum_weighted.to('cuda', non_blocking=True)
            #         batch_data_blosum=batch_data_blosum.to('cuda', non_blocking=True)
            #         batch_embedding=batch_embedding.to('cuda', non_blocking=True)
            #         batch_seq_representation=batch_seq_representation.to('cuda', non_blocking=True)



    elif method == "batch_by_clade": #todo: remove
        clades_labels = []
        clades_datasets = []
        clades_patristic = []
        clades_blosums_weighted = []  # this is where the weighted average goes
        clades_data_blosums = []  # this is the dataset in blosum-encoding goes
        clades_embeddings = []
        clades_sequences_representations = []
        for key, values in clades_dict.items():
            clades_labels.append(key)
            if isinstance(values, list) and len(values) > 1:
                clades_indexes = (datasets["int"][:, 0, 1][..., None] == torch.Tensor(values)).any(-1)
                patristic_indexes = (patristic_matrix_train[:, 0][..., None] == torch.Tensor(values).cpu()).any(-1)
            else:
                clades_indexes = (datasets["int"][:, 0, 1][..., None] == values).any(-1)
                patristic_indexes = (patristic_matrix_train[:, 0][..., None] == values).any(-1)

            clade_dataset = datasets["int"][clades_indexes]
            clades_datasets.append(clade_dataset.cpu())
            clade_embedding = datasets["embedding"][clades_indexes]
            clades_embeddings.append(clade_embedding)
            clade_seq_representations = datasets["sequences_representations"][clades_indexes]
            clades_sequences_representations.append(clade_seq_representations)

            patristic_indexes[0] = True  # To re-add the node names
            clade_patristic = patristic_matrix_train[patristic_indexes]
            clade_patristic = clade_patristic[:, patristic_indexes]
            clades_patristic.append(clade_patristic)
            clade_aa_frequencies = DraupnirUtils.calculate_aa_frequencies(clade_dataset[:, 2:, 0].cpu().numpy(), build_config.aa_probs)
            blosum_max, blosum_weighted, variable_score = DraupnirUtils.process_blosum(blosum.cpu(),
                                                                         torch.from_numpy(clade_aa_frequencies),
                                                                         build_config.align_seq_len,
                                                                         build_config.aa_probs)
            clades_blosums_weighted.append(blosum_weighted)
            clade_data_blosum = DraupnirUtils.blosum_encoding(blosum, clade_aa_frequencies, build_config.align_seq_len,
                                                         build_config.aa_probs, clade_dataset, False)
            clades_data_blosums.append(clade_data_blosum.cpu())
        Clades_Datasets = CladesDataset(clades_labels, clades_datasets, clades_patristic, clades_blosums_weighted,
                                        clades_data_blosums,clades_embeddings)
        train_loader = DataLoader(Clades_Datasets, **kwargs)
        if use_cuda:
            for batch_number, dataset in enumerate(train_loader):
                for clade_name, clade_dataset, clade_patristic, clade_blosum, clade_data_blosum, clade_embedding, clade_seq_representations in zip(
                        dataset["clade_name"], dataset["clade_data"], dataset["clade_patristic"],
                        dataset["clade_blosum_weighted"], dataset["clade_data_blosum"],dataset["clade_embedding"], dataset["clade_seq_representation"]): #todo: why this does not work with the .values() trick, try again?
                    clade_dataset.to('cuda:0', non_blocking=True)
                    clade_patristic.to('cuda:0', non_blocking=True)
                    clade_blosum.to('cuda:0', non_blocking=True)
                    clade_data_blosum.to('cuda:0', non_blocking=True)
                    clade_embedding.to('cuda:0', non_blocking=True)

    print(' Train_loader size: ', len(train_loader), 'batches')

    return train_loader



def upper_diagonal_idx(blocks):
    """
    :param blocks :
    """

    nsplits = len(blocks)
    row_blocks = []
    column_blocks = []
    for i,split in enumerate(blocks):
        row_blocks.append([split]*(nsplits-i))
        column_blocks.append(blocks[i:])

    row_blocks = list(itertools.chain(*row_blocks))
    column_blocks = list(itertools.chain(*column_blocks))

    return row_blocks, column_blocks



def setup_data_loaders_NEW_BATCHING_attempt(datasets, patristic_matrix_train, clades_dict, blosum, build_config, args, method="batch_dim_0",
                       use_cuda=True):
    """Loads the data set into the model. There are 3 modalities of data loading:
    a) No batching, load the entire data set (batch_dim_0, batch_size=1)
    b) batching, split evenly the data set, the batch size is automatically calculated if None is given (batch_dim_0, batch_size>1 or None)
    c) Clade batching, loads the data divided by the tree's clades: the latent space is not splitted, full inference (args.batch_by_clade=True: 1 batch= 1 clade )
    :param dataset
    :param patristic_matrix_train
    :param clades_dict: dictionary containing the tree nodes divided by clades
    :param blosum: BLOSUM matrix
    :param namedtuple build_config
    :param namedtuple args
    :param method: batching method
    NOTES: If a clade_dict is present it will Load each clade one at the time. Otherwise a predefined batch size is used"""
    # torch.manual_seed(0)    # For same random split of train/test set every time the code runs!
    # kwargs = {'num_workers': 0, 'pin_memory': use_cuda}  # pin-memory has to do with transferring CPU tensors to GPU
    kwargs = {'num_workers': 0, 'pin_memory': use_cuda}  # pin-memory has to do with transferring CPU tensors to GPU
    patristic_matrix_train = patristic_matrix_train.detach().cpu()  # otherwise it cannot be used with the train loader
    n_seqs = datasets["int"].shape[0]
    if method == "batch_dim_0":
        if args.batch_size == 1:  # only 1 batch // plating
            # train_loader = DataLoader(dataset.cpu(),batch_size=build_config.batch_size,shuffle=False,**kwargs)
            train_loader = CustomDataset(datasets["blosum"].cpu(), datasets["int"].cpu(), datasets["onehot"].cpu(),
                                         datasets["embedding"].cpu(), datasets["sequences_representations"].cpu())
            train_loader = DataLoader(train_loader, batch_size=build_config.batch_size, shuffle=False, **kwargs)
            if use_cuda:
                for dataset in train_loader:
                    for x in dataset.values():
                        x.to('cuda', non_blocking=True)
                # train_loader = [x.to('cuda', non_blocking=True) for x in train_loader]
        else:  # split the dataset for batching

            blocks = DraupnirModelUtils.intervals(n_seqs // build_config.batch_size, n_seqs)

            row_blocks, column_blocks = upper_diagonal_idx(blocks)

            batch_labels = ["batch_{}".format(i) for i in range(len(blocks))]
            batch_embeddings = []
            batch_sequences_representations = []
            batch_datasets = []
            batch_patristics = []
            batch_aa_freqs = []
            batch_blosums_max = []
            batch_blosums_weighted = []
            batch_data_blosums = []
            for row_idx, col_idx in zip(row_blocks, column_blocks):
                #TODO: if it does not allow different batch sizes, then , duplicate the diagonal sequences

                if row_idx == col_idx:
                    batch_data = datasets["int"][int(row_idx[0]):int(row_idx[1])] #we take 2 blocks of sequences (atm also with the diagonal, need to see if i can have different batch sizes)
                    batch_embedding = datasets["embedding"][int(row_idx[0]):int(row_idx[1])]
                    batch_seq_representation = datasets["sequences_representations"][int(row_idx[0]):int(row_idx[1])]
                    # batch_data = torch.concat([datasets["int"][int(row_idx[0]):int(row_idx[1])],datasets["int"][int(row_idx[0]):int(row_idx[1])]]) #we take 2 blocks of sequences (atm also with the diagonal, need to see if i can have different batch sizes)
                    # batch_embedding = torch.concat([datasets["embedding"][int(row_idx[0]):int(row_idx[1])],datasets["embedding"][int(row_idx[0]):int(row_idx[1])]])
                    # batch_seq_representation = torch.concat([datasets["sequences_representations"][int(row_idx[0]):int(row_idx[1])],datasets["sequences_representations"][int(row_idx[0]):int(row_idx[1])]])
                else:
                    batch_data = torch.concat([datasets["int"][int(row_idx[0]):int(row_idx[1])],datasets["int"][int(col_idx[0]):int(col_idx[1])]])
                    batch_embedding = torch.concat([datasets["embedding"][int(row_idx[0]):int(row_idx[1])],datasets["embedding"][int(col_idx[0]):int(col_idx[1])]])
                    batch_seq_representation = torch.concat([datasets["sequences_representations"][int(row_idx[0]):int(row_idx[1])],datasets["sequences_representations"][int(col_idx[0]):int(col_idx[1])]])

                batch_datasets.append(batch_data.cpu())
                batch_nodes = batch_data[:, 0, 1]
                patristic_indexes = (patristic_matrix_train[:, 0][..., None] == batch_nodes.cpu()).any(-1)
                patristic_indexes[0] = True  # To re-add the node names
                batch_patristic = patristic_matrix_train[patristic_indexes]
                batch_patristic = batch_patristic[:, patristic_indexes]
                batch_patristics.append(batch_patristic)
                batch_embeddings.append(batch_embedding.cpu())
                batch_sequences_representations.append(batch_seq_representation.cpu())

                batch_aa_frequencies = DraupnirUtils.calculate_aa_frequencies(batch_data[:, 2:, 0].cpu().numpy(),
                                                                              build_config.aa_probs)
                batch_aa_freqs.append(batch_aa_frequencies)
                batch_blosum_max, batch_blosum_weighted, batch_variable_score = DraupnirUtils.process_blosum(
                    blosum.cpu(),
                    torch.from_numpy(batch_aa_frequencies),
                    build_config.align_seq_len,
                    build_config.aa_probs)
                batch_blosums_max.append(batch_blosum_max)
                batch_blosums_weighted.append(batch_blosum_weighted)
                batch_data_blosum = DraupnirUtils.blosum_encoding(blosum, batch_aa_frequencies,
                                                                  build_config.align_seq_len, build_config.aa_probs,
                                                                  batch_data, False)
                batch_data_blosums.append(batch_data_blosum.cpu())

            Splitted_Datasets = SplittedDataset(batch_labels,
                                                batch_datasets,
                                                batch_patristics,
                                                batch_blosums_weighted,
                                                batch_data_blosums,
                                                batch_embeddings,
                                                batch_sequences_representations)

            train_loader = DataLoader(Splitted_Datasets, **kwargs)
            # for batch_number, dataset in enumerate(train_loader): #TODO: not necessary, since in the end I opted to transfer everything in the training loop
            #     #train_loader = [ x.to('cuda', non_blocking=True) if isinstance(x,torch.Tensor) else x for x in dataset.values()]
            #     for batch_label, batch_dataset, batch_patristic, batch_blosum_weighted, batch_data_blosum, batch_embedding, batch_seq_representation in zip(
            #             dataset["batch_name"], dataset["batch_data_int"], dataset["batch_patristic"],
            #             dataset["batch_blosum_weighted"], dataset["batch_data_blosum"], dataset["batch_embedding"], dataset["batch_sequence_representation"]):
            #
            #         batch_dataset = batch_dataset.to('cuda', non_blocking=True)
            #         batch_patristic = batch_patristic.to('cuda', non_blocking=True)
            #         batch_blosum_weighted= batch_blosum_weighted.to('cuda', non_blocking=True)
            #         batch_data_blosum=batch_data_blosum.to('cuda', non_blocking=True)
            #         batch_embedding=batch_embedding.to('cuda', non_blocking=True)
            #         batch_seq_representation=batch_seq_representation.to('cuda', non_blocking=True)



    elif method == "batch_by_clade":  # todo: remove
        clades_labels = []
        clades_datasets = []
        clades_patristic = []
        clades_blosums_weighted = []  # this is where the weighted average goes
        clades_data_blosums = []  # this is the dataset in blosum-encoding goes
        clades_embeddings = []
        clades_sequences_representations = []
        for key, values in clades_dict.items():
            clades_labels.append(key)
            if isinstance(values, list) and len(values) > 1:
                clades_indexes = (datasets["int"][:, 0, 1][..., None] == torch.Tensor(values)).any(-1)
                patristic_indexes = (patristic_matrix_train[:, 0][..., None] == torch.Tensor(values).cpu()).any(-1)
            else:
                clades_indexes = (datasets["int"][:, 0, 1][..., None] == values).any(-1)
                patristic_indexes = (patristic_matrix_train[:, 0][..., None] == values).any(-1)

            clade_dataset = datasets["int"][clades_indexes]
            clades_datasets.append(clade_dataset.cpu())
            clade_embedding = datasets["embedding"][clades_indexes]
            clades_embeddings.append(clade_embedding)
            clade_seq_representations = datasets["sequences_representations"][clades_indexes]
            clades_sequences_representations.append(clade_seq_representations)

            patristic_indexes[0] = True  # To re-add the node names
            clade_patristic = patristic_matrix_train[patristic_indexes]
            clade_patristic = clade_patristic[:, patristic_indexes]
            clades_patristic.append(clade_patristic)
            clade_aa_frequencies = DraupnirUtils.calculate_aa_frequencies(clade_dataset[:, 2:, 0].cpu().numpy(),
                                                                          build_config.aa_probs)
            blosum_max, blosum_weighted, variable_score = DraupnirUtils.process_blosum(blosum.cpu(),
                                                                                       torch.from_numpy(
                                                                                           clade_aa_frequencies),
                                                                                       build_config.align_seq_len,
                                                                                       build_config.aa_probs)
            clades_blosums_weighted.append(blosum_weighted)
            clade_data_blosum = DraupnirUtils.blosum_encoding(blosum, clade_aa_frequencies, build_config.align_seq_len,
                                                              build_config.aa_probs, clade_dataset, False)
            clades_data_blosums.append(clade_data_blosum.cpu())
        Clades_Datasets = CladesDataset(clades_labels, clades_datasets, clades_patristic, clades_blosums_weighted,
                                        clades_data_blosums, clades_embeddings)
        train_loader = DataLoader(Clades_Datasets, **kwargs)
        if use_cuda:
            for batch_number, dataset in enumerate(train_loader):
                for clade_name, clade_dataset, clade_patristic, clade_blosum, clade_data_blosum, clade_embedding, clade_seq_representations in zip(
                        dataset["clade_name"], dataset["clade_data"], dataset["clade_patristic"],
                        dataset["clade_blosum_weighted"], dataset["clade_data_blosum"], dataset["clade_embedding"],
                        dataset[
                            "clade_seq_representation"]):  # todo: why this does not work with the .values() trick, try again?
                    clade_dataset.to('cuda:0', non_blocking=True)
                    clade_patristic.to('cuda:0', non_blocking=True)
                    clade_blosum.to('cuda:0', non_blocking=True)
                    clade_data_blosum.to('cuda:0', non_blocking=True)
                    clade_embedding.to('cuda:0', non_blocking=True)

    print(' Train_loader size: ', len(train_loader), 'batches')

    return train_loader
class UnpicklerDraupnir(Unpickler):
    """ Overwriting dill's unpickler to deal with re-naming of modules that interferes with the process of serialization of the pickled dictionaries
    python's Unpickler extended to interpreter sessions and more types"""
    _session = False
    def find_class(self, module, name):
        from pickle import Pickler as StockPickler, Unpickler as StockUnpickler
        if (module, name) == ('__builtin__', '__main__'):
            return self._main.__dict__ #XXX: above set w/save_module_dict
        elif (module, name) == ('__builtin__', 'NoneType'):
            return type(None) #XXX: special case: NoneType missing
        if module == 'dill.dill': module = 'dill._dill'
        if module == 'Draupnir_utils':module='draupnir.utils'
        return StockUnpickler.find_class(self, module, name)
def load_serialized(file, ignore=None, **kwds):
    """unpickle an object from a file"""
    return UnpicklerDraupnir(file, ignore=ignore, **kwds).load()


