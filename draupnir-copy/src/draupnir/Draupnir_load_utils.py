from collections import defaultdict
import os,sys
import numpy as np
import pandas as pd
import torch
from Bio.SeqRecord import SeqRecord
from Bio.Phylo.TreeConstruction import DistanceCalculator
from Bio.Seq import Seq
from Bio import SeqIO
sys.path.append("./draupnir/draupnir")
import Draupnir_utils as DraupnirUtils
from collections import namedtuple
SamplingOutput = namedtuple("SamplingOutput",["aa_sequences","latent_space","logits","phis","psis","mean_phi","mean_psi","kappa_phi","kappa_psi"])

def convert_clades_dict(name,clades_dict,leave_nodes_dict,internal_nodes_dict, only_leaves):
    "Transforms the names of the nodes to their tree transversal level order number"
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
    "Transforms the names of the nodes to their tree transversal level order number"
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
    "Transform the nodes names to tree level order"
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

def convert_nearest_leaf(name, nearest_leaf_dict,leaves_nodes_dict):
    nearest_leaf_renamed_dict = defaultdict()
    if name == "benchmark_randall_original_naming": #leaves keep their numeration , internal nodes have A in front
        for key, val in nearest_leaf_dict.items():
            nearest_leaf_renamed_dict[int(key.replace("A",""))] = int(val.replace("A",""))
    else:
        for key, val in nearest_leaf_dict.items():

            nearest_leaf_renamed_dict[leaves_nodes_dict[key]] = leaves_nodes_dict[val]
    return nearest_leaf_renamed_dict

def convert_descendants(name, descendants_dict,internal_nodes_dict,leave_nodes_dict):
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
def create_children_array(Dataset,ancestor_info_numbers):
    """Group nodes by common ancestors"""
    c = ancestor_info_numbers[:, [0, 2]]
    unique_nodes = np.unique(c[:, 1].astype(str), return_index=True)  # returns ordered nodes and indexes
    v = np.split(c[:, 0], unique_nodes[1])[1:]  # Highlight: cast the values as string type, otherwise np.unique messes up with the string types
    length = max(map(len, v))
    children_array = np.array([xi.tolist() + [None] * (length - len(xi)) for xi in v])
    # unique_nodes = np.array(pd.DataFrame(c[:, 1]).drop_duplicates().values).squeeze(-1) #Highlight: Replaces np.unique(c[:, 1]
    children_array = np.concatenate((unique_nodes[0].astype(float)[:, np.newaxis], children_array), axis=1)
    # Add ancestors to Dataset, to the 1 row, in position 3 ---> Skip row 0 (name)
    Dataset[:, 1, 3] = [c[np.isin(c[:, 0], node)][0][1] for node in Dataset[:, 1, 1]]
    return Dataset,children_array
def convert_ancestor_info(name,ancestor_info,tree_levelorder_names):
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
    """Validating that the selected aa probs are correct"""
    alignment_ids = []
    alignment_seqs = []
    for i, aligned in enumerate(alignment):
        alignment_ids.append(alignment[i].id)
        alignment_seqs.append(alignment[i].seq.strip("*"))
    dict_alignment = dict(zip(alignment_ids, alignment_seqs))
    summary_aa_probs = [DraupnirUtils.validate_sequence_alphabet(value) for key, value in dict_alignment.items()]  # finds the alphabets of each of the sequences in the alignment, checks for dna
    aa_probs_updated = max(build_config.aa_prob,
                           max(summary_aa_probs))  # if the input aa_probs is different from those found, the aa_probs change. And also the aa substitution  matrix
    return aa_probs_updated
def pairwise_distance_matrix(name,script_dir):
    """Reads any of the available pairwise distances file and sorts them in pairs in ascending order"""
    distance_matrix_file = "{}/{}_distance_matrix.csv".format(script_dir, name) #file given by iqtree
    pairwise_distance_matrix_file = "{}/{}_pairwise_distance_matrix.csv".format(script_dir,name) #file computed while creating the dataset
    if os.path.isfile(distance_matrix_file) :
        distance_matrix = pd.read_csv(distance_matrix_file, index_col=0)
    elif os.path.isfile(pairwise_distance_matrix_file):
        distance_matrix = pd.read_csv(pairwise_distance_matrix_file, index_col=0)
    else:
        distance_matrix = None
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
    else:
        sorted_distance_matrix = None
    return sorted_distance_matrix
def remove_nan2(dataset):
    """Detect and remove nan angle pair, where either phi or psi are nan , due mostly to nh3 or coo terminals. We assign the aminoacid where there are any nan values to gap"""
    aa_angles = dataset[:,3:,0:3].astype(float) #[n_seq,len_seq,[phi,psi]]
    #print(angles[np.isnan(angles)])
    indexes_nan = np.argwhere(np.isnan(aa_angles))#.squeeze(0) #n_matches, dim0,dim1,dim2 ---> we only want dim0 and dim1
    #nan_angles = angles[indexes_nan[:,0],indexes_nan[:,1],indexes_nan[:,2]] #all angles that are nan
    aa_angles[indexes_nan[:, 0], indexes_nan[:, 1]] = 0. #update all the angles to 0 and asign the aa  to 0 (gap), will it fail when more than one match?
    #simply update the angles inside dataset
    dataset[:, 3:, 0:3] = aa_angles
    return dataset
def remove_nan(dataset):
    """Detect and remove nan angle pair, where either phi or psi are nan , due mostly to nh3 or coo terminals. We assign the aminoacid where there are any nan values to gap"""
    aa_angles = dataset[:,3:,0:3].astype(float) #[n_seq,len_seq,[phi,psi]]
    aa_angles = np.apply_along_axis(lambda r: np.zeros_like(r) if np.isnan(r).any() else r, 2, aa_angles)
    dataset[:, 3:, 0:3] = aa_angles
    return dataset
def processing(Results_dir,Dataset,patristic_matrix,cladistic_matrix,sorted_distance_matrix,n_seq_train,n_seq_test,now,name,aa_probs,leaves_nodes,one_hot_encoding,nodes=[],ancestral =True):
    """Divides the Dataset into train and test, also the evolutionary/patristic matrices.
     ancestral : Keeps(true) or discard (false) the information on the ancestral nodes"""
    if n_seq_test == 0:
        Dataset_train = Dataset
        # Write the train sequences to a fasta file
        with open("{}/{}_training.fasta".format(Results_dir, name), "w") as output_handle,open("{}/{}_training_aligned.fasta".format(Results_dir, name), "w") as output_handle2:
            records = []
            records_aligned = []
            for sequence in Dataset_train:
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
                                   description="")  # TODO: There is an error here for the one-hot encoded version!!!
                records_aligned.append(record_aligned)
            SeqIO.write(records, output_handle, "fasta")
            SeqIO.write(records_aligned, output_handle2, "fasta")

        # Eliminate the name part in the row, so that not everything needs to be changed
        Dataset_train = Dataset_train[:, 1:].astype('float64')
        Dataset_test = None
        patristic_matrix_train = DraupnirUtils.symmetrize_and_clean(patristic_matrix,ancestral=ancestral)  # Highlight: if ancestral is true, patristic_matrix_train = patristic_matrix_full
        patristic_matrix_train = DraupnirUtils.rename_axis(patristic_matrix_train, nodes, name_file=name)
        if cladistic_matrix is not None:
            cladistic_matrix_train = DraupnirUtils.symmetrize_and_clean(cladistic_matrix, ancestral=ancestral)
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
        evolutionary_matrix_test = None
        position_test = [None, None]
        leaves_names_test = None

    else:
        n_train = Dataset.shape[0]
        n_test = int(Dataset.shape[0] * n_seq_test/ 100)
        #find the leaves in the 25%-75% area, in the middle, not the extremes
        n_25 = int((n_train*25)/100)
        n_75 = int((n_train*75)/100)
        test_indexes = np.random.choice(np.arange(n_25,n_75),n_test,replace=False)
        leaves_names_test = [leaves_nodes[index] for index in test_indexes]
        #leaves_names_test = ["5vei","2m0y","1wdx","6pbc","1gbr","3uat","4a65","2cud","7d7s","1x2p","1x27"]


        # rank = 100  # Higher rank, more pairwise distance
        # sequence_0 = sorted_distance_matrix.loc[rank].Sequence_0
        # sequence_1 = sorted_distance_matrix.loc[rank].Sequence_1
        #
        print("Using {} leaves for testing".format(len(leaves_names_test)))
        file_hyperparams = open("{}/Hyperparameters_{}_{}.txt".format(Results_dir, now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"), Results_dir.split("_")[-1]), "a")
        file_hyperparams.write('Test sequences names: {} \n'.format(leaves_names_test))
        file_hyperparams.close()

        Dataset_test = Dataset[np.isin(Dataset[:, 0, 0], leaves_names_test)]
        Dataset_train = Dataset[np.logical_not(np.isin(Dataset[:, 0, 0], leaves_names_test))]

        # Write the train sequences to a fasta file
        with open("{}/{}_training.fasta".format(Results_dir, name), "w") as output_handle:
            records = []
            for sequence in Dataset_train:
                train_sequence = DraupnirUtils.convert_to_letters(sequence[2:, 0],aa_probs)
                record = SeqRecord(Seq(''.join(train_sequence).replace("-", "")),
                                   annotations={"molecule_type": "protein"},
                                   id=sequence[0, 0], description="")
                records.append(record)
            SeqIO.write(records, output_handle, "fasta")


        patristic_matrix = DraupnirUtils.symmetrize_and_clean(patristic_matrix, ancestral=ancestral)
        patristic_matrix_test = patristic_matrix.loc[patristic_matrix.index.isin(leaves_names_test)]
        patristic_matrix_test = patristic_matrix_test.loc[:,patristic_matrix.index.isin(leaves_names_test)]

        patristic_matrix_train = patristic_matrix.loc[~patristic_matrix.index.isin(leaves_names_test)]
        patristic_matrix_train = patristic_matrix_train.loc[:,~patristic_matrix.index.isin(leaves_names_test)]

        position_test = patristic_matrix_test.index.tolist()
        position_test = [np.where(patristic_matrix.index == name)[0][0] for name in position_test]

        patristic_matrix_train = DraupnirUtils.rename_axis(patristic_matrix_train, nodes, name_file=name)
        patristic_matrix_test = DraupnirUtils.rename_axis(patristic_matrix_test, nodes, name_file=name)

        if cladistic_matrix is not None:
            cladistic_matrix= DraupnirUtils.symmetrize_and_clean(cladistic_matrix,ancestral=ancestral) #check
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
        Dataset_test = Dataset_test[:, 1:].astype('float64')
        Dataset_train = Dataset_train[:, 1:].astype('float64')


    #train_mask = np.where(~Dataset_train[:, 3:].any(axis=2), 0, 1)
    patristic_matrix_full = DraupnirUtils.symmetrize(patristic_matrix)
    patristic_matrix_full = DraupnirUtils.rename_axis(patristic_matrix_full, nodes, name_file=name)
    patristic_matrix_full = DraupnirUtils.pandas_to_numpy(patristic_matrix_full)

    if cladistic_matrix is not None:
        cladistic_matrix_full = DraupnirUtils.symmetrize(cladistic_matrix)
        cladistic_matrix_full = DraupnirUtils.rename_axis(cladistic_matrix_full, nodes, name_file=name)
        cladistic_matrix_full = DraupnirUtils.pandas_to_numpy(cladistic_matrix_full)
    else:
        cladistic_matrix_full = cladistic_matrix

    return Dataset_train,Dataset_test,evolutionary_matrix_train,evolutionary_matrix_test,patristic_matrix_train,patristic_matrix_test,patristic_matrix_full, cladistic_matrix_train,cladistic_matrix_test,cladistic_matrix_full,position_test,leaves_names_test


def pretreatment(dataset_train,patristic_matrix_full,cladistic_matrix_full,build_config):
    #Highlight: AA freqs

    # Highlight: alternative aa_freqs = DraupnirUtils.calculate_aa_frequencies_torch(dataset_train[:,2:,0],freq_bins=build_config.aa_prob)
    aa_frequencies = DraupnirUtils.calculate_aa_frequencies(dataset_train[:,2:,0].cpu().detach().numpy(),freq_bins=build_config.aa_prob)
    aa_frequencies = torch.from_numpy(aa_frequencies)
    aa_properties = DraupnirUtils.aa_properties(build_config.aa_prob,build_config.script_dir)

    def matrix_sort(matrix,trim=False):
        # Highlight: Sort by descent the patristic distances by node id
        matrix_sorted, matrix_sorted_idx = torch.sort(matrix[:, 0])
        matrix_sorted = matrix[matrix_sorted_idx]  # sorted rows
        matrix_sorted = matrix_sorted[:, matrix_sorted_idx]  # sorted columns
        if trim:
            # Highlight: Find only the observed/train/leaves nodes indexes on the patristic matrix
            obs_indx = (matrix_sorted[:, 0][..., None] == dataset_train[:, 0, 1]).any(-1)
            obs_indx[0] = True  # To re-add the node names
            matrix_sorted = matrix_sorted[obs_indx]
            matrix_sorted = matrix_sorted[:, obs_indx]
        return matrix_sorted
    patristic_matrix_train = matrix_sort(patristic_matrix_full,trim=True)
    patristic_matrix_full = matrix_sort(patristic_matrix_full)
    if cladistic_matrix_full is not None:
        cladistic_matrix_train = matrix_sort(cladistic_matrix_full,trim=True)
        cladistic_matrix_full = matrix_sort(cladistic_matrix_full)
    else:
        cladistic_matrix_train = None

    # Highlight: Sort by descent the family data by node id, so that the order of the patristic distances and the sequences are matching
    dataset_train_sorted_vals, dataset_train_sorted_idx = torch.sort(dataset_train[:, 0, 1])
    dataset_train_sorted = dataset_train[dataset_train_sorted_idx]

    return dataset_train_sorted,patristic_matrix_full,patristic_matrix_train,cladistic_matrix_full,cladistic_matrix_train,aa_frequencies
def pretreatment_Bayes(training_Dataset, patristic_matrix,aa_prob):
    "Works when the nodes have their name from their order in the traversal tree order (otherwise the names could be repeated and the sorting will not work)"
    #Highlight: AA freqs
    aa_frequencies = DraupnirUtils.calculate_aa_frequencies(training_Dataset[:,2:,0],freq_bins=aa_prob)
    # Highlight: Sort by descent the patristic distances by node id
    patristic_matrix_sorted = patristic_matrix[patristic_matrix[:,0].argsort()] #sort rows
    patristic_matrix_sorted = patristic_matrix_sorted[:,patristic_matrix[0,:].argsort(axis=0)] #sort columns
    # Highlight: Find only the observed node indexes on the patristic matrix
    obs_indx = (patristic_matrix_sorted[:, 0][..., None] == training_Dataset[:, 0, 1]).any(-1)
    obs_indx[0] = True  # To re-add the node names
    patristic_matrix_sorted = patristic_matrix_sorted[obs_indx]
    patristic_matrix_sorted = patristic_matrix_sorted[:, obs_indx]
    # Highlight: Sort by descent the family data by node id, so that the order of the patristic distances and the sequences are matching
    #training_Dataset_sorted, training_Dataset_sorted_idx = torch.sort(training_Dataset[:, 0, 1])
    training_Dataset = training_Dataset[training_Dataset[:, 0, 1].argsort()]
    aminoacid_sequences = training_Dataset[:, 2:, 0]
    # aminoacid_sequences = aminoacid_sequences.unsqueeze(2)  # to add a 3rd dimension to allow for the gru to do time series
    angles = training_Dataset[:, 2:, 1:3]

    return training_Dataset, aminoacid_sequences, angles, patristic_matrix_sorted,aa_frequencies
def pretreatment_Benchmark(Dataset_test,Dataset_train,patristic_matrix,cladistic_matrix,test_nodes_observed,device,inferred=True,original_naming=True):
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
    n_obs = Dataset_train.shape[0]
    patristic_matrix_test = patristic_matrix[:n_obs-1] #removed n_obs-1 ---> could mess up the problem with the inferred tree
    patristic_matrix_test = patristic_matrix_test[:, :n_obs-1]
    if cladistic_matrix is not None:
        cladistic_matrix_test = cladistic_matrix[:n_obs - 1]  # removed n_obs-1 ---> could mess up the problem with the inferred tree
        cladistic_matrix_test = cladistic_matrix_test[:, :n_obs - 1]

    test_nodes = torch.tensor(test_nodes_inferred_list_correspondence).cpu()
    #Highlight: replace also in the original dataset with the correspondent node names
    Dataset_test[:,0,1] = test_nodes
    vals, idx = torch.sort(test_nodes)
    test_nodes = test_nodes[idx]
    Dataset_test = Dataset_test[idx] #also order! same as the patristic!
    test_indx_patristic = (patristic_matrix_test[:, 0][..., None] == test_nodes).any(-1)
    test_indx_patristic[0] = True  # To re-add the node names
    patristic_matrix_test = patristic_matrix_test[test_indx_patristic]
    patristic_matrix_test = patristic_matrix_test[:, test_indx_patristic]
    if cladistic_matrix is not None: #TODO: I do not think is necessary, the benchmark has cladistic
        cladistic_matrix_test = cladistic_matrix_test[test_indx_patristic]
        cladistic_matrix_test = cladistic_matrix_test[:, test_indx_patristic]
    #Highlight:  Sort by descent the patristic distances by node id ( for the train sequences is done in Draupnir.preprocessing)
    vals_test_sorted, idx_test_sorted = torch.sort(patristic_matrix_test[:, 0])
    patristic_matrix_test= patristic_matrix_test[idx_test_sorted]  # sorted rows
    patristic_matrix_test = patristic_matrix_test[:, idx_test_sorted]  # sorted columns
    assert patristic_matrix_test[1:, 0].tolist() == Dataset_test[:, 0, 1].tolist()
    if cladistic_matrix is not None:
        vals_test_sorted, idx_test_sorted = torch.sort(cladistic_matrix_test[:, 0])
        cladistic_matrix_test= cladistic_matrix_test[idx_test_sorted]  # sorted rows
        cladistic_matrix_test = cladistic_matrix_test[:, idx_test_sorted]  # sorted columns
    #Highlight : Training matrix

    obs_node_names = Dataset_train[:,0,1]
    train_indx_patristic = (patristic_matrix[:, 0][..., None] == obs_node_names).any(-1)
    train_indx_patristic[0] = True

    # Highlight: Skip ancestral number 19 (repeated!!!)
    if not original_naming:
        train_indx_patristic[1] = False
    patristic_matrix_train = patristic_matrix[train_indx_patristic]
    patristic_matrix_train = patristic_matrix_train[:, train_indx_patristic]
    vals,idx = torch.sort(patristic_matrix_train[:,0]) #sort just in case
    patristic_matrix_train = patristic_matrix_train[idx]
    if cladistic_matrix is not None:
        cladistic_matrix_train = cladistic_matrix[train_indx_patristic]
        cladistic_matrix_train = cladistic_matrix_train[:, train_indx_patristic]
        cladistic_matrix_train = cladistic_matrix_train[idx]
    assert Dataset_train[:,0,1].tolist() == patristic_matrix_train[1:,0].tolist()
    #Highlight: Need to invert the dict mapping for later
    correspondence_dict = {v: k for k, v in correspondence_dict.items()}
    return patristic_matrix_train,patristic_matrix_test,cladistic_matrix_train,cladistic_matrix_test,Dataset_test,Dataset_train,correspondence_dict
def pretreatment_Benchmark_Bayes(Dataset_test,training_Dataset,patristic_matrix,test_nodes_observed,inferred=False,original_naming=True):

    if inferred:
        test_nodes_observed_correspondence = [21, 30, 32, 31, 22, 33, 34, 35, 28, 23, 36, 29, 27, 24, 26,
                                              25]  # numbers in the benchmark dataset paper/original names
        test_nodes_inferred_list = [19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34,
                                    35]  # iqtree correspondence
    else:
        if original_naming:
            test_nodes_observed_correspondence = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]
            test_nodes_inferred_list = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]
        else:
            test_nodes_observed_correspondence = [37, 22, 30, 23, 28, 31, 32, 24, 27, 29, 33, 34, 25, 26, 35,
                                                  36]  # original names
            test_nodes_inferred_list = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                                        36]  # new names for ancestral nodes (it's the observed tree but ete3 changes the names of the ancestral)
    correspondence_dict = dict(zip(test_nodes_observed_correspondence, test_nodes_inferred_list))
    test_nodes_inferred_list_correspondence = [correspondence_dict[val] for val in test_nodes_observed]

    # Highlight: Keep only the ancestral nodes, because , some of the ancestral nodes have the same number as the leaves
    n_obs = training_Dataset.shape[0]
    patristic_matrix_test = patristic_matrix[:n_obs - 1]
    patristic_matrix_test = patristic_matrix_test[:, :n_obs - 1]
    test_nodes = np.array(test_nodes_inferred_list_correspondence)
    #Highlight: replace also in the original dataset with the correspondent node names
    Dataset_test[:,0,1] = test_nodes
    Dataset_test = Dataset_test[Dataset_test[:,0,1].argsort()] #sort rows (in case they are not sorted)
    test_indx_patristic = (patristic_matrix_test[:, 0][..., None] == test_nodes).any(-1)
    test_indx_patristic[0] = True  # To re-add the node names
    patristic_matrix_test = patristic_matrix_test[test_indx_patristic]
    patristic_matrix_test = patristic_matrix_test[:, test_indx_patristic]
    #Highlight:  Sort by descent the patristic distances by node id ( for the train sequences is done in Draupnir.preprocessing)
    patristic_matrix_test = patristic_matrix_test[patristic_matrix_test[:, 0].argsort()]  # sort rows
    patristic_matrix_test = patristic_matrix_test[:, patristic_matrix_test[0, :].argsort(axis=0)]  # sort columns
    #Highlight : Training matrix
    obs_node_names = patristic_matrix[n_obs - 1:, 0]
    train_indx_patristic = (patristic_matrix[:, 0][..., None] == obs_node_names).any(-1)
    train_indx_patristic[0] = True
    # Highlight: Skip ancestral number 19 (repeated!!!)
    train_indx_patristic[1] = False
    patristic_matrix = patristic_matrix[train_indx_patristic]
    patristic_matrix = patristic_matrix[:, train_indx_patristic]
    #Highlight: Need to invert the dict mapping for later
    correspondence_dict = {v: k for k, v in correspondence_dict.items()}
    return patristic_matrix,patristic_matrix_test,Dataset_test,correspondence_dict

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
                                # TODO: the old results have the name as predictions instead of aa_sequences
                                latent_space=load_dict["latent_space"],
                                logits=load_dict["logits"],
                                phis=tryexcept(load_dict, "phis"),  # TODO: try , excep None
                                psis=tryexcept(load_dict, "psis"),
                                mean_phi=tryexcept(load_dict, "mean_phi"),
                                mean_psi=tryexcept(load_dict, "mean_psi"),
                                kappa_phi=tryexcept(load_dict, "kappa_phi"),
                                kappa_psi=tryexcept(load_dict, "kappa_psi"))

    return sample_out

