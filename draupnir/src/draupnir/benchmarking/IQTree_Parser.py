import os.path

import numpy as np
import pandas as pd
import torch
from Bio.Seq import Seq
from ete3 import Tree
from collections import defaultdict
import pyro.distributions as dist
import re
def parse_IQTree(name,simulation_folder, dataset_number, root_sequence_name, sequence_input_type,sites_count,script_dir_level_up, script_dir_2level_up,indel_calculation=False):

    if name.startswith("simulations"):
        original_tree_file = "/home/lys/Dropbox/PhD/DRAUPNIR/Datasets_Simulations/{}/Dataset{}/{}_True_Rooted_tree_node_labels.format7newick".format(simulation_folder,dataset_number,root_sequence_name)
        if sequence_input_type == "DNA":
            iqtree_seqs = os.path.join(script_dir_level_up,"IQTree/{}/Dataset{}/{}/{}_True_alignment.FASTA.state".format(simulation_folder,dataset_number,sequence_input_type,root_sequence_name))
            iqtree_tree_file = os.path.join(script_dir_level_up,"IQTree/{}/Dataset{}/{}/{}_True_alignment.FASTA.treefile".format(simulation_folder,dataset_number,sequence_input_type,root_sequence_name))
        else:
            iqtree_seqs = os.path.join(script_dir_level_up,"IQTree/{}/Dataset{}/{}/{}_True_Pep_alignment.FASTA.state".format(simulation_folder,dataset_number,sequence_input_type,root_sequence_name))
            iqtree_tree_file = os.path.join(script_dir_level_up,"IQTree/{}/Dataset{}/{}/{}_True_Pep_alignment.FASTA.treefile".format(simulation_folder, dataset_number, sequence_input_type, root_sequence_name))
    elif name.startswith("Coral"):
        original_tree_file = "/home/lys/Dropbox/PhD/DRAUPNIR/GPFCoralDataset/{}/c90.format7newick".format(name)
        if sequence_input_type == "DNA":
            iqtree_tree_file = os.path.join(script_dir_level_up,"IQTree/{}/DNA/{}_CodonAlignment_DNA.fasta.treefile".format(name,name))
            iqtree_seqs = os.path.join(script_dir_level_up,"IQTree/{}/DNA/{}_CodonAlignment_DNA.fasta.state".format(name,name))
        else:
            iqtree_tree_file = os.path.join(script_dir_level_up,"IQTree/{}/PROTEIN/{}_Aligned_Protein.fasta.treefile".format(name, name))
            iqtree_seqs = os.path.join(script_dir_level_up,"IQTree/{}/PROTEIN/{}_Aligned_Protein.fasta.state".format(name, name))
    elif name.endswith("_subtree"):
        iqtree_seqs = os.path.join(script_dir_level_up,"IQTree/{}/{}/{}_ALIGNED.mafft.state".format(name,sequence_input_type,name.replace("_subtree","")))
        original_tree_file = os.path.join(script_dir_2level_up,"Douglas_SRC_Dataset/{}/{}_ALIGNED.format7newick".format(name,name.replace("_subtree","")))
        iqtree_tree_file = os.path.join(script_dir_level_up,"IQTree/{}/PROTEIN/{}_ALIGNED.mafft.treefile".format(name,name.replace("_subtree","")))
    elif name in ["PF00400"]:
        iqtree_seqs = os.path.join(script_dir_level_up,"IQTree/{}/{}/{}.mafft.state".format(name,sequence_input_type,name))
        original_tree_file = os.path.join(script_dir_2level_up,"Mixed_info_Folder/{}/{}.format7newick".format(name,name.replace("_subtree","")))
        iqtree_tree_file = os.path.join(script_dir_level_up,"IQTree/{}/PROTEIN/{}.mafft.treefile".format(name,name.replace("_subtree","")))

    else:
        indel_calculation = False
        print("Only implemented for Randall's benchmark dataset!")
        original_tree_file = os.path.join(script_dir_level_up,"AncestralResurrectionStandardDataset/RandallBenchmarkTree_OriginalNaming.format7newick")
        if sequence_input_type == "DNA":
            iqtree_seqs = os.path.join(script_dir_level_up,"IQTree/AncestralResurrectionStandardDataset/DNA/RandallExperimentalPhylogenyDNASeqsLEAVES.fasta.state")
            iqtree_tree_file = os.path.join(script_dir_level_up,"IQTree/AncestralResurrectionStandardDataset/DNA/RandallExperimentalPhylogenyDNASeqsLEAVES.fasta.treefile")
        else:
            iqtree_seqs = os.path.join(script_dir_level_up,"IQTree/AncestralResurrectionStandardDataset/PROTEIN/benchmark_randall_original_naming_corrected_leaves_names.mafft.state")
            iqtree_tree_file = os.path.join(script_dir_level_up,"IQTree/AncestralResurrectionStandardDataset/PROTEIN/benchmark_randall_original_naming_corrected_leaves_names.mafft.treefile")
    #
    # print(iqtree_tree_file)
    # iqtree_tree = Tree(iqtree_tree_file,format=7)
    # original_tree = Tree(original_tree_file, format=8)
    # # # print(iqtree_tree.get_ascii(show_internal=True))
    # # # print(original_tree.get_ascii(show_internal=True))
    # # # exit()
    # correspondence_iqtree_to_original = defaultdict()
    # correspondence_original_to_iqtree = defaultdict()
    # for node_iqtree, node_original in zip(iqtree_tree.traverse(), original_tree.traverse()):
    #     if not node_original.is_leaf():
    #         # print(node_iqtree.name)
    #         # print(node_original.name)
    #         # print("..................")
    #         correspondence_iqtree_to_original[node_iqtree.name] = node_original.name
    #         correspondence_original_to_iqtree[node_original.name] = node_iqtree.name
    probabilities_df = pd.read_csv(iqtree_seqs,skiprows=8,sep = "\t", index_col=False)
    probabilities_df = probabilities_df[["Node","State"]]
    most_probable_sequences = probabilities_df.groupby('Node')['State'].apply(lambda x: x.sum()).to_dict()
    if sequence_input_type == "DNA":
        most_probable_sequences = {key: list(str(Seq(val.replace("-","")).translate()).replace("*","")) for key,val in most_probable_sequences.items()}
    else:
        most_probable_sequences = {key: list(str(val).replace("*","")) for key, val in most_probable_sequences.items()}
    if indel_calculation: #Heuristic calculation of indels---> not necessary, seems to insert gap when it does not know what to do. When doing it manually the results get worse
        #Highlight: Where there was a gap, assign to the most frequent aa or gap in the input/train alignment
        most_probable_sequences = pd.DataFrame.from_dict(most_probable_sequences, orient='index')
        #positions_with_gaps = most_probable_sequences.iloc[:,[int(key) for key in sites_count.keys()]]
        for site in sites_count.keys():
            if max(sites_count[site], key=sites_count[site].get) == "-":
                most_probable_sequences.iloc[:,int(site)] = max(sites_count[site], key=sites_count[site].get)
        most_probable_sequences = most_probable_sequences.apply(lambda x: ''.join(x), axis=1).to_dict()

    return most_probable_sequences, None, None#, correspondence_iqtree_to_original,correspondence_original_to_iqtree

def sample_IQTree(name,simulation_folder, dataset_number, root_sequence_name, sequence_input_type,sites_count,script_dir_level_up, script_dir_2level_up,plot_folder_name,indel_calculation=False):
    print("Sampling from IQTree")
    if name.startswith("simulations"):
        original_tree_file = "/home/lys/Dropbox/PhD/DRAUPNIR/Datasets_Simulations/{}/Dataset{}/{}_True_Rooted_tree_node_labels.format7newick".format(simulation_folder,dataset_number,root_sequence_name)
        if sequence_input_type == "DNA":
            iqtree_seqs = os.path.join(script_dir_level_up,"IQTree/{}/Dataset{}/{}/{}_True_alignment.FASTA.state".format(simulation_folder,dataset_number,sequence_input_type,root_sequence_name))
            iqtree_tree_file = os.path.join(script_dir_level_up,"IQTree/{}/Dataset{}/{}/{}_True_alignment.FASTA.treefile".format(simulation_folder,dataset_number,sequence_input_type,root_sequence_name))
        else:
            iqtree_seqs = os.path.join(script_dir_level_up,"IQTree/{}/Dataset{}/{}/{}_True_Pep_alignment.FASTA.state".format(simulation_folder,dataset_number,sequence_input_type,root_sequence_name))
            iqtree_tree_file = os.path.join(script_dir_level_up,"IQTree/{}/Dataset{}/{}/{}_True_Pep_alignment.FASTA.treefile".format(simulation_folder, dataset_number, sequence_input_type, root_sequence_name))
    elif name.startswith("Coral"):
        original_tree_file = "/home/lys/Dropbox/PhD/DRAUPNIR/GPFCoralDataset/{}/c90.format7newick".format(name)
        if sequence_input_type == "DNA":
            iqtree_tree_file = os.path.join(script_dir_level_up,"IQTree/{}/DNA/{}_CodonAlignment_DNA.fasta.treefile".format(name,name))
            iqtree_seqs = os.path.join(script_dir_level_up,"IQTree/{}/DNA/{}_CodonAlignment_DNA.fasta.state".format(name,name))
        else:
            iqtree_tree_file = os.path.join(script_dir_level_up,"IQTree/{}/PROTEIN/{}_Aligned_Protein.fasta.treefile".format(name, name))
            iqtree_seqs = os.path.join(script_dir_level_up,"IQTree/{}/PROTEIN/{}_Aligned_Protein.fasta.state".format(name, name))
    elif name.endswith("_subtree"):
        iqtree_seqs = os.path.join(script_dir_level_up,"IQTree/{}/{}/{}_ALIGNED.mafft.state".format(name,sequence_input_type,name.replace("_subtree","")))
        original_tree_file = os.path.join(script_dir_2level_up,"Douglas_SRC_Dataset/{}/{}_ALIGNED.format7newick".format(name,name.replace("_subtree","")))
        iqtree_tree_file = os.path.join(script_dir_level_up,"IQTree/{}/PROTEIN/{}_ALIGNED.mafft.treefile".format(name,name.replace("_subtree","")))
    elif name in ["PF00400"]:
        iqtree_seqs = os.path.join(script_dir_level_up,"IQTree/{}/{}/{}.mafft.state".format(name,sequence_input_type,name))
        original_tree_file = os.path.join(script_dir_2level_up,"Mixed_info_Folder/{}/{}.format7newick".format(name,name.replace("_subtree","")))
        iqtree_tree_file = os.path.join(script_dir_level_up,"IQTree/{}/PROTEIN/{}.mafft.treefile".format(name,name.replace("_subtree","")))

    else:
        indel_calculation = False
        print("Only implemented for Randall's benchmark dataset!")
        original_tree_file = os.path.join(script_dir_level_up,"AncestralResurrectionStandardDataset/RandallBenchmarkTree_OriginalNaming.format7newick")
        if sequence_input_type == "DNA":
            iqtree_seqs = os.path.join(script_dir_level_up,"IQTree/AncestralResurrectionStandardDataset/DNA/RandallExperimentalPhylogenyDNASeqsLEAVES.fasta.state")
            iqtree_tree_file = os.path.join(script_dir_level_up,"IQTree/AncestralResurrectionStandardDataset/DNA/RandallExperimentalPhylogenyDNASeqsLEAVES.fasta.treefile")
        else:
            iqtree_seqs = os.path.join(script_dir_level_up,"IQTree/AncestralResurrectionStandardDataset/PROTEIN/benchmark_randall_original_naming_corrected_leaves_names.mafft.state")
            iqtree_tree_file = os.path.join(script_dir_level_up,"IQTree/AncestralResurrectionStandardDataset/PROTEIN/benchmark_randall_original_naming_corrected_leaves_names.mafft.treefile")

    probabilities_df = pd.read_csv(iqtree_seqs,skiprows=8,sep = "\t", index_col=False)
    max_likelihood_seq = np.array(probabilities_df[["State"]])
    probabilities_df = probabilities_df[["Node","Site","p_A","p_R","p_N","p_D","p_C","p_Q","p_E","p_G","p_H","p_I","p_L","p_K","p_M","p_F","p_P","p_S","p_T","p_W","p_Y","p_V"]]
    align_length = probabilities_df.tail(1)["Site"].item()

    n_nodes = int(probabilities_df.shape[0]//align_length)
    n_samples = n_nodes
    nodes_names = probabilities_df["Node"].unique() #IQtree node order
    #Re-order IQtree node's order into tree_level order
    ordered_idx = np.argsort(nodes_names) #indices to put everything in the right order
    nodes_names = nodes_names[ordered_idx]
    nodes_names_expanded = nodes_names[None,:].repeat(n_samples,0).reshape(n_samples,n_nodes)

    aminoacids = probabilities_df.columns[2:].str.replace("p_","").tolist()
    aa_dict = dict(zip(list(range(len(aminoacids))),aminoacids))
    probabilities = probabilities_df[probabilities_df.columns.difference(['Node', 'Site','State'])].to_numpy(dtype=float)
    probabilities = torch.from_numpy(probabilities).reshape(n_nodes,align_length,20)
    max_likelihood_seq = max_likelihood_seq.reshape(n_nodes,align_length)
    probabilities = probabilities[ordered_idx]
    sampled_aa_sequences = dist.Categorical(probs=probabilities).sample([n_samples])
    aa_sequences = np.vectorize(aa_dict.get)(sampled_aa_sequences.numpy())
    aa_sequences_named = np.concatenate([nodes_names_expanded[:,:,None],aa_sequences],axis=2)
    root_name = aa_sequences_named[0,0,0]
    n=50
    if indel_calculation:
        file_name_root = "{}/IQTree_{}_root_sampled_ancestors_seq_{}.fasta".format(plot_folder_name, name,sequence_input_type)
        file_name = "{}/IQTree_{}_sampled_ancestors_seq_{}.fasta".format(plot_folder_name, name, sequence_input_type)
        with open(file_name, "w+") as f, open(file_name_root, "w+") as f2:
            for sample_idx in range(n_samples):
                for node_idx in range(n_nodes):
                  gaps_idx = np.where(max_likelihood_seq[node_idx] == "-")[0]
                  node_name = aa_sequences_named[sample_idx, node_idx, 0]
                  sequence = aa_sequences_named[sample_idx, node_idx, 1:]
                  sequence[gaps_idx] = "-" #Gaps added by IQtree
                  #Highlight: Indels added by me heuristically. If the leaves have almost all gaps, I assign a gap to the ancestral sequence
                  for site in sites_count.keys():
                      if max(sites_count[site], key=sites_count[site].get) == "-":
                        sequence[int(site)] = max(sites_count[site], key=sites_count[site].get)
                  if node_name == root_name:
                      f2.write(">Node_{}_sample_{}\n".format(node_name.replace("I", "A"), sample_idx))
                  f.write(">Node_{}_sample_{}\n".format(node_name, sample_idx))
                  splitted_seq = [sequence[i:i + n] for i in range(0, len(sequence), n)]
                  for segment in splitted_seq:
                      f.write("{}\n".format("".join(segment)))
                      if node_name == root_name:
                          f2.write("{}\n".format("".join(segment)))
    else:
        file_name_root = "{}/IQTree_{}_root_sampled_ancestors_seq_{}_noindels.fasta".format(plot_folder_name, name,sequence_input_type)
        file_name = "{}/IQTree_{}_sampled_ancestors_seq_{}_noindels.fasta".format(plot_folder_name, name, sequence_input_type)
        with open(file_name, "a+") as f, open(file_name_root, "a+") as f2:
            for sample_idx in range(n_samples):
                for node_idx in range(n_nodes):
                  gaps_idx = np.where(max_likelihood_seq[node_idx] == "-")[0]
                  node_name = aa_sequences_named[sample_idx, node_idx, 0]
                  sequence = aa_sequences_named[sample_idx, node_idx, 1:]
                  sequence[gaps_idx] = "-" #Highligth: Gaps added by IQtree
                  if node_name == root_name:
                      f2.write(">Node_{}_sample_{}\n".format(node_name.replace("I", "A"), sample_idx))
                  f.write(">Node_{}_sample_{}\n".format(node_name, sample_idx))
                  splitted_seq = [sequence[i:i + n] for i in range(0, len(sequence), n)]
                  for segment in splitted_seq:
                      f.write("{}\n".format("".join(segment)))
                      if node_name == root_name:
                          f2.write("{}\n".format("".join(segment)))




