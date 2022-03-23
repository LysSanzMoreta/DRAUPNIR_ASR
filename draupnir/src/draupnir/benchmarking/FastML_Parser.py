import pandas as pd
import numpy as np
from ete3 import Tree
from Bio import AlignIO
from Bio.Seq import Seq
from collections import defaultdict
import os
import pyro.distributions as dist
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def parse_FastML(name,simulation_folder,dataset_number,root_sequence_name,sequence_input_type,script_dir_level_up, script_dir_2level_up):
    "If the input alignment contains gaps use seq.marginal_IndelAndChars.txt, otherwise seq.marginal.txt"
    if name.startswith("simulations"):
        fastml_tree_file = os.path.join(script_dir_level_up,"FastML/{}/Dataset{}/{}/tree.newick.txt".format(simulation_folder,dataset_number,sequence_input_type))
        fastml_sequences = os.path.join(script_dir_level_up,"FastML/{}/Dataset{}/{}/seq.marginal_IndelAndChars.txt".format(simulation_folder,dataset_number,sequence_input_type))
        original_tree_file = os.path.join(script_dir_2level_up,"Datasets_Simulations/{}/Dataset{}/{}_True_Rooted_tree_node_labels.format8newick".format(simulation_folder,dataset_number,root_sequence_name))
    elif name.startswith("Coral"):
        fastml_tree_file = os.path.join(script_dir_level_up,"FastML/{}/{}/tree.newick.txt".format(name,sequence_input_type))
        fastml_sequences = os.path.join(script_dir_level_up,"FastML/{}/{}/seq.marginal_IndelAndChars.txt".format(name,sequence_input_type))
        original_tree_file = os.path.join(script_dir_2level_up,"GPFCoralDataset/{}/c90.format8newick".format(name))
    elif name.startswith("Cnidarian"):
        fastml_tree_file = os.path.join(script_dir_level_up,"FastML/{}/{}/tree.newick.txt".format(name,sequence_input_type))
        fastml_sequences = os.path.join(script_dir_level_up,"FastML/{}/{}/seq.marginal_IndelAndChars.txt".format(name,sequence_input_type))
        original_tree_file = os.path.join(script_dir_2level_up,"GPFCoralDataset/Cnidarian_FP/c90.format8newick".format(name))
    elif name.endswith("_subtree"):
        fastml_tree_file = os.path.join(script_dir_level_up,"FastML/{}/{}/tree.newick.txt".format(name,sequence_input_type))
        fastml_sequences = os.path.join(script_dir_level_up,"FastML/{}/{}/seq.marginal_IndelAndChars.txt".format(name,sequence_input_type))
        original_tree_file = os.path.join(script_dir_2level_up,"Douglas_SRC_Dataset/{}/{}_ALIGNED.format8newick".format(name,name.replace("_subtree","")))

    elif name in ["PF00400","SH3_pf00018_larger_than_30aa","PF00096","PF00400_200","aminopeptidase","PKinase_PF07714","Douglas_SRC"]:
        fastml_tree_file = os.path.join(script_dir_level_up,"FastML/{}/{}/tree.newick.txt".format(name, sequence_input_type))
        fastml_sequences = os.path.join(script_dir_level_up,"FastML/{}/{}/seq.marginal_IndelAndChars.txt".format(name, sequence_input_type))
        if name == "aminopeptidase":
            original_tree_file = os.path.join(script_dir_2level_up, "AminoPeptidase/2MAT_BLAST90.format8newick")
        else:
            original_tree_file = os.path.join(script_dir_2level_up, "Mixed_info_Folder/{}.format8newick".format(name))


    else:
        print("Only implemented for Randall's benchmark dataset!")
        fastml_tree_file = os.path.join(script_dir_level_up,"FastML/AncestralResurrectionStandardDataset/{}/tree.newick.txt".format(sequence_input_type))
        fastml_sequences = os.path.join(script_dir_level_up,"FastML/AncestralResurrectionStandardDataset/{}/seq.marginal.txt".format(sequence_input_type))
        original_tree_file = os.path.join(script_dir_2level_up,"AncestralResurrectionStandardDataset/RandallBenchmarkTree_OriginalNaming.format8newick")

    #Obtain both trees in tree traversal
    fastml_tree = Tree(fastml_tree_file,format=1)
    original_tree = Tree(original_tree_file, format=8)
    # print(fastml_tree.get_ascii(attributes=[ 'name']))
    # print(original_tree.get_ascii(attributes=['name']))
    # exit()

    correspondence_fastml_to_original = defaultdict()
    correspondence_original_to_fastml = defaultdict()
    for node_fastml, node_original in zip(fastml_tree.traverse(), original_tree.traverse()):
        if not node_original.is_leaf():
            correspondence_fastml_to_original[node_fastml.name.replace("N","I")] = node_original.name.replace("A","I")
            correspondence_original_to_fastml[node_original.name.replace("A","I")] = node_fastml.name.replace("N","I")

    #fastml_internal_nodes = [node.name for node in fastml_tree.traverse() if not node.is_leaf()]
    fastml_internal_nodes = correspondence_fastml_to_original.keys()
    if os.path.exists(fastml_sequences):
        alignment = AlignIO.read(fastml_sequences, "fasta")
    else:
        try:
            alignment = AlignIO.read(fastml_sequences.replace("_IndelAndChars",""), "fasta")
        except:
            return None, None,None
    if sequence_input_type == "DNA":
        ancestral_sequences = {seq.id.replace("N","I"):Seq(str(seq.seq).replace("-","")).translate(gap="-") for seq in alignment if seq.id.replace("N","I") in fastml_internal_nodes}
    else:
        ancestral_sequences = {seq.id.replace("N", "I"): seq.seq for seq in alignment if seq.id.replace("N", "I") in fastml_internal_nodes}

    return ancestral_sequences,correspondence_fastml_to_original,correspondence_original_to_fastml

def sample_FastML(name,plot_folder_name,simulation_folder,dataset_number,root_sequence_name,sequence_input_type,correspondence_fastml_to_original,script_dir_level_up, script_dir_2level_up):
    print("Sampling from FastML...")

    if name.startswith("simulations"):
        fastml_marginal_probabilities = os.path.join(script_dir_level_up,"FastML/{}/Dataset{}/{}/prob.marginal.csv".format(simulation_folder,dataset_number,sequence_input_type))
        fastml_indels_probabilities = os.path.join(script_dir_level_up,"FastML/{}/Dataset{}/{}/IndelsMarginalProb.txt".format(simulation_folder,dataset_number,sequence_input_type))
    elif name.startswith("Coral"):
        fastml_marginal_probabilities = os.path.join(script_dir_level_up,"FastML/{}/{}/prob.marginal.csv".format(name,sequence_input_type))
        fastml_indels_probabilities = os.path.join(script_dir_level_up,"FastML/{}/{}/IndelsMarginalProb.txt".format(name,sequence_input_type))
    elif name.endswith("_subtree"):
        fastml_marginal_probabilities = os.path.join(script_dir_level_up,"FastML/{}/{}/prob.marginal.csv".format(name,sequence_input_type))
        fastml_indels_probabilities = os.path.join(script_dir_level_up,"FastML/{}/{}/IndelsMarginalProb.txt".format(name,sequence_input_type))
    elif name in ["PF00400","SH3_pf00018_larger_than_30aa","PF00096","PF00400_200","aminopeptidase","PKinase_PF07714","Douglas_SRC","Cnidarian"]:
        fastml_marginal_probabilities = os.path.join(script_dir_level_up,"FastML/{}/{}/prob.marginal.csv".format(name,sequence_input_type))
        fastml_indels_probabilities = os.path.join(script_dir_level_up,"FastML/{}/{}/IndelsMarginalProb.txt".format(name,sequence_input_type))
    else:
        print("Only implemented for Randall's benchmark dataset!")
        fastml_marginal_probabilities = os.path.join(script_dir_level_up,"FastML/AncestralResurrectionStandardDataset/{}/prob.marginal.csv".format(name,sequence_input_type))
        fastml_indels_probabilities = os.path.join(script_dir_level_up,"FastML/AncestralResurrectionStandardDataset/{}/IndelsMarginalProb.txt".format(name,sequence_input_type))


    marginal_probs = pd.read_csv(fastml_marginal_probabilities,sep=",")
    aa_names = np.array(marginal_probs.columns[2:]).tolist()

    aa_dict = dict(zip(list(range(len(aa_names))),aa_names))
    marginal_probs = np.array(marginal_probs)

    align_length = marginal_probs[-1][1]
    n_nodes = marginal_probs.shape[0]//align_length
    n_samples = 125
    marginal_probs = marginal_probs.reshape((n_nodes,align_length,22)) #[n_nodes,L,aa_probs + 2]


    nodes_names = marginal_probs[:,0,0]
    nodes_names_expanded = nodes_names[None,:].repeat(n_samples,0).reshape(n_samples,n_nodes)
    marginal_probs = torch.from_numpy(marginal_probs[:,:,2:].astype(np.float64))
    aa_sequences = dist.Categorical(probs=marginal_probs).sample([n_samples]) #[n_samples,n_nodes,L]

    aa_sequences = np.vectorize(aa_dict.get)(aa_sequences.numpy())
    aa_sequences_named = np.concatenate([nodes_names_expanded[:,:,None],aa_sequences],axis=2)
    a = marginal_probs.cpu().numpy().flatten()
    a = a[~np.isnan(a)]
    sns.histplot(a, bins=30)
    plt.title("FastML all probabilities (gaps + no gaps probabilities)")
    plt.savefig('{}/FastML_Histogram_ALL_marginalprobs.png'.format(plot_folder_name))
    plt.clf()
    a_entropy = marginal_probs.cpu().numpy()
    a_entropy = -np.sum(np.log(a_entropy)*a_entropy,axis=2)
    sns.histplot(a_entropy.flatten(), bins=30)
    plt.title("FastML all entropy (gaps + no gaps probabilities)")
    plt.savefig('{}/FastML_Histogram_ALL_entropy.png'.format(plot_folder_name))
    plt.clf()
    indels_array = pd.read_csv(fastml_indels_probabilities,sep="\t")
    #indels_array.hist('Prob_Of_Indel', ax=ax,bins=30)

    indels_array = np.array(indels_array[indels_array['Node'].isin(nodes_names)])
    n_positions_with_gaps = int(indels_array.shape[0]/n_nodes) #the indel file does not contain all positions all the time

    indels_array = indels_array.reshape((n_nodes,n_positions_with_gaps,3)) #[:,:,1:] #.transpose(1,0,2)  ---> [n_nodes,L,prob]
    indels_array_entropy = indels_array[:,:,2].astype(float)
    fig, ax = plt.subplots()
    sns.histplot(indels_array[:,:,2].astype(float).flatten(), bins=30)
    plt.title("FastML GAPS probabilities")
    fig.savefig('{}/FastML_Histogram_GAPS.png'.format(plot_folder_name))
    plt.clf()
    indels_array_entropy = -indels_array_entropy*np.log(indels_array_entropy)
    fig, ax = plt.subplots()
    sns.histplot(indels_array_entropy.flatten(), bins=30)
    plt.title("FastML GAPS entropy")
    fig.savefig('{}/FastML_Histogram_GAPS_entropy.png'.format(plot_folder_name))
    plt.clf()
    aa_sequences_indels = np.zeros_like(aa_sequences_named)
    file_name_noindels = "{}/FastML_{}_sampled_ancestors_seq_{}_noindels.fasta".format(plot_folder_name, name,sequence_input_type)
    file_name = "{}/FastML_{}_sampled_ancestors_seq_{}.fasta".format(plot_folder_name, name,sequence_input_type)
    file_name_root = "{}/FastML_{}_root_sampled_ancestors_seq_{}.fasta".format(plot_folder_name, name,sequence_input_type)
    file_name_root_noindels = "{}/FastML_{}_root_sampled_ancestors_seq_{}_noindels.fasta".format(plot_folder_name, name,sequence_input_type)

    root_name = list(correspondence_fastml_to_original.values())[0]
    n = 50
    indels_reconstruction = True
    marginal_probs_no_gaps = marginal_probs #initialize
    if indels_reconstruction:
        with open(file_name, "w+") as f, open(file_name_root, "w+") as f2:
            for sample in range(n_samples):
                for node in range(n_nodes):
                    node_name = aa_sequences_named[sample, node, 0]
                    sequence = aa_sequences_named[sample,node,1:]
                    rows_idx = np.where(indels_array[:, :,1] == node_name)
                    indels_info_node = indels_array[rows_idx]
                    gap_idx = np.where(indels_info_node[:, 2] > 0.9)
                    # prob_gap = dist.Binomial(probs=torch.from_numpy(indels_info_node[:, 2].astype(float))).sample()
                    # gap_idx = np.where(prob_gap == 1) #we have to do this because it does not output gap probs for all positions
                    position_with_gaps = indels_info_node[gap_idx,0] -1 # -1 because the indels are indicated in 1-index and python is 0-index
                    sequence[position_with_gaps.squeeze(0).astype(int)] = "-"
                    marginal_probs_no_gaps[node,position_with_gaps.squeeze(0).astype(int)]  = torch.full((len(aa_names),),np.nan).double()
                    new_node_name = correspondence_fastml_to_original[node_name.replace("N","I")]
                    aa_sequences_indels[sample,node,0] = new_node_name
                    aa_sequences_indels[sample,node,1:] = sequence
                    if new_node_name == root_name:
                        f2.write(">Node_{}_sample_{}\n".format(new_node_name.replace("I","A"), sample))
                    f.write(">Node_{}_sample_{}\n".format(new_node_name.replace("I","A"), sample))
                    splitted_seq = [sequence[i:i + n] for i in range(0, len(sequence), n)]
                    for segment in splitted_seq:
                        f.write("{}\n".format("".join(segment)))
                        if new_node_name == root_name:
                            f2.write("{}\n".format("".join(segment)))
    else:
        print("Not using gaps probabilities")
        with open(file_name_noindels, "w+") as f, open(file_name_root_noindels, "w+") as f2:
            for sample in range(n_samples):
                for node in range(n_nodes):
                    node_name = aa_sequences_named[sample, node, 0]
                    sequence = aa_sequences_named[sample, node, 1:]
                    new_node_name = correspondence_fastml_to_original[node_name.replace("N", "I")]
                    aa_sequences_indels[sample, node, 0] = new_node_name
                    aa_sequences_indels[sample, node, 1:] = sequence
                    if new_node_name == root_name:
                        f2.write(">Node_{}_sample_{}\n".format(new_node_name.replace("I", "A"), sample))
                    f.write(">Node_{}_sample_{}\n".format(new_node_name.replace("I", "A"), sample))
                    splitted_seq = [sequence[i:i + n] for i in range(0, len(sequence), n)]
                    for segment in splitted_seq:
                        f.write("{}\n".format("".join(segment)))
                        if new_node_name == root_name:
                            f2.write("{}\n".format("".join(segment)))

    try:
        a = marginal_probs_no_gaps.cpu().numpy().flatten()
        a = a[~np.isnan(a)]
        sns.histplot(a,bins=30)
        plt.title("FastML no gaps probabilities")
        plt.savefig('{}/FastML_Histogram_NOGAPS_marginalprobs.png'.format(plot_folder_name))
        plt.clf()
        a_entropy = marginal_probs_no_gaps.cpu().numpy()
        a_entropy =  -np.sum(np.log(a_entropy)*a_entropy,axis=0)
        sns.histplot(a_entropy.flatten(), bins=30)
        plt.title("FastML no gaps entropies")
        plt.savefig('{}/FastML_Histogram_NOGAPS_entropy.png'.format(plot_folder_name))
        plt.clf()
        idx = torch.argmax(marginal_probs_no_gaps,dim=2).cpu().numpy()
        b = marginal_probs_no_gaps[idx]
        sns.histplot(b.flatten(),bins=30)
        #plt.ylim(0,6000000)
        plt.title("FastML most likely sequence probabilities")
        plt.savefig('{}/FastML_Histogram_MAPprobs.png'.format(plot_folder_name))
        plt.clf()
        b_entropy = -np.sum(np.log(b)*b,axis=0)
        sns.histplot(b_entropy.flatten(), bins=30)
        plt.title("FastML most likely sequence entropy")
        plt.savefig('{}/FastML_Histogram_MAPlogits.png'.format(plot_folder_name))
        plt.clf()
    except:
        print("could not make last plots")
        pass





def parse_FastML_probabilities(name,simulation_folder,dataset_number,root_sequence_name,sequence_input_type):
    if name.startswith("simulations"):
        fastml_tree_file= "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_results/FastML/{}/Dataset{}/{}/tree.newick.txt".format(simulation_folder,dataset_number,sequence_input_type)
        fastml_prob_marginal = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_results/FastML/{}/Dataset{}/{}/prob.marginal.csv".format(simulation_folder,dataset_number,sequence_input_type)
        original_tree_file = "/home/lys/Dropbox/PhD/DRAUPNIR/Datasets_Simulations/{}/Dataset{}/{}_True_Rooted_tree_node_labels.format8newick".format(simulation_folder,dataset_number,root_sequence_name)
    else:
        print("Only implemented for Randall's benchmark dataset!")
        fastml_tree_file = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_results/FastML/AncestralResurrectionStandardDataset/{}/tree.newick.txt".format(sequence_input_type)
        fastml_prob_marginal = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_results/FastML/AncestralResurrectionStandardDataset/{}/prob.marginal.csv".format(sequence_input_type)
        original_tree_file = "/home/lys/Dropbox/PhD/DRAUPNIR/AncestralResurrectionStandardDataset/RandallBenchmarkTree_OriginalNaming.format8newick"

    #Obtain both trees in tree traversal
    fastml_tree = Tree(fastml_tree_file,format=1)
    original_tree = Tree(original_tree_file, format=8)
    # print(fastml_tree.get_ascii(attributes=[ 'name']))
    # print(original_tree.get_ascii(attributes=['name']))

    correspondence_fastml_to_original = defaultdict()
    correspondence_original_to_fastml = defaultdict()
    for node_fastml, node_original in zip(fastml_tree.traverse(), original_tree.traverse()):
        if not node_original.is_leaf():
            correspondence_fastml_to_original[node_fastml.name.replace("N","I")] = node_original.name.replace("A","I")
            correspondence_original_to_fastml[node_original.name.replace("A","I")] = node_fastml.name.replace("N","I")

    fastml_prob_marginal_df = pd.read_csv(fastml_prob_marginal,sep=",")
    fastml_internal_nodes = correspondence_fastml_to_original.keys()
    fastml_probability_dict = defaultdict()
    for node in fastml_internal_nodes:
        node_probs = fastml_prob_marginal_df.loc[fastml_prob_marginal_df['Ancestral Node'] == node.replace("I","N")]
        node_probs = node_probs[["A","C","G","T"]]
        node_probs = np.array(node_probs).astype(float)
        max_prob = np.amax(node_probs, axis=1)
        fastml_probability_dict[node] = np.prod(max_prob)
    return fastml_probability_dict,correspondence_fastml_to_original,correspondence_original_to_fastml

