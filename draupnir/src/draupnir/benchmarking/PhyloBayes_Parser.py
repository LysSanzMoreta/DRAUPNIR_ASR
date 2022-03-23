import pandas as pd
import numpy as np
from ete3 import Tree
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from collections import defaultdict
import glob,os, re
def parse_PhyloBayes(name,simulation_folder,dataset_number,root_sequence_name,sequence_input_type,script_dir_level_up, script_dir_2level_up):
    #TODO: Inherit variables, pass them to named tuple
    if name.startswith("simulations"):
        #folder = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_results/PhyloBayes/{}/Dataset{}/{}".format(simulation_folder,dataset_number,sequence_input_type)
        folder = os.path.join(script_dir_level_up,"PhyloBayes/{}/Dataset{}/{}".format(simulation_folder,dataset_number,sequence_input_type))
        os.chdir(folder)
        #output_file = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_results/PhyloBayes/{}/Dataset{}/{}/Ancestral_Sequences.fasta".format(simulation_folder,dataset_number,sequence_input_type)
        output_file = os.path.join(script_dir_level_up,"PhyloBayes/{}/Dataset{}/{}/Ancestral_Sequences.fasta".format(simulation_folder,dataset_number,sequence_input_type))
        #tree_file = "/home/lys/Dropbox/PhD/DRAUPNIR/Datasets_Simulations/{}/Dataset{}/{}_True_Rooted_tree_node_labels.format8newick".format(simulation_folder,dataset_number,root_sequence_name)
        tree_file = os.path.join(script_dir_2level_up,"Datasets_Simulations/{}/Dataset{}/{}_True_Rooted_tree_node_labels.format8newick".format(simulation_folder,dataset_number,root_sequence_name))
    elif name.startswith("Coral"):
        #folder = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_results/PhyloBayes/{}/{}".format(name,sequence_input_type)
        folder = os.path.join(script_dir_level_up,"PhyloBayes/{}/{}".format(name,sequence_input_type))
        os.chdir(folder)
        #output_file = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_results/PhyloBayes/{}/{}/Ancestral_Sequences.fasta".format(name,sequence_input_type)
        output_file = os.path.join(script_dir_level_up,"PhyloBayes/{}/{}/Ancestral_Sequences.fasta".format(name,sequence_input_type))
        #tree_file = "/home/lys/Dropbox/PhD/DRAUPNIR/GPFCoralDataset/{}/c90.format8newick".format(name)
        tree_file = os.path.join(script_dir_2level_up,"GPFCoralDataset/{}/c90.format8newick".format(name))
    elif name.endswith("_subtree"):
        #folder = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_results/PhyloBayes/{}/{}".format(name,sequence_input_type)
        folder = os.path.join(script_dir_level_up,"PhyloBayes/{}/{}".format(name,sequence_input_type))
        os.chdir(folder)
        #output_file = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_results/PhyloBayes/{}/{}/Ancestral_Sequences.fasta".format(name,sequence_input_type)
        output_file = os.path.join(script_dir_level_up,"PhyloBayes/{}/{}/Ancestral_Sequences.fasta".format(name,sequence_input_type))
        #tree_file = "/home/lys/Dropbox/PhD/DRAUPNIR/Douglas_SRC_Dataset/{}/{}_ALIGNED.format8newick".format(name,name.replace("_subtree",""))
        tree_file = os.path.join(script_dir_2level_up,"Douglas_SRC_Dataset/{}/{}_ALIGNED.format8newick".format(name,name.replace("_subtree","")))
    elif name in ["PF00400"]:
        folder = os.path.join(script_dir_level_up,"PhyloBayes/{}/{}".format(name,sequence_input_type))
        os.chdir(folder)
        #output_file = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_results/PhyloBayes/{}/{}/Ancestral_Sequences.fasta".format(name,sequence_input_type)
        output_file = os.path.join(script_dir_level_up,"PhyloBayes/{}/{}/Ancestral_Sequences.fasta".format(name,sequence_input_type))
        #tree_file = "/home/lys/Dropbox/PhD/DRAUPNIR/Douglas_SRC_Dataset/{}/{}_ALIGNED.format8newick".format(name,name.replace("_subtree",""))
        tree_file = os.path.join(script_dir_2level_up,"Mixed_info_Folder/{}.format8newick".format(name))

    else:
        print("Only implemented for Randall's AncestralResurrectionStandardDataset dataset")
        #folder = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_results/PhyloBayes/AncestralResurrectionStandardDataset/{}".format(sequence_input_type)
        folder = os.path.join(script_dir_level_up,"PhyloBayes/AncestralResurrectionStandardDataset/{}".format(sequence_input_type))
        os.chdir(folder)
        #output_file = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_results/PhyloBayes/AncestralResurrectionStandardDataset/{}/Ancestral_Sequences.fasta".format(sequence_input_type)
        output_file = os.path.join(script_dir_level_up,"PhyloBayes/AncestralResurrectionStandardDataset/{}/Ancestral_Sequences.fasta".format(sequence_input_type))
        #tree_file = "/home/lys/Dropbox/PhD/DRAUPNIR/AncestralResurrectionStandardDataset/RandallBenchmarkTree_OriginalNaming.format8newick"
        tree_file = os.path.join(script_dir_2level_up,"AncestralResurrectionStandardDataset/RandallBenchmarkTree_OriginalNaming.format8newick")


    tree = Tree(tree_file,format=8)
    if sequence_input_type == "DNA":
        records = []
        phylobayes_dict={}
        for file in glob.glob("*.ancstatepostprob*"): #*.ancstatepostprob
            if name.startswith("Coral"):  # The coral sequences contain "_" in the name #TODO: move outside the loop
                nodes_names = re.split(r"A_sample_[0-9]+_", file.replace(".ancstatepostprob", ""))[1]
                number_underscores = nodes_names.count("_")
                if number_underscores == 1:  # no extra underscores in the node names
                    node_a, node_b = nodes_names.split("_")
                else:
                    nodes_names_matches = re.findall(r"[a-z]+[0-9]+_[0-9]+", nodes_names,
                                                     flags=re.IGNORECASE)  # highlight, this only works because it seems like all the names start with an alpha character
                    if len(nodes_names_matches) == 2:
                        node_a, node_b = nodes_names_matches
                    else:
                        node_a = nodes_names_matches[0]
                        node_b = nodes_names.replace(node_a, "").replace("_", "")
            else:  # for when the node name does not contain "_"
                node_a, node_b = file.replace(".ancstatepostprob", "").split("_")[3:5]
            if node_a != node_b:
                #print("Pair {} {}".format(node_a, node_b))

                if  name == "benchmark_randall_original_naming":
                    node_a = "%02d"% (int(node_a.replace("A","")))
                    node_b = "%02d"% (int(node_b.replace("A", "")))
                ancestor = tree.get_common_ancestor(node_a,node_b).name
                #print("Ancestor {}".format(ancestor))
                nt_seq = pd.read_csv(file,sep="\t")#,usecols=["A","C","G","T"])
                nt_seq = nt_seq[["A","C","G","T"]]
                nt_seq_array = np.array(nt_seq).astype(float)
                variable_sites = np.where((nt_seq_array >0 ) & (nt_seq_array <1))[0] #in the rows where there is a prob != 1, that means the probabilities are spread
                nt_seq_array = nt_seq_array[variable_sites]
                ambiguous_sites = defaultdict()
                for index, row in pd.DataFrame(nt_seq_array).iterrows():
                    duplicates_max = row.index[row == row.max()].tolist() #Highlight: Find all the indexes of the largest number in the row
                    #indexes_duplicated = row.duplicated(keep=False)
                    if len(duplicates_max) > 1: #I am not doing anything about this, just pointing out some nt have the same probabilities
                        nt_types = pd.Series(["A","C","G","T"])
                        nt_alternatives = nt_types[duplicates_max]
                        ambiguous_sites[index] = nt_alternatives
                print(" {} ambigous sites in phylobayes".format(len(ambiguous_sites)))

                #different_probabilities_sites = np.unique(nt_seq_array,axis=1)#nt_seq.apply(pd.Series.nunique, axis=1)
                coding_dna = nt_seq.idxmax(axis=1,skipna=True) #pick the nucleotides with the highest probabilities
                coding_dna = Seq("".join(coding_dna.values.tolist()))
                protein = coding_dna.translate()  # to_stop=True, if we want to split the protein when reaching a stop codon
                record = SeqRecord(Seq(''.join(protein)),
                                   annotations={"molecule_type": "DNA"},
                                   id=ancestor,
                                   description="")
                records.append(record)
                phylobayes_dict[ancestor] = protein

        SeqIO.write(records, output_file, "fasta")
        return phylobayes_dict
    else:
        records = []
        phylobayes_dict = {}
        for file in glob.glob("*.ancstatepostprob*"):  # *.ancstatepostprob
            if name.startswith("Coral"): #The coral sequences contain "_" in the name #TODO: move outside the loop
                nodes_names = re.split(r"A_sample_[0-9]+_", file.replace(".ancstatepostprob",""))[1]
                number_underscores = nodes_names.count("_")
                if number_underscores == 1: #no extra underscores in the node names
                    node_a, node_b = nodes_names.split("_")
                else:
                    nodes_names_matches = re.findall(r"[a-z]+[0-9]+_[0-9]+",nodes_names,flags=re.IGNORECASE) #highlight, this only works because it seems like all the names start with an alpha character
                    if len(nodes_names_matches) == 2:
                        node_a,node_b = nodes_names_matches
                    else:
                        node_a = nodes_names_matches[0]
                        node_b = nodes_names.replace(node_a,"").replace("_","")
            else: #for when the node name does not contain "_"
                node_a, node_b = file.replace(".ancstatepostprob","").split("_")[3:5]
            if node_a != node_b:
                if name == "benchmark_randall_original_naming":
                    node_a = "%02d" % (int(node_a.replace("A", "")))
                    node_b = "%02d" % (int(node_b.replace("A", "")))
                ancestor = tree.get_common_ancestor(node_a, node_b).name
                # print("Ancestor {}".format(ancestor))
                aa_seq = pd.read_csv(file, sep="\t")  # ,usecols=["A","C","G","T"])
                aa_seq = aa_seq[["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]]
                aa_seq_array = np.array(aa_seq).astype(float)
                variable_sites = np.where((aa_seq_array > 0) & (aa_seq_array < 1))[0]  # in the rows where there is a prob != 1, that means the probabilities are spread
                aa_seq_array = aa_seq_array[variable_sites]
                ambiguous_sites = defaultdict()
                for index, row in pd.DataFrame(aa_seq_array).iterrows():
                    duplicates_max = row.index[row == row.max()].tolist()  # Highlight: Find all the indexes of the largest number in the row
                    # indexes_duplicated = row.duplicated(keep=False)
                    if len(duplicates_max) > 1:  # I am not doing anything about this, just pointing out some aa have the same probabilities
                        aa_types = pd.Series(["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"])
                        aa_alternatives = aa_types[duplicates_max]
                        ambiguous_sites[index] = aa_alternatives
                print(" {} ambigous sites in phylobayes".format(len(ambiguous_sites)))

                # different_probabilities_sites = np.unique(nt_seq_array,axis=1)#nt_seq.apply(pd.Series.nunique, axis=1)
                protein = aa_seq.idxmax(axis=1, skipna=True)  # pick the nucleotides with the highest probabilities
                protein = Seq("".join(protein.values.tolist()))
                record = SeqRecord(Seq(''.join(protein)),
                                   annotations={"molecule_type": "protein"},
                                   id=ancestor,
                                   description="")
                records.append(record)
                phylobayes_dict[ancestor] = protein

        SeqIO.write(records, output_file, "fasta")
        return phylobayes_dict


def parse_PhyloBayes_probabilities(name,simulation_folder,dataset_number,root_sequence_name,sequence_input_type):
    #TODo: Fix for proteins/Dna comparison system
    if name.startswith("simulations"):
        folder = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_results/PhyloBayes/{}/Dataset{}/{}".format(simulation_folder,dataset_number,sequence_input_type)
        os.chdir(folder)
        tree_file = "/home/lys/Dropbox/PhD/DRAUPNIR/Datasets_Simulations/{}/Dataset{}/{}_True_Rooted_tree_node_labels.format8newick".format(simulation_folder,dataset_number,root_sequence_name)
    else:
        print("Only implemented for Randall's AncestralResurrectionStandardDataset dataset")
        folder = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_results/PhyloBayes/AncestralResurrectionStandardDataset/{}".format(sequence_input_type)
        os.chdir(folder)
        tree_file = "/home/lys/Dropbox/PhD/DRAUPNIR/AncestralResurrectionStandardDataset/RandallBenchmarkTree_OriginalNaming.format8newick"

    tree = Tree(tree_file,format=8)

    phylobayes_dict={}
    for file in glob.glob("*.ancstatepostprob*"): #*.ancstatepostprob
        node_a,node_b = file.strip(".ancstatepostprob").split("_")[3:5]

        if node_a != node_b:
            #print("Pair {} {}".format(node_a, node_b))
            if  name == "benchmark_randall_original_naming":
                node_a = "%02d"% (int(node_a.replace("A","")))
                node_b = "%02d"% (int(node_b.replace("A", "")))
            ancestor = tree.get_common_ancestor(node_a,node_b).name
            nt_seq = pd.read_csv(file,sep="\t")#,usecols=["A","C","G","T"])
            nt_seq = nt_seq[["A","C","G","T"]]
            nt_seq_array = np.array(nt_seq).astype(float)
            max_entropy = np.amax(nt_seq_array, axis=1)
            phylobayes_dict[ancestor] = np.exp(-np.prod(max_entropy))

    return phylobayes_dict


