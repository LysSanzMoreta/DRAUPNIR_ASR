"""
=======================
2022: Lys Sanz Moreta
Draupnir : Ancestral protein sequence reconstruction using a tree-structured Ornstein-Uhlenbeck variational autoencoder
=======================
"""

import re
import os,sys,shutil
from os import listdir
from os.path import isfile, join
import subprocess
import warnings
import _pickle as cPickle
import bz2
os.environ['QT_QPA_PLATFORM']='offscreen' #TODO: remove?
#Ete 3
from ete3 import Tree as TreeEte3
import dgl
try:
    from ete3 import Tree, faces, AttrFace, TreeStyle,NodeStyle
except:
    pass
from scipy.sparse import coo_matrix
import argparse
import dill
import ast
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import LogNorm
import torch
from torch.utils.data import Dataset, DataLoader
import statistics
import seaborn as sns
from collections import defaultdict,namedtuple
import pickle
import draupnir.models_utils as DraupnirModelUtils
#Biopython
import Bio.PDB as PDB
from Bio.PDB.Polypeptide import PPBuilder, CaPPBuilder
from Bio.Data.SCOPData import protein_letters_3to1
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import BiopythonWarning
from Bio import AlignIO, SeqIO
from Bio.PDB.PDBList import PDBList
import Bio.Align
from Bio.Align.Applications import MafftCommandline
from Bio.Phylo.TreeConstruction import *
from Bio import Phylo
#Numpy
import numpy as np
import numpy.random as npr
import pandas as pd
import matplotlib
matplotlib.use('Agg')
def aa_properties(aa_probs,data_folder):
    """ Creates a dictionary with amino acid properties, extracted from
    https://www.sigmaaldrich.com/life-science/metabolomics/learning-center/amino-acid-reference-chart.html
    :param int aa_probs: amino acid probabilities used to extract the types of amino acids present in the dataset
    :param str data_folder: path to where draupnir is been executed from"""

    aa_types = list(aminoacid_names_dict(aa_probs).keys())
    storage_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
    aa_properties = pd.read_csv("{}/AA_properties.txt".format(storage_folder),sep="\s+")
    aa_info = defaultdict()
    for indx,aa in aa_properties.iterrows():
        if aa["Abbr."] in aa_types:
            aa_number = aa_types.index(aa["Abbr."])
            aa_info[aa_number] = [float(aa["Molecular_Weight"]),float(aa["pKa1"]),float(aa["pl4"])]
    return aa_info

def validate_sequence_alphabet(seq):
    """
    Checks that the sequences from an alignment only contains values from one of the protein alphabets (protein21 or protein21plus). Reject DNA or RNA sequences
    :param str seq: sequence of characters
    """
    alphabets = {'dna': re.compile('^[acgtn]*$', re.I),
             'protein21': re.compile('^[-acdefghiklmnpqrstvwy]*$', flags=re.IGNORECASE),
            'protein21plus': re.compile('^[-acdefghiklmnpqrstvwybzx]*$', flags= re.IGNORECASE)}

    if alphabets["dna"].search(str(seq)) is not None: raise ValueError("Please use amino acids in your sequences, accepted alphabets are protein21: -acdefghiklmnpqrstvwy or protein21plus: -*acdefghiklmnpqrstvwybzx")

    if alphabets["protein21"].search(str(seq)) is not None:
        aa_probs = 21
        return aa_probs
    if alphabets["protein21plus"].search(str(seq)) is not None:
        aa_probs = 24
        return aa_probs
    else:
        raise ValueError("Your sequences contain not allowed characters. Available alphabets are: {protein21}: -acdefghiklmnpqrstvwy or {protein21plus} -*acdefghiklmnpqrstvwybzx. If your sequence contains stop codons perhaps you can trim them.")

def aminoacid_names_dict(aa_probs):
    """ Returns an aminoacid associated to a integer value
    :param int aa_probs: amino acid probabilities, this number correlates to the number of different aa types in the input alignment"""
    if aa_probs == 21:
        aminoacid_names = {"-":0,"R":1,"H":2,"K":3,"D":4,"E":5,"S":6,"T":7,"N":8,"Q":9,"C":10,"G":11,"P":12,"A":13,"V":14,"I":15,"L":16,"M":17,"F":18,"Y":19,"W":20}
        return aminoacid_names
    if aa_probs == 22: #includes stop codons---> fix in Create blosum
        aminoacid_names = {"-":0,"*":0,"R":1,"H":2,"K":3,"D":4,"E":5,"S":6,"T":7,"N":8,"Q":9,"C":10,"G":11,"P":12,"A":13,"V":14,"I":15,"L":16,"M":17,"F":18,"Y":19,"W":20}
        return aminoacid_names
    elif aa_probs > 22:
        aminoacid_names = {"-":0,"R":1,"H":2,"K":3,"D":4,"E":5,"S":6,"T":7,"N":8,"Q":9,"C":10,"G":11,"P":12,"A":13,"V":14,"I":15,"L":16,"M":17,"F":18,"Y":19,"W":20,"B":21,"Z":22,"X":23}
        return aminoacid_names
def create_blosum(aa_probs,subs_matrix_name):
    """
    Builds an array containing the blosum scores per character
    :param aa_probs: amino acid probabilities, determines the choice of BLOSUM matrix
    :param str subs_matrix_name: name of the substitution matrix, check availability at /home/lys/anaconda3/pkgs/biopython-1.76-py37h516909a_0/lib/python3.7/site-packages/Bio/Align/substitution_matrices/data"""

    if aa_probs > 21 and not subs_matrix_name.startswith("PAM"):
        warnings.warn("Your dataset contains special amino acids. Switching your substitution matrix to PAM70")
        subs_matrix_name = "PAM70"
    subs_matrix = Bio.Align.substitution_matrices.load(subs_matrix_name)
    aa_list = list(aminoacid_names_dict(aa_probs).keys())
    index_gap = aa_list.index("-")
    aa_list[index_gap] = "*" #in the blosum matrix gaps are represanted as *

    subs_dict = defaultdict()
    subs_array = np.zeros((len(aa_list) , len(aa_list) ))
    for i, aa_1 in enumerate(aa_list):
        for j, aa_2 in enumerate(aa_list):
            if aa_1 != "*" and aa_2 != "*":
                subs_dict[(aa_1,aa_2)] = subs_matrix[(aa_1, aa_2)]
                subs_dict[(aa_2, aa_1)] = subs_matrix[(aa_1, aa_2)]
            subs_array[i, j] = subs_matrix[(aa_1, aa_2)]
            subs_array[j, i] = subs_matrix[(aa_2, aa_1)]

    names = np.concatenate((np.array([float("-inf")]), np.arange(0,aa_probs)))
    subs_array = np.c_[ np.arange(0,aa_probs), subs_array ]
    subs_array = np.concatenate((names[None,:],subs_array),axis=0)

    return subs_array, subs_dict
def divide_into_monophyletic_clades(tree,storage_folder,name):
    """
    Divides the tree into monophyletic clades:
    See https://www.mun.ca/biology/scarr/Taxon_types.html
    Implementation based on: https://www.biostars.org/p/97409/

    The reasonable clade division criteria seems to group all those nodes whose distance to the internal is lower to the overall average distance from each leaf to the root

    :param ete3-tree tree: ete3 tree class
    :param str storage_folder: folder where to store the results, in this case a dictionary containing {"clade_number": [nodes list]}
    :name str name: name of the data set project

    """

    def mean(array):
        """Calculates branch length average"""
        return sum(array) / float(len(array))

    def cache_distances(tree):
        """Precalculate distances of all nodes to the root"""
        node2rootdist = {tree: 0}
        for node in tree.iter_descendants('preorder'):
            node2rootdist[node] = node.dist + node2rootdist[node.up]
        return node2rootdist

    def build_clades(tree,name):
        """When a clustering condition is met, it collapses the tree at that node to unify all the leaves in that cluster into 'one' leaves. After, we read
        the collapsed tree into a dictionary that contains {clade number:{"internal":[nodes numbers],"leaves":[node numbers]}}"""
        # cache the tip content of each node to reduce the number of times the tree is traversed
        node2tips = tree.get_cached_content()
        root_distance = cache_distances(tree)  # distances of each of the nodes to the root
        average_root_distance = mean(root_distance.values())
        std_root_distance = statistics.stdev(root_distance.values())
        n_leaves = len(tree.get_leaves())
        #TODO: automatize clustering condition
        if n_leaves >= 100 or name.endswith("_subtree"):
            if name in ["PF00096","PF00400"]:
                clustering_condition = average_root_distance #+ 0.3*std_root_distance
            else:
                clustering_condition = average_root_distance -2*std_root_distance
        elif name in ["Coral_all","Coral_Faviina","SH3_pf00018_larger_than_30aa"] or "calcitonin" in name:
            clustering_condition = average_root_distance - std_root_distance
        else:
            clustering_condition = average_root_distance

        for node in tree.get_descendants('preorder'):
            if not node.is_leaf():  # for internal nodes
                avg_distance_to_tips = mean([root_distance[tip] - root_distance[node] for tip in node2tips[node]])  # average distance from the internal node to all it's possible derived leaves
                if avg_distance_to_tips < clustering_condition:
                    #node.name += ' COLLAPSED avg_d:%g {%s}' % (avg_distance_to_tips, ','.join([tip.name for tip in node2tips[node]]))
                    node.name += ' COLLAPSED avg_d:%g leaves:{%s} internal:{%s}' % (avg_distance_to_tips, ','.join([tip.name for tip in node2tips[node]]),','.join([internal.name for internal in node.iter_descendants() if not internal.is_leaf()]))
                    node.add_features(collapsed=True)
                    node.img_style['draw_descendants'] = False

        for n in tree.search_nodes(collapsed=True):
            for child in n.get_children():
                child.detach()
        #print(tree.get_ascii(show_internal=True))
        i = 0
        clade_dict_all = defaultdict(lambda: defaultdict())
        for n in tree.traverse():
                if n.is_leaf() and "COLLAPSED" in n.name: #collapsed leaf (it is a clade on it's own)
                    clade_names_leaves = n.name[n.name.find("leaves:{") + 8:n.name.find("}")].split(",")
                    clade_names_internal = [n.name.split(" ")[0]]
                    clade_names_internal += n.name[n.name.find("internal:{") + 10:].strip("}").split(",")
                    clade_dict_all["Clade_{}".format(i)]["leaves"] = set(clade_names_leaves)  # remove duplicates
                    clade_dict_all["Clade_{}".format(i)]["internal"] = list(filter(None,set(clade_names_internal))) #sometimes the  node strings are empty
                    i += 1
                elif not n.is_leaf(): #if the node is internal
                    clade_names_leaves = []
                    clade_names_internal = []
                    clade_names_internal += [n.name]
                    for descendant in n.iter_descendants():
                        if descendant.is_leaf():
                            if "{" not in descendant.name: #it was a pure leaf
                                clade_names_leaves += [descendant.name]
                            else: #is a collapsed leave
                                clade_names_leaves += descendant.name[descendant.name.find("leaves:{")+8:descendant.name.find("}")].split(",")
                                clade_names_internal += [descendant.name.split(" ")[0]]
                                clade_names_internal += descendant.name[descendant.name.find("internal:{")+10:].strip("}").split(",")
                        else: #add the internal node also to it's clade
                            clade_names_internal += [descendant.name]

                    clade_dict_all["Clade_{}".format(i)]["leaves"] = set(clade_names_leaves) #remove duplicates
                    clade_dict_all["Clade_{}".format(i)]["internal"] = list(filter(None,set(clade_names_internal))) #sometimes the  node strings are empty
                    i += 1
                else:#Non collapsed leaves
                    pass
        clade_dict_leaves = defaultdict()
        i = 0
        for n in tree.traverse("preorder"):
            if n.is_leaf():
                if "{" not in n.name:
                    clade_names_leaves = [n.name]
                else:
                    clade_names_leaves = n.name[n.name.find("leaves:{") + 8:n.name.find("}")].split(",")
                clade_dict_leaves["Clade_{}".format(i)] = clade_names_leaves
                i += 1

        return clade_dict_leaves,clade_dict_all

    clade_dict_leaves,clade_dict_all = build_clades(tree,name)
    #Highlight: clades_dict_all contains each clade's internal and leaves nodes, clades_dict_leaves only contains the leaves of each clade

    dill.dump(clade_dict_all, open('{}/{}_Clades_dict_all.p'.format(storage_folder,name), 'wb'))#,protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(clade_dict_leaves, open('{}/{}_Clades_dict_leaves.p'.format(storage_folder,name), 'wb'),protocol=pickle.HIGHEST_PROTOCOL)
def calculate_closest_leaves(name,tree,storage_folder):
    """ Creates a dictionary that contains the closest leave to an internal node {internal_node:leave}
    :param str name: data set project name
    :param ete3-tree tree: Ete3 tree class object
    :param str storage_folder: folder where to dump the output
    """
    closest_leaves_dict=defaultdict() #closest leave to an internal node
    for node in tree.traverse():
        if not node.is_leaf(): #if it's an internal node
            terminal_node = all(node.is_leaf() for node in node.get_children())
            if terminal_node:
                closest_leaves_dict[node.name] = [node.name for node in node.get_children()]
            else:
                closest_leaves_dict[node.name] = [node.get_closest_leaf()[0].name]
    pickle.dump(closest_leaves_dict, open('{}/{}_Closest_leaves_dict.p'.format(storage_folder,name), 'wb'),protocol=pickle.HIGHEST_PROTOCOL)
def calculate_directly_linked_nodes(name,tree,storage_folder):
    """Creates a dictionary that contains the 2 children nodes directly linked to a node (not all the children from that node) {node:children}
    :param str name: data set project name
    :param ete3-tree tree: Ete3 tree class object
    :param str storage_folder: folder where to dump the output"""
    closest_children_dict=defaultdict()
    for node in tree.traverse():
        closest_children_dict[node.name] = [node.name for node in node.get_children()]
    pickle.dump(closest_children_dict, open('{}/{}_Closest_children_dict.p'.format(storage_folder,name), 'wb'),protocol=pickle.HIGHEST_PROTOCOL)
def calculate_descendants(name,tree,storage_folder):
    """Creates a dictionary that contains all the internal nodes and leaves that descend from that internal node {internal_node:descendants}
    :param str name: data set project name
    :param ete3-tree tree: Ete3 tree class object
    :param str storage_folder: folder where to dump the output"""
    closest_descendants_dict = defaultdict(lambda: defaultdict())
    for node in tree.traverse():
        if not node.is_leaf():
            descendant_leaves = []
            descendant_internal = [node.name]
            for descendant in node.iter_descendants():
                if descendant.is_leaf():
                    descendant_leaves.append(descendant.name)
                else:
                    descendant_internal.append(descendant.name)
            closest_descendants_dict[node.name]["internal"] = descendant_internal
            closest_descendants_dict[node.name]["leaves"] = descendant_leaves
    dill.dump(closest_descendants_dict, open('{}/{}_Descendants_dict.p'.format(storage_folder,name), 'wb'))#,protocol=pickle.HIGHEST_PROTOCOL)
def Pfam_parser(family_name,data_folder,first_match=False,update_pfam=False):
    """Creates a dictionary containing the PDB files and the sequence information (chain, residues...)
    :param str family_name: pfam family name
    :param bool first_match: True -> Picks only the first found protein and not it's duplicates, False -> Takes every available structure, including duplicates
    :param bool update_pfam: True -> Downloads and save the latest pfam version from http://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/pdbmap.gz"; False -> Use the stored version
    """
    if update_pfam:
        try:
            subprocess.call("rm -rf pdbmap.gz pdbmap")
        except:
            pass
        subprocess.call("wget http://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/pdbmap.gz",shell=True)
        subprocess.call("gunzip pdbmap.gz",shell=True)
        subprocess.call("mv pdbmap PfamPdbMap.txt", shell=True)

    print("Reading and creating pfam dictionary....")
    storage_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
    df = pd.read_csv("{}/PfamPdbMap.txt".format(storage_folder), sep="\t", error_bad_lines=False, engine='python', index_col=False,
                     names=["PDB_1", "Chain", "Empty", "Function", "Family", "Uniprot", "Residues"]) #TODO: move to data
    family = df[df['Family'].str.contains(family_name)]
    if first_match: #115 #only takes the first pdb file found for each sequence
        first_unique_match = family.groupby('Uniprot').head(1).reset_index(drop=True)
        first_unique_match["PDB_1"] = first_unique_match['PDB_1'].str.replace(r';', '')
        pdb_list = first_unique_match["PDB_1"].tolist()
        res = ["Residues"] * len(pdb_list)
        residues_list = first_unique_match["Residues"].str.replace(r';', '').tolist()
        chain = ["Chain"] * len(pdb_list)
        chains_list = first_unique_match["Chain"].str.replace(r';', '').tolist()
        pfam_dict = {a: {b: c, d: e} for a, b, c, d, e in zip(pdb_list, res, residues_list, chain, chains_list)}
    else: #641 #contains all pdb files (duplicates) for each sequence
        family_groups = family.groupby('Uniprot')[["PDB_1", "Chain", "Function", "Family", "Residues"]].agg(lambda x: list(x))
        pdb_list =sum(family_groups["PDB_1"].tolist(),[]) #map(str, value_list) #.str.replace(r';', '')
        pdb_list = [pdb.replace(r';', '') for pdb in pdb_list]
        res = ["Residues"]*len(pdb_list)
        residues_list=sum(family_groups["Residues"].tolist(),[])
        residues_list = [res.replace(r';', '') for res in residues_list]
        chain = ["Chain"] * len(pdb_list)
        chains_list = sum(family_groups["Chain"].tolist(),[])
        chains_list = [chain[-2] for chain in chains_list]
        pfam_dict = {a:{b:c,d:e} for a, b, c,d,e in zip(pdb_list, res,residues_list,chain, chains_list)}
    return pfam_dict, pdb_list
def tree_pair_for_rooting(distance_matrix):
    """Finds the pair of sequences with the largest pairwise distance to set as an outgroup to form the root of an unrooted tree
    :param pandas dataframe distance_matrix: contains pairwise distances across the sequences in the tree"""

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
    rank = sorted_distance_matrix.shape[0] -1
    sequence_0 = sorted_distance_matrix.loc[rank].Sequence_0
    sequence_1 = sorted_distance_matrix.loc[rank].Sequence_1
    return sequence_0,sequence_1
def download_PDB(files_list, directory):
    """Reads a list of PDB_files and downloads them to a folder
    :param list files_list: list of PDB files names to download
    :param str directory: folder or directory where to download the files"""
    pdbl = PDBList()
    for i, file in enumerate(files_list):
        pdbl.retrieve_pdb_file(file, pdir=directory, obsolete=False, file_format="pdb")
def convert_to_pandas(distance_matrix):
    """Converts a biopython pairwise distance matrix into a symmetric pandas dataframe
    :param biopython calculator matrix: pairwise distance matrix
    """
    b = np.zeros([len(distance_matrix.matrix), len(max(distance_matrix.matrix, key=lambda x: len(x)))])
    for i, j in enumerate(distance_matrix.matrix):
        b[i][0:len(j)] = j
    #df = pd.DataFrame(b, index=DistanceMatrix.names, columns=DistanceMatrix.names)
    b_transpose = np.transpose(b)
    b = b + b_transpose
    df = pd.DataFrame(b, index=distance_matrix.names, columns=distance_matrix.names)
    return  df
def infer_tree(alignment, alignment_file_name,name,method=None,tree_file_name=None,tree_file=None,storage_folder=""):
    """ Performs tree inference or reads an input given tree, returns an ete3 tree formated tree
    :param biopython alignment alignment: biopython alignment class
    :param str alignment_file_name: path to alignment file
    :param str name: dataset project name
    :param str method: tree inference method (if tree is not given)
    :param str tree_file_name: name to give to the tree file
    .param str tree_file:path to a possible given tree in newick format 1
    :param str storage_folder: folder where to store the results of the tree inference

    """
    if tree_file:
        print("Using given tree file...")
        tree = TreeEte3(tree_file,format=1,quoted_node_names=True)
        return tree

    else:
        # Pairwise distance matrix
        print("Building distance matrices and {} tree...".format(method))
        if len(alignment) < 200 and method in ["nj","nj_rooted","upgma"]:
            calculator = DistanceCalculator('blosum62')  # DNA ---> Identity// Protein ---> blosum62
            distance_matrix_cal = calculator.get_distance(alignment)
            distance_matrix_cal_pandas = convert_to_pandas(distance_matrix_cal)
            distance_matrix_cal_pandas.to_csv("{}/{}_distance_matrix.csv".format(storage_folder,name))
            #https://stackoverflow.com/questions/30247359/how-does-biopython-determine-the-root-of-a-phylogenetic-tree
            if method == "nj":
                print("Tree inference via Neighbour Joining NOT rooted method...")
                constructor = DistanceTreeConstructor(method="nj")
                tree = constructor.nj(distance_matrix_cal)
                tree = to_ete3(tree)
                return tree
            elif method == "nj_rooted":
                print("Tree inference via Neighbour Joining with additional rooting method...")
                constructor = DistanceTreeConstructor(method="nj")
                tree = constructor.nj(distance_matrix_cal)
                tree = to_ete3(tree)
                # Making a root:
                sequence_0, sequence_1 = tree_pair_for_rooting(distance_matrix_cal_pandas)
                tree.set_outgroup(tree & sequence_0)
                # ancestor = tree.get_common_ancestor(sequence_0, sequence_1)
                # tree.set_outgroup(ancestor)
                return tree
            elif method == "upgma":
                print("Tree inference via Upgma rooted method...")
                constructor = DistanceTreeConstructor(method="upgma") # nj method is unrooted in biopython. upgma is rooted
                tree = constructor.upgma(distance_matrix_cal)
                tree = to_ete3(tree)
                return tree
        elif method == "iqtree":
            print("Iqtree ML method...")
            alignment_f = [alignment_file_name if alignment_file_name else "{}/{}.mafft".format(storage_folder,name)][0]
            tree_file_name = alignment_f.split(".")[0] + ".treefile"

            if not os.path.exists(tree_file_name):
                #-o	Specify an outgroup taxon name to root the tree. The output tree in .treefile will be rooted accordingly. DEFAULT: first taxon in alignment
                taxon_root = False
                if taxon_root:
                    root=21
                    subprocess.run(args=["iqtree","-s",alignment_f.split(".")[0],"--aln",alignment_f,"-nt","AUTO","-o",root],stderr=sys.stderr, stdout=sys.stdout)
                else:
                    subprocess.run(args=["iqtree","-s",alignment_f.split(".")[0],"--aln",alignment_f,"-nt","AUTO"],stderr=sys.stderr, stdout=sys.stdout)
                os.remove(alignment_f + ".log")
                os.remove(alignment_f + ".bionj")
                os.remove(alignment_f + ".ckp.gz")
                os.remove(alignment_f + ".model.gz")
            distance_matrix_cal = pd.read_csv(alignment_f+".mldist", sep="\\s+", skiprows=1, header=None)
            distance_matrix_cal.columns = ["rows"] + distance_matrix_cal.iloc[:,0].to_list()
            distance_matrix_cal.set_index("rows", inplace=True)
            distance_matrix_cal.index.name = ""
            distance_matrix_cal.to_csv("{}/{}_distance_matrix.csv".format(storage_folder,name))
            tree = TreeEte3(alignment_f+ ".treefile")
            return tree
        elif method == "rapidnj":
            print("Using Rapidnj to build NOT rooted tree...")
            tree_file_name = ["{}/{}.tree".format(storage_folder,name) if not tree_file_name else tree_file_name][0]
            alignment_f = [alignment_file_name if alignment_file_name else "{}/{}.mafft".format(storage_folder,name)][0]
            with open(tree_file_name, "w") as tree_file_out:
                subprocess.run(args=["rapidnj",alignment_f, "-i", "fa"], stdout=tree_file_out)
            tree_file_out.close()
            tree = TreeEte3(tree_file_name)
            return tree
def infer_alignment(alignment_file,input_name_file,output_name_file):
    """
    Reads and alignment or performs alignment using MAFFT [MAFFT multiple sequence alignment software version 7: improvements in performance and usability]. Returns a dictionary with the sequence name
    and the sequence and a biopython alignment object
    :param str or None alignment_file: path to pre-computed alignment to read
    :param str input_file_name: path to the file containing unaligned sequences,in fasta format
    :param str output_file_name: name of the file that will contain the aligned sequences"""
    # Align the polypeptides/sequences and write to a fasta file
    print("Analyzing alignment...")
    if alignment_file: #The alignment file should contain the polypeptides of the PDB structures and sequences without structures
        print("Reading given alignment file ...")
        # Read the aligned sequences
        alignment = AlignIO.read("{}".format(alignment_file), "fasta")
        alignment_ids = []
        alignment_seqs = []
        for i,aligned in enumerate(alignment):
            alignment_ids.append(alignment[i].id)
            alignment_seqs.append(alignment[i].seq.strip("*")) #Highlight: Remove stop codons
        dict_alignment = dict(zip(alignment_ids, alignment_seqs))
        return dict_alignment, alignment
    else:
        print("Using mafft to align...")
        mafft_cline = MafftCommandline(input=input_name_file)
        stdout, stderr = mafft_cline()
        with open(output_name_file, "w") as handle:
            handle.write(stdout)
        handle.close()

        # Read the aligned sequences
        alignment = AlignIO.read(output_name_file, "fasta")
        alignment_ids = [alignment[i].id for i, aligned in enumerate(alignment)]
        alignment_seqs = [alignment[i].seq for i, aligned in enumerate(alignment)]
        dict_alignment = dict(zip(alignment_ids, alignment_seqs))
        return dict_alignment, alignment
def calculate_pairwise_distance(name,alignment,storage_folder):
    """Calculates the pairwise distance matrix accross the sequences in the alignment
    :param str name: data set project name
    :param biopython alignment object: object containing aligned sequences
    :param str storage_folder: path where to store the calculated pairwise matrix"""
    print("Building pairwise distance matrix ...")
    if len(alignment) <= 200: #very slow method
        calculator = DistanceCalculator('identity')
        distance_matrix_biopython = calculator.get_distance(alignment)
        distance_df = pd.DataFrame(index=distance_matrix_biopython.names, columns=distance_matrix_biopython.names)
        distance_df = distance_df.fillna(0)
        for i, t1 in enumerate(distance_matrix_biopython.names):
            for j, t2 in enumerate(list(distance_matrix_biopython.names)[i + 1:]):
                distance_df.loc[[t1], [t2]] = distance_matrix_biopython[t1,t2]
                distance_df.loc[[t2], [t1]] = distance_matrix_biopython[t1,t2]
        distance_df.to_csv("{}/{}_pairwise_distance_matrix.csv".format(storage_folder,name))

    else: #TODO: faster implementation---> it's done elsewhere
        print("Finish implementing for larger datasets")
        #Highlight: Turn alignment into numpy array, vectorize to numbers, computer pairwise in fast manner
        pass
def calculate_patristic_distance(name_file,combined_dict,nodes_and_leafs_names,tree,tree_file, storage_folder):
    """Calculates the patristic distances or branch lengths across the nodes in a tree. It also saves the tree in different formats needed for benchmarking etc
    :param str name_file: data set project name
    :param dict combined_dict #TODO: Remove?
    :param list nodes_and_leafs_names: tree nodes in tree-level order stored in a list
    :param ete3-tree tree: ete3 object containing tree
    :param str tree_file: path to the stored tree file
    :param str storage_folder: folder where to store the results
    """

    n_seqs = len(combined_dict)
    #work_dir = os.path.dirname(os.path.abspath(__file__))
    work_dir = ""
    if n_seqs > 200:
        print("Dataset larger than 200 sequences: Using R script for patristic distances (cladistic matrix is NOT available)!")
        warnings.warn("Dataset larger than 200 sequences: Requires R and the ape library")
        command = 'Rscript'
        #path2script = '/home/lys/Dropbox/PhD/DRAUPNIR/Calculate_Patristic.R'
        path2script = "Calculate_Patristic.R"
        if tree_file:
            new_tree = work_dir +tree_file.split(".")[0]+".newick"
            new_tree_format8 = work_dir  + tree_file.split(".")[0] + ".format8newick"
            new_tree_format6 = work_dir  + tree_file.split(".")[0] + ".format6newick"
            new_tree_format7 = work_dir  + tree_file.split(".")[0] + ".format7newick"
        else:
            new_tree = work_dir + "{}/{}.newick".format(storage_folder,name_file)
            new_tree_format8 = work_dir + "{}/{}.format8newick".format(storage_folder,name_file)
            new_tree_format6 = work_dir + "{}/{}.format6newick".format(storage_folder,name_file)
            new_tree_format7 = work_dir + "{}/{}.format7newick".format(storage_folder,name_file)

        tree.write(outfile=new_tree_format8, format=8,format_root_node=True) # format 8 all nodes names
        tree.write(outfile=new_tree_format6, format=6,format_root_node=True)
        tree.write(outfile=new_tree_format7, format=7,format_root_node=True) #all nodes names + branch lengths
        tree.write(outfile=new_tree,format=1) #save the renamed tree, format 9 to not save the internal nodes names
        patristic_file = "{}/{}_patristic_distance_matrix.csv".format(storage_folder,name_file)
        if not os.path.exists(patristic_file):
            # Build subprocess command
            subprocess.check_call([command,path2script,new_tree,patristic_file])
        else:
            print("Patristic matrix already exists at {}, not calculated. Delete it otherwise".format(patristic_file))
        #Highlight: PHYLOCOM
        # # #Highlight:Transform the file to Nexus format
        # new_tree = tree_file.split(".")[0]+".newick"
        # new_tree_filename = ntpath.basename(new_tree)
        # working_directory = os.path.dirname(os.path.abspath(new_tree))
        # tree.write(outfile=new_tree,format=1) #save the renamed tree, format 9 to not save the internal nodes names
        # patristic_file = "Datasets_Folder/Patristic_distance_matrix_{}.txt".format(name_file)
        # with open(patristic_file, "w") as patristic_dist_out:
        #     subprocess.run(args=["phylocom","phydist", "-f", new_tree_filename],stderr=sys.stderr, stdout=patristic_dist_out,cwd=working_directory)
        # patristic_matrix = pd.read_csv(patristic_file,sep="\t",index_col=0)
        # patristic_matrix.to_csv("Datasets_Folder/Patristic_distance_matrix_{}.csv".format(name_file),index_label="rows")

    else:
        if tree_file:
            new_tree = work_dir + "/" + tree_file.split(".")[0] +".newick"
            new_tree_format8 = work_dir + "/" + tree_file.split(".")[0] + ".format8newick"
            new_tree_format6 = work_dir + "/" + tree_file.split(".")[0] + ".format6newick"
            new_tree_format7 = work_dir + "/" + tree_file.split(".")[0] + ".format7newick"
        else:
            new_tree = work_dir + "/{}/{}.newick".format(storage_folder,name_file)
            new_tree_format8 = work_dir + "/{}/{}.format8newick".format(storage_folder,name_file)
            new_tree_format6 = work_dir + "/{}/{}.format6newick".format(storage_folder,name_file)
            new_tree_format7 = work_dir + "/{}/{}.format7newick".format(storage_folder,name_file)


        tree.write(outfile=new_tree_format8, format=8,format_root_node=True)
        tree.write(outfile=new_tree_format6, format=6,format_root_node=True)
        tree.write(outfile=new_tree_format7, format=7,format_root_node=True)
        tree.write(outfile=new_tree,format=1)  # save the renamed tree, format 9 to not save the internal nodes names. format 8 all nodes names

        n_elements = len(nodes_and_leafs_names)
        I = pd.Index(nodes_and_leafs_names, name="rows")
        C = pd.Index(nodes_and_leafs_names, name="columns")
        patristic_matrix = pd.DataFrame(data=np.zeros((n_elements, n_elements)), index=I, columns=C)
        cladistic_matrix = pd.DataFrame(data=np.zeros((n_elements, n_elements)), index=I, columns=C)
        if not os.path.exists("{}/{}_patristic_distance_matrix.csv".format(storage_folder,name_file)):
            for i, t1 in enumerate(nodes_and_leafs_names):
                for j, t2 in enumerate(list(nodes_and_leafs_names)[i + 1:]):
                    cladistic_matrix.loc[[t1], [t2]] = tree.get_distance(t1, t2, topology_only=True)
                    patristic_matrix.loc[[t1], [t2]] = tree.get_distance(t1, t2, topology_only=False)
            cladistic_matrix.to_csv("{}/{}_cladistic_distance_matrix.csv".format(storage_folder,name_file))
            patristic_matrix.to_csv("{}/{}_patristic_distance_matrix.csv".format(storage_folder,name_file))
        else:
            print("Patristic matrix file already exists, not calculated")
def my_layout(node):
    """Ete3 layout that adds the internal nodes names. It is a plug-in for rendering tree images
    :param ete3-node node: node from an ete3 tree"""
    if node.is_leaf():
        # If terminal node, draws its name
        name_face = AttrFace("name",fsize=8,fgcolor="blue")
    else:
        # If internal node, draws label with smaller font size
        name_face = AttrFace("name", fsize=8,fgcolor="red")
    # Adds the name face to the image at the preferred position
    faces.add_face_to_node(name_face, node, column=0, position="branch-right")
def render_tree(tree,storage_folder,name_file):
    """Function to render an ete3 tree into an image
    :param ete3-tree tree: Ete3 tree object
    :param str storage_folder: path to folder where to store the results
    :param name_file: data set project name"""
    ts = TreeStyle()
    ns = NodeStyle()
    #Make thicker lines
    ns["vt_line_width"] = 5
    ns["hz_line_width"] = 5
    # Do not add leaf names automatically
    ts.show_leaf_name = False
    # Use my custom layout
    ts.layout_fn = my_layout
    #print the branch lengths
    ts.show_branch_length = True
    for n in tree.traverse():
            n.set_style(ns)
    try:
        tree.render("{}/return_{}.png".format(storage_folder,name_file),w=1000, units="mm",tree_style=ts)
    except:
        tree.render("{}/return_{}.png".format(storage_folder,name_file), w=1000, units="mm")
def rename_tree_internal_nodes_simulations(tree,with_indexes=False):
    """Rename the internal nodes of an ete3 tree to a label I + number, in simulations the leaves have the prefix A. With indexes shows the names used when transferring to an array
    :param ete3-tree tree: Ete3 tree
    :param bool with_indexes: True --> adds the tree level order indexes to the name
    """
    print("Renaming tree from simulations")
    leafs_names = tree.get_leaf_names()
    edge = len(leafs_names)
    internal_nodes_names = []
    if with_indexes:
        for idx,node in enumerate(tree.traverse()):  # levelorder (nodes are visited in zig zag order from root to leaves)
            if not node.is_leaf():
                node.name = "I" + node.name + "/{}".format(idx)
                internal_nodes_names.append(node.name)
                edge += 1
            else:
                node.name = node.name + "/{}".format(idx)
                #edge += 1
    else:
        for node in tree.traverse():  # levelorder (nodes are visited in zig zag order from root to leaves)
            if not node.is_leaf():
                node.name = "I" + node.name
                internal_nodes_names.append(node.name)
                edge += 1
    return tree
def rename_tree_internal_nodes(tree):
    """Rename the internal nodes of a tree to a label A + number, unless the given newick file already has the names on it
    :param ete3-tree tree: Ete3 tree constructor"""

    #Rename the internal nodes
    leafs_names = tree.get_leaf_names()
    edge = len(leafs_names)
    internal_nodes_names = []
    for node in tree.traverse(): #levelorder (nodes are visited in zig zag order from root to leaves)
        if not node.is_leaf():
            node.name = "A%d" % edge
            internal_nodes_names.append(node.name)
            edge += 1
    return tree
def process_pdb_files(PDB_folder,aa_probs,pfam_dict,one_hot_encoding,min_len):
    """"Extract information from the PDB files
    :param str PDB_folder: path to folder containing the PFB files
    :param int aa_probs: amino acid probabilities
    :param dict pfam_dict: dictionary containing the information on how to extract the sequences from the pdb file
    :param bool one_hot_encoding
    :param int min_len: Lower bound on the sequence length
    """
    parser = PDB.PDBParser()
    #ppb = PPBuilder()
    capp = CaPPBuilder()
    prot_info_dict = {}
    prot_aa_dict = {}
    aminoacid_names = aminoacid_names_dict(aa_probs)  # TODO: generalize to include all types of amino acids. For onehot encodings simply use aa_names_dict.keys()
    files_list = [f for f in listdir(PDB_folder) if isfile(join(PDB_folder, f))]
    duplicates = []
    for i, PDB_file in enumerate(files_list):
        structure = parser.get_structure('{}'.format(PDB_file), join(PDB_folder, PDB_file))
        if pfam_dict:  # contains information on which chain to take and which residues to select from the PDB file
            chain_name = pfam_dict[PDB_file[3:7].upper()]["Chain"]
            Chain_0 = structure[0][chain_name]
            start_residue, end_residue = pfam_dict[PDB_file[3:7].upper()]["Residues"].split("-")
            list_residues = list(range(int(start_residue), int(end_residue) + 1))
        else:  # when there is not Information not available on which chains to pick
            chains = [chain for chain in structure[0]]
            Chain_0 = chains[0]
        polypeptides = capp.build_peptides(Chain_0)  # C_alpha-C-alpha polypeptide
        angles_list = []
        aa_list_embedded = []
        aa_list = []
        # coordinates_list = []
        for poly_index, poly in enumerate(polypeptides):
            if not pfam_dict:
                list_residues = [residue.get_id()[1] for residue in poly]  # TODO: Change to .get_resname()?
            correspondence_index = [index for index, residue in enumerate(poly) if residue.get_id()[1] in list_residues]
            phipsi_list = poly.get_phi_psi_list()
            # Gotta chop also the angles list according to the range of desired residues
            if correspondence_index and pfam_dict:
                angles_list += phipsi_list[correspondence_index[0]:correspondence_index[
                                                                       -1] + 1]  # chop also the angles_list according to the selected residues
            elif not correspondence_index and pfam_dict:  # if there is not a corresponding residue do not append anything
                angles_list += []
            else:  # keep appending all residues in the polypeptide
                angles_list += phipsi_list
            for residue in poly:
                residue_position = residue.get_id()[1]
                if residue_position in list_residues:  # Only get the desired residues (though, if pfam_dict is absent it will append the entire polypeptide)
                    aa_name = protein_letters_3to1[residue.get_resname()]
                    if aa_name not in aminoacid_names.keys():
                        raise ValueError("Please select aa_probs > 21 , in order to allow using special amino acids")
                    if one_hot_encoding:
                        aa_name_index = aminoacid_names[aa_name] #0-index
                        one_hot = np.zeros(aa_probs)  # one position, one character
                        one_hot[aa_name_index] = 1.0
                        aa_list_embedded.append(one_hot)
                    else:
                        aa_name_index = aminoacid_names[aa_name]
                        aa_list_embedded.append(aa_name_index)
                    aa_list.append(aa_name)
        # Replace None values from NT and CT angles by npr.normal(np.pi,0.1)----> actually, perhaps just chop them, too much missinformation
        angles_list_filled = []
        for tpl in angles_list:
            tpl = list(tpl)
            if None in tpl:
                tpl[np.where(np.array(tpl) == None)[0][0]] = npr.normal(np.pi, 0.1)
            angles_list_filled.append(tpl)
        seq_len = len(angles_list_filled)
        aa_info = np.zeros((seq_len + 2, 30))  # will contain information about all the aminoacids in this current sequence
        aa_info[0] = [seq_len] + [0] * 29
        for index in range(2,seq_len + 2):  # First dimension contains some custom information (i.e name), second is the git vector and the rest should have the aa type and the angles
            if one_hot_encoding:
                aa_info[index] = np.hstack([aa_list_embedded[index - 2], angles_list_filled[index - 2], [0] * 7])
            else:
                aa_info[index] = np.hstack([aa_list_embedded[index - 2], angles_list_filled[index - 2], [0] * 27])
        if seq_len > min_len and not all(v == (None, None) for v in angles_list):  # skip the proteins that are empty or too small | skip the proteins with all None values in the angles
            if "".join(aa_list) not in duplicates:
                prot_info_dict[files_list[i][3:7]] = aa_info
                prot_aa_dict[files_list[i][3:7]] = aa_list
                duplicates.append("".join(aa_list))
    return prot_aa_dict,prot_info_dict
def create_dataset(name_file,
                   one_hot_encoding,
                   min_len=30,
                   fasta_file=None,
                   PDB_folder=None,
                   alignment_file=None,
                   tree_file = None,
                   pfam_dict= None,
                   method="iqtree",
                   aa_probs=21,
                   rename_internal_nodes=False,
                   storage_folder=""):
    """ Complex function to create the dataset and additional files (i.e dictionaries) that Draupnir uses for inference
    in:
        :param str name_file : dataset name
        :param bool one_hot_encoding: {True,False} WARNING: One hot encoding is faulty, needs to be fixed, DO NOT USE
        :param int min_len: minimum length of the sequence, drops out sequences smaller than this
        :param str fasta_file: path to fasta with unaligned sequences
        :param str or None PDB_folder: Folder with PDB files from where to extract sequences and angles
        :param str or None alignment_file: path to fasta with aligned sequences
        :param str or None tree_file: path to newick tree, format 1 (ete3 nomenclature)
        :param dict or None pfam_dict: dictionary with PDB files names
        :param str method: tree inference methodology,
                          "iqtree": for ML tree inference by IQtree (make sure is installed globally),
                          "nj": for neighbour joining unrooted tree inference (biopython),
                          "nj_rooted": for NJ rooted tree inference (selects a root based on the distances beetwen nodes) (biopython),
                          "upgma": UPGMA (biopython),
                          "rapidnj": inference of Fast NJ unrooted inference (make sure is installed globally),
        aa_probs: amino acid probabilities
        rename_internal_nodes: {True,False} use different names for the internal/ancestral nodes from the ones given in the tree
        storage_folder: "draupnir/src/draupnir/data/dataset_name" or folder where the fasta file is located (recommended to put in a specific folder)
    out:
        if one_hot_encoding: where gap is [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
               Tensor with size: [Nsequences]x[Alignment length + 2]x[30] --> [[length,tree_position,dist_to_root,?,0,0,0...],[GIT vector],[aa1 one hot, phi, psi],[aa2 one hot, phi, psi],....[]]
        else: Amino acids are assigned numbers from 1-20, 0 means gap
               Tensor with size: [Nsequences]x[Alignment length + 3]x[30] --> [[length,tree_position,dist_to_root,0,0,0,0...],[GIT vector],[aa1 number, phi, psi],[aa2 number, phi, psi],....[]]
    """

    warnings.simplefilter('ignore', BiopythonWarning)
    one_hot_label= ["onehot" if one_hot_encoding else "integers"]

    prot_info_dict = {}
    prot_aa_dict = {}
    if PDB_folder:# and not alignment_file:---> We allow to have sequences that have and don't have 3D structure
        print("Creating dataset from PDB files...")
        prot_aa_dict,prot_info_dict = process_pdb_files(PDB_folder,aa_probs,pfam_dict,one_hot_encoding,min_len)
    #Remove duplicated sequences in both dictionaries
    if prot_aa_dict:
        fasta_file = "{}/{}/{}.fasta".format(storage_folder,name_file,name_file)
        print("Writing polypeptides to fasta file to {}".format(fasta_file))
        with open(fasta_file, "w") as output_handle:
            pdb_records =[]
            for id,sequence in prot_aa_dict.items():
                pdb_record = SeqRecord(Seq(''.join(sequence)),
                                   id=str(id),
                                   description="",
                                   annotations={"molecule_type": "protein"})
                pdb_records.append(pdb_record)
            SeqIO.write(pdb_records, output_handle, "fasta")
        output_handle.close()
    # Highlight: Align the polypeptides/sequences and write to a fasta angles_list file
    dict_alignment,alignment = infer_alignment(alignment_file,input_name_file=fasta_file,output_name_file="{}/{}/{}.mafft".format(storage_folder,name_file,name_file))
    #calculate_pairwise_distance(name_file,alignment)
    alignment_file = [alignment_file if alignment_file else "{}/{}/{}.mafft".format(storage_folder,name_file,name_file)][0]
    #Highlight: checking that the selected number of probabilities is correct
    summary_aa_probs = [validate_sequence_alphabet(value) for key,value in dict_alignment.items()] #finds the alphabets of each of the sequences in the alignment, checks for dna
    aa_probs = max(aa_probs,max(summary_aa_probs)) #if the input aa_probs is different from those found, the aa_probs change. And also the aa substitution  matrix
    aa_names_dict = aminoacid_names_dict(aa_probs)
    #Highlight: If the aa sequences do not come from  PDB files, they come from an alignment file that needs to be processed
    #dict_alignment_2 = dict.fromkeys(dict_alignment.keys())
    not_aligned_seqs_from_alignment_file ={}
    for key,value in dict_alignment.items():
        aligned_seq = list(dict_alignment[key])
        #no_gap_indexes = np.where(np.array(aligned_seq) != "-")[0] + 2  # plus 2 in order to make the indexes fit in the final dataframe
        not_aligned_seq =list(filter(lambda a: a != "-", aligned_seq))
        seq_len = len(not_aligned_seq)
        git_vector = np.zeros(30) #fake git vector
        aa_info = np.zeros((seq_len + 2, 30))
        aa_info[0] = np.hstack([seq_len,[0]*29]) #first row contains some sequence length information
        aa_info[1] = git_vector #second row contains a git vector (never used but we could use it for something else)
        if one_hot_encoding:
            for index_a, aa_name in enumerate(not_aligned_seq):
                one_hot = np.zeros(aa_probs)
                index = aa_names_dict[aa_name]
                one_hot[index] = 1
                extra_space = 30-aa_probs
                aa_info[index_a+2] = np.hstack([one_hot,[0]*extra_space]) #extra_space = 10 for 20 aa_probs, 9 for 21 aa_probs, 8 for 22 aa_probs
        else:
            for index_aa, aa_name in enumerate(not_aligned_seq):
                index = aa_names_dict[aa_name]
                aa_info[index_aa+2] =np.hstack([index, [0]*29])
        not_aligned_seqs_from_alignment_file[key] = aa_info
        #dict_alignment_2[key] = aa_info

    tree = infer_tree(alignment=alignment,
                      alignment_file_name=alignment_file,
                      name=name_file,
                      method=method,
                      tree_file_name="{}/{}/{}.tree".format(storage_folder,name_file,name_file),
                      tree_file=tree_file,
                      storage_folder="{}/{}".format(storage_folder,name_file))
    max_lenght = alignment.get_alignment_length()

    #Highlight: Combining sequences in the alignment that have a PDB structure and those who don't
    if prot_info_dict: #Otherwise it does not loop over empty dictionaries and does nothing
        not_aligned_seqs_from_alignment_file.update((k, prot_info_dict[k]) for k, v in not_aligned_seqs_from_alignment_file.items() if k in prot_info_dict.keys())  #update the alignment keys with their homolog with pdb information. Update only those sequences in the alignment. Mafft and the IQ tree program might discard different proteins (for example they drop different identical proteins)
        Combined_dict = not_aligned_seqs_from_alignment_file
    else:
        Combined_dict = not_aligned_seqs_from_alignment_file


    if rename_internal_nodes:
        if name_file.startswith("simulations"):
           tree = rename_tree_internal_nodes_simulations(tree,with_indexes=False)
        else:
           tree = rename_tree_internal_nodes(tree)

    leafs_names = tree.get_leaf_names()
    pickle.dump(leafs_names,open('{}/{}/{}_Leafs_names_list.p'.format(storage_folder,name_file,name_file), 'wb'))
    if len(leafs_names) <= 200:
        print("Rendering tree...")
        render_tree(tree, "{}/{}".format(storage_folder,name_file), name_file)
    internal_nodes_names = [node.name for node in tree.traverse() if not node.is_leaf()]

    ancestors_all =[]
    for node in tree.traverse():
        ancestors_node =[node.name.replace("'","")]+[node.dist] +[ancestor.name.replace("'","") for ancestor in node.get_ancestors()]
        ancestors_all.append(ancestors_node)
    length = max(map(len, ancestors_all))
    ancestors_info = np.array([xi + [None] * (length - len(xi)) for xi in ancestors_all])

    tree_levelorder_names = np.asarray([node.name.replace("'","") for node in tree.traverse()])

    tree_levelorder_dist =np.asarray([node.dist for node in tree.traverse()])
    #Add the index of the sequence in the tree to the seq length array info
    Dataset = np.zeros((len(Combined_dict), max_lenght + 1 + 1 +1, 30),dtype=object)  # 30 dim to accomodate git vectors. Careful with the +3 (to include git, seqlen and row/protein name)
    for i, (key,val) in enumerate(Combined_dict.items()):
        aligned_seq = list(dict_alignment[key].strip(","))
        no_gap_indexes = np.where(np.array(aligned_seq) != "-")[0] + 3  # plus 3 in order to make the indexes fit in the final dataframe
        Dataset[i,0,0] = key.replace("'","") #row name/sequence name
        Dataset[i, 1:3] = Combined_dict[key][:2] #Insert seq len and git vector
        if name_file in ["benchmark_randall","benchmark_randall_original","benchmark_randall_original_naming"]:#their leaves have number names, so we keep them instead of changing them for the tree level order ones
            Dataset[i, 1, 1] = int(key.replace("'", ""))
        elif name_file in ["PF01038_lipcti_msa_fungi"]:
            Dataset[i, 1, 1] = np.where(tree_levelorder_names == key.replace(":","_"))[0][0]  # the node name will be its position in the tree
        else:
            Dataset[i,1,1] = np.where(tree_levelorder_names == key.replace("'",""))[0][0] #the node name will be its position in the tree
        Dataset[i, 1, 2] =  tree_levelorder_dist[Dataset[i,1,1]] #distance to the root? that's according to the documentation yes, but is different from the patristic distances
        Dataset[i, no_gap_indexes] = Combined_dict[key][2:]  # Assign the aa info (including angles) to those positions where there is not a gap
        if one_hot_encoding:
            #Highlight: no_gap indexes is not a boolean, we have to convert it
            gaps_mask = np.ones(max_lenght+3,np.bool)
            gaps_mask[no_gap_indexes] = False #do not keep the positions where there is not a gap
            gaps_mask[0] = False #not the first two rows
            gaps_mask[1] = False
            gaps_mask[2] = False
            Dataset[i,gaps_mask,0] = 1. #if one hot encoding the gaps with be assigned the first position in one hot encoding

    #  Reconstruct the tree from the distance matrix
    print("Building patristic and cladistic matrices ...")
    tree_save = pd.DataFrame(ancestors_info)
    #tree_save = pd.DataFrame({"Nodes_Names":tree_levelorder_names.tolist(),"Distance_to_root":tree_levelorder_dist.tolist()})
    tree_save.to_csv("{}/{}/{}_tree_levelorder_info.csv".format(storage_folder,name_file,name_file),sep="\t")
    nodes_and_leafs_names = internal_nodes_names + leafs_names
    calculate_patristic_distance(name_file,Combined_dict,nodes_and_leafs_names,tree,tree_file,"{}/{}".format(storage_folder,name_file))
    calculate_closest_leaves(name_file,tree,"{}/{}".format(storage_folder,name_file))
    calculate_directly_linked_nodes(name_file, tree,"{}/{}".format(storage_folder,name_file))
    calculate_descendants(name_file,tree,"{}/{}".format(storage_folder,name_file))
    print("Ready and saved!")
    print("Building clades (warning: collapses the original tree!)")
    divide_into_monophyletic_clades(tree,"{}/{}".format(storage_folder,name_file),name_file)
    np.save("{}/{}/{}_dataset_numpy_aligned_{}.npy".format(storage_folder,name_file,name_file,one_hot_label[0]), Dataset)
    max_lenght_not_aligned = max([int(sequence[0][0]) for idx,sequence in Combined_dict.items()]) #Find the largest sequence without being aligned
    print("Creating not aligned dataset...")
    Dataset_not_aligned = np.zeros((len(Combined_dict), max_lenght_not_aligned +3, 30), dtype=object)  # 30 for future git vectors. Careful with the +3
    for i,(key,value) in enumerate(Combined_dict.items()):
        Dataset_not_aligned[i,0,0] = key
        Dataset_not_aligned[i, 1:3] = Combined_dict[key][:2] #Fill in the sequence lenght and the git vector
        if name_file in ["benchmark_randall_original_naming"]:
            Dataset[i, 1, 1] = int(key.replace("'", ""))
        else:
            Dataset[i,1,1] = np.where(tree_levelorder_names == key.replace("'",""))[0][0] #position in the tree
        Dataset_not_aligned[i, 1, 2] =  tree_levelorder_dist[Dataset[i,1,1]]
        Dataset_not_aligned[i, 3:int(Combined_dict[key][0][0]) +3] = Combined_dict[key][2:] #Fill in the amino acids "letters"/"numbers" and their angles
        if one_hot_encoding:
            # #Highlight: no_gap indexes is not a boolean, we have to convert it
            # gaps_mask = np.ones(max_lenght+3,np.bool)
            # gaps_mask[no_gap_indexes] = False #do not keep the positions where there is not a gap
            # gaps_mask[0] = False #not the first two rows
            # gaps_mask[1] = False
            # gaps_mask[2] = False
            # Dataset[i,gaps_mask,0] = 1. #if one hot encoding the gaps with be assigned the first position in one hot encoding
            Dataset_not_aligned[i, (int(Combined_dict[key][0][0]) + 3):,0] = 1.
    np.save("{}/{}/{}_dataset_numpy_NOT_aligned_{}.npy".format(storage_folder,name_file,name_file,one_hot_label[0]), Dataset_not_aligned)

    return tree_file

def symmetrize_and_clean(matrix,ancestral=True):
    """Remove, if necessary the ancestral nodes from the train matrix
    :param pandas-array matrix: patristic or cladistic matrices with columns and indexes as node names """
    if not ancestral:#Drop the ancestral nodes information
        matrix = matrix[~matrix.index.str.contains('^A{1}[0-9]+(?![A-Z])+')]  # repeat with [a-z] if problems
        matrix = matrix.loc[:,~matrix.columns.str.contains('^A{1}[0-9]+(?![A-Z])+')]
    matrix = symmetrize(matrix)
    return matrix
def rename_axis(matrix,nodes,name_file = None):
    """
    Rename the nodes names to their tree level index
    nodes: tree level order node names"""
    if len(nodes) != 0 and name_file not in ["benchmark_randall","benchmark_randall_original","benchmark_randall_original_naming"]: #and not name.startswith("simulations"): #use the level order transversal tree information
        #Highlight: If nan (pd.isnull) is found, is because the root name is messed up and missing, just write it down in the patristic matrix file!!!! Or the names in the matrix != names in trevel order

        matrix.index = [np.where(nodes == node_name)[0][0] for node_name in matrix.index] #TODO; why am I doing this twice?
        matrix.columns = [np.where(nodes == node_name)[0][0] for node_name in matrix.columns]
        return matrix
    elif  name_file in ["benchmark_randall","benchmark_randall_original","benchmark_randall_original_naming"] :
        matrix.index = matrix.index.str.replace("A","").astype("int")
        matrix.columns = matrix.columns.str.replace("A", "").astype("int")
        return matrix

    else:
        return matrix
def sum_matrices(matrix1,matrix2):
    """Sums cladistic and patristic matrices to form an evolutionary matrix
    :param: pandas-array matrix1
    :param: pandas-array matrix2
    """
    column_names = matrix1.columns.values  # 82 + 1
    column_names = np.concatenate((np.array([float("-inf")]), column_names))  # 82 + 1 (to fit the numpy array
    rows_names = matrix1.index.values  # 82
    matrix1 = matrix1.to_numpy()
    matrix2 = matrix2.to_numpy()
    matrix = matrix1 + matrix2
    matrix = normalize_standarize(matrix)
    matrix = np.column_stack((rows_names, matrix))
    matrix = np.row_stack((column_names, matrix))
    return matrix
def pandas_to_numpy(matrix):
    """Converts pandas array to numpy array, in this case by transforming the nodes names
    :param matrix"""
    column_names = matrix.columns.values.astype("int")  # 82 + 1
    column_names = np.concatenate((np.array([float("-inf")]), column_names))  # 82 + 1 (to fit the numpy array
    rows_names = matrix.index.values.astype("int")  # 82
    matrix = np.column_stack((rows_names, matrix))
    matrix = np.row_stack((column_names, matrix))
    return matrix
def convert_to_letters(seq,aa_probs):
    """Converts back the integers assigned to the amino acids to letters
    :param seq: seq
    :param int aa_probs: amino acid probabilities"""

    aa_names_dict = aminoacid_names_dict(aa_probs)
    aa_names_dict_reverse = {val:key for key,val in aa_names_dict.items()}
    if not isinstance(seq[0],float):
        seq_letters = [aa_names_dict_reverse[position.item()] for position in seq if position.item() in aa_names_dict_reverse]

    else:
        seq_letters = [aa_names_dict_reverse[position] for position in seq if position in aa_names_dict_reverse]

    return ''.join(seq_letters)
def score_match(pair, matrix):
    """Returns the corresponding blosum scores between the pair of amino acids
    :param tuple pair: pair of amino acids to compare
    :param matrix: Blosum matrix """
    if pair not in matrix:
        return matrix[(tuple(reversed(pair)))]
    else:
        return matrix[pair]
def score_pairwise(seq1, seq2, matrix, gap_s, gap_e):
    #TODO: https://stackoverflow.com/questions/5686211/is-there-a-function-that-can-calculate-a-score-for-aligned-sequences-given-the-a
    """
    Calculates the blosum score of the true sequence against the predictions
    :param matrix matrix: Blosum matrix containing the log-odds scores (he logarithm for the ratio of the
       likelihood of two amino acids appearing with a biological sense and the likelihood of the same amino acids appearing by chance)
        (the higher the score, the more likely the corresponding amino-acid substitution is)
    :param int gap_s: gap penalty
    :param int gap_e : mismatch penalty
    """
    score = 0
    gap = False
    for i in range(len(seq1)):
        pair = (seq1[i], seq2[i])
        if not gap:
            if '-' in pair:
                gap = True
                score += gap_s
            # elif "*" in pair: #TODO: Keep?
            #     score +=gap_e
            else:
                score += score_match(pair, matrix)
        else:
            if '-' not in pair:# and "*" not in pair:
                gap = False
                score += score_match(pair, matrix)
            else:
                score += gap_e
    return score
def score_pairwise_2(seq1, seq2, matrix, gap_s, gap_e, gap = True):
    """Alternative method to calculate the blosum score between 2 sequences"""
    for A,B in zip(seq1, seq2):
        diag = ('-'==A) or ('-'==B)
        yield (gap_e if gap else gap_s) if diag else matrix[(A,B)]
        gap = diag
def normalize_standarize(x):
    "Normalizes and standarizes an input matrix"
    norm = np.linalg.norm(x)
    normal_array = x / norm
    return normal_array
def folders(folder_name,basepath):
    """ Creates a folder at the indicated location. It rewrites folders with the same name
    :param str folder_name: name of the folder
    :param str basepath: indicates the place where to create the folder
    """
    #basepath = os.getcwd()

    if not basepath:
        newpath = folder_name
    else:
        newpath = basepath + "/%s" % folder_name

    if not os.path.exists(newpath):

        try:
            original_umask = os.umask(0)
            os.makedirs(newpath, 0o777)
        finally:
            os.umask(original_umask)
    else:
        shutil.rmtree(newpath)  # removes all the subdirectories!
        os.makedirs(newpath,0o777)
def divide_batches(Dataset,number_splits):
    """ Divides the training dataset in a given number of splits"""
    Dataset = Dataset[Dataset[:,0,0].argsort()]
    Dataset_splits =np.array_split(Dataset,number_splits)
    return Dataset_splits
def perc_identity_alignment(aln):
    """Calculates the %ID of an alignment
    :param aln: biopython alignment object"""
    i = 0
    for a in range(0,len(aln[0])):
        s = aln[:,a]
        if s == len(s) * s[0]:
            i += 1
    return 100*i/float(len(aln[0]))
def perc_identity_pair_seq(seq1,seq2):
    """Calculates the percent identity among a pair of sequences
    :param str seq1
    :param str seq2"""
    i = 0
    seq1 =list(seq1)
    seq2=list(seq2)

    aln = np.row_stack([seq1,seq2])
    for a in range(0, aln.shape[1]):
        s = aln[:,a].tolist()
        s = "".join(s)
        if s == len(s) * s[0]:
            i += 1
    return 100*i/float(len(aln[0]))
def incorrectly_predicted_aa(seq1,seq2):
    """"""
    i = 0
    seq1 =list(seq1)
    seq2=list(seq2)

    aln = np.row_stack([seq1,seq2])
    for a in range(0, aln.shape[1]):
        s = aln[:,a].tolist()
        s = "".join(s)
        if s == len(s) * s[0]:
            i += 1 #same aa
    return len(aln[0])-i
def to_ete3(tree):
    import tempfile
    from ete3 import Tree as EteTree
    with tempfile.NamedTemporaryFile(mode="w") as tmp:
        Phylo.write(tree, tmp, 'newick')
        tmp.flush()
        return EteTree(tmp.name,format=1)
def to_biopythonTree(tree):
    import tempfile
    from ete3 import Tree as EteTree
    from Bio import Phylo
    with tempfile.NamedTemporaryFile(mode="w+") as tmp:
        tree.write(outfile=tmp,format=1,format_root_node=True)
        tmp.flush()
        return Phylo.read(tmp.name, 'newick')
def load_obj(name):
    # with open(name, 'rb') as f:
    #     return pickle.load(f)
    data = bz2.BZ2File(name, "rb")
    data = cPickle.load(data)
    # with open(name,"rb") as json_file:
    #     data = json.load(json_file, encoding="utf-8")
    # data_arrays = {}
    # for key,value in data.items():
    #     data_arrays[key] = np_jax.array(value)
    # return data_arrays
    return data
def save_obj(obj, name):
    # with open(name, 'wb') as f:
    #     pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    with bz2.BZ2File(name, "wb") as f:
        cPickle.dump(obj, f)
    # obj_serialized={}
    # for key,value in obj.items():
    #     obj_serialized[key] = value.tolist()
    #
    # with open(name, 'w',encoding='utf-8') as outfile:
    #     json.dump(obj_serialized, outfile,ensure_ascii=False, indent=4)
def autolabel(rects, ax, blosum_dict=None,percent_id_dict=None,blosum_true_dict=None):
    """Helper function to assign a integer value on top of the histogram bar
    :param rects: matplotlib rectangles
    :param ax: matplotlib axis
    :param blosum_dict: dictionary containing the blosum scores of the true vs the prediction sequences
    :param blosum_true_dict: dictionary containing the blosum scores of the true vs true sequences"""
    # Get y-axis height to calculate label position from.
    (y_bottom, y_top) = ax.get_ylim()
    y_height = y_top - y_bottom
    if blosum_dict is not None:
        for rect,blosum_score,pid,blosum_score_true in zip(rects,blosum_dict.values(),percent_id_dict.values(),blosum_true_dict.values()):
            height = rect.get_height()
            # Fraction of axis height taken up by this rectangle
            p_height = (height / y_height)
            # If we can fit the label above the column, do that;
            # otherwise, put it inside the column.
            if p_height > 0.95:  # arbitrary; 95% looked good to me.
                label_position = height - (y_height * 0.05)
            else:
                label_position = height + (y_height * 0.01)

            ax.text(rect.get_x() + rect.get_width() / 2., label_position,
                    'InAA:%d \n Bs:%d | BsTrue:%d \n PID:%1f' % (int(height),blosum_score,blosum_score_true,pid),
                    ha='center', va='bottom')
    else:
        for rect in rects:
            height = rect.get_height()
            # Fraction of axis height taken up by this rectangle
            p_height = (height / y_height)
            # If we can fit the label above the column, do that;
            # otherwise, put it inside the column.
            if p_height > 0.95:  # arbitrary; 95% looked good to me.
                label_position = height - (y_height * 0.05)
            else:
                label_position = height + (y_height * 0.01)

            ax.text(rect.get_x() + rect.get_width() / 2., label_position,
                    '%d' % int(height),
                    ha='center', va='bottom')
def symmetrize(a):
    "Make a triangular upper matrix symmetric along the diagonal. Valid for numpy arrays and pandas dataframes"
    a = np.maximum(a, a.transpose())
    return a
def build_predicted_tree(index,sampled_sequences,leaf_names,name,results_directory,method): #TODO: Correct
    """Builds a tree between the train (leaves) data and the predicted ancestors
    :param index: node index
    :param sampled_sequences: dataset_test with sampled sequences
    :param leaf_names: leaf node names
    :param name: datasetproject name
    :param results_directory
    :param method: tree inference engine"""
    training_seqs = SeqIO.parse("{}/{}_training.fasta".format(results_directory, name), "fasta")
    with open("{}/Tree_Alignment_Sampled/{}_combined_sample_index_{}.fasta".format(results_directory, name,index), 'w') as w_file:
        SeqIO.write(training_seqs,w_file,"fasta")
        records=[]
        for sampled_seq,leaf_name in zip(sampled_sequences,leaf_names):
            record = SeqRecord.SeqRecord(Seq(''.join(sampled_seq).replace("-", "")),annotations={"molecule_type": "protein"},id=leaf_name, description="")
            records.append(record)
        SeqIO.write(records,w_file,"fasta")
    dict_align,alignment = infer_alignment(input_name_file="{}/Tree_Alignment_Sampled/{}_combined_sample_index_{}.fasta".format(results_directory, name,index),
                                           output_name_file="{}/Tree_Alignment_Sampled/{}_combined_sample_index_{}.mafft".format(results_directory,name,index),alignment_file=None)

    #alignment, alignment_file_name,name,method=None,tree_file_name=None,tree_file=None,storage_folder=""
    tree = infer_tree(alignment=alignment,
                      alignment_file_name="{}/Tree_Alignment_Sampled/{}_combined_sample_index_{}.mafft".format(results_directory,name,index),
                      tree_file_name="{}/Tree_Alignment_Sampled/{}_combined_sample_index_{}.tree".format(results_directory,name,index),
                      method=method,
                      name="{}_combined_sample_index_{}.tree".format(name,index))
    return tree
def heatmaps(predictions_samples,dataset_test,name, num_samples,nodes_indexes,results_directory,aa_probs,additional_load,additional_info,correspondence_dict=None):
    """Build heatmaps between the percent identity true sequences (leaves or internal nodes) and the predicted sequences
    :param tensor predictions_samples
    :param dataset_test
    :param str name: dataset project name
    :param int num_samples
    :param tensor nodes_indexes
    :param str results_directory
    :param int aa_probs
    :param namedtuple additional_load
    :param namedtuple additional_info
    :param correspondence_dict"""
    print("Building %ID and Blosum heatmaps & Incorrect aa histogram...")
    #blosum = MatrixInfo.blosum62
    blosum = additional_info.blosum_dict
    nodes_indexes = nodes_indexes.tolist()
    #node_names = dataset_test[:, 0, 1]
    folder = os.path.basename(results_directory).split("_")[0]
    def Percent_ID_Test_SAMPLED_SAMPLED(multiindex=False):
        "Generate the Average and STD %ID among the sampled seqs"
        #TODO: Function for the multiindex. Slice out tensor outside of for loop. "Store" node names outside loop (not waste time looking up multidim arrays)
        percent_id_SAMPLED_SAMPLED = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        blosum_SAMPLED_SAMPLED = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        predictions = predictions_samples[:, :, 3:]
        if multiindex:
            def storage():
                percent_id_SAMPLED_SAMPLED[t1][t2] ["Average"] = np.mean(np.array(percent_id_i), axis=0)
                percent_id_SAMPLED_SAMPLED[t1][t2] ["Std"] = np.std(np.array(percent_id_i), axis=0)
                blosum_SAMPLED_SAMPLED[t1][t2] ["Average"] = np.mean(np.array(blosum_score_i), axis=0)
                blosum_SAMPLED_SAMPLED[t1][t2] [ "Std"] = np.std(np.array(blosum_score_i), axis=0)
        else:
            def storage():
                percent_id_SAMPLED_SAMPLED[t1][t2]  = np.mean(np.array(percent_id_i), axis=0)
                blosum_SAMPLED_SAMPLED[t1][t2]  = np.mean(np.array(blosum_score_i), axis=0)

        for i, t1 in enumerate(nodes_indexes):#for test node
            all_sampled_i = predictions[:,i]  # All samples for the same test seq
            for j, t2 in enumerate(nodes_indexes[i:]):
                all_sampled_j = predictions[:, i+j]  # All samples for the same test seq
                percent_id_i=[] #all samples for the same test sequence
                blosum_score_i=[]
                for w in range(num_samples):
                    seq_sampled_i = convert_to_letters(all_sampled_i[w],aa_probs)
                    seq_sampled_j = convert_to_letters(all_sampled_j[w],aa_probs)
                    pid = perc_identity_pair_seq(seq_sampled_i,seq_sampled_j)
                    blos = score_pairwise(seq_sampled_i,seq_sampled_j,blosum,gap_s=11, gap_e=1)
                    percent_id_i.append(pid)
                    blosum_score_i.append(blos)
                storage()
        return percent_id_SAMPLED_SAMPLED,blosum_SAMPLED_SAMPLED

    def Percent_ID_Test_SAMPLED_OBSERVED(multiindex=False):
        "Generate the Average and STD %ID and Blosum score of the sampled seqs vs obs test"
        percent_id_SAMPLED_OBSERVED = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        blosum_SAMPLED_OBSERVED = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        incorrect_SAMPLED_OBSERVED = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        Dataset_test_sliced = dataset_test[:,2:,0]
        predictions_samples_sliced = predictions_samples[:,:,3:]
        if multiindex:
            def storage():
                percent_id_SAMPLED_OBSERVED[t1][t2]["Average"] = np.mean(np.array(percent_id_i), axis=0)
                percent_id_SAMPLED_OBSERVED[t1][t2]["Std"] = np.std(np.array(percent_id_i), axis=0)
                blosum_SAMPLED_OBSERVED[t1][t2]["Average"] = np.mean(np.array(blosum_score_i), axis=0)
                blosum_SAMPLED_OBSERVED[t1][t2]["Std"] = np.std(np.array(blosum_score_i), axis=0)
                # incorrect_SAMPLED_OBSERVED[node_names[i].item()][node_names[j].item()] ["Average"] = np.mean(np.array(incorrect_aa_i), axis=0)
                # incorrect_SAMPLED_OBSERVED[node_names[i].item()][node_names[j].item()] ["Std"] = np.std(np.array(incorrect_aa_i), axis=0)
        else:
            def storage():
                percent_id_SAMPLED_OBSERVED[t1][t2]  = np.mean(np.array(percent_id_i), axis=0)
                blosum_SAMPLED_OBSERVED[t1][t2]  = np.mean(np.array(blosum_score_i), axis=0)
                #incorrect_SAMPLED_OBSERVED[node_names[i].item()][node_names[j].item()]  = np.mean(np.array(incorrect_aa_i), axis=0)
        for i, t1 in enumerate(nodes_indexes):
            test_obs_i = convert_to_letters(Dataset_test_sliced[i],aa_probs)
            for j, t2 in enumerate(nodes_indexes[i:]):  # for test node
                all_sampled_test = predictions_samples_sliced[:, i+j]  # All samples for the same test seq
                percent_id_i = []
                blosum_score_i = []
                #incorrect_aa_i = []
                for w in range(num_samples):
                    seq_sampled_test = convert_to_letters(all_sampled_test[w],aa_probs)
                    pid = perc_identity_pair_seq(test_obs_i, seq_sampled_test)
                    #wrong_pred = incorrectly_predicted_aa(test_obs_i,seq_sampled_test)
                    blos = score_pairwise(test_obs_i, seq_sampled_test, blosum, gap_s=11, gap_e=1)
                    percent_id_i.append(pid)
                    blosum_score_i.append(blos)
                    #incorrect_aa_i.append(wrong_pred)
                storage()
        return percent_id_SAMPLED_OBSERVED,blosum_SAMPLED_OBSERVED,incorrect_SAMPLED_OBSERVED

    def Percent_ID_Test_OBS_OBSERVED():
        "Generate the Average and STD %ID among obs test"
        percent_id_OBS = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        blosum_OBS = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        Dataset_test_sliced = dataset_test[:,2:,0]
        for i, t1 in enumerate(nodes_indexes):
            test_obs_i = convert_to_letters(Dataset_test_sliced[i],aa_probs)
            #percent_id_OBS[Dataset_test[i, 0, 1].item()][Dataset_test[i, 0, 1].item()] = perc_identity_pair_seq(test_obs_i, test_obs_i)
            for j, t2 in enumerate(nodes_indexes[i:]):
                test_obs_j = convert_to_letters(Dataset_test_sliced[i+j],aa_probs)
                pid_score = perc_identity_pair_seq(test_obs_i,test_obs_j)
                percent_id_OBS[t1][t2] = pid_score
                blosum_score = score_pairwise(test_obs_i,test_obs_j, blosum, gap_s=11, gap_e=1)
                blosum_OBS[t1][t2] = blosum_score
        return percent_id_OBS,blosum_OBS

    def Plot_Heatmap(df,title,title2,annot,mask,vmax):
        fig,ax =  plt.subplots(1, 1,figsize=(10,10))
        hmap = sns.heatmap(df, cmap="Spectral",annot=annot,annot_kws={"fontsize":6},mask=mask,vmax=vmax)
        ax.set_title("{} ;\n".format(title) + r"{}".format(additional_load.full_name))
        ax.set_xlabel(title2.split("_")[-1])
        ax.set_ylabel(title2.split("_")[-2])
        hmap.figure.savefig("{}/Heatmap_{}.png".format(results_directory,title2),
                            format='png',
                            dpi=150)
        plt.clf()
        plt.close()

    multiindex=False
    if multiindex:
        #Highlight: sampled vs observed
        dict_pid_SAMPLED_OBS, dict_blosum_SAMPLED_OBS,dict_incorrect_SAMPLED_OBS = Percent_ID_Test_SAMPLED_OBSERVED(multiindex)
        df_pid_SAMPLED_OBS = pd.concat({k: pd.DataFrame.from_dict(v, 'index') for k, v in dict_pid_SAMPLED_OBS.items()}, axis=0)
        df_blosum_SAMPLED_OBS = pd.concat({k: pd.DataFrame.from_dict(v, 'index') for k, v in dict_blosum_SAMPLED_OBS.items()},axis=0)
        #df_incorrect_SAMPLED_OBS = pd.concat({k: pd.DataFrame.from_dict(v, 'index') for k, v in dict_incorrect_SAMPLED_OBS.items()}, axis=0)
        #Highlight: Sampled vs Sampled
        dict_pid_SAMPLED_SAMPLED,dict_blosum_SAMPLED_SAMPLED=Percent_ID_Test_SAMPLED_SAMPLED(multiindex)
        df_pid_SAMPLED_SAMPLED = pd.concat({k: pd.DataFrame.from_dict(v, 'index') for k, v in dict_pid_SAMPLED_SAMPLED.items()}, axis=0)
        df_blosum_SAMPLED_SAMPLED = pd.concat({k: pd.DataFrame.from_dict(v, 'index') for k, v in dict_blosum_SAMPLED_SAMPLED.items()}, axis=0)

    else:
        # Highlight: sampled vs observed
        dict_pid_SAMPLED_OBS, dict_blosum_SAMPLED_OBS,dict_incorrect_SAMPLED_OBS = Percent_ID_Test_SAMPLED_OBSERVED(multiindex)
        df_pid_SAMPLED_OBS = pd.DataFrame(dict_pid_SAMPLED_OBS)
        df_blosum_SAMPLED_OBS = pd.DataFrame(dict_blosum_SAMPLED_OBS)
        #df_incorrect_SAMPLED_OBS = pd.DataFrame(dict_incorrect_SAMPLED_OBS).to_numpy()
        #Highlight: Sampled vs Sampled
        dict_pid_SAMPLED_SAMPLED, dict_blosum_SAMPLED_SAMPLED = Percent_ID_Test_SAMPLED_SAMPLED(multiindex)
        df_pid_SAMPLED_SAMPLED = pd.DataFrame(dict_pid_SAMPLED_SAMPLED)
        df_blosum_SAMPLED_SAMPLED = pd.DataFrame(dict_blosum_SAMPLED_SAMPLED)


    dict_pid_OBS_OBS,dict_blosum_OBS_OBS = Percent_ID_Test_OBS_OBSERVED()
    df_pid_OBS_OBS = pd.DataFrame(dict_pid_OBS_OBS)
    df_blosum_OBS_OBS = pd.DataFrame(dict_blosum_OBS_OBS)

    #df_lt = df_OBS_OBS.where(np.tril(np.ones(df_OBS_OBS.shape)).astype(np.bool))
    annot = False
    Plot_Heatmap(df_pid_OBS_OBS,"OBS seqs vs OBS seqs %ID","PID_{}_OBS_OBS".format(folder),annot=annot,mask=None,vmax=100)
    Plot_Heatmap(df_pid_SAMPLED_OBS, "OBS vs SAMPLED seqs AVERAGE %ID", "PID_{}_OBS_SAMPLED".format(folder),annot=annot,mask=None,vmax=100)
    Plot_Heatmap(df_pid_SAMPLED_SAMPLED, "SAMPLED vs SAMPLED seqs AVERAGE %ID", "PID_{}_SAMPLED_SAMPLED".format(folder),annot=annot,mask=None,vmax=100)

    vmax = max([df_blosum_OBS_OBS.max().max(),df_blosum_SAMPLED_SAMPLED.max().max(),df_blosum_SAMPLED_OBS.max().max()])
    Plot_Heatmap(df_blosum_OBS_OBS, "OBS seqs vs OBS seqs Blosum Score", "Blosum_{}_OBS_OBS".format(folder),annot=annot,mask=None,vmax=vmax)
    Plot_Heatmap(df_blosum_SAMPLED_OBS, "OBS vs SAMPLED seqs AVERAGE Blosum Score", "Blosum_{}_OBS_SAMPLED".format(folder),annot=annot,mask=None,vmax=vmax)
    Plot_Heatmap(df_blosum_SAMPLED_SAMPLED, "SAMPLED vs SAMPLED seqs AVERAGE Blosum Score", "Blosum_{}_SAMPLED_SAMPLED".format(folder),annot=annot,mask=None,vmax = vmax)
def build_dataframes_overlapping_histograms(aa_sequences_predictions, dataset_train,dataset_test, name, n_samples,nodes_indexes,results_directory,aa_probs,correspondence_dict=None):
    """Plot histogram whose bars represent the percentage identity between the true ancestors vs true leaves and the predicted ancestors vs true leaves
    :param tensor aa_sequences_predictions:
    :param tensor dataset_train
    :param tensor dataset_test
    :param str name: dataset name project
    :param tensor aa_sequences_predictions
    :param int n_samples: number of sampled predictions
    :param nodes_indexes:
    :param str results_directory
    :param int aa_prob: amino acid probabilities (i.e 21)
    :param dict correspondence_dict: correspondence between the tree level order indexes and the --> only useful for benchmark_randall_original_naming
    """
    "https://matplotlib.org/3.1.1/gallery/units/bar_unit_demo.html#sphx-glr-gallery-units-bar-unit-demo-py"
    print("Building Overlapping Histogram...")
    n_children = len(nodes_indexes)
    def Percent_ID_OBS_OBS():
        "Generate the  %ID of the OBS TEST sequences against the OBS train"
        percent_id_OBS = dict.fromkeys(dataset_train[:,0,1].tolist(), dict.fromkeys(dataset_test[:,0,1].tolist(),[]))
        for i,trainseq in enumerate(dataset_train):
            seq_TRAIN_obs_letters_i = convert_to_letters(dataset_train[i, 2:,0],aa_probs)
            for j,testseq in enumerate(dataset_test):
                seq_TEST_obs_letters_i = convert_to_letters(dataset_test[j, 2:,0],aa_probs)
                pid =perc_identity_pair_seq(seq_TRAIN_obs_letters_i,seq_TEST_obs_letters_i)
                percent_id_OBS[dataset_train[i,0,1].item()][dataset_test[j,0,1].item()] = pid
        return percent_id_OBS
    def Percent_ID_PRED_OBS():
        "Generate the Average and STD %ID of the sampled sequences against the OBS train"
        #percent_id_PRED = dict.fromkeys(Dataset_train[:,0,1].tolist(), dict.fromkeys(predictions_samples[0,:,1].tolist(),{"Average":[],"Std":[]})) #{"Average":[],"Std":[]}
        percent_id_PRED = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for i, seq in enumerate(dataset_train):
            seq_TRAIN_obs_letters_i = convert_to_letters(dataset_train[i, 2:,0],aa_probs)
            for j in range(n_children):#for test node #n_children
                all_sampled_index = aa_sequences_predictions[:,j,3:]  # All samples for same test seq
                percent_id_i=[] #all samples for the same test sequence
                for w in range(n_samples):
                    seq_sampled_test = convert_to_letters(all_sampled_index[w],aa_probs)
                    pid = perc_identity_pair_seq(seq_TRAIN_obs_letters_i, seq_sampled_test)
                    percent_id_i.append(pid)
                percent_id_PRED[dataset_train[i,0,1].item()][aa_sequences_predictions[0,j,1]]["Average"] = np.mean(np.array(percent_id_i),axis=0)
                percent_id_PRED[dataset_train[i,0,1].item()][aa_sequences_predictions[0,j,1]]["Std"] = np.std(np.array(percent_id_i), axis=0)
        return percent_id_PRED


    percent_id_PRED = Percent_ID_PRED_OBS()
    percent_id_OBS = Percent_ID_OBS_OBS()

    fig = plt.figure(figsize=(18, 5))
    ax = fig.add_subplot(111)

    ## Indexes where the bars will go
    ind_lines = np.arange(10,len(dataset_train)*10,10 )
    ind_train = np.arange(5,len(dataset_train)*10,10 )  # the x locations for the train nodes
    width = 10/len(dataset_test) -0.3 # the width of the bars
    ind_test = np.arange(0,10,width + 0.3)  # the x locations for the test nodes, taking into account the space between them?
    start = 0
    blue, = sns.color_palette("muted", 1)
    for train_node in dataset_train[:,0,1]:
        for idx,test_node in enumerate(dataset_test[:,0,1]): #TODO: out of range
            rects2 = ax.bar(ind_test[idx] + start ,
                            percent_id_OBS[train_node.item()][test_node.item()],
                            width,
                            color='green',
                            yerr=[0],
                            error_kw=dict(elinewidth=0.01, ecolor='red'),
                            label="OBSERVED")
            rects1 = ax.bar(ind_test[idx] + start,
                            percent_id_PRED[train_node.item()][test_node.item()]["Average"],
                            width,
                            color="orange",
                            alpha=1,
                            yerr=percent_id_PRED[train_node.item()][test_node.item()]["Std"],
                            error_kw=dict(elinewidth=0.01, ecolor='red'),
                            label="SAMPLED")

        start += 10
    # axes and labels
    ax.set_xlim(-width, np.max(ind_train) +5)
    ax.set_ylim(0, 100)
    ax.set_ylabel('%ID')
    ax.set_title('Overlapping Histogram')
    xTickMarks = ['TrainNode_{}'.format(int(i.item())) for i in dataset_train[:,0,1]]
    ax.set_xticks(ind_train)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=45, fontsize=6)
    plt.vlines(ind_lines,ymin=0,ymax=100)
    ## add a legend--> Not working (inside or outside loop)
    ax.legend((rects1,rects2), ('Sampled',"Observed"),loc=1,bbox_to_anchor=(1.15, 1.1))
    plt.savefig("{}/OverlappingHistogram".format(results_directory))
def tree_positional_embeddings(ancestor_info_dict,tree_by_levels_dict):
    """The nodes ordered by tree level order are converted into somewhat one-hot encoded like embeddings
    Implementated following
    https://papers.nips.cc/paper/2019/file/6e0917469214d8fbd8c517dcdc6b8dcf-Paper.pdf
    Input: Tree by levels in traversal order
    Output: Representations of the position of the node in the tree in one hot bit encoding. They represent the one hot encoded path from root to the node """
    degree_tree =2
    n_levels = len(tree_by_levels_dict)
    nodes_representations=defaultdict()
    node_representation = [0]*(degree_tree*n_levels) #initiate the root to all 0
    for level_name, level_nodes in sorted(tree_by_levels_dict.items()):
        all_ancestors_in_level = {index:ancestor_info_dict[node] for index,node in enumerate(level_nodes)} #find the indexes of the nodes sharing the same ancestors
        if np.nan in all_ancestors_in_level.values():#identify root, because it has nan as an ancestor
            nodes_representations[level_nodes[0]] = node_representation
        else:
            node_representation = node_representation[:-degree_tree]
            level_pairs = [level_nodes[i:i + degree_tree] for i in range(0, len(level_nodes), degree_tree)] #Divide level in groups of 2, the tree is in tree traversal, so everything shoudl be ordered
            for level_pair in level_pairs: #these pairs have same ancestor because of tree levels are organized in traversal order
                #get the ancestor node representation
                ancestor = ancestor_info_dict[level_pair[0]] #just use one of them to get the ancestor
                node_representation = nodes_representations[ancestor][:-degree_tree]
                added_representation =[0]*2
                for index,node in enumerate(level_pair):
                    added_representation[index] =1
                    node_representation = added_representation + node_representation
                    nodes_representations[node] = node_representation
                    added_representation = [0] * 2 #restart
                    node_representation = nodes_representations[ancestor][:-degree_tree] #restart
    nodes_representations_array = np.vstack(list(nodes_representations.values()))
    nodes_representations_array = np.c_[ np.array(list(nodes_representations.keys()))[:,None], nodes_representations_array]
    return nodes_representations_array
def extra_processing(ancestor_info,patristic_matrix,results_dir,args,build_config):
    """Computes some dictionaries with the nodes ordered in tree level order, processes the blosum matrix and build a graph with the tree nodes and the branch lengths as edges
     :param namedtuple ancestor_info
     :param tensor patristic_matrix: [n_leaves + n_internal + 1, n_leaves + n_internal + 1]
     :param str results_dir
     :param namedtuple args
     :param namedtuple build_config"""
    AdditionalInfo = namedtuple("AdditionalInfo",
                            ["blosum" ,"blosum_dict","children_dict","ancestor_info_dict", "tree_by_levels_dict", "tree_by_levels_array",
                             "patristic_info", "graph_coo","patristic_full_sparse","nodes_representations_array","dgl_graph"])

    ancestor_info_dict = dict(zip(ancestor_info[:,0].tolist(),ancestor_info[:,2].tolist()))
    pickle.dump(ancestor_info_dict, open('{}/Ancestor_info_dict.p'.format(results_dir), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    if isinstance(patristic_matrix,np.ndarray):
        patristic_info = dict(zip(list(range(len(patristic_matrix)-1)),patristic_matrix[1:,0].astype(int).tolist())) #Skip the fake header
    else:

        patristic_info =dict(zip(list(range(len(patristic_matrix)-1)),patristic_matrix[1:,0].type(torch.int).tolist()))

    ancestors_info_flipped = np.flip(ancestor_info, axis=1) #So we can evaluate in reverse, we start with the last column

    tree_by_levels = []
    for column in ancestors_info_flipped.T:
        indexes = np.where(column == 0)
        level = ancestors_info_flipped.T[-1][indexes]
        tree_by_levels.append(level)

    tree_by_levels = tree_by_levels[:-1] #the root gets added twice because the first column are the node indexes, cannot be fixed, otherwise we cannot do the trick

    tree_by_levels_dict = dict(zip(list(reversed(range(len(tree_by_levels)))),tree_by_levels))
    pickle.dump(tree_by_levels_dict, open('{}/Tree_by_levels_dict.p'.format(results_dir), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    length = max(map(len, tree_by_levels_dict.values()))
    tree_by_levels_array = np.array([xi.tolist() + [None] * (length - len(xi)) for xi in tree_by_levels])
    # tree_levels_2 = Create_tree_by_levels(children_dict)

    children_dict = {}
    for k, v in ancestor_info_dict.items():
        children_dict[v] = children_dict.get(v, [])
        children_dict[v].append(k)
    pickle.dump(children_dict, open('{}/Children_dict.p'.format(results_dir), 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)

    #"https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html"
    #Highlight:Build the matrix representation of the tree graph: Directed (root-> leaves) weighted (patristic dist) graph
    if build_config.build_graph: #TODO: Fix in server
        print("Building a tree-graph")
        #make graph adjacent matrix
        patristic_matrix = patristic_matrix[patristic_matrix[:, 0].argsort()]
        patristic_matrix = patristic_matrix[:, patristic_matrix[0, :].argsort()]

        graph_node_matrix = np.zeros_like(patristic_matrix) #Directed (root-> leaves) NOT weighted graph
        edge_weight_matrix = np.zeros_like(patristic_matrix) #Directed (root-> leaves) weighted (patristic dist) graph
        graph_node_matrix[0,:] = patristic_matrix[0,:]
        graph_node_matrix[:,0] = patristic_matrix[:,0]
        edge_weight_matrix[0, :] = patristic_matrix[0, :]
        edge_weight_matrix[:, 0] = patristic_matrix[:, 0]
        for ancestor,children in children_dict.items():
            ancestor_idx = np.in1d(graph_node_matrix[0,:], ancestor).nonzero()[0]
            children_idx= np.in1d(graph_node_matrix[0,:], children).nonzero()[0]
            graph_node_matrix[ancestor_idx,children_idx] = 1 #contains no edges weights
            graph_node_matrix[children_idx,ancestor_idx] = 1
            edge_weight_matrix[ancestor_idx, children_idx] = patristic_matrix[ancestor_idx, children_idx]
            #edge_weight_matrix[children_idx, ancestor_idx] = patristic_matrix[children_idx, ancestor_idx] #Highlight: Not use to avoid creating a self looped graph or bidirectional graphs
        #Create a coordinates matrix, basically points out which regions from the matrix do not have 0 values and therefore are connected
        weights_coo = coo_matrix(edge_weight_matrix[1:,1:])
        try:
            #The graph connectivity (edge index) should be confined with the COO format,
            # i.e. the first list contains the index of the source nodes, while the index of target nodes is specified in the second list.
            #Note that the order of the edge index is irrelevant to the Data object you create since such information is only for computing the adjacency matrix.
            from torch_geometric.utils.convert import from_scipy_sparse_matrix
            edge_index,edge_weight = from_scipy_sparse_matrix(weights_coo)
            # dgl_graph = dgl.DGLGraph() #graph for TreeLSTM
            # dgl_graph = dgl.DGLHeteroGraph()
            # dgl_graph.add_nodes(patristic_matrix[1:, 1:].shape[0])
            # dgl_graph.add_edges(edge_index[0].cuda(), edge_index[1].cuda())
            # dgl_graph.edata['y'] = edge_weight.cuda()
            if args.use_cuda:
                graph_coo = (edge_index.cuda(), edge_weight.cuda())  # graph for pytorch geometric for GNN
                dgl_graph = dgl.convert.from_scipy(weights_coo).to("cuda")
                dgl_graph.edata['y'] = edge_weight.to("cuda")
            else:
                graph_coo = (edge_index.cpu(), edge_weight.cpu())  # graph for pytorch geometric for GNN
                dgl_graph = dgl.convert.from_scipy(weights_coo).to("cpu")
                dgl_graph.edata['y'] = edge_weight.cpu()
            print("tree-graph finished")

        except:
            graph_coo = None
            dgl_graph = None

    else:
        graph_coo=None
        edge_weight_matrix=None
        dgl_graph = None
    blosum_array,blosum_dict = create_blosum(build_config.aa_probs,args.subs_matrix)

    #dgl_graph.ndata['x'] = torch.zeros((3, 5)) #to add later in the model it will be the latent space that gets transformed to logits
    #G.nodes[[0, 2]].data['x'] = th.ones((2, 5))

    nodes_representations_array = tree_positional_embeddings(ancestor_info_dict,tree_by_levels_dict)
    additional_info = AdditionalInfo(blosum=torch.from_numpy(blosum_array),
                                     blosum_dict=blosum_dict,
                                     children_dict = children_dict,
                                     ancestor_info_dict=ancestor_info_dict,
                                     tree_by_levels_dict=tree_by_levels_dict,
                                     tree_by_levels_array=tree_by_levels_array,
                                     patristic_info=patristic_info,
                                     graph_coo=graph_coo,
                                     patristic_full_sparse = edge_weight_matrix,
                                     nodes_representations_array=torch.from_numpy(nodes_representations_array),
                                     dgl_graph=dgl_graph)


    return additional_info
def create_tree_by_levels(children_dict):
    """Alternative method to build the tree by levels as suggested by Robert"""
    tree_levels = [[0]]
    current_nodes = [0]
    while len(current_nodes) > 0:
        children_nodes = []
        for node in current_nodes:
            if node in children_dict.keys():
                children_nodes += children_dict[node]

        tree_levels.append(children_nodes)
        current_nodes = children_nodes

    return tree_levels
class MyDataset(Dataset):#TODO: remove
    def __init__(self,labels,data_arrays):
        self.labels = labels
        self.data_arrays = data_arrays

    def __getitem__(self, index): #sets a[i]
        label = self.labels[index]
        data = self.data_arrays[index]
        return {'family_name': label, 'family_data': data}
    def __len__(self):
        return len(self.labels)
def define_batch_size(n,batch_size=True, benchmarking=False):
    """Automatic calculation the available divisors of the number of training data points (n).
    This helps to suggest an appropiate integer batch size number, that splits evenly the data
    :param int n: number of sequences
    :param bool batch_size
    :param benchmarking"""
    if not benchmarking:
        assert n >= 100 ,"Not worth batching, number of sequences is < 100 "
    divisors = DraupnirModelUtils.print_divisors(n)
    n_digits = len(list(str(n))) # 100 has 3 digits, 1000 has 4 digits, 10000 has 5 digits and so on
    n_chunks = [min(divisors[n_digits-1:]) if len(divisors) > n_digits-1 else 1][0]  # smallest divisor that is not 1, or just use 1 , when no other divisors are available (prime numbers)
    batchsize = int(DraupnirModelUtils.intervals(n_chunks, n)[0][1])
    if batch_size:return batchsize
    else:return n_chunks
def covariance(x,y):
    """Computes cross-covariance between vectors, and is defined by cov[X,Y]=E[(X−μX)(Y−μY)T]"""
    A = torch.sqrt(torch.arange(12).reshape(3, 4))  # some 3 by 4 array
    b = torch.tensor([[2], [4], [5]])  # some 3 by 1 vector
    cov = torch.dot(b.T - b.mean(), A - A.mean(dim=0)) / (b.shape[0] - 1)
    return cov
def find_nan(array,nan=True):
    """Drops rows in a tensor containing any nan values
    :param tensor array"""
    shape = array.shape
    tensor_reshaped = array.reshape(shape[0], -1)
    # Drop all rows containing any nan:
    if not nan:
        tensor_reshaped = tensor_reshaped[~torch.any(tensor_reshaped.isnan(), dim=1)]
    else:
        tensor_reshaped = tensor_reshaped[torch.any(tensor_reshaped.isnan(), dim=1)]
    # Reshape back:
    array = tensor_reshaped.reshape(tensor_reshaped.shape[0], *shape[1:])
    return array
def ramachandran_plot( data_angles,save_directory,plot_title, one_hot_encoded = False ):
    """Plots Ramachandran plots between the phi and psi angles in the dataset
    :param numpy-array data_angles : 3d array [n_seq, align_len, 30], where columns 1 and 2 or 21 and 22 contain the angles in the case of NOT one-hot encoded and one-hot encoded, respectively
    :param str save_directory: path to saving directory
    :param str plot_title
    :param bool one_hot_encoded
    """
    plt.clf()
    if one_hot_encoded:
        phi = data_angles[:,:,21].reshape((data_angles.shape[0]*data_angles.shape[1])).astype(float) #- np.pi

        psi = data_angles[:,:,22].reshape((data_angles.shape[0]*data_angles.shape[1])).astype(float) #- np.pi
    else:
        phi = data_angles[:,:,1].reshape((data_angles.shape[0]*data_angles.shape[1])).astype(float)
        psi = data_angles[:,:,2].reshape((data_angles.shape[0]*data_angles.shape[1])).astype(float)

    fig = plt.figure(figsize=(5,5))
    plt.hist2d( phi, psi, bins = 628, norm = LogNorm())#, cmap = plt.cm.jet )
    plt.ylim(-np.pi, np.pi)
    plt.xlim(-np.pi, np.pi)
    plt.title(plot_title,fontsize=12)
    plt.xlabel('φ')
    plt.ylabel('ψ')
    plt.savefig(save_directory)
    plt.clf()
    plt.close(fig)
def ramachandran_plot_sampled(phi,psi,save_directory,plot_title,plot_kappas=False):
    """Plots Ramachandran plots between the phi and psi angles in the dataset
    :param tensor phi
    :param tensor psi
    :param str save_directory: path to saving directory
    :param str plot_title
    :param bool plot_kappas
    """
    if isinstance(phi,torch.Tensor):
        phi = phi.view(-1).cpu().detach().numpy()
        psi = psi.view(-1).cpu().detach().numpy()
    #axes = [[-np.pi, np.pi], [-np.pi, np.pi]]
    plt.figure(figsize=(5,5))
    plt.hist2d( phi, psi, bins = 628, norm = LogNorm())#, cmap = plt.cm.jet )
    if plot_kappas:
        plt.xlabel('Kappa φ')
        plt.ylabel('Kappa ψ')
    else:
        plt.ylim(-np.pi, np.pi)
        plt.xlim(-np.pi, np.pi)
        plt.xlabel('φ')
        plt.ylabel('ψ')
    plt.title(plot_title,fontsize=12)
    plt.savefig(save_directory)
    plt.clf()
    plt.close()
def gradients_plot(gradient_norms,epochs,directory):
    """Visualization of the gradient descent per model parameter
    :param dict gradient_norms: dictionary with the model parameters and the partial derivatives
    :param int epochs"""

    fig = plt.figure(figsize=(16, 9), dpi=100).set_facecolor('white')
    ax = plt.subplot(121)
    color_map = cm.rainbow(np.linspace(0, 1, 42))
    for (name_i, grad_norms), color in zip(gradient_norms.items(), color_map):
        ax.plot(grad_norms, label=name_i, c=color)
    plt.xlabel('iters')
    plt.ylabel('gradient norm')
    plt.yscale('log')
    ax.legend(loc='upper center', bbox_to_anchor=(1.5, 1), shadow=True, ncol=1, prop={'size': 8})
    plt.title('Gradient norms during SVI')
    plt.savefig("{}/Gradients_{}_epochs.png".format(directory,epochs))
    plt.clf()
    plt.close()

def renaming(tree):
        """Rename the internal nodes, unless the given newick file already has the names on it
        :param tree: ete3 format 1 tree"""
        #Rename the internal nodes
        leafs_names = tree.get_leaf_names()
        edge = len(leafs_names)
        internal_nodes_names = []
        for node in tree.traverse(): #levelorder (nodes are visited in zig zag order from root to leaves)
            if not node.is_leaf():
                node.name = "A%d" % edge
                internal_nodes_names.append(node.name)
                edge += 1

def calculate_aa_frequencies(dataset,freq_bins):
    """Calculates a frequency for each of the aa & gap at each position.The number of bins (of size 1) is one larger than the largest value in x. This is done for numpy arrays
    :param tensor dataset
    :param int freq_bins
    """
    freqs = np.apply_along_axis(lambda x: np.bincount(x, minlength=freq_bins), axis=0, arr=dataset.astype("int64")).T
    freqs = freqs/dataset.shape[0]
    return freqs
def calculate_aa_frequencies_torch(dataset,freq_bins):
    """Calculates a frequency for each of the aa & gap at each position.The number of bins (of size 1) is one larger than the largest value in x. This is done for torch tensors
    :param tensor dataset
    :param int freq_bins
    """
    freqs = torch.stack([torch.bincount(x_i, minlength=freq_bins) for i, x_i in enumerate(torch.unbind(dataset.type(torch.int64), dim=1), 0)], dim=1)
    freqs = freqs.T
    freqs = freqs / dataset.shape[0]
    return freqs
def compare_trees(t1,t2):
    """Computes the Rf distance among two tree, it calculates the number of changes requires to transform one tree into the other
    :param str t1: path to first netwick tree
    :param str t2: path to second netwick tree"""
    columns = ["RF distance","Maximum RF distance","Common leaves","PartitionsIn_t1_NOT_t2","PartitionsIn_t2_NOT_t1","Discarded_partitions_t1","Discarded_partitions_t2"]
    # t1  = TreeEte3(t1)
    # t2 = TreeEte3(t2)
    # results = t1.robinson_foulds(t2,unrooted_trees=True)
    command = 'Rscript'
    path2script = '/home/lys/Dropbox/PhD/DRAUPNIR/Tree_Distance.R'
    # Build subprocess command
    args =[t1,t2]
    cmd = [command, path2script] + args
    distance = subprocess.check_output(cmd, universal_newlines=True)
    return distance

def remove_stop_codons(sequence_file,is_prot=False):
    """Remove the stop codons(*) from an alignment file"""
    if is_prot:
        print("Removing stop codons from protein")
        print(sequence_file)
        seq_file = open(sequence_file, 'r+')
        seq = seq_file.read()
        seq_file.seek(0)
        no_codons = [i if i!="*" else "-" for i in seq] #replace for a gap
        #TODO: Truncate the sequence if there is a stop codon in the middle (stop adding gaps when \n is reached)
        # no_codons= []
        # for index,i in enumerate(seq):
        #     if i == "*":
        #         i= "-"
        #         next_break_index = seq[index:].index("\n")
        #         seq[index:next_break_index] = "-"*(next_break_index-index)

        seq_file.write("".join(no_codons))
        seq_file.truncate()  # remove contents
    else:
        stop_codons = ["TGA","TAG","TAA"]
        seq_file = open(sequence_file, 'r+')
        seq = seq_file.read()
        seq_file.seek(0)
        #coding_dna = Seq(seq)
        #protein = coding_dna.translate()
        codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
        codons =[cod for cod in codons if len(cod) ==3 and cod not in stop_codons]
        #check = any(item in codons for item in stop_codons)
        seq_file.write("".join(codons))
        seq_file.truncate()  # remove contents

def convert_to_integers(dataset,aa_probs,axis):
    """Transforms one hot encoded amino acids into 0-indexed integers i.e [1,0,0,0] --> 0  [0,1,0,0] -> 1
    :param numpy dataset
    :param aa_probs: amino acid probabilities, dimensions of the one-hot encoding"""
    if axis==3:#use for predictions
        b = torch.argmax(dataset, dim=axis)
    else:
        integers = torch.argmax(dataset[:,2:,0:aa_probs],dim=axis)
        b = torch.zeros(dataset[:,:,0].shape + (30,)).cpu()
        #b = np.zeros(Dataset[:,:,0].shape + (30,))
        b[:, 2:, 0] = integers
        b[:,:2] = dataset[:,:2]
    return b
def process_blosum(blosum,aa_freqs,align_seq_len,aa_probs):
    """
    Computes the matrices required to build a blosum embedding
    :param tensor blosum: BLOSUM likelihood  scores
    :param tensor aa_freqs : amino acid frequencies per position
    :param align_seq_len: alignment length
    :param aa_probs: amino acid probabilities, types of amino acids in the alignment
    :out tensor blosum_max [align_len,aa_prob]: blosum likelihood scores for the most frequent aa in the alignment position
    :out tensor blosum_weighted [align_len,aa_prob: weighted average of blosum likelihoods according to the aa frequency
    :out variable_core: [align_len] : counts the number of different elements (amino acid diversity) per alignment position"""

    aa_freqs_max = torch.argmax(aa_freqs, dim=1).repeat(aa_probs, 1).permute(1, 0) #[max_len, aa_probs]
    blosum_expanded = blosum[1:, 1:].repeat(align_seq_len, 1, 1)  # [max_len,aa_probs,aa_probs]
    blosum_max = blosum_expanded.gather(1, aa_freqs_max.unsqueeze(1)).squeeze(1)  # [align_seq_len,21] Seems correct

    blosum_weighted = aa_freqs[:,:,None]*blosum_expanded #--> replace 0 with nans? otherwise the 0 are in the mean as well....
    blosum_weighted = blosum_weighted.mean(dim=1)

    variable_score = torch.count_nonzero(aa_freqs, dim=1)/aa_probs #higher score, more variable

    return blosum_max,blosum_weighted, variable_score
def blosum_embedding_encoder(blosum,aa_freqs,align_seq_len,aa_probs,dataset_train, one_hot_encoding):
    """Transforms the train dataset, which is formed by a tensor of integers that represent the amino acids, into a dataset where each amino acid (integer) is exchanged with its blosum score
    :out tensor aa_train_blosum : Training dataset with the blosum vectors instead of the amino acids (numbers or one hot representation)"""
    if one_hot_encoding:
        dataset_train = convert_to_integers(dataset_train,aa_probs,axis=2)
    aminoacids_seqs = dataset_train[:,2:,0].repeat(aa_probs,1,1).permute(1,2,0) #[N,max_len,aa repeated aa_probs times]--seems correct
    blosum_expanded = blosum[1:, 1:].repeat(dataset_train.shape[0],align_seq_len, 1, 1)  # [N,max_len,aa_probs,aa_probs]
    aa_train_blosum = blosum_expanded.gather(3, aminoacids_seqs.to(torch.int64).unsqueeze(3)).squeeze(-1)  #[N,max_len,aa_probs]


    return aa_train_blosum

def str2bool(v):
    """Converts a string into a boolean, useful for boolean arguments
    :param str v"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1','True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0','False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def str2None(v):
    """Converts a string into None
    :param str v"""

    if v.lower() in ('None','none'):
        return None
    elif isinstance(v,str):
        return v
    else:
        v = ast.literal_eval(v)
        return v






