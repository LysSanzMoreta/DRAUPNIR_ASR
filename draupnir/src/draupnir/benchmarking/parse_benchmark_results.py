"""Parser based on:
https://evosite3d.blogspot.com/2014/09/tutorial-on-ancestral-sequence.html
https://github.com/romainstuder/evosite3d
"""
import pandas.plotting
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO,Phylo
from Bio.SeqUtils import ProtParam
from ete3 import Tree
from collections import defaultdict
import pandas as pd
import numpy as np
from Bio import AlignIO
import seaborn as sns
import matplotlib.pyplot as plt
import re,sys,pickle,os, time,datetime
from collections import namedtuple
from functools import partial
sys.path.append('/home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src/draupnir/other')
sys.path.append('../../')
from prody import *
import draupnir.utils as DraupnirUtils
from PhyloBayes_Parser import parse_PhyloBayes
from FastML_Parser import parse_FastML,sample_FastML
from IQTree_Parser import parse_IQTree,sample_IQTree
import  matplotlib
matplotlib.use('TkAgg')
now = datetime.datetime.now()
np.set_printoptions(None)
PlotLoad = namedtuple("PlotLoad",["name","dataset_number","simulation_folder","root_sequence_name","rst_file","true_ancestral_file","paml_predictions_file","gap_positions",
                                  "tree_by_levels","children_dict","draupnir_incorrectly_predicted_sites","draupnir_incorrectly_predicted_sites_complete","draupnir_sequences_predictions_msa","ancestor_info","sites_count"])
PlotLoad2 = namedtuple("PlotLoad2",["fastml_dict","phylobayes_dict","iqtree_dict","correspondence_dict_sim_to_paml","correspondence_dict_paml_to_sim","correspondence_fastml_to_original",
                                    "correspondence_original_to_fastml","correspondence_iqtree_to_original","correspondence_original_to_iqtree","plot_folder_name","plot_full_name","name","sequence_input_type","test_mode","draupnir_alig_length",
                                    "plot_only_sample"])
PlotLoad3 = namedtuple("PlotLoad3",["paml_incorrectly_predicted_sites_dict","draupnir_average_incorrectly_predicted_sites","draupnir_std_incorrectly_predicted_sites",
                                    "phylobayes_incorrectly_predicted_sites_dict","fastml_incorrectly_predicted_sites_dict","iqtree_incorrectly_predicted_sites_dict","true_alignment_length","plot_name","plot_folder_name",
                                    "name","sequence_input_type","test_mode"])

def autolabel(rects, ax):
    # Get y-axis height to calculate label position from.
    (y_bottom, y_top) = ax.get_ylim()
    y_height = y_top - y_bottom

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
                ha='center', va='bottom',fontsize=8,fontweight="bold")
def Load_Tree(rst_file_name,name,simulation_folder,dataset_number,sequence_input_type):
    """parse rst file from PAML. Find the correspondence in internal nodes naming in PAML to the original ones"""
    if name.startswith("simulation"):
        #out_file_name = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_results/PAML/CodeML/{}/Dataset{}/Paml.tree".format(simulation_folder,dataset_number)
        out_file_name = os.path.join(script_dir,"CodeML/{}/Dataset{}/Paml.tree".format(simulation_folder,dataset_number))
    elif name.startswith("Coral"):
        #out_file_name = "/home/lys/Dropbox/PhD/DRAUPNIR/GPFCoralDataset/{}/c90.format6newick".format(name,name)
        out_file_name = os.path.join(script_dir_2level_up,"GPFCoralDataset/{}/c90.format6newick".format(name,name))
    elif name.endswith("_subtree"):
        out_file_name = os.path.join(script_dir_2level_up,"Douglas_SRC_Dataset/{}/{}_ALIGNED.format6newick".format(name,name.replace("_subtree","")))
    else:
        #out_file_name = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_results/PAML/CodeML/AncestralResurrectionStandardDataset/RandallBenchmarkTree_OriginalNaming.format6newick"
        out_file_name = os.path.join(script_dir,"CodeML/AncestralResurrectionStandardDataset/RandallBenchmarkTree_OriginalNaming.format6newick")
    with open(rst_file_name, 'r') as f:
        data = f.read().split('\n')
        tree_str_1 = data[data.index("tree with node labels for Rod Page's TreeView")+1] #contains new internal nodes names
        tree_1 = Tree(tree_str_1, format=1) #contains branch lengths
        if sequence_input_type == "PROTEIN":
            tree_str_2 = data[data.index("Ancestral reconstruction by AAML.") + 2] #contains original tree
        else:
            tree_str_2 = data[data.index("Ancestral reconstruction by CODONML.") + 2]  # contains original tree
        tree_2 = Tree(tree_str_2,format=1)

    #Highlight:Rename the tree according to the new names given by PAML to the internal nodes (which actually start from last leaf count +1 and so on in tree traversal order)
    correspondence_dict_sim_to_paml = defaultdict()
    correspondence_dict_paml_to_sim = defaultdict()
    if name.startswith("simulation"):
        i=1
        for node_1,node_2 in zip(tree_1.traverse(),tree_2.traverse()):
            if not node_2.is_leaf():
                node_2.name = "I{}".format(i)
                correspondence_dict_sim_to_paml["I{}".format(i)] = node_1.name #Store the correspondence between our naming in the simulation and paml's
                correspondence_dict_paml_to_sim[node_1.name] = "I{}".format(i)
                i+=1
    else:
        i = 0
        for node_1, node_2 in zip(tree_1.traverse(), tree_2.traverse()):
            if not node_2.is_leaf():
                node_2.name = str(i) #use tree level order
                correspondence_dict_sim_to_paml["I{}".format(i)] = node_1.name  # Store the correspondence between our naming in the simulation and paml's
                correspondence_dict_paml_to_sim[node_1.name] = i
            i += 1
    tree_2.write(format=1, outfile=out_file_name, format_root_node=True)  # Readable by TreeFig
    #print(tree_1.get_ascii(attributes=[ 'name']))
    return correspondence_dict_sim_to_paml,correspondence_dict_paml_to_sim
def Save_Sequences(file_name,output_file,name):
    """Saves ancestral sequences predicted by PAML and their pI
    https://www.sciencedirect.com/book/9780444636881/proteomic-profiling-and-analytical-chemistry
    The isoelectric point (pI) is the pH of a solution at which the net charge of a protein becomes zero.
    At solution pH that is above the pI, the surface of the protein is predominantly negatively charged, and therefore like-charged molecules will exhibit repulsive forces.
    Likewise, at a solution pH that is below the pI, the surface of the protein is predominantly positively charged, and repulsion between proteins occurs. However, at the pI,
    the negative and positive charges are balanced, reducing repulsive electrostatic forces, and the attraction forces predominate, causing aggregation and precipitation.
    The pI of most proteins is in the pH range of 4 to 7. """
    records= []
    file_in = open(file_name, "r")
    while 1:
        line = file_in.readline()
        if line == "":
            break
        if line[0:4] == "node":
            #tab = line.split("          ")
            #tab = line.split("         ")
            tab = re.split(r'(node\s{1}#[0-9]*\s*)', line)[1:] #skip annoying space at the beginning
            id_name = tab[0].replace(" ", "").replace("#", "")
            sequence = tab[1].replace(" ", "").strip("\n")
            #Highlight: Compute isoelectric point
            analysed_protein = ProtParam.ProteinAnalysis(sequence)
            # Compute some properties
            pI = analysed_protein.isoelectric_point()
            MW = analysed_protein.molecular_weight()
            record = SeqRecord(Seq(sequence),annotations={"molecule_type": "protein"},
                               id=id_name,
                               description="Molecular Weight: {}, Isoelectric Point: {}".format(MW,pI))
            records.append(record)

    SeqIO.write(records, output_file, "fasta")
    file_in.close()
def Extract_Sites_Probabilities(file_name,node_name):
    """
    PAML
    Returns the aa with the highest probability
    Deals with the fact that PAML does not predict gaps and when clean_data=0, to positions with gaps it just gives it the aa with the gives probability"""
    import operator

    tag = 0
    file_in = open(file_name, "r")
    proba_dict = {} #contains the aa probabilities per site of the sequence, it also stores the aa with the highest probability and whether there is a gap in the column
    while 1:
        line = file_in.readline()
        if line == "" or "Prob of best state at each node, listed by site" in line: #make it stop, otherwise it adds crap to the dict
            break
        line = line.rstrip()

        if "Prob distribution at node " + str(node_name + 1) + ", by site" in line:
            tag = 0
        if tag == 1:
            tab = line.split()
            if len(tab) > 3:
                # print(tab)
                site = tab[0]
                align_column = tab[2]
                prob = tab[3:24]
                gap = ["-"]
                contains_gap = any(i in align_column for i in gap)
                proba_dict[site] = (prob,max(set(align_column), key=align_column.count),contains_gap) #Stores the prob list per aa and the most frequent aa in that alignment column
        if "Prob distribution at node " + str(node_name) + ", by site" in line:
            tag = 1

    file_in.close()
    highest_prob_dict = {}
    for site, site_info in proba_dict.items():
        prob_aa_dict = {}
        prob_list = site_info[0]
        for aa_prob in prob_list:
            prob_aa_dict[aa_prob[0]] = float(aa_prob[2:7])
        sorted_x = sorted(prob_aa_dict.items(), key=operator.itemgetter(1), reverse=True)
        site_domain = int(site)
        highest_prob_dict[site_domain] = (sorted_x[0][0],round(sorted_x[0][1], 2),site_info[2])

    return highest_prob_dict
def Extract_Sites_Probabilities_Codons(file_name,node_name):
    """
    PAML
    Returns dictionary with keys == site numbers and the values are ( the aa with the highest probability,aa prob, contains gap bool)
    Deals with the fact that PAML does not predict gaps and when clean_data=0, to positions with gaps it just gives it the aa with the gives probability"""
    import operator
    #TODO: Check that codon table is correct, copy pasted from https://www.geeksforgeeks.org/dna-protein-python-3/
    codon_table = {
        'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
        'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
        'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
        'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
        'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
        'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
        'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
        'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
        'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
        'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
        'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
        'TAC': 'Y', 'TAT': 'Y', 'TAA': '_', 'TAG': '_',
        'TGC': 'C', 'TGT': 'C', 'TGA': '_', 'TGG': 'W',
    }
    tag = 0
    file_in = open(file_name, "r")
    highest_prob_dict = {} #contains the aa probabilities per site of the sequence (one node only), it also stores the aa with the highest probability and whether there is a gap in the column
    while 1:
        line = file_in.readline()
        if line == "" or "Prob of best state at each node, listed by site" in line: #make it stop, otherwise it adds crap to the dict
            break
        line = line.rstrip()

        if "Prob distribution at node " + str(node_name + 1) + ", by site" in line:
            tag = 0
        if tag == 1:
            tab = line.split()
            if len(tab) > 3:
                site = tab[0]
                prob_index = tab.index(":") + 1
                align_column = tab[2:prob_index-1]
                align_column = align_column[1::2]
                prob = tab[prob_index:]
                prob_dict = {codon_prob.split("(")[0]:float(codon_prob.split("(")[1].strip(")")) for codon_prob in prob} #dictionary containing codons and their prob values
                gap = ["(-)"]
                contains_gap = any(i in align_column for i in gap)
                codon_site_max_prob = max(prob_dict, key=prob_dict.get) #retrieve the codon with the highest probability
                highest_prob_dict[int(site)] = (codon_table[codon_site_max_prob],prob_dict[codon_site_max_prob],contains_gap) #Stores the prob list per aa and the most frequent aa in that alignment column
        if "Prob distribution at node " + str(node_name) + ", by site" in line:
            tag = 1

    file_in.close()

    return highest_prob_dict
def chunks(data,size):
    from itertools import islice
    it = iter(data)
    for i in range(0, len(data), size):
        yield {k:data[k] for k in islice(it, size)}
def translate(seq):
    table = {
        'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
        'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
        'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
        'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
        'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
        'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
        'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
        'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
        'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
        'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
        'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
        'TAC': 'Y', 'TAT': 'Y', 'TAA': '_', 'TAG': '_',
        'TGC': 'C', 'TGT': 'C', 'TGA': '_', 'TGG': 'W',
    }
    protein = ""
    if len(seq) % 3 == 0:
        for i in range(0, len(seq), 3):
            codon = seq[i:i + 3]
            protein += table[codon]
    return protein
def splitted_plot(plot_load3, n_plots):
    n_nodes = len(plot_load3.paml_incorrectly_predicted_sites_dict)
    # n_plots = DraupnirUtils.Define_batch_size(len(plot_load3.paml_incorrectly_predicted_sites_dict), batch_size=False,benchmarking= True)
    # n_plots = [2 if n_plots == 1 else n_plots][0] #otherwise we cannot subscript AxesSubPlots
    fig, axs = plt.subplots(n_plots, 1,figsize=(30,15))
    N = int(n_nodes/n_plots)
    ## necessary variables
    width = 0.35  # the width of the bars
    # ind = np.arange(0,N*2,2)   # the x locations for the groups index = np.arange(0, n_groups * 2, 2)
    ind = np.arange(0, N * 10 * width, 10 * width)
    blue, orange, green, red, purple = sns.color_palette("muted", 5)
    paml_incorrectly_predicted_sites_dict = [item for item in chunks(plot_load3.paml_incorrectly_predicted_sites_dict,N)]
    phylobayes_incorrectly_predicted_sites_dict = [item for item in chunks(plot_load3.phylobayes_incorrectly_predicted_sites_dict,N)]
    fastml_incorrectly_predicted_sites_dict = [item for item in chunks(plot_load3.fastml_incorrectly_predicted_sites_dict,N)]
    iqtree_incorrectly_predicted_sites_dict = [item for item in chunks(plot_load3.iqtree_incorrectly_predicted_sites_dict, N)]
    draupnir_average_incorrectly_predicted_sites = np.array_split(plot_load3.draupnir_average_incorrectly_predicted_sites, n_plots) #the other dicts are ordered with respect to this one
    draupnir_std_incorrectly_predicted_sites = np.array_split(plot_load3.draupnir_std_incorrectly_predicted_sites, n_plots)

    for n in range(n_plots):
        assert paml_incorrectly_predicted_sites_dict[n].keys() == phylobayes_incorrectly_predicted_sites_dict[n].keys() == fastml_incorrectly_predicted_sites_dict[n].keys() == iqtree_incorrectly_predicted_sites_dict[n].keys()
        assert len(paml_incorrectly_predicted_sites_dict[n].keys()) == draupnir_average_incorrectly_predicted_sites[n].shape[0]
        rects1 = axs[n].bar(ind, paml_incorrectly_predicted_sites_dict[n].values(),
                        width,
                        color=blue,
                        alpha=0.75,
                        edgecolor="black",
                        yerr=[0] * len(paml_incorrectly_predicted_sites_dict[n]),
                        error_kw=dict(elinewidth=2, ecolor='red'),
                        label="PAML-CodeML")
        rects2 = axs[n].bar(ind + 2 * width, draupnir_average_incorrectly_predicted_sites[n],
                        width,
                        color=green,
                        alpha=0.75,
                        edgecolor="black",
                        yerr=draupnir_std_incorrectly_predicted_sites[n],
                        error_kw=dict(elinewidth=1, ecolor='red'),
                        label="Draupnir")
        rects3 = axs[n].bar(ind + 4 * width, phylobayes_incorrectly_predicted_sites_dict[n].values(),
                        width,
                        color=orange,
                        alpha=0.75,
                        edgecolor="black",
                        yerr=[0] * len(phylobayes_incorrectly_predicted_sites_dict[n]),
                        error_kw=dict(elinewidth=2, ecolor='red'),
                        label="PhyloBayes")

        rects4 = axs[n].bar(ind + 6 * width, fastml_incorrectly_predicted_sites_dict[n].values(),
                        width,
                        color=red,
                        alpha=0.75,
                        edgecolor="black",
                        yerr=[0] * len(fastml_incorrectly_predicted_sites_dict[n]),
                        error_kw=dict(elinewidth=2, ecolor='red'),
                        label="FastML")
        rects5 = axs[n].bar(ind + 8 * width, iqtree_incorrectly_predicted_sites_dict[n].values(),
                            width,
                            color=purple,
                            alpha=0.75,
                            edgecolor="black",
                            yerr=[0] * len(iqtree_incorrectly_predicted_sites_dict[n]),
                            error_kw=dict(elinewidth=2, ecolor='red'),
                            label="IQtree")
        autolabel(rects1, axs[n])
        autolabel(rects2, axs[n])
        autolabel(rects3, axs[n])
        autolabel(rects4, axs[n])
        autolabel(rects5, axs[n])
        # axes and labels
        axs[n].set_xlim(-2 * width, np.max(ind) + 8 * width)
        axs[n].set_ylim(0, plot_load3.true_alignment_length + 5)
        #axs[n].set_ylabel('Number of incorrect sites')
        #axs[n].set_title('{} Incorrectly predicted aa sites (%ID)'.format(plot_load3.plot_name))
        xTickMarks = paml_incorrectly_predicted_sites_dict[n].keys()
        axs[n].set_xticks(ind + 4*width)
        xtickNames = axs[n].set_xticklabels(xTickMarks)
        axs[n].hlines(plot_load3.true_alignment_length, linestyles="dashed", xmin=0, xmax=np.max(ind), color=purple,
                   label="max number sites")
        axs[n].vlines(ind + 7 * width, linestyles="dashed", ymax=plot_load3.true_alignment_length, ymin=0, alpha=0.5)
        plt.setp(xtickNames, rotation=10, fontsize=8)
        #handles, _ = axs[n].get_legend_handles_labels()
        #axs[-1].legend((rects1[-1], rects2[-1], rects3[-1], rects4[-1]), ("PAML-CodemL", 'Draupnir', "PhyloBayes", "FastML"))
    # legend = plt.legend(loc="upper right", edgecolor="black")
    # legend.get_frame().set_alpha(None)
    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc = "upper center",bbox_to_anchor=(0.5, 0.96),bbox_transform=plt.gcf().transFigure)
    plt.ylabel("Number of incorrectly predicted sites")
    plt.xlabel("Internal nodes displayed in tree traverse order (root → leaves)")
    plt.suptitle('Incorrectly predicted aa sites; \n' + r"{}".format(plot_load3.plot_name))
    plt.savefig("{}/{}/Comparison_{}_{}_{}input_{}.png".format(plot_load3.plot_folder_name,save_folder,plot_load3.name,now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),plot_load3.sequence_input_type,plot_load3.test_mode), bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
def splitted_plot_800(plot_load3, n_plots):
    n_nodes = len(plot_load3.iqtree_incorrectly_predicted_sites_dict)
    # n_plots = DraupnirUtils.Define_batch_size(len(plot_load3.paml_incorrectly_predicted_sites_dict), batch_size=False,benchmarking= True)
    # n_plots = [2 if n_plots == 1 else n_plots][0] #otherwise we cannot subscript AxesSubPlots
    fig, axs = plt.subplots(n_plots, 1,figsize=(30,15))
    N = int(n_nodes/n_plots)
    ## necessary variables
    width = 0.35  # the width of the bars
    # ind = np.arange(0,N*2,2)   # the x locations for the groups index = np.arange(0, n_groups * 2, 2)
    ind = np.arange(0, N * 10 * width, 10 * width)
    blue, orange, green, red, purple = sns.color_palette("muted", 5)
    iqtree_incorrectly_predicted_sites_dict = [item for item in chunks(plot_load3.iqtree_incorrectly_predicted_sites_dict, N)]
    draupnir_average_incorrectly_predicted_sites = np.array_split(plot_load3.draupnir_average_incorrectly_predicted_sites, n_plots) #the other dicts are ordered with respect to this one
    draupnir_std_incorrectly_predicted_sites = np.array_split(plot_load3.draupnir_std_incorrectly_predicted_sites, n_plots)

    for n in range(n_plots):
        assert len(iqtree_incorrectly_predicted_sites_dict[n].keys()) == draupnir_average_incorrectly_predicted_sites[n].shape[0]

        rects1 = axs[n].bar(ind + 2 * width, draupnir_average_incorrectly_predicted_sites[n],
                        width,
                        color=green,
                        alpha=0.75,
                        edgecolor="black",
                        yerr=draupnir_std_incorrectly_predicted_sites[n],
                        error_kw=dict(elinewidth=1, ecolor='red'),
                        label="Draupnir")

        rects2 = axs[n].bar(ind + 8 * width, iqtree_incorrectly_predicted_sites_dict[n].values(),
                            width,
                            color=purple,
                            alpha=0.75,
                            edgecolor="black",
                            yerr=[0] * len(iqtree_incorrectly_predicted_sites_dict[n]),
                            error_kw=dict(elinewidth=2, ecolor='red'),
                            label="IQtree")
        autolabel(rects1, axs[n])
        autolabel(rects2, axs[n])
        # axes and labels
        axs[n].set_xlim(-2 * width, np.max(ind) + 8 * width)
        axs[n].set_ylim(0, plot_load3.true_alignment_length + 5)
        #axs[n].set_ylabel('Number of incorrect sites')
        #axs[n].set_title('{} Incorrectly predicted aa sites (%ID)'.format(plot_load3.plot_name))
        xTickMarks = iqtree_incorrectly_predicted_sites_dict[n].keys()
        axs[n].set_xticks(ind + 4*width)
        xtickNames = axs[n].set_xticklabels(xTickMarks)
        axs[n].hlines(plot_load3.true_alignment_length, linestyles="dashed", xmin=0, xmax=np.max(ind), color=purple,
                   label="max number sites")
        axs[n].vlines(ind + 7 * width, linestyles="dashed", ymax=plot_load3.true_alignment_length, ymin=0, alpha=0.5)
        plt.setp(xtickNames, rotation=10, fontsize=8)
        #handles, _ = axs[n].get_legend_handles_labels()
        #axs[-1].legend((rects1[-1], rects2[-1], rects3[-1], rects4[-1]), ("PAML-CodemL", 'Draupnir', "PhyloBayes", "FastML"))
    # legend = plt.legend(loc="upper right", edgecolor="black")
    # legend.get_frame().set_alpha(None)
    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc = "upper center",bbox_to_anchor=(0.5, 0.96),bbox_transform=plt.gcf().transFigure)
    plt.ylabel("Number of incorrectly predicted sites")
    plt.xlabel("Internal nodes displayed in tree traverse order (root → leaves)")
    plt.suptitle('Incorrectly predicted aa sites; \n' + r"{}".format(plot_load3.plot_name))
    plt.savefig("{}/{}/Comparison_{}_{}_{}input_{}.png".format(plot_load3.plot_folder_name,save_folder,plot_load3.name,now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),plot_load3.sequence_input_type,plot_load3.test_mode), bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
def small_plot(plot_load3):
    "Unique plot for number of nodes smaller than 50"
    fig = plt.figure(figsize=(26, 8))
    ax = fig.add_subplot(111)
    ## the data
    N = len(plot_load3.paml_incorrectly_predicted_sites_dict)
    ## necessary variables
    width = 0.35  # the width of the bars
    # ind = np.arange(0,N*2,2)   # the x locations for the groups index = np.arange(0, n_groups * 2, 2)
    ind = np.arange(0, N * 8 * width, 8 * width)
    blue, orange, green, red, purple = sns.color_palette("muted", 5)

    ## the bars
    rects1 = ax.bar(ind, plot_load3.paml_incorrectly_predicted_sites_dict.values(),
                    width,
                    color=blue,
                    alpha=0.75,
                    edgecolor="black",
                    yerr=[0] * len(plot_load3.paml_incorrectly_predicted_sites_dict),
                    error_kw=dict(elinewidth=2, ecolor='red'),
                    label="PAML-CodeML")
    rects2 = ax.bar(ind + 2 * width, plot_load3.draupnir_average_incorrectly_predicted_sites,
                    width,
                    color=green,
                    alpha=0.75,
                    edgecolor="black",
                    yerr=plot_load3.draupnir_std_incorrectly_predicted_sites,
                    error_kw=dict(elinewidth=1, ecolor='red'),
                    label="Draupnir")
    rects3 = ax.bar(ind + 4 * width, plot_load3.phylobayes_incorrectly_predicted_sites_dict.values(),
                    width,
                    color=orange,
                    alpha=0.75,
                    edgecolor="black",
                    yerr=[0] * len(plot_load3.phylobayes_incorrectly_predicted_sites_dict),
                    error_kw=dict(elinewidth=2, ecolor='red'),
                    label="PhyloBayes")

    rects4 = ax.bar(ind + 6 * width, plot_load3.fastml_incorrectly_predicted_sites_dict.values(),
                    width,
                    color=red,
                    alpha=0.75,
                    edgecolor="black",
                    yerr=[0] * len(plot_load3.fastml_incorrectly_predicted_sites_dict),
                    error_kw=dict(elinewidth=2, ecolor='red'),
                    label="FastML")
    rects5 = ax.bar(ind + 8 * width, plot_load3.iqtree_incorrectly_predicted_sites_dict.values(),
                    width,
                    color=purple,
                    alpha=0.75,
                    edgecolor="black",
                    yerr=[0] * len(plot_load3.fastml_incorrectly_predicted_sites_dict),
                    error_kw=dict(elinewidth=2, ecolor='red'),
                    label="IQTree")
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)
    autolabel(rects4, ax)
    autolabel(rects5, ax)
    # axes and labels
    ax.set_xlim(-2 * width, np.max(ind) + 8 * width)
    ax.set_ylim(0, plot_load3.true_alignment_length + 5)
    ax.set_ylabel('Number of incorrectly predicted sites')
    ax.set_title('Incorrectly predicted aa sites; \n' + r"{}".format(plot_load3.plot_name))
    xTickMarks = plot_load3.paml_incorrectly_predicted_sites_dict.keys()
    ax.set_xticks(ind + 4*width)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.xlabel("Internal nodes displayed in tree traverse order (zigzag root→ leaves)")
    plt.hlines(plot_load3.true_alignment_length, linestyles="dashed", xmin=0, xmax=np.max(ind), color=purple,
               label="max number sites")
    plt.vlines(ind + 7 * width, linestyles="dashed", ymax=plot_load3.true_alignment_length, ymin=0, alpha=0.5)
    plt.setp(xtickNames, rotation=45, fontsize=12)
    # for key, x in zip(packetsNeeded.keys(), range(len(packetsNeeded))):
    #     pplot.text(x, packetsNeeded[key], packetsNeeded[key])
    handles, _ = ax.get_legend_handles_labels()
    ## add a legend--> Not working (inside or outside loop)
    # https://stackoverflow.com/questions/20532614/multiple-lines-of-x-tick-labels-in-matplotlib
    ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]), ("PAML-CodemL", 'Draupnir', "PhyloBayes", "FastML"))
    legend = plt.legend(loc="upper right", edgecolor="black")
    legend.get_frame().set_alpha(None)
    plt.savefig("{}/{}/Comparison_{}_{}_{}_{}.png".format(plot_load3.plot_folder_name,save_folder,plot_load3.name,now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),plot_load3.sequence_input_type,plot_load3.test_mode), bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
def small_plot_800(plot_load3):
    "Unique plot for number of nodes smaller than 50"
    fig = plt.figure(figsize=(26, 8))
    ax = fig.add_subplot(111)
    ## the data
    N = len(plot_load3.paml_incorrectly_predicted_sites_dict)
    ## necessary variables
    width = 0.35  # the width of the bars
    # ind = np.arange(0,N*2,2)   # the x locations for the groups index = np.arange(0, n_groups * 2, 2)
    ind = np.arange(0, N * 8 * width, 8 * width)
    blue, orange, green, red, purple = sns.color_palette("muted", 5)

    ## the bars
    rects1 = ax.bar(ind, plot_load3.paml_incorrectly_predicted_sites_dict.values(),
                    width,
                    color=blue,
                    alpha=0.75,
                    edgecolor="black",
                    yerr=[0] * len(plot_load3.paml_incorrectly_predicted_sites_dict),
                    error_kw=dict(elinewidth=2, ecolor='red'),
                    label="PAML-CodeML")
    rects2 = ax.bar(ind + 2 * width, plot_load3.draupnir_average_incorrectly_predicted_sites,
                    width,
                    color=green,
                    alpha=0.75,
                    edgecolor="black",
                    yerr=plot_load3.draupnir_std_incorrectly_predicted_sites,
                    error_kw=dict(elinewidth=1, ecolor='red'),
                    label="Draupnir")
    rects3 = ax.bar(ind + 4 * width, plot_load3.phylobayes_incorrectly_predicted_sites_dict.values(),
                    width,
                    color=orange,
                    alpha=0.75,
                    edgecolor="black",
                    yerr=[0] * len(plot_load3.phylobayes_incorrectly_predicted_sites_dict),
                    error_kw=dict(elinewidth=2, ecolor='red'),
                    label="PhyloBayes")

    rects4 = ax.bar(ind + 6 * width, plot_load3.fastml_incorrectly_predicted_sites_dict.values(),
                    width,
                    color=red,
                    alpha=0.75,
                    edgecolor="black",
                    yerr=[0] * len(plot_load3.fastml_incorrectly_predicted_sites_dict),
                    error_kw=dict(elinewidth=2, ecolor='red'),
                    label="FastML")
    rects5 = ax.bar(ind + 8 * width, plot_load3.iqtree_incorrectly_predicted_sites_dict.values(),
                    width,
                    color=purple,
                    alpha=0.75,
                    edgecolor="black",
                    yerr=[0] * len(plot_load3.fastml_incorrectly_predicted_sites_dict),
                    error_kw=dict(elinewidth=2, ecolor='red'),
                    label="IQTree")
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)
    autolabel(rects4, ax)
    autolabel(rects5, ax)
    # axes and labels
    ax.set_xlim(-2 * width, np.max(ind) + 8 * width)
    ax.set_ylim(0, plot_load3.true_alignment_length + 5)
    ax.set_ylabel('Number of incorrectly predicted sites')
    ax.set_title('Incorrectly predicted aa sites; \n' + r"{}".format(plot_load3.plot_name))
    xTickMarks = plot_load3.paml_incorrectly_predicted_sites_dict.keys()
    ax.set_xticks(ind + 4*width)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.xlabel("Internal nodes displayed in tree traverse order (zigzag root→ leaves)")
    plt.hlines(plot_load3.true_alignment_length, linestyles="dashed", xmin=0, xmax=np.max(ind), color=purple,
               label="max number sites")
    plt.vlines(ind + 7 * width, linestyles="dashed", ymax=plot_load3.true_alignment_length, ymin=0, alpha=0.5)
    plt.setp(xtickNames, rotation=45, fontsize=12)
    # for key, x in zip(packetsNeeded.keys(), range(len(packetsNeeded))):
    #     pplot.text(x, packetsNeeded[key], packetsNeeded[key])
    handles, _ = ax.get_legend_handles_labels()
    ## add a legend--> Not working (inside or outside loop)
    # https://stackoverflow.com/questions/20532614/multiple-lines-of-x-tick-labels-in-matplotlib
    ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]), ("PAML-CodemL", 'Draupnir', "PhyloBayes", "FastML"))
    legend = plt.legend(loc="upper right", edgecolor="black")
    legend.get_frame().set_alpha(None)
    plt.savefig("{}/{}/Comparison_{}_{}_{}_{}.png".format(plot_load3.plot_folder_name,save_folder,plot_load3.name,now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),plot_load3.sequence_input_type,plot_load3.test_mode), bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
def comparison_table(plot_load3,big_data=False):
    """Builds a table that shows how many nodes were better, equal or worse predicted than the competitors"""
    print("Build Comparison table")
    if big_data:
        prob_dict = {
            "Draupnir": plot_load3.draupnir_average_incorrectly_predicted_sites.tolist(),
            "PAML_CodeML":len(plot_load3.draupnir_average_incorrectly_predicted_sites.tolist())*[np.nan],
            "PhyloBayes": len(plot_load3.draupnir_average_incorrectly_predicted_sites.tolist()) * [np.nan],
            "FastML": len(plot_load3.draupnir_average_incorrectly_predicted_sites.tolist()) * [np.nan],
            "IQTree": plot_load3.iqtree_incorrectly_predicted_sites_dict.values(),
        }
        headers = ["PAML_CodeML","Phylobayes","FastML","IQTree"]
    else:
        prob_dict = {
            "Draupnir": plot_load3.draupnir_average_incorrectly_predicted_sites.tolist(),
            "PAML_CodeML": plot_load3.paml_incorrectly_predicted_sites_dict.values(),
            "PhyloBayes": plot_load3.phylobayes_incorrectly_predicted_sites_dict.values(),
            "FastML": plot_load3.fastml_incorrectly_predicted_sites_dict.values(),
            "IQTree":plot_load3.iqtree_incorrectly_predicted_sites_dict.values(),
        }
        headers = ["PAML_CodeML","Phylobayes","FastML","IQTree"]

    incorrectly_predicted_df = pd.DataFrame.from_dict(prob_dict, orient='index', columns=plot_load3.iqtree_incorrectly_predicted_sites_dict.keys())
    comparison_df = pd.DataFrame(index=headers,columns=["Improved","Equal","Worse"])
    greater = incorrectly_predicted_df.iloc[1:].gt(incorrectly_predicted_df.iloc[0],axis = 1).sum(axis=1)
    equal = incorrectly_predicted_df.iloc[1:].eq(incorrectly_predicted_df.iloc[0],axis = 1).sum(axis=1)
    smaller = incorrectly_predicted_df.iloc[1:].lt(incorrectly_predicted_df.iloc[0],axis = 1).sum(axis=1)
    comparison_df.loc[:,"Improved"] = greater.tolist() #Highlight: Had to turn the series into list in order to avoid nan values
    comparison_df.loc[:, "Equal"] = equal.tolist()#TODO: perhaps make an interval??
    comparison_df.loc[:, "Worse"] = smaller.tolist()
    print("Performance evaluation")
    print(comparison_df.to_latex())
def percendid_table(plot_load3,big_data=False):
    """Builds a table that shows how many nodes were better, equal or worse predicted than the competitors"""
    print("Build percent id table")
    if big_data:
        prob_dict = {
            "Draupnir": plot_load3.draupnir_average_incorrectly_predicted_sites.tolist(),
            "PAML_CodeML": len(plot_load3.draupnir_average_incorrectly_predicted_sites.tolist()) * [np.nan],
            "PhyloBayes": len(plot_load3.draupnir_average_incorrectly_predicted_sites.tolist()) * [np.nan],
            "FastML": len(plot_load3.draupnir_average_incorrectly_predicted_sites.tolist()) * [np.nan],
            "IQTree": plot_load3.iqtree_incorrectly_predicted_sites_dict.values(),
        }
        headers = ["PAML_CodeML", "Phylobayes", "FastML", "IQTree"]
    else:
        prob_dict = {
            "Draupnir": plot_load3.draupnir_average_incorrectly_predicted_sites.tolist(),
            "PAML_CodeML": plot_load3.paml_incorrectly_predicted_sites_dict.values(),
            "PhyloBayes": plot_load3.phylobayes_incorrectly_predicted_sites_dict.values(),
            "FastML": plot_load3.fastml_incorrectly_predicted_sites_dict.values(),
            "IQTree": plot_load3.iqtree_incorrectly_predicted_sites_dict.values(),
        }
        headers = ["Draupnir","PAML\_CodeML", "Phylobayes", "FastML", "IQTree"]

    dataframe_path = plot_load3.plot_folder_name +"/{}".format(save_folder) + "/PercentID_comparison_{}_{}_{}.tex".format(now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),plot_load3.sequence_input_type,plot_load3.test_mode)
    percendid_df = pd.DataFrame.from_dict(prob_dict, orient='index', columns=plot_load3.iqtree_incorrectly_predicted_sites_dict.keys()) #.T

    def bold_formatter(x, value, num_decimals=2):
        """Format a number in bold when (almost) identical to a given value.
        Args:
            x: Input number.
            value: Value to compare x with.
            num_decimals: Number of decimals to use for output format.
        Returns:
            String converted output.
        """
        # Consider values equal, when rounded results are equal
        # otherwise, it may look surprising in the table where they seem identical
        if round(x, num_decimals) == round(value, num_decimals):
            return '\\textbf{' + f'{x:0.2f}' + '}'
        else:
            return f'{x:0.2f}'
    format_dict = {column: partial(bold_formatter, value=percendid_df[column].max(), num_decimals=2) for column in percendid_df.columns}
    percendid_df.to_latex(buf=dataframe_path,
                          bold_rows=True,
                          escape=False,
                          formatters= format_dict,
                          na_rep="Not available",
                          index_names=headers)

    #Highlight: weird trick to transpose the table
    percendid_df_transpose = pd.read_csv(dataframe_path,
                     sep='&',
                     header=None,
                     skiprows=4,
                     skipfooter=2,
                     engine='python',
                     index_col=0)
    percendid_df_transpose.columns = plot_load3.iqtree_incorrectly_predicted_sites_dict.keys()

    percendid_df_transpose = percendid_df_transpose.T
    percendid_df_transpose.to_latex(dataframe_path, escape=False,na_rep="Not available") #escape false to avoid probles with \
def dict_to_fasta(predictions, file_name):
    with open(file_name, "w") as handle:
        for key, val in predictions.items():
            record = SeqRecord(Seq(val), id=key,description="")
            SeqIO.write(record, handle, "fasta")
def merge_to_align(true_seq_dict,predicted_dict,plot_load2,program,root_node_name):

    z = {**true_seq_dict, **predicted_dict}
    records = []
    for key, sequence in z.items():
        record = SeqRecord(Seq(''.join(sequence).replace("-", "")),
                           annotations={"molecule_type": plot_load2.sequence_input_type},
                           id=str(key), description="")
        records.append(record)
    unaligned_sequences_file = "{}/{}/Unaligned_{}_{}_predictions".format(plot_load2.plot_folder_name, save_folder,plot_load2.name,program)
    SeqIO.write(records, unaligned_sequences_file, "fasta")
    aligned_sequences_out_file = "{}/{}/Aligned_{}_{}_predictions".format(plot_load2.plot_folder_name,save_folder, plot_load2.name,program)
    dict_alignment, alignment = DraupnirUtils.Infer_alignment(None, unaligned_sequences_file,aligned_sequences_out_file)
    #root_node_name = ["allcor" if name == "Coral_all" else "all-fav"][0]
    true_sequences_aligned_to_predictions = defaultdict()
    aligned_predictions = defaultdict()
    for key, val in dict_alignment.items():
        if key.startswith(root_node_name):
            true_sequences_aligned_to_predictions[key] = val
        else:
            aligned_predictions[key] = val

    return true_sequences_aligned_to_predictions,aligned_predictions
def merge_to_align_simulationsDNA(true_seq_dict,predicted_dict,plot_load2,program):

    true_seq_dict = {"{}_true".format(key):val for key,val in true_seq_dict.items()}
    z = {**true_seq_dict, **predicted_dict}
    records = []
    for key, sequence in z.items():
        record = SeqRecord(Seq(''.join(sequence).replace("-", "")),
                           annotations={"molecule_type": plot_load2.sequence_input_type},
                           id=str(key), description="")
        records.append(record)
    unaligned_sequences_file = "{}/{}/Unaligned_{}_{}_predictions".format(plot_load2.plot_folder_name,save_folder, plot_load2.name,program)
    SeqIO.write(records, unaligned_sequences_file, "fasta")
    aligned_sequences_out_file = "{}/{}/Aligned_{}_{}_predictions".format(plot_load2.plot_folder_name,save_folder, plot_load2.name,program)
    dict_alignment, alignment = DraupnirUtils.infer_alignment(None, unaligned_sequences_file,aligned_sequences_out_file)
    #root_node_name = ["allcor" if name == "Coral_all" else "all-fav"][0]
    true_sequences_aligned_to_predictions = defaultdict()
    aligned_predictions = defaultdict()
    for key, val in dict_alignment.items():
        if key.endswith("_true"):
            true_sequences_aligned_to_predictions[key.strip("_true")] = val
        else:
            aligned_predictions[key] = val

    return true_sequences_aligned_to_predictions,aligned_predictions
def scatterplot_matrix(data, names, **kwargs):
    import  itertools
    """Plots a scatterplot matrix of subplots.  Each row of "data" is plotted
    against other rows, resulting in a nrows by nrows grid of subplots with the
    diagonal subplots labeled with "names".  Additional keyword arguments are
    passed on to matplotlib's "plot" command. Returns the matplotlib figure
    object containg the subplot grid."""
    numvars, numdata = data.shape
    fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(8,8))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for ax in axes.flat:
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # Set up ticks only on one side for the "edge" subplots...
        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')

    # Plot the data.
    for i, j in zip(*np.triu_indices_from(axes, k=1)):
        for x, y in [(i,j), (j,i)]:
            axes[x,y].plot(data[x], data[y], **kwargs)

    # Label the diagonal subplots...
    for i, label in enumerate(names):
        axes[i,i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                ha='center', va='center')

    # Turn on the proper x or y axes ticks.
    for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
        axes[j,i].xaxis.set_visible(True)
        axes[i,j].yaxis.set_visible(True)

    return fig
def plot_incorrectly_predicted_kinases(plot_load, plot_load2, consider_gaps_predictions=True):
    """Summary tables and histograms of the predictions of all state of the art programs
    consider_gaps_predictions: If True,it keeps untouched the prediction by the program, otherwise, if there is a gap in the leaves it substitutes it by a gap """

    print("Building comparison Incorrectly Predicted aa...")
    paml_predictions_file = plot_load.paml_predictions_file
    true_ancestral_file = plot_load.true_ancestral_file
    ancestor_info = plot_load.ancestor_info
    draupnir_incorrectly_predicted_sites = plot_load.draupnir_incorrectly_predicted_sites
    correspondence_dict_paml_to_sim = plot_load2.correspondence_dict_paml_to_sim
    rst_file = plot_load.rst_file
    correspondence_dict_sim_to_paml = plot_load2.correspondence_dict_sim_to_paml
    phylobayes_dict = plot_load2.phylobayes_dict
    fastml_dict = plot_load2.fastml_dict
    gap_positions = plot_load.gap_positions
    name = plot_load.name
    correspondence_fastml_to_original = plot_load2.correspondence_fastml_to_original
    correspondence_original_to_fastml = plot_load2.correspondence_original_to_fastml
    gaps = ["GAPS" if consider_gaps_predictions else "withoutGAPS"][0]
    plot_name = plot_load2.plot_full_name
    sequence_input_type = plot_load2.sequence_input_type
    test_mode = plot_load2.test_mode

    paml_predicted_alignment = AlignIO.read(paml_predictions_file, "fasta")
    true_alignment = SeqIO.parse(true_ancestral_file, "fasta")
    nodes_id_list = []
    nodes_seq_list = []
    for seq in true_alignment: #Highlight: I had to do the loop outside because for some strange reason (prob some memory pointers) using list comprehensions retunned empty
        nodes_id_list.append(seq.id)
        nodes_seq_list.append(seq.seq)
    true_alignment_dict = dict(zip(nodes_id_list,nodes_seq_list))

    # Highlight: order predicted alignment nodes according to draupnir results
    ancestor_info["0"] = ancestor_info["0"].astype(str)
    ancestor_info.drop('Unnamed: 0', inplace=True, axis=1)
    nodes_names = ancestor_info["0"].tolist()
    root_name = name.replace("_subtree", "").replace("_", "-")
    #tree_file = "/home/lys/Dropbox/PhD/DRAUPNIR/Douglas_SRC_Dataset/{}_subtree/{}_ALIGNED.mafft.treefile".format(name.replace("_subtree", ""),name.replace("_subtree", ""))
    tree_file = os.path.join(script_dir_2level_up,"Douglas_SRC_Dataset/{}_subtree/{}_ALIGNED.mafft.treefile".format(name.replace("_subtree", ""),name.replace("_subtree", "")))
    tree = Tree(tree_file, format=1)  # ,quoted_node_names=True)
    DraupnirUtils.Renaming(tree)
    root_number = [node.name for node in tree.traverse() if node.is_root()][0]
    true_alignment_dict = dict(zip([root_name],[true_alignment_dict[root_name]]))


    draupnir_nodes_treelevel_order = [0] # is always the root
    # Highlight: PAML CodeML ##########################################################################################################
    paml_predicted_alignment_dict = dict(zip([correspondence_dict_paml_to_sim[seq.id.replace("node", "")] for seq in paml_predicted_alignment],[seq.seq for seq in paml_predicted_alignment]))
    paml_predicted_alignment_dict_ordered = dict(zip(draupnir_nodes_treelevel_order,
                                                     [val for key, val in paml_predicted_alignment_dict.items() if
                                                      key in draupnir_nodes_treelevel_order]))  # until here seems correct

    # if not consider_gaps_predictions:  # if there is any gap in the paml prediction, ignore that site , replace with gap
    #     def build_site():
    #         if highest_prob_dict[idx][2]:  # if there is any gap in the paml prediction, ignore that site , replace with gap
    #             site = "-"
    #         else:
    #             site = highest_prob_dict[idx][0]
    #         return site
    # else:
    #     def build_site():
    #         site = highest_prob_dict[idx][0]
    #         return site

    if sequence_input_type == "DNA":
        def esp(rst, index):
            return Extract_Sites_Probabilities_Codons(rst, index)
    else:
        def esp(rst, index):
            return Extract_Sites_Probabilities(rst, index)
    #Highlight : First, the root needs to be aligned to its prediction in order to be comparable
    if sequence_input_type == "DNA":
        paml_predicted_alignment_dict_ordered = {key:translate(seq) for key,seq in paml_predicted_alignment_dict_ordered.items()}

    true_sequences_aligned_to_paml_predictions,paml_aligned_predictions = merge_to_align(true_alignment_dict,paml_predicted_alignment_dict_ordered,plot_load2,"PAML",root_name)
    paml_incorrectly_predicted_sites_dict = defaultdict()
    paml_percent_id_dict = defaultdict()

    for equivalent_true_node_number, pred_seq in paml_aligned_predictions.items():
        # highest_prob_dict = esp(rst_file, int(correspondence_dict_sim_to_paml["I" + str(equivalent_true_node_number)]))
        # fixed_predicted_seq = []
        # for idx, site in enumerate(pred_seq, 1):
        #     site = build_site()
        #     fixed_predicted_seq.append(site)
        # fixed_predicted_seq = "".join(fixed_predicted_seq)
        fixed_predicted_seq = pred_seq
        for true_name, true_node_seq in true_sequences_aligned_to_paml_predictions.items():
            paml_incorrectly_predicted_sites_dict[true_name] = DraupnirUtils.incorrectly_predicted_aa(fixed_predicted_seq, true_node_seq)
            paml_percent_id_dict[true_name] = DraupnirUtils.perc_identity_pair_seq(fixed_predicted_seq,true_node_seq)

    # Highlight: FastML###########################################################################

    fastml_dict_ordered = {0:fastml_dict[correspondence_original_to_fastml[root_number.replace("A","I")]]} #pick only the root

    # Highlight : First, the root needs to be aligned to its prediction in order to be comparable
    true_sequences_aligned_to_fastml_predictions,fastml_aligned_predictions = merge_to_align(true_alignment_dict,fastml_dict_ordered,plot_load2,"FastML",root_name)

    if not consider_gaps_predictions:
        def build_seq(in_seq):
            " Correct prediction, re-assign gaps, sicen fastml does not handle gaps"
            out_seq = "".join(["-" if idx in gap_positions else letter for idx, letter in enumerate(in_seq)])  # correct prediction, re-assign gaps, sicen fastml does not handle gaps (they apply a simple heuristic)
            return out_seq
    else:
        def build_seq(in_seq):
            out_seq = "".join([letter for idx, letter in enumerate(in_seq)])
            return out_seq

    fastml_incorrectly_predicted_sites_dict = defaultdict()
    fastml_percent_id_dict = defaultdict()
    for internal_node, fastml_seq in fastml_aligned_predictions.items():
        fastml_seq = build_seq(fastml_seq).strip("*") #TODO: .replace("*","-")
        for true_seq_name, true_seq in true_sequences_aligned_to_fastml_predictions.items():
            fastml_percent_id_dict[true_seq_name] = DraupnirUtils.perc_identity_pair_seq(fastml_seq,true_seq)
            fastml_incorrectly_predicted_sites_dict[true_seq_name] = DraupnirUtils.incorrectly_predicted_aa(fastml_seq,true_seq)

    # Highlight: PhyloBayes############################################################################
    # phylobayes_dict_ordered = defaultdict()
    # for node in draupnir_nodes_treelevel_order:
    #     node_name = draupnir_correspondence_dict_reversed[node]
    #     phylobayes_dict_ordered[node_name] = phylobayes_dict[node_name.replace("I", "A")]

    phylobayes_dict_ordered = {0:phylobayes_dict[root_number]}
    # Highlight : First, the root needs to be aligned to its prediction in order to be comparable
    true_sequences_aligned_to_phylobayes_predictions, phylobayes_aligned_predictions = merge_to_align(true_alignment_dict,
                                                                                              phylobayes_dict_ordered,
                                                                                              plot_load2, "PhyloBayes",root_name)

    phylobayes_incorrectly_predicted_sites_dict = defaultdict()
    phylobayes_percent_id_dict = defaultdict()
    for internal_node, phylo_bayes_seq in phylobayes_aligned_predictions.items():
        #equivalent_true_node_sequence = true_alignment_dict[internal_node]
        #phylo_bayes_seq = build_seq(phylo_bayes_seq)#.strip("*")
        for true_seq_name, true_seq in true_sequences_aligned_to_phylobayes_predictions.items():
            phylobayes_percent_id_dict[true_seq_name] = DraupnirUtils.perc_identity_pair_seq(phylo_bayes_seq,true_seq)
            phylobayes_incorrectly_predicted_sites_dict[true_seq_name] = DraupnirUtils.incorrectly_predicted_aa(phylo_bayes_seq, true_seq)

    # # PhyloBayes is missing some nodes (those internal not calculated from distance among leaves), we assign them to not predicted
    # for key in paml_incorrectly_predicted_sites_dict.keys():
    #     if key not in phylobayes_incorrectly_predicted_sites_dict.keys():
    #         if sequence_input_type == "DNA":
    #             phylobayes_incorrectly_predicted_sites_dict[key] = int(
    #                 len(paml_predicted_alignment_dict_ordered[key]) // 3)
    #         else:
    #             phylobayes_incorrectly_predicted_sites_dict[key] = len(paml_predicted_alignment_dict_ordered[key])
    # # Order again:
    # phylobayes_incorrectly_predicted_sites_dict = {
    #     "I" + str(int(key)): phylobayes_incorrectly_predicted_sites_dict["I" + str(int(key))] for key in
    #     draupnir_nodes_order if
    #     "I" + str(int(key)) in phylobayes_incorrectly_predicted_sites_dict.keys()}
    # Highlight: Order Iqtree #TODO: review
    iqtree_dict_ordered = {0: plot_load2.iqtree_dict[root_number]}

    true_sequences_aligned_to_iqtree_predictions, iqtree_aligned_predictions = merge_to_align(true_alignment_dict,
                                                                                              iqtree_dict_ordered,
                                                                                              plot_load2, "IQTree",root_name)
    iqtree_incorrectly_predicted_sites_dict = defaultdict()
    iqtree_percent_id_dict = defaultdict()
    for internal_node, iqtree_seq in iqtree_aligned_predictions.items():
        for true_seq_name, true_seq in true_sequences_aligned_to_phylobayes_predictions.items():
            iqtree_percent_id_dict[true_seq_name] = DraupnirUtils.perc_identity_pair_seq(iqtree_seq, true_seq)
            iqtree_incorrectly_predicted_sites_dict[true_seq_name] = DraupnirUtils.incorrectly_predicted_aa(iqtree_seq, true_seq)
    # Highlight: Draupnir ##############################################################################################################
    if test_mode == "Test_argmax":
        draupnir_average_incorrectly_predicted_sites = draupnir_incorrectly_predicted_sites[:,0].tolist()
        draupnir_std_incorrectly_predicted_sites = np.zeros(draupnir_incorrectly_predicted_sites.shape[0]).tolist()
    else:
        draupnir_average_incorrectly_predicted_sites = np.mean(draupnir_incorrectly_predicted_sites, axis=1).tolist()
        draupnir_std_incorrectly_predicted_sites = np.std(draupnir_incorrectly_predicted_sites, axis=1).tolist()


    plot_load_incorrectly_predicted_sites = PlotLoad3(paml_incorrectly_predicted_sites_dict=paml_incorrectly_predicted_sites_dict,
                           draupnir_average_incorrectly_predicted_sites=draupnir_average_incorrectly_predicted_sites,
                           draupnir_std_incorrectly_predicted_sites=draupnir_std_incorrectly_predicted_sites,
                           phylobayes_incorrectly_predicted_sites_dict=phylobayes_incorrectly_predicted_sites_dict,
                           fastml_incorrectly_predicted_sites_dict=fastml_incorrectly_predicted_sites_dict,
                           iqtree_incorrectly_predicted_sites_dict=iqtree_incorrectly_predicted_sites_dict,
                           true_alignment_length=len(true_alignment_dict[root_name]),
                           plot_name=plot_name,
                           plot_folder_name=plot_load2.plot_folder_name,
                           name = plot_load2.name,
                           sequence_input_type=plot_load2.sequence_input_type,
                           test_mode=plot_load2.test_mode)
    draupnir_average_percent_id = (plot_load2.draupnir_alig_length - np.array(draupnir_average_incorrectly_predicted_sites)) * 100 / plot_load2.draupnir_alig_length
    draupnir_std_percent_id = draupnir_std_incorrectly_predicted_sites
    plot_load_percentid = PlotLoad3(paml_incorrectly_predicted_sites_dict=paml_percent_id_dict,
                                    draupnir_average_incorrectly_predicted_sites=draupnir_average_percent_id,
                                    draupnir_std_incorrectly_predicted_sites=draupnir_std_percent_id,
                                    phylobayes_incorrectly_predicted_sites_dict=phylobayes_percent_id_dict,
                                    fastml_incorrectly_predicted_sites_dict=fastml_percent_id_dict,
                                    iqtree_incorrectly_predicted_sites_dict=iqtree_percent_id_dict,
                                    true_alignment_length=100,
                                    plot_name=plot_name,
                                    plot_folder_name=plot_load2.plot_folder_name,
                                    name=plot_load2.name + "%ID",
                                    sequence_input_type=plot_load2.sequence_input_type,
                                    test_mode=plot_load2.test_mode)

    small_plot(plot_load_incorrectly_predicted_sites)
    percendid_table(plot_load_percentid)
def plot_incorrectly_predicted_coral(plot_load, plot_load2, consider_gaps_predictions=True):
    "Bar plot that conciles all programs we benchmark against"

    print("Building comparison Incorrectly Predicted aa...")
    paml_predictions_file = plot_load.paml_predictions_file
    true_ancestral_file = plot_load.true_ancestral_file
    ancestor_info = plot_load.ancestor_info
    draupnir_incorrectly_predicted_sites = plot_load.draupnir_incorrectly_predicted_sites
    correspondence_dict_paml_to_sim = plot_load2.correspondence_dict_paml_to_sim
    rst_file = plot_load.rst_file
    correspondence_dict_sim_to_paml = plot_load2.correspondence_dict_sim_to_paml
    phylobayes_dict = plot_load2.phylobayes_dict
    fastml_dict = plot_load2.fastml_dict
    gap_positions = plot_load.gap_positions
    name = plot_load.name
    correspondence_fastml_to_original = plot_load2.correspondence_fastml_to_original
    correspondence_original_to_fastml = plot_load2.correspondence_original_to_fastml
    gaps = ["GAPS" if consider_gaps_predictions else "withoutGAPS"][0]
    sequence_input_type = plot_load2.sequence_input_type
    test_mode = plot_load2.test_mode

    plot_name = plot_load2.plot_full_name

    paml_predicted_alignment = AlignIO.read(paml_predictions_file, "fasta")
    true_alignment = AlignIO.read(true_ancestral_file, "fasta")

    # Highlight: order predicted alignment nodes according to draupnir results
    ancestor_info["0"] = ancestor_info["0"].astype(str)
    ancestor_info.drop('Unnamed: 0', inplace=True, axis=1)
    nodes_names = ancestor_info["0"].tolist()
    true_alignment_dict = dict(zip([seq.id for seq in true_alignment], [seq.seq for seq in true_alignment]))
    if name == "Coral_Faviina":
        root = "A35"
        ancestral_names = {"all-fav0":root,"all-fav1":root,"all-fav2":root,"all-fav3":root,"all-fav4":root}
    else:
        root = "A71"
        ancestral_names = {"allcor0": root, "allcor1": root, "allcor2": root, "allcor3": root, "allcor4": root}

    true_alignment_dict = {key:value for key,value in true_alignment_dict.items() if key in ancestral_names.keys()}

    draupnir_nodes_treelevel_order = [0] #for the coral sequences is always the root
    # Highlight: PAML CodeML ##########################################################################################################
    paml_predicted_alignment_dict = dict(zip([correspondence_dict_paml_to_sim[seq.id.replace("node", "")] for seq in paml_predicted_alignment],[seq.seq for seq in paml_predicted_alignment]))
    paml_predicted_alignment_dict_ordered = dict(zip(draupnir_nodes_treelevel_order,
                                                     [val for key, val in paml_predicted_alignment_dict.items() if
                                                      key in draupnir_nodes_treelevel_order]))  # until here seems correct

    if not consider_gaps_predictions:  # if there is any gap in the paml prediction, ignore that site , replace with gap
        def build_site():
            if highest_prob_dict[idx][2]:  # if there is any gap in the paml prediction, ignore that site , replace with gap
                site = "-"
            else:
                site = highest_prob_dict[idx][0]
            return site
    else:
        def build_site():
            site = highest_prob_dict[idx][0]
            return site

    if sequence_input_type == "DNA":
        def esp(rst, index):
            return Extract_Sites_Probabilities_Codons(rst, index)
    else:
        def esp(rst, index):
            return Extract_Sites_Probabilities(rst, index)
    #Highlight : First, the root needs to be aligned to its prediction in order to be comparable
    if sequence_input_type == "DNA":
        paml_predicted_alignment_dict_ordered = {key:translate(seq) for key,seq in paml_predicted_alignment_dict_ordered.items()}
    root_node_name = ["allcor" if name == "Coral_all" else "all-fav"][0]
    true_sequences_aligned_to_paml_predictions,paml_aligned_predictions = merge_to_align(true_alignment_dict,paml_predicted_alignment_dict_ordered,plot_load2,"PAML",root_node_name)

    paml_incorrectly_predicted_sites_dict = defaultdict()
    paml_percent_id_dict = defaultdict()
    paml_sequences_predictions_dict = defaultdict()
    for equivalent_true_node_number, pred_seq in paml_aligned_predictions.items():
        highest_prob_dict = esp(rst_file, int(correspondence_dict_sim_to_paml["I" + str(equivalent_true_node_number)]))
        fixed_predicted_seq = []
        for idx, site in enumerate(pred_seq, 1):
            site = build_site()
            fixed_predicted_seq.append(site)
        fixed_predicted_seq = "".join(fixed_predicted_seq)
        #fixed_predicted_seq = pred_seq #Highlight: Uses the raw prediction, not the most likely
        for true_name, true_node_seq in true_sequences_aligned_to_paml_predictions.items():
            paml_incorrectly_predicted_sites_dict[true_name] = DraupnirUtils.incorrectly_predicted_aa(fixed_predicted_seq, true_node_seq)
            paml_percent_id_dict[true_name] = DraupnirUtils.perc_identity_pair_seq(fixed_predicted_seq,true_node_seq)
            paml_sequences_predictions_dict[true_name] = fixed_predicted_seq
    dict_to_fasta(paml_sequences_predictions_dict,"{}/{}/PAML_predictions_{}.fasta".format(plot_load2.plot_folder_name,save_folder, name))

    for true_sample in true_alignment_dict.keys():
        dict_to_fasta({true_sample:"".join(true_alignment_dict[true_sample])}, "{}/{}/True_seq_{}_{}.fasta".format(plot_load2.plot_folder_name,save_folder, name, true_sample))

    # Highlight: FastML###########################################################################
    fastml_dict_ordered = {0:fastml_dict[correspondence_original_to_fastml[root.replace("A","I")]]}
    # Highlight : First, the root needs to be aligned to its prediction in order to be comparable
    true_sequences_aligned_to_fastml_predictions,fastml_aligned_predictions = merge_to_align(true_alignment_dict,fastml_dict_ordered,plot_load2,"FastML",root_node_name)

    if not consider_gaps_predictions:
        def build_seq(in_seq):
            " Correct prediction, re-assign gaps, sicen fastml does not handle gaps"
            out_seq = "".join(["-" if idx in gap_positions else letter for idx, letter in enumerate(in_seq)])  # correct prediction, re-assign gaps, sicen fastml does not handle gaps (they apply a simple heuristic)
            return out_seq
    else:
        def build_seq(in_seq):
            out_seq = "".join([letter for idx, letter in enumerate(in_seq)])
            return out_seq

    fastml_incorrectly_predicted_sites_dict = defaultdict()
    fastml_percent_id_dict = defaultdict()
    fastml_sequences_predictions_dict = defaultdict()
    for internal_node, fastml_seq in fastml_aligned_predictions.items():
        fastml_seq = build_seq(fastml_seq).strip("*")
        fastml_sequences_predictions_dict[str(internal_node)] = fastml_seq
        for true_seq_name, true_seq in true_sequences_aligned_to_fastml_predictions.items():
            fastml_percent_id_dict[true_seq_name] = DraupnirUtils.perc_identity_pair_seq(fastml_seq,true_seq)
            fastml_incorrectly_predicted_sites_dict[true_seq_name] = DraupnirUtils.incorrectly_predicted_aa(fastml_seq,true_seq)


    dict_to_fasta(fastml_sequences_predictions_dict,"{}/{}/FastML_predictions_{}.fasta".format(plot_load2.plot_folder_name,save_folder, name))

    # Highlight: PhyloBayes############################################################################

    phylobayes_dict_ordered = {0:phylobayes_dict[root]}
    # Highlight : First, the root needs to be aligned to its prediction in order to be comparable
    true_sequences_aligned_to_phylobayes_predictions, phylobayes_aligned_predictions = merge_to_align(true_alignment_dict,
                                                                                              phylobayes_dict_ordered,
                                                                                              plot_load2, "PhyloBayes",root_node_name)

    phylobayes_incorrectly_predicted_sites_dict = defaultdict()
    phylobayes_percent_id_dict = defaultdict()
    phylobayes_sequences_predictions_dict = defaultdict()
    for internal_node, phylo_bayes_seq in phylobayes_aligned_predictions.items():
        #equivalent_true_node_sequence = true_alignment_dict[internal_node]
        phylo_bayes_seq = build_seq(phylo_bayes_seq).strip("*")
        phylobayes_sequences_predictions_dict[internal_node] = phylo_bayes_seq
        for true_seq_name, true_seq in true_sequences_aligned_to_phylobayes_predictions.items():
            phylobayes_percent_id_dict[true_seq_name] = DraupnirUtils.perc_identity_pair_seq(phylo_bayes_seq,true_seq)
            phylobayes_incorrectly_predicted_sites_dict[true_seq_name] = DraupnirUtils.incorrectly_predicted_aa(phylo_bayes_seq, true_seq)



    dict_to_fasta(phylobayes_sequences_predictions_dict,"{}/{}/PhyloBayes_predictions_{}.fasta".format(plot_load2.plot_folder_name, save_folder,name))


    # Highlight: Order Iqtree #TODO: review
    iqtree_dict_ordered = {0: plot_load2.iqtree_dict[root]}
    true_sequences_aligned_to_iqtree_predictions, iqtree_aligned_predictions = merge_to_align(true_alignment_dict,
                                                                                              iqtree_dict_ordered,
                                                                                              plot_load2, "IQTree",root_node_name)
    iqtree_incorrectly_predicted_sites_dict = defaultdict()
    iqtree_percent_id_dict = defaultdict()
    iqtree_sequences_predictions_dict = defaultdict()
    for internal_node, iqtree_seq in iqtree_aligned_predictions.items():
        iqtree_sequences_predictions_dict[internal_node] = "".join(iqtree_seq)
        for true_seq_name, true_seq in true_sequences_aligned_to_iqtree_predictions.items():
            iqtree_percent_id_dict[true_seq_name] = DraupnirUtils.perc_identity_pair_seq(iqtree_seq, true_seq)
            iqtree_incorrectly_predicted_sites_dict[true_seq_name] = DraupnirUtils.incorrectly_predicted_aa(iqtree_seq, true_seq)
    dict_to_fasta(iqtree_sequences_predictions_dict,"{}/{}/IQTree_predictions_{}.fasta".format(plot_load2.plot_folder_name,save_folder, name))

    # Highlight: Draupnir ##############################################################################################################
    if test_mode == "Test_argmax":
        draupnir_average_incorrectly_predicted_sites = draupnir_incorrectly_predicted_sites[:,0].tolist()
        draupnir_std_incorrectly_predicted_sites = np.zeros(draupnir_incorrectly_predicted_sites.shape[0]).tolist()
    else:
        draupnir_average_incorrectly_predicted_sites = np.mean(draupnir_incorrectly_predicted_sites, axis=1).tolist()
        draupnir_std_incorrectly_predicted_sites = np.std(draupnir_incorrectly_predicted_sites, axis=1).tolist()

    plot_load_incorrectly_predicted_sites = PlotLoad3(paml_incorrectly_predicted_sites_dict=paml_incorrectly_predicted_sites_dict,
                           draupnir_average_incorrectly_predicted_sites=draupnir_average_incorrectly_predicted_sites,
                           draupnir_std_incorrectly_predicted_sites=draupnir_std_incorrectly_predicted_sites,
                           phylobayes_incorrectly_predicted_sites_dict=phylobayes_incorrectly_predicted_sites_dict,
                           fastml_incorrectly_predicted_sites_dict=fastml_incorrectly_predicted_sites_dict,
                           iqtree_incorrectly_predicted_sites_dict=iqtree_incorrectly_predicted_sites_dict,
                           true_alignment_length=len(true_alignment[0].seq),
                           plot_name=plot_name,
                           plot_folder_name=plot_load2.plot_folder_name,
                           name = plot_load2.name,
                           sequence_input_type=plot_load2.sequence_input_type,
                           test_mode=plot_load2.test_mode)


    draupnir_average_percent_id = (plot_load2.draupnir_alig_length - np.array(draupnir_average_incorrectly_predicted_sites)) * 100 / plot_load2.draupnir_alig_length

    draupnir_std_percent_id = draupnir_std_incorrectly_predicted_sites
    plot_load_percentid = PlotLoad3(paml_incorrectly_predicted_sites_dict=paml_percent_id_dict,
                                    draupnir_average_incorrectly_predicted_sites=draupnir_average_percent_id,
                                    draupnir_std_incorrectly_predicted_sites=draupnir_std_percent_id,
                                    phylobayes_incorrectly_predicted_sites_dict=phylobayes_percent_id_dict,
                                    fastml_incorrectly_predicted_sites_dict=fastml_percent_id_dict,
                                    iqtree_incorrectly_predicted_sites_dict=iqtree_percent_id_dict,
                                    true_alignment_length=100,
                                    plot_name=plot_name,
                                    plot_folder_name=plot_load2.plot_folder_name,
                                    name=plot_load2.name,
                                    test_mode=plot_load2.test_mode,
                                    sequence_input_type=plot_load2.sequence_input_type)

    small_plot(plot_load_incorrectly_predicted_sites)
    percendid_table(plot_load_percentid)
    writeMSA('{}/{}/Draupnir_{}_{}.fasta'.format(plot_load2.plot_folder_name, save_folder,name, test_mode),
             plot_load.draupnir_sequences_predictions_msa, format='fasta')
def plot_incorrectly_predicted_aa_randall(plot_load,plot_load2,consider_gaps_predictions=True):
        "Bar plot that conciles all programs we benchmark against"

        print("Building comparison Incorrectly Predicted aa...")
        paml_predictions_file = plot_load.paml_predictions_file
        true_ancestral_file = plot_load.true_ancestral_file
        ancestor_info = plot_load.ancestor_info
        draupnir_incorrectly_predicted_sites = plot_load.draupnir_incorrectly_predicted_sites
        correspondence_dict_paml_to_sim = plot_load2.correspondence_dict_paml_to_sim
        correspondence_original_to_iqtree = plot_load2.correspondence_original_to_iqtree
        correspondence_iqtree_to_original = plot_load2.correspondence_iqtree_to_original
        rst_file = plot_load.rst_file
        correspondence_dict_sim_to_paml = plot_load2.correspondence_dict_sim_to_paml
        phylobayes_dict = plot_load2.phylobayes_dict
        fastml_dict = plot_load2.fastml_dict
        gap_positions = plot_load.gap_positions
        name = plot_load.name
        correspondence_fastml_to_original = plot_load2.correspondence_fastml_to_original
        correspondence_original_to_fastml = plot_load2.correspondence_original_to_fastml
        gaps = ["GAPS" if consider_gaps_predictions else "withoutGAPS"][0]
        sequence_input_type = plot_load2.sequence_input_type
        test_mode = plot_load2.test_mode
        plot_name = plot_load2.plot_full_name

        paml_predicted_alignment = AlignIO.read(paml_predictions_file, "fasta")
        true_alignment = AlignIO.read(true_ancestral_file, "fasta")

        # Highlight: order predicted alignment nodes according to draupnir results
        ancestor_info["0"] = ancestor_info["0"].astype(str)
        ancestor_info.drop('Unnamed: 0', inplace=True, axis=1)
        nodes_names = ancestor_info["0"].tolist()
        true_alignment_dict = dict(zip(["I" + seq.id for seq in true_alignment], [seq.seq for seq in true_alignment]))
        nodes_names = [node.replace("A", "I") if node.startswith("A") else "A" + str(int(node)) for node in nodes_names]  # new naming systam where A is for leaves

        internal_nodes_indexes = [i for i, node in enumerate(nodes_names) if re.search('^(?!^A{1}[0-9]+(?![A-Z])+)', str(node))]
        internal_nodes_names = [node for i, node in enumerate(nodes_names) if re.search('^(?!^A{1}[0-9]+(?![A-Z])+)', str(node))]

        draupnir_correspondence_dict = dict(zip(internal_nodes_names, internal_nodes_indexes))
        draupnir_correspondence_dict_reversed = {value: key for key, value in draupnir_correspondence_dict.items()}
        draupnir_nodes_order = draupnir_incorrectly_predicted_sites[0].tolist()
        draupnir_nodes_treelevel_order= [draupnir_correspondence_dict["I"+str(int(node_name))]  for node_name in draupnir_nodes_order if "I"+str(int(node_name)) in draupnir_correspondence_dict.keys()]

        # Highlight: PAML CodeML ##########################################################################################################
        paml_predicted_alignment_dict = dict(zip([correspondence_dict_paml_to_sim[seq.id.replace("node", "")] for seq in paml_predicted_alignment],[seq.seq for seq in paml_predicted_alignment]))
        paml_predicted_alignment_dict_ordered = dict(zip(draupnir_nodes_treelevel_order,
                                                         [val for key, val in paml_predicted_alignment_dict.items() if
                                                          key in draupnir_nodes_treelevel_order])) #until here seems correct

        true_alignment_dict_paml = { draupnir_correspondence_dict[key]:value for key,value in true_alignment_dict.items()} #Todo, change the keys should be A21
        if not consider_gaps_predictions:  # if there is any gap in the paml prediction, ignore that site , replace with gap
            def build_site():
                if highest_prob_dict[idx][2]:  # if there is any gap in the paml prediction, ignore that site , replace with gap
                    site = "-"
                else:
                    site = highest_prob_dict[idx][0]
                return site
        else:
            def build_site():
                site = highest_prob_dict[idx][0]
                return site

        if sequence_input_type == "DNA":
            def esp(rst,index): return Extract_Sites_Probabilities_Codons(rst,index)
        else:
            def esp(rst,index): return Extract_Sites_Probabilities(rst,index)
        paml_incorrectly_predicted_sites_dict = defaultdict()
        paml_percent_id_dict = defaultdict()
        paml_sequences_predictions_dict = defaultdict()
        trueseq_dict = defaultdict()
        for equivalent_true_node_number, pred_seq in paml_predicted_alignment_dict_ordered.items():
            equivalent_true_node_sequence = true_alignment_dict_paml[equivalent_true_node_number]
            trueseq_dict[equivalent_true_node_number] = "".join(equivalent_true_node_sequence)
            #highest_prob_dict = Extract_Sites_Probabilities(rst_file, int(correspondence_dict_sim_to_paml["I"+str(equivalent_true_node_number)]))
            highest_prob_dict = esp(rst_file, int(correspondence_dict_sim_to_paml["I"+str(equivalent_true_node_number)]))
            if sequence_input_type == "DNA":
                pred_seq = translate(pred_seq)
            fixed_predicted_seq = [] #sequence with highest likelihood, so far it seems the same sequence as pred_seq, but it was recommended to do it like this...
            for idx, site in enumerate(pred_seq, 1):
                new_site = build_site()
                fixed_predicted_seq.append(new_site)
            fixed_predicted_seq = "".join(fixed_predicted_seq)
            paml_incorrectly_predicted_sites_dict[equivalent_true_node_number] = DraupnirUtils.incorrectly_predicted_aa(fixed_predicted_seq, equivalent_true_node_sequence)
            paml_percent_id_dict[equivalent_true_node_number] = DraupnirUtils.perc_identity_pair_seq(fixed_predicted_seq, equivalent_true_node_sequence)
            paml_sequences_predictions_dict[equivalent_true_node_number] = fixed_predicted_seq
        paml_incorrectly_predicted_sites_dict = {draupnir_correspondence_dict_reversed[key]:value for key,value in paml_incorrectly_predicted_sites_dict.items()} #change naming again, necessary for later
        paml_sequences_predictions_dict = {draupnir_correspondence_dict_reversed[key]:value for key,value in paml_sequences_predictions_dict.items()}
        #Highlight: Save the PAML predicted sequences in a fasta file to calculate mutual information
        dict_to_fasta(paml_sequences_predictions_dict,"{}/{}/PAML_predictions_{}_{}.fasta".format(plot_load2.plot_folder_name,save_folder,name,sequence_input_type))
        trueseq_dict = {str(draupnir_correspondence_dict_reversed[key]):value for key,value in trueseq_dict.items()}
        dict_to_fasta(trueseq_dict,"{}/{}/True_seq_{}.fasta".format(plot_load2.plot_folder_name,save_folder,name))

        #Highlight: FastML###########################################################################
        correspondence_fastml_to_original_treelevelorder = {key1:draupnir_correspondence_dict[value1] for key1,value1 in correspondence_fastml_to_original.items()}  #all internal nodes
        correspondence_original_to_fastml_treelevelorder = {value:key for key,value in correspondence_fastml_to_original_treelevelorder.items()}
        fastml_dict_ordered = defaultdict()
        for value in draupnir_nodes_treelevel_order:
            node_name = correspondence_original_to_fastml_treelevelorder[value]
            fastml_dict_ordered[node_name] = fastml_dict[node_name]

        if not consider_gaps_predictions:
            def build_seq(in_seq):
                " Correct prediction, re-assign gaps, sicen fastml does not handle gaps"
                out_seq = "".join(["-" if idx in gap_positions else letter for idx, letter in enumerate(
                    in_seq)])  # correct prediction, re-assign gaps, sicen fastml does not handle gaps (they apply a simple heuristic)
                return out_seq
        else:
            def build_seq(in_seq):
                out_seq = "".join([letter for idx, letter in enumerate(in_seq)])
                return out_seq

        true_alignment_dict_fastml = {correspondence_original_to_fastml[key]:value for key,value in true_alignment_dict.items() }
        fastml_incorrectly_predicted_sites_dict = defaultdict()
        fastml_percent_id_dict = defaultdict()
        fastml_sequences_predictions_dict = defaultdict()
        for internal_node, fastml_seq in fastml_dict_ordered.items():
            equivalent_true_node_sequence = true_alignment_dict_fastml[internal_node]
            fastml_seq = build_seq(fastml_seq).strip("*")
            fastml_percent_id_dict[internal_node] = DraupnirUtils.perc_identity_pair_seq(fastml_seq,equivalent_true_node_sequence)
            fastml_incorrectly_predicted_sites_dict[internal_node] = DraupnirUtils.incorrectly_predicted_aa(fastml_seq,equivalent_true_node_sequence)
            fastml_sequences_predictions_dict[internal_node] = fastml_seq
        dict_to_fasta(fastml_sequences_predictions_dict,"{}/{}/FastML_predictions_{}_{}.fasta".format(plot_load2.plot_folder_name,save_folder,name,sequence_input_type))
        # Highlight: PhyloBayes############################################################################
        phylobayes_dict_ordered = defaultdict()
        for node in draupnir_nodes_treelevel_order:
            node_name = draupnir_correspondence_dict_reversed[node]
            phylobayes_dict_ordered[node_name] = phylobayes_dict[node_name.replace("I","A")]

        phylobayes_incorrectly_predicted_sites_dict = defaultdict()
        phylobayes_percent_id_dict = defaultdict()
        phylobayes_sequences_predictions_dict = defaultdict()
        for internal_node, phylo_bayes_seq in phylobayes_dict_ordered.items():
            equivalent_true_node_sequence = true_alignment_dict[internal_node]
            phylo_bayes_seq = build_seq(phylo_bayes_seq).strip("*")
            phylobayes_percent_id_dict[internal_node] = DraupnirUtils.perc_identity_pair_seq(phylo_bayes_seq,
                                                                                             equivalent_true_node_sequence)
            phylobayes_incorrectly_predicted_sites_dict[internal_node] = DraupnirUtils.incorrectly_predicted_aa(
                phylo_bayes_seq, equivalent_true_node_sequence)
            phylobayes_sequences_predictions_dict[internal_node] = phylo_bayes_seq
        # PhyloBayes is missing some nodes (those internal not calculated from distance among leaves), we assign them to not predicted
        for key in paml_incorrectly_predicted_sites_dict.keys():
            if key not in phylobayes_incorrectly_predicted_sites_dict.keys():
                print("Missing key in phylobayes")
                if sequence_input_type == "DNA":
                    phylobayes_incorrectly_predicted_sites_dict[key] = int(len(paml_predicted_alignment_dict_ordered[key]) // 3)
                else:
                    phylobayes_incorrectly_predicted_sites_dict[key] = len(paml_predicted_alignment_dict_ordered[key])
                phylobayes_percent_id_dict[key] = np.nan
                phylobayes_sequences_predictions_dict[key] = np.nan
        # Order again & fix naming
        phylobayes_incorrectly_predicted_sites_dict = {"I"+str(int(key)): phylobayes_incorrectly_predicted_sites_dict["I"+str(int(key))] for key in
                                                       draupnir_nodes_order if
                                                       "I"+str(int(key)) in phylobayes_incorrectly_predicted_sites_dict.keys()}
        phylobayes_sequences_predictions_dict = {"I"+str(int(key)): phylobayes_sequences_predictions_dict["I"+str(int(key))] for key in
                                                       draupnir_nodes_order if
                                                       "I"+str(int(key)) in phylobayes_sequences_predictions_dict.keys()}
        dict_to_fasta(phylobayes_sequences_predictions_dict,"{}/{}/PhyloBayes_predictions_{}_{}.fasta".format(plot_load2.plot_folder_name,save_folder,name,sequence_input_type))

        # Highlight: Order Iqtree #TODO: all of this can be done way faster?
        iqtree_dict = plot_load2.iqtree_dict
        iqtree_incorrectly_predicted_sites_dict = defaultdict()
        iqtree_percent_id_dict = defaultdict()
        iqtree_sequences_predictions_dict = defaultdict()
        for key in true_alignment_dict.keys():
            true_seq = true_alignment_dict[key]
            if key in iqtree_dict.keys():
                iqtree_seq = "".join(iqtree_dict[key])
                iqtree_percent_id_dict[key] = DraupnirUtils.perc_identity_pair_seq(iqtree_seq, true_seq)
                iqtree_incorrectly_predicted_sites_dict[key] = DraupnirUtils.incorrectly_predicted_aa(iqtree_seq, true_seq)
                iqtree_sequences_predictions_dict[key] = iqtree_seq
            else:
                print("Missing key {}".format(key))
                iqtree_percent_id_dict[key] = np.nan
                iqtree_incorrectly_predicted_sites_dict[key] = len(true_seq) #todo: change to np.nan, does not matter anymnore since we are using %id
                iqtree_sequences_predictions_dict[key]  = np.nan

        dict_to_fasta(iqtree_sequences_predictions_dict,"{}/{}/IQTree_predictions_{}_{}.fasta".format(plot_load2.plot_folder_name,save_folder,name,sequence_input_type))
        # Highlight: Draupnir ##############################################################################################################
        if test_mode == "Test_argmax":
            draupnir_average_incorrectly_predicted_sites = draupnir_incorrectly_predicted_sites[1]
            draupnir_std_incorrectly_predicted_sites = np.zeros(draupnir_incorrectly_predicted_sites.shape[1])
        else:
            draupnir_average_incorrectly_predicted_sites = np.mean(draupnir_incorrectly_predicted_sites[1:], axis=0)
            draupnir_std_incorrectly_predicted_sites = np.std(draupnir_incorrectly_predicted_sites[1:], axis=0)

        plot_load_incorrect_aa = PlotLoad3(paml_incorrectly_predicted_sites_dict=paml_incorrectly_predicted_sites_dict,
                               draupnir_average_incorrectly_predicted_sites=draupnir_average_incorrectly_predicted_sites,
                               draupnir_std_incorrectly_predicted_sites=draupnir_std_incorrectly_predicted_sites,
                               phylobayes_incorrectly_predicted_sites_dict=phylobayes_incorrectly_predicted_sites_dict,
                               fastml_incorrectly_predicted_sites_dict=fastml_incorrectly_predicted_sites_dict,
                               iqtree_incorrectly_predicted_sites_dict=iqtree_incorrectly_predicted_sites_dict,
                               true_alignment_length=len(true_alignment[0].seq),
                               plot_name=plot_name,
                               plot_folder_name=plot_load2.plot_folder_name,
                               name = plot_load2.name,
                               sequence_input_type=plot_load2.sequence_input_type,
                               test_mode=plot_load2.test_mode)

        draupnir_average_percent_id = (plot_load2.draupnir_alig_length - draupnir_average_incorrectly_predicted_sites) * 100 / plot_load2.draupnir_alig_length
        draupnir_std_percent_id = draupnir_std_incorrectly_predicted_sites
        #I re-use the named tuple from incorrectly predicted sites
        plot_load_percentid = PlotLoad3(paml_incorrectly_predicted_sites_dict=paml_percent_id_dict,
                                        draupnir_average_incorrectly_predicted_sites=draupnir_average_percent_id,
                                        draupnir_std_incorrectly_predicted_sites=draupnir_std_percent_id,
                                        phylobayes_incorrectly_predicted_sites_dict=phylobayes_percent_id_dict,
                                        fastml_incorrectly_predicted_sites_dict=fastml_percent_id_dict,
                                        iqtree_incorrectly_predicted_sites_dict=iqtree_percent_id_dict,
                                        true_alignment_length=100, # 100% identity
                                        plot_name=plot_name,
                                        plot_folder_name=plot_load2.plot_folder_name,
                                        name=plot_load2.name + "%ID",
                                        sequence_input_type=plot_load2.sequence_input_type,
                                        test_mode=plot_load2.test_mode)
        small_plot(plot_load_incorrect_aa)
        comparison_table(plot_load_incorrect_aa)
        percendid_table(plot_load_percentid)
        writeMSA('{}/{}/Draupnir_{}_{}.fasta'.format(plot_load2.plot_folder_name,save_folder,name,test_mode), plot_load.draupnir_sequences_predictions_msa, format='fasta')
def plot_incorrectly_predicted_aa_simulations(plot_load,plot_load2,consider_gaps_predictions=True):
    "if consider_gaps_predictions is False, turn the position into a gap (using the colum sites from the leaf nodes) "

    print("Building comparison Incorrectly Predicted aa...")
    paml_predictions_file = plot_load.paml_predictions_file
    true_ancestral_file = plot_load.true_ancestral_file
    ancestor_info = plot_load.ancestor_info
    draupnir_incorrectly_predicted_sites = plot_load.draupnir_incorrectly_predicted_sites
    draupnir_incorrectly_predicted_sites_complete = plot_load.draupnir_incorrectly_predicted_sites_complete
    correspondence_dict_paml_to_sim = plot_load2.correspondence_dict_paml_to_sim
    rst_file = plot_load.rst_file
    correspondence_dict_sim_to_paml = plot_load2.correspondence_dict_sim_to_paml
    phylobayes_dict = plot_load2.phylobayes_dict
    fastml_dict = plot_load2.fastml_dict
    gap_positions = plot_load.gap_positions
    name = plot_load.name
    correspondence_original_to_iqtree = plot_load2.correspondence_original_to_iqtree
    correspondence_iqtree_to_original = plot_load2.correspondence_iqtree_to_original
    #plot_name = "{}_Dataset{}_{}".format(name,plot_load.dataset_number,gaps)
    plot_name = plot_load2.plot_full_name
    sequence_input_type = plot_load2.sequence_input_type
    test_mode = plot_load2.test_mode
    paml_predicted_alignment = AlignIO.read(paml_predictions_file, "fasta")
    true_alignment = AlignIO.read(true_ancestral_file, "fasta")

    #Highlight: order predicted alignment nodes according to draupnir results
    #Highlight: Find back the correspondence of the nodes names
    ancestor_info["0"] = ancestor_info["0"].astype(str)
    ancestor_info.drop('Unnamed: 0', inplace=True, axis=1)
    nodes_names = ancestor_info["0"].tolist()
    true_alignment_dict = dict(zip([seq.id.replace("Node", "I") for seq in true_alignment], [seq.seq for seq in true_alignment]))

    # internal_nodes_indexes = [i for i, node in enumerate(nodes_names) if re.search('^(?!^A{1}[0-9]+(?![A-Z])+)', str(node))]
    # internal_nodes_names = [node for i, node in enumerate(nodes_names) if re.search('^(?!^A{1}[0-9]+(?![A-Z])+)', str(node))]
    # draupnir_correspondence_dict = dict(zip(internal_nodes_names,internal_nodes_indexes))
    #
    draupnir_correspondence_dict = {node:i for i, node in enumerate(nodes_names) if re.search('^(?!^A{1}[0-9]+(?![A-Z])+)', str(node))}

    draupnir_nodes_order_complete = draupnir_incorrectly_predicted_sites_complete[0,:,0].tolist()
    draupnir_nodes_order_sample = draupnir_incorrectly_predicted_sites[0].tolist() #for large datasets this is just a subsample from all the nodes
    if draupnir_incorrectly_predicted_sites_complete.shape[1] != draupnir_incorrectly_predicted_sites.shape[1]:
        draupnir_nodes_order = draupnir_nodes_order_complete
    else:
        draupnir_nodes_order = draupnir_nodes_order_sample

    draupnir_nodes_order_reverse = [key for key, value in draupnir_correspondence_dict.items() if value in draupnir_nodes_order] #TODO: This is valied just because everything has already the same order as the draupnir nodes
    #Highlight: PAML CodeML ##########################################################################################################
    paml_predicted_alignment_dict = dict(zip([correspondence_dict_paml_to_sim[seq.id.replace("node","")] for seq in paml_predicted_alignment], [seq.seq for seq in paml_predicted_alignment]))
    paml_predicted_alignment_dict_ordered = dict(zip(draupnir_nodes_order_reverse,[val for key,val in paml_predicted_alignment_dict.items() if key in draupnir_nodes_order_reverse]))

    if not consider_gaps_predictions:  # if there is any gap in the leaves alignment, ignore that site , replace with gap
        def build_site():
            if highest_prob_dict[idx][2]:
                site = "-"
            else:
                site = highest_prob_dict[idx][0]
            return site
    else:
        def build_site():
            site = highest_prob_dict[idx][0]
            return site

    if sequence_input_type == "DNA":
        def esp(rst, index):
            return Extract_Sites_Probabilities_Codons(rst, index)
    else:
        def esp(rst, index):
            return Extract_Sites_Probabilities(rst, index)
    paml_incorrectly_predicted_sites_dict = defaultdict()
    paml_percent_id_dict = defaultdict()
    paml_sequences_predictions_dict = defaultdict()
    trueseq_dict = defaultdict()
    for equivalent_true_node_number,pred_seq in paml_predicted_alignment_dict_ordered.items():
        equivalent_true_node_sequence = true_alignment_dict[equivalent_true_node_number]
        highest_prob_dict = esp(rst_file, int(correspondence_dict_sim_to_paml[equivalent_true_node_number]))
        trueseq_dict[equivalent_true_node_number] = "".join(equivalent_true_node_sequence)
        if sequence_input_type=="DNA":
            pred_seq = translate(pred_seq)
        fixed_predicted_seq = []
        for idx,site in enumerate(pred_seq,1):
            site = build_site()
            fixed_predicted_seq.append(site)
        fixed_predicted_seq = "".join(fixed_predicted_seq)
        paml_incorrectly_predicted_sites_dict[equivalent_true_node_number] = DraupnirUtils.incorrectly_predicted_aa(fixed_predicted_seq,equivalent_true_node_sequence)
        paml_percent_id_dict[equivalent_true_node_number] = DraupnirUtils.perc_identity_pair_seq(fixed_predicted_seq,equivalent_true_node_sequence)
        paml_sequences_predictions_dict[equivalent_true_node_number] = "".join(fixed_predicted_seq)
    dict_to_fasta(paml_sequences_predictions_dict,"{}/{}/PAML_predictions_{}_{}.fasta".format(plot_load2.plot_folder_name,save_folder, name,sequence_input_type))
    dict_to_fasta(trueseq_dict, "{}/{}/True_seq_{}.fasta".format(plot_load2.plot_folder_name,save_folder, name))
    #Highlight: FastML###########################################################################
    # if not consider_gaps_predictions:
    #     def build_seq(in_seq):
    #         ""
    #         out_seq = "".join(["-" if idx in gap_positions else letter for idx, letter in enumerate(
    #             in_seq)])  # correct prediction, re-assign gaps, sicen fastml does not handle gaps
    #         return out_seq
    # else:
    def build_seq(in_seq):
        out_seq = "".join([letter for idx, letter in enumerate(in_seq)])
        return out_seq
    fastml_incorrectly_predicted_sites_dict = defaultdict()
    fastml_percent_id_dict = defaultdict()
    fastml_sequences_predictions_dict = defaultdict()
    if plot_load2.fastml_dict is not None:
        fastml_dict_ordered = {key: fastml_dict[key] for key in draupnir_nodes_order_reverse if
                                   key in fastml_dict.keys()}

        if sequence_input_type == "DNA": #Highlight: we need to realign the sequences ...
            true_alignment_dict, fastml_dict_ordered = merge_to_align_simulationsDNA(true_alignment_dict,
                                                                                                  fastml_dict_ordered,
                                                                                                  plot_load2, "FastML")

        for internal_node,fastml_seq in fastml_dict_ordered.items():
            equivalent_true_node_sequence = true_alignment_dict[internal_node]
            fastml_seq = build_seq(fastml_seq)
            fastml_percent_id_dict[internal_node] = DraupnirUtils.perc_identity_pair_seq(fastml_seq, equivalent_true_node_sequence)
            fastml_incorrectly_predicted_sites_dict[internal_node] = DraupnirUtils.incorrectly_predicted_aa(fastml_seq,equivalent_true_node_sequence)
            fastml_sequences_predictions_dict[internal_node] = fastml_seq
        dict_to_fasta(fastml_sequences_predictions_dict,"{}/{}/FastML_predictions_{}_{}.fasta".format(plot_load2.plot_folder_name,save_folder,name,sequence_input_type))
    else:
        print("fastml not found!!!")
        fastml_incorrectly_predicted_sites_dict = dict(zip(draupnir_nodes_order_reverse,[plot_load2.draupnir_alig_length]*len(draupnir_nodes_order_reverse)))
        fastml_percent_id_dict = dict(zip(draupnir_nodes_order_reverse, [np.nan] * len(draupnir_nodes_order_reverse)))
    #Highlight: PhyloBayes############################################################################
    phylobayes_dict_ordered = {key: phylobayes_dict[key] for key in draupnir_nodes_order_reverse if
                               key in phylobayes_dict.keys()}

    #if sequence_input_type == "DNA": #Highlight: we need to realign the sequences, because of some stop codons
    true_alignment_dict, phylobayes_dict_ordered = merge_to_align_simulationsDNA(true_alignment_dict,
                                                                                          phylobayes_dict_ordered,
                                                                                          plot_load2, "Phylobayes")
    phylobayes_incorrectly_predicted_sites_dict = defaultdict()
    phylobayes_percent_id_dict = defaultdict()
    phylobayes_sequences_predictions_dict = defaultdict()
    for internal_node,phylo_bayes_seq in phylobayes_dict_ordered.items():
        equivalent_true_node_sequence = true_alignment_dict[internal_node]
        phylo_bayes_seq = build_seq(phylo_bayes_seq)
        phylobayes_percent_id_dict[internal_node] = DraupnirUtils.perc_identity_pair_seq(phylo_bayes_seq, equivalent_true_node_sequence)
        phylobayes_incorrectly_predicted_sites_dict[internal_node] = DraupnirUtils.incorrectly_predicted_aa(phylo_bayes_seq,equivalent_true_node_sequence)
        phylobayes_sequences_predictions_dict[internal_node] = phylo_bayes_seq
    #Highlight: PhyloBayes is missing some nodes (those internal not calculated from distance among leaves), we assign them to not predicted
    for key in paml_incorrectly_predicted_sites_dict.keys():
        if key not in phylobayes_incorrectly_predicted_sites_dict.keys():
            if sequence_input_type == "DNA":
                phylobayes_incorrectly_predicted_sites_dict[key] = int(len(paml_predicted_alignment_dict_ordered[key])//3)
                phylobayes_percent_id_dict[key] = np.nan
                phylobayes_sequences_predictions_dict[key] = "-"*int(len(paml_predicted_alignment_dict_ordered[key])//3)
            else:
                phylobayes_incorrectly_predicted_sites_dict[key] = len(paml_predicted_alignment_dict_ordered[key])
                phylobayes_percent_id_dict[key] = np.nan
                phylobayes_sequences_predictions_dict[key] = "-"*len(paml_predicted_alignment_dict_ordered[key])


    #Highlight: Order again:
    phylobayes_incorrectly_predicted_sites_dict = {key: phylobayes_incorrectly_predicted_sites_dict[key] for key in draupnir_nodes_order_reverse if key in phylobayes_incorrectly_predicted_sites_dict.keys()}
    phylobayes_sequences_predictions_dict = {key: phylobayes_sequences_predictions_dict[key] for key in draupnir_nodes_order_reverse if key in phylobayes_incorrectly_predicted_sites_dict.keys()}

    dict_to_fasta(phylobayes_sequences_predictions_dict,
                  "{}/{}/PhyloBayes_predictions_{}_{}.fasta".format(plot_load2.plot_folder_name, save_folder,name,sequence_input_type))

    #Highlight: Iqtree ---> also missing some node's predictions
    iqtree_incorrectly_predicted_sites_dict = defaultdict()
    iqtree_percent_id_dict = defaultdict()
    iqtree_sequences_predictions_dict = defaultdict()
    iqtree_dict = plot_load2.iqtree_dict

    true_sequences_aligned_to_iqtree_predictions, iqtree_aligned_predictions = merge_to_align_simulationsDNA(true_alignment_dict,
                                                                                              iqtree_dict,
                                                                                              plot_load2, "IQTree")
    for key in draupnir_nodes_order_reverse:
        if key in iqtree_dict.keys():
            true_seq = true_alignment_dict[key]
            if len(true_seq) != len(iqtree_dict[key]): #sometimes iqtree predicts gaps,we use the aligned sequences
                true_seq = true_sequences_aligned_to_iqtree_predictions[key]
                iqtree_prediction = iqtree_aligned_predictions[key]
            else:
                iqtree_prediction = iqtree_dict[key]
            iqtree_percent_id_dict[key] = DraupnirUtils.perc_identity_pair_seq(iqtree_prediction,true_seq)
            iqtree_incorrectly_predicted_sites_dict[key] = DraupnirUtils.incorrectly_predicted_aa(iqtree_prediction, true_seq)
            iqtree_sequences_predictions_dict[key] = "".join(iqtree_prediction)
        else:
            print("Missing key in IQtree {}".format(key))
            iqtree_percent_id_dict[key] = np.nan
            if sequence_input_type == "DNA":
                iqtree_sequences_predictions_dict[key] = "-" * int(len(paml_predicted_alignment_dict_ordered[key])//3)
                iqtree_incorrectly_predicted_sites_dict[key] = int(len(paml_predicted_alignment_dict_ordered[key])//3)
            else:
                iqtree_sequences_predictions_dict[key] = "-" * len(paml_predicted_alignment_dict_ordered[key])
                iqtree_incorrectly_predicted_sites_dict[key] = len(paml_predicted_alignment_dict_ordered[key])
    dict_to_fasta(iqtree_sequences_predictions_dict,"{}/{}/IQTree_predictions_{}_{}.fasta".format(plot_load2.plot_folder_name,save_folder, name,sequence_input_type))
    #Highlight: Draupnir ##############################################################################################################
    if draupnir_incorrectly_predicted_sites_complete.shape[1] != draupnir_incorrectly_predicted_sites.shape[1]:
        if test_mode == "Test_argmax":
            draupnir_average_incorrectly_predicted_sites = draupnir_incorrectly_predicted_sites_complete[:,:,1].squeeze(0)
            draupnir_std_incorrectly_predicted_sites = np.zeros(draupnir_incorrectly_predicted_sites_complete.shape[1])
        else:
            draupnir_average_incorrectly_predicted_sites = np.mean(draupnir_incorrectly_predicted_sites_complete[:,:,1], axis=0)
            draupnir_std_incorrectly_predicted_sites = np.std(draupnir_incorrectly_predicted_sites_complete[:,:,1], axis=0)
    else:
        if test_mode == "Test_argmax":
            draupnir_average_incorrectly_predicted_sites = draupnir_incorrectly_predicted_sites[1]
            draupnir_std_incorrectly_predicted_sites = np.zeros(draupnir_incorrectly_predicted_sites.shape[1])
        else:
            draupnir_average_incorrectly_predicted_sites = np.mean(draupnir_incorrectly_predicted_sites[1:], axis=0)
            draupnir_std_incorrectly_predicted_sites = np.std(draupnir_incorrectly_predicted_sites[1:], axis=0)

    plot_load_incorrect_aa = PlotLoad3(paml_incorrectly_predicted_sites_dict=paml_incorrectly_predicted_sites_dict,
                           draupnir_average_incorrectly_predicted_sites=draupnir_average_incorrectly_predicted_sites,
                           draupnir_std_incorrectly_predicted_sites=draupnir_std_incorrectly_predicted_sites,
                           phylobayes_incorrectly_predicted_sites_dict=phylobayes_incorrectly_predicted_sites_dict,
                           fastml_incorrectly_predicted_sites_dict=fastml_incorrectly_predicted_sites_dict,
                           iqtree_incorrectly_predicted_sites_dict = iqtree_incorrectly_predicted_sites_dict,
                           true_alignment_length=len(true_alignment[0].seq),
                           plot_name=plot_name,
                           plot_folder_name=plot_load2.plot_folder_name,
                           name = plot_load2.name,
                           sequence_input_type=plot_load2.sequence_input_type,
                           test_mode=plot_load2.test_mode)

    draupnir_average_percent_id = (plot_load2.draupnir_alig_length-draupnir_average_incorrectly_predicted_sites)*100/plot_load2.draupnir_alig_length
    draupnir_std_percent_id = draupnir_std_incorrectly_predicted_sites

    plot_load_percentid = PlotLoad3(paml_incorrectly_predicted_sites_dict=paml_percent_id_dict,
                           draupnir_average_incorrectly_predicted_sites=draupnir_average_percent_id,
                           draupnir_std_incorrectly_predicted_sites=draupnir_std_percent_id,
                           phylobayes_incorrectly_predicted_sites_dict=phylobayes_percent_id_dict,
                           fastml_incorrectly_predicted_sites_dict=fastml_percent_id_dict,
                           iqtree_incorrectly_predicted_sites_dict=iqtree_percent_id_dict,
                           true_alignment_length=100,
                           plot_name=plot_name,
                           plot_folder_name=plot_load2.plot_folder_name ,
                           name=plot_load2.name + "%ID",
                           sequence_input_type=plot_load2.sequence_input_type,
                           test_mode=plot_load2.test_mode)


    if len(paml_incorrectly_predicted_sites_dict) > 50:
        if plot_load2.plot_only_sample: #Highlight: pick only the nodes that were randomly plotted
            draupnir_nodes_order_reverse = [key for key, value in draupnir_correspondence_dict.items() if value in draupnir_nodes_order_sample]
            paml_incorrectly_predicted_sites_dict_sample = {key:value for key,value in paml_incorrectly_predicted_sites_dict.items() if key in draupnir_nodes_order_reverse}
            phylobayes_incorrectly_predicted_sites_dict_sample = {key:value for key,value in phylobayes_incorrectly_predicted_sites_dict.items() if key in draupnir_nodes_order_reverse}
            fastml_incorrectly_predicted_sites_dict_sample = {key:value for key,value in fastml_incorrectly_predicted_sites_dict.items() if key in draupnir_nodes_order_reverse}
            iqtree_incorrectly_predicted_sites_dict_sample = {key: value for key, value in
                                                              iqtree_incorrectly_predicted_sites_dict.items() if
                                                              key in draupnir_nodes_order_reverse}
            paml_percent_id_dict_sample = {key:value for key,value in paml_percent_id_dict.items() if key in draupnir_nodes_order_reverse}
            phylobayes_percent_id_dict_sample = {key:value for key,value in phylobayes_percent_id_dict.items() if key in draupnir_nodes_order_reverse}
            fastml_percent_id_dict_sample = {key:value for key,value in fastml_percent_id_dict.items() if key in draupnir_nodes_order_reverse}
            iqtree_percent_id_dict_sample = {key: value for key, value in iqtree_percent_id_dict.items() if
                                             key in draupnir_nodes_order_reverse}
            if test_mode == "Test_argmax":
                draupnir_average_incorrectly_predicted_sites_sample = draupnir_incorrectly_predicted_sites[1]
                draupnir_std_incorrectly_predicted_sites_sample = np.zeros(draupnir_incorrectly_predicted_sites.shape[1])
                draupnir_average_percent_id_sample = (plot_load2.draupnir_alig_length - draupnir_average_incorrectly_predicted_sites_sample) * 100 / plot_load2.draupnir_alig_length
                draupnir_std_percent_id_sample = draupnir_std_incorrectly_predicted_sites_sample
            else:
                draupnir_average_incorrectly_predicted_sites_sample = np.mean(draupnir_incorrectly_predicted_sites[1:], axis=0)
                draupnir_std_incorrectly_predicted_sites_sample = np.std(draupnir_incorrectly_predicted_sites[1:], axis=0)
                draupnir_average_percent_id_sample = (plot_load2.draupnir_alig_length - draupnir_average_incorrectly_predicted_sites_sample) * 100 / plot_load2.draupnir_alig_length
                draupnir_std_percent_id_sample = draupnir_std_incorrectly_predicted_sites_sample
            plot_load_incorrect_aa = PlotLoad3(paml_incorrectly_predicted_sites_dict=paml_incorrectly_predicted_sites_dict_sample,
                                   draupnir_average_incorrectly_predicted_sites=draupnir_average_incorrectly_predicted_sites_sample,
                                   draupnir_std_incorrectly_predicted_sites=draupnir_std_incorrectly_predicted_sites_sample,
                                   phylobayes_incorrectly_predicted_sites_dict=phylobayes_incorrectly_predicted_sites_dict_sample,
                                   fastml_incorrectly_predicted_sites_dict=fastml_incorrectly_predicted_sites_dict_sample,
                                   iqtree_incorrectly_predicted_sites_dict=iqtree_incorrectly_predicted_sites_dict_sample,
                                   true_alignment_length=len(true_alignment[0].seq),
                                   plot_name=plot_name,
                                   plot_folder_name=plot_load2.plot_folder_name,
                                   name = plot_load2.name,
                                   sequence_input_type=plot_load2.sequence_input_type,
                                   test_mode=plot_load2.test_mode)
            plot_load_percentid = PlotLoad3(paml_incorrectly_predicted_sites_dict=paml_percent_id_dict_sample,
                                            draupnir_average_incorrectly_predicted_sites=draupnir_average_percent_id_sample,
                                            draupnir_std_incorrectly_predicted_sites=draupnir_std_percent_id_sample,
                                            phylobayes_incorrectly_predicted_sites_dict=phylobayes_percent_id_dict_sample,
                                            fastml_incorrectly_predicted_sites_dict=fastml_percent_id_dict_sample,
                                            iqtree_incorrectly_predicted_sites_dict=iqtree_percent_id_dict_sample,
                                            true_alignment_length=100,
                                            plot_name=plot_name,
                                            plot_folder_name=plot_load2.plot_folder_name,
                                            name=plot_load2.name + "%ID",
                                            sequence_input_type=plot_load2.sequence_input_type,
                                            test_mode=plot_load2.test_mode)
            small_plot(plot_load_incorrect_aa)
            comparison_table(plot_load_incorrect_aa)
            percendid_table(plot_load_percentid)
        else:#Highlight:Plot all the nodes
            n_plots = DraupnirUtils.Define_batch_size(len(plot_load_incorrect_aa.paml_incorrectly_predicted_sites_dict),batch_size=False, benchmarking=True) #sometimes we cannot divide equally
            if n_plots >= 2:
                splitted_plot(plot_load_incorrect_aa,n_plots)
            else:
                small_plot(plot_load_incorrect_aa)
            comparison_table(plot_load_incorrect_aa)
            percendid_table(plot_load_percentid)
    else:
        small_plot(plot_load_incorrect_aa)
        comparison_table(plot_load_incorrect_aa)
        percendid_table(plot_load_percentid)
    writeMSA('{}/{}/Draupnir_{}_{}.fasta'.format(plot_load2.plot_folder_name,save_folder, name, test_mode),
             plot_load.draupnir_sequences_predictions_msa, format='fasta')
    #draupnir_mi = applyMutinfoCorr(buildMutinfoMatrix(plot_load.draupnir_sequences_predictions_msa), corr="apc")
    #plot_mutual_information(trueseq_mi, paml_mi, fastml_mi, phylobayes_mi, draupnir_mi)
def plot_incorrectly_predicted_aa_simulations_800(plot_load,plot_load2,consider_gaps_predictions=True):
    """For larger datasets.if consider_gaps_predictions is False, turn the position into a gap (using the column sites from the leaf nodes) """

    print("Building comparison Incorrectly Predicted aa...")
    paml_predictions_file = plot_load.paml_predictions_file
    true_ancestral_file = plot_load.true_ancestral_file
    ancestor_info = plot_load.ancestor_info
    draupnir_incorrectly_predicted_sites = plot_load.draupnir_incorrectly_predicted_sites
    draupnir_incorrectly_predicted_sites_complete = plot_load.draupnir_incorrectly_predicted_sites_complete
    correspondence_original_to_iqtree = plot_load2.correspondence_original_to_iqtree
    correspondence_iqtree_to_original = plot_load2.correspondence_iqtree_to_original
    plot_name = plot_load2.plot_full_name
    sequence_input_type = plot_load2.sequence_input_type
    test_mode = plot_load2.test_mode
    true_alignment = AlignIO.read(true_ancestral_file, "fasta")
    name = plot_load.name

    #Highlight: order predicted alignment nodes according to draupnir results
    #Highlight: Find back the correspondence of the nodes names
    ancestor_info["0"] = ancestor_info["0"].astype(str)
    ancestor_info.drop('Unnamed: 0', inplace=True, axis=1)
    nodes_names = ancestor_info["0"].tolist()
    true_alignment_dict = dict(zip([seq.id.replace("Node", "I") for seq in true_alignment], [seq.seq for seq in true_alignment]))

    internal_nodes_indexes = [i for i, node in enumerate(nodes_names) if re.search('^(?!^A{1}[0-9]+(?![A-Z])+)', str(node))]
    internal_nodes_names = [node for i, node in enumerate(nodes_names) if re.search('^(?!^A{1}[0-9]+(?![A-Z])+)', str(node))]

    draupnir_correspondence_dict = dict(zip(internal_nodes_names,internal_nodes_indexes))
    draupnir_nodes_order_complete = draupnir_incorrectly_predicted_sites_complete[0,:,0].tolist()
    draupnir_nodes_order_sample = draupnir_incorrectly_predicted_sites[0].tolist() #for large datasets this is just a subsample from all the nodes
    if draupnir_incorrectly_predicted_sites_complete.shape[1] != draupnir_incorrectly_predicted_sites.shape[1]:
        draupnir_nodes_order = draupnir_nodes_order_complete
    else:
        draupnir_nodes_order = draupnir_nodes_order_sample

    draupnir_nodes_order_reverse = [key for key, value in draupnir_correspondence_dict.items() if value in draupnir_nodes_order] #TODO: This is valied just because everything has already the same order as the draupnir nodes

    #Highlight: Order Iqtree ---> also missing some node's predictions
    iqtree_incorrectly_predicted_sites_dict = defaultdict()
    iqtree_sequences_predictions_dict = defaultdict()
    iqtree_percent_id_dict = defaultdict()
    iqtree_dict = plot_load2.iqtree_dict
    trueseq_dict={key:"".join(val) for key,val in true_alignment_dict.items()}
    true_sequences_aligned_to_iqtree_predictions, iqtree_aligned_predictions = merge_to_align_simulationsDNA(
        true_alignment_dict,
        iqtree_dict,
        plot_load2, "IQTree")
    for key in draupnir_nodes_order_reverse:
        if key in iqtree_dict.keys():
            true_seq = true_alignment_dict[key]
            if len(true_seq) != len(iqtree_dict[key]):  # sometimes iqtree predicts gaps, so we need
                true_seq = true_sequences_aligned_to_iqtree_predictions[key]
                iqtree_prediction = iqtree_aligned_predictions[key]
            else:
                iqtree_prediction = iqtree_dict[key]
            iqtree_percent_id_dict[key] = DraupnirUtils.perc_identity_pair_seq(iqtree_prediction, true_seq)
            iqtree_incorrectly_predicted_sites_dict[key] = DraupnirUtils.incorrectly_predicted_aa(iqtree_prediction,
                                                                                                  true_seq)
            iqtree_sequences_predictions_dict[key] = "".join(iqtree_prediction)
        else:
            print("Missing key in IQtree {}".format(key))
            iqtree_percent_id_dict[key] = np.nan
            iqtree_incorrectly_predicted_sites_dict[key] = len(true_alignment_dict[key])
            iqtree_sequences_predictions_dict[key] = "-"*len(true_alignment_dict[key])

    dict_to_fasta(iqtree_sequences_predictions_dict,"{}/{}/IQTree_predictions_{}_{}.fasta".format(plot_load2.plot_folder_name,save_folder, name,sequence_input_type))
    dict_to_fasta(trueseq_dict, "{}/{}/True_seq_{}.fasta".format(plot_load2.plot_folder_name,save_folder, name))

    #Highlight: Draupnir ##############################################################################################################
    if draupnir_incorrectly_predicted_sites_complete.shape[1] != draupnir_incorrectly_predicted_sites.shape[1]:
        if test_mode == "Test_argmax":
            draupnir_average_incorrectly_predicted_sites = draupnir_incorrectly_predicted_sites_complete[:,:,1].squeeze(0)
            draupnir_std_incorrectly_predicted_sites = np.zeros(draupnir_incorrectly_predicted_sites_complete.shape[1])
        else:
            draupnir_average_incorrectly_predicted_sites = np.mean(draupnir_incorrectly_predicted_sites_complete[:,:,1], axis=0)
            draupnir_std_incorrectly_predicted_sites = np.std(draupnir_incorrectly_predicted_sites_complete[:,:,1], axis=0)
    else:
        if test_mode == "Test_argmax":
            draupnir_average_incorrectly_predicted_sites = draupnir_incorrectly_predicted_sites[1]
            draupnir_std_incorrectly_predicted_sites = np.zeros(draupnir_incorrectly_predicted_sites.shape[1])
        else:
            draupnir_average_incorrectly_predicted_sites = np.mean(draupnir_incorrectly_predicted_sites[1:], axis=0)
            draupnir_std_incorrectly_predicted_sites = np.std(draupnir_incorrectly_predicted_sites[1:], axis=0)

    plot_load_incorrect_aa = PlotLoad3(paml_incorrectly_predicted_sites_dict=None,
                           draupnir_average_incorrectly_predicted_sites=draupnir_average_incorrectly_predicted_sites,
                           draupnir_std_incorrectly_predicted_sites=draupnir_std_incorrectly_predicted_sites,
                           phylobayes_incorrectly_predicted_sites_dict=None,
                           fastml_incorrectly_predicted_sites_dict=None,
                           iqtree_incorrectly_predicted_sites_dict = iqtree_incorrectly_predicted_sites_dict,
                           true_alignment_length=len(true_alignment[0].seq),
                           plot_name=plot_name,
                           plot_folder_name=plot_load2.plot_folder_name,
                           name = plot_load2.name,
                           sequence_input_type=plot_load2.sequence_input_type,
                           test_mode=plot_load2.test_mode)

    draupnir_average_percent_id = (plot_load2.draupnir_alig_length-draupnir_average_incorrectly_predicted_sites)*100/plot_load2.draupnir_alig_length
    draupnir_std_percent_id = draupnir_std_incorrectly_predicted_sites

    plot_load_percentid = PlotLoad3(paml_incorrectly_predicted_sites_dict=None,
                           draupnir_average_incorrectly_predicted_sites=draupnir_average_percent_id,
                           draupnir_std_incorrectly_predicted_sites=draupnir_std_percent_id,
                           phylobayes_incorrectly_predicted_sites_dict=None,
                           fastml_incorrectly_predicted_sites_dict=None,
                           iqtree_incorrectly_predicted_sites_dict=iqtree_percent_id_dict,
                           true_alignment_length=100,
                           plot_name=plot_name,
                           plot_folder_name=plot_load2.plot_folder_name ,
                           name=plot_load2.name + "%ID",
                           sequence_input_type=plot_load2.sequence_input_type,
                           test_mode=plot_load2.test_mode)

    n_plots = DraupnirUtils.Define_batch_size(len(draupnir_average_percent_id),
                                              batch_size=False, benchmarking=True)  # sometimes we cannot divide equally
    if n_plots >= 2:
        splitted_plot_800(plot_load_incorrect_aa, n_plots)
    else:
        small_plot(plot_load_incorrect_aa)
    comparison_table(plot_load_incorrect_aa,big_data=True)
    percendid_table(plot_load_percentid,big_data=True)
    writeMSA('{}/{}/Draupnir_{}_{}.fasta'.format(plot_load2.plot_folder_name,save_folder, name, test_mode),
             plot_load.draupnir_sequences_predictions_msa, format='fasta')
def save_sequences(plot_load,plot_load2,consider_gaps_predictions=True):
    "For datasets without true ancestral sequences, we just save the parsed and ordered sequences from all programs"
    print("Saving parsed sequences...")
    paml_predictions_file = plot_load.paml_predictions_file
    ancestor_info = plot_load.ancestor_info
    correspondence_dict_paml_to_sim = plot_load2.correspondence_dict_paml_to_sim
    rst_file = plot_load.rst_file
    correspondence_dict_sim_to_paml = plot_load2.correspondence_dict_sim_to_paml
    phylobayes_dict = plot_load2.phylobayes_dict
    fastml_dict = plot_load2.fastml_dict
    gap_positions = plot_load.gap_positions
    name = plot_load.name
    correspondence_original_to_iqtree = plot_load2.correspondence_original_to_iqtree
    correspondence_iqtree_to_original = plot_load2.correspondence_iqtree_to_original
    correspondence_fastml_to_original = plot_load2.correspondence_fastml_to_original
    correspondence_original_to_fastml = plot_load2.correspondence_original_to_fastml
    #plot_name = "{}_Dataset{}_{}".format(name,plot_load.dataset_number,gaps)
    plot_name = plot_load2.plot_full_name
    sequence_input_type = plot_load2.sequence_input_type
    test_mode = plot_load2.test_mode
    paml_predicted_alignment = AlignIO.read(paml_predictions_file, "fasta")

    #Highlight: order predicted alignment nodes according to draupnir results
    #Highlight: Find back the correspondence of the nodes names
    ancestor_info["0"] = ancestor_info["0"].astype(str)
    ancestor_info.drop('Unnamed: 0', inplace=True, axis=1)
    nodes_names = ancestor_info["0"].tolist()
    #
    draupnir_correspondence_dict = {node:i for i, node in enumerate(nodes_names) if re.search('^A{1}[0-9]+(?![A-Z])+', node)}
    draupnir_correspondence_dict_reverse = {i: node for i, node in enumerate(nodes_names) if re.search('^A{1}[0-9]+(?![A-Z])+', node)}
    draupnir_ancestors_fasta_alignment = SeqIO.parse(plot_load.draupnir_incorrectly_predicted_sites,"fasta")
    draupnir_nodes_order = [float(seq.id.split("//")[1].split("_")[0]) for seq in draupnir_ancestors_fasta_alignment]
    draupnir_nodes_order_reverse = [key for key, value in draupnir_correspondence_dict.items() if value in draupnir_nodes_order] #get the names

    #Highlight: PAML CodeML ##########################################################################################################

    #paml_predicted_alignment_dict = dict(zip([correspondence_dict_paml_to_sim[seq.id.replace("node","")] for seq in paml_predicted_alignment], [seq.seq for seq in paml_predicted_alignment]))
    paml_predicted_alignment_dict = {correspondence_dict_paml_to_sim[seq.id.replace("node","")]:seq.seq for seq in paml_predicted_alignment}
    paml_predicted_alignment_dict_ordered = dict(zip(draupnir_nodes_order,[val for key,val in paml_predicted_alignment_dict.items() if key in draupnir_nodes_order]))
    if not consider_gaps_predictions:  # if there is any gap in the leaves alignment, ignore that site , replace with gap
        def build_site():
            if highest_prob_dict[idx][2]:
                site = "-"
            else:
                site = highest_prob_dict[idx][0]
            return site
    else:
        def build_site():
            site = highest_prob_dict[idx][0]
            return site

    if sequence_input_type == "DNA":
        def esp(rst, index):
            return Extract_Sites_Probabilities_Codons(rst, index)
    else:
        def esp(rst, index):
            return Extract_Sites_Probabilities(rst, index)
    paml_sequences_predictions_dict = defaultdict()
    for equivalent_true_node_number,pred_seq in paml_predicted_alignment_dict_ordered.items():
        highest_prob_dict = esp(rst_file, int(correspondence_dict_sim_to_paml["I"+str(int(equivalent_true_node_number))]))
        if sequence_input_type=="DNA":
            pred_seq = translate(pred_seq)
        fixed_predicted_seq = []
        for idx,site in enumerate(pred_seq,1):
            site = build_site()
            fixed_predicted_seq.append(site)
        fixed_predicted_seq = "".join(fixed_predicted_seq)
        draupnir_node_name = draupnir_correspondence_dict_reverse[equivalent_true_node_number]
        paml_sequences_predictions_dict[draupnir_node_name] = "".join(fixed_predicted_seq)

    dict_to_fasta(paml_sequences_predictions_dict,"{}/{}/PAML_predictions_{}_{}.fasta".format(plot_load2.plot_folder_name, save_folder,name,sequence_input_type))
    #Highlight: FastML###########################################################################
    # consider_gaps_predictions_fastml = True
    if not consider_gaps_predictions:
        def build_seq(in_seq):
            " Correct prediction, re-assign gaps, sicen fastml does not handle gaps"
            out_seq = "".join(["-" if idx in gap_positions else letter for idx, letter in enumerate(
                in_seq)])  # correct prediction, re-assign gaps, sicen fastml does not handle gaps
            return out_seq
    else:
        def build_seq(in_seq):
            out_seq = "".join([letter for idx, letter in enumerate(in_seq)])
            return out_seq
    fastml_sequences_predictions_dict = defaultdict()

    if plot_load2.fastml_dict is not None:
        #fastml_dict_ordered = {key: fastml_dict[key] for key in draupnir_nodes_order_reverse if key in fastml_dict.keys()}
        fastml_dict_ordered = defaultdict()
        for key in draupnir_nodes_order_reverse:
            fastml_seq = fastml_dict[correspondence_original_to_fastml[key.replace("A","I")]]
            fastml_dict_ordered[key] = fastml_seq
        for internal_node,fastml_seq in fastml_dict_ordered.items():
            fastml_seq = build_seq(fastml_seq)
            fastml_sequences_predictions_dict[internal_node] = fastml_seq
        dict_to_fasta(fastml_sequences_predictions_dict,"{}/{}/FastML_predictions_{}_{}.fasta".format(plot_load2.plot_folder_name,save_folder,name,sequence_input_type))
    else:
        print("fastml not found!!!")
    #Highlight: PhyloBayes############################################################################

    phylobayes_dict_ordered = {key: phylobayes_dict[key] for key in draupnir_nodes_order_reverse if
                               key in phylobayes_dict.keys()}

    #Highlight: we need to realign the sequences, because of some stop codons--> THIS IS A PROBLEM FOR COMPARISON OF PREDICTIONS-Mutual Information
    # true_alignment_dict, phylobayes_dict_ordered = merge_to_align_simulationsDNA(true_alignment_dict,
    #                                                                                       phylobayes_dict_ordered,
    #                                                                                       plot_load2, "Phylobayes")
    phylobayes_sequences_predictions_dict = defaultdict()
    for internal_node,phylo_bayes_seq in phylobayes_dict_ordered.items():
        phylo_bayes_seq = build_seq(phylo_bayes_seq)
        phylobayes_sequences_predictions_dict[internal_node] = phylo_bayes_seq
    #Highlight: PhyloBayes is missing some nodes (those internal not calculated from distance among leaves), we assign them to not predicted
    for key in paml_sequences_predictions_dict.keys():
        if key not in phylobayes_sequences_predictions_dict.keys():
            if sequence_input_type == "DNA":
                phylobayes_sequences_predictions_dict[key] = "-"*int(len(paml_predicted_alignment_dict_ordered[key])//3)
            else:
                phylobayes_sequences_predictions_dict[key] = "-"*len(paml_predicted_alignment_dict_ordered[key])


    #Highlight: Order again:
    # phylobayes_sequences_predictions_dict = {key: phylobayes_sequences_predictions_dict[key] for key in draupnir_nodes_order_reverse if key in phylobayes_incorrectly_predicted_sites_dict.keys()}
    #
    dict_to_fasta(phylobayes_sequences_predictions_dict,
                  "{}/{}/PhyloBayes_predictions_{}_{}.fasta".format(plot_load2.plot_folder_name,save_folder, name,sequence_input_type))
    #Highlight: Iqtree ---> also missing some node's predictions
    iqtree_sequences_predictions_dict = defaultdict()
    iqtree_dict = plot_load2.iqtree_dict

    for key in draupnir_nodes_order_reverse:
        if key in iqtree_dict.keys():
            iqtree_prediction = iqtree_dict[key]
            iqtree_sequences_predictions_dict[key] = "".join(iqtree_prediction)
        else:
            print("Missing key in IQtree {}".format(key))
            if sequence_input_type == "DNA":
                iqtree_sequences_predictions_dict[key] = "-" * int(len(paml_predicted_alignment_dict_ordered[key])//3)
            else:
                iqtree_sequences_predictions_dict[key] = "-" * len(paml_predicted_alignment_dict_ordered[key])
    dict_to_fasta(iqtree_sequences_predictions_dict,"{}/{}/IQTree_predictions_{}_{}.fasta".format(plot_load2.plot_folder_name,save_folder, name,sequence_input_type))
    #Highlight: Draupnir ##############################################################################################################

    writeMSA('{}/{}/Draupnir_{}_{}.fasta'.format(plot_load2.plot_folder_name,save_folder, name, test_mode),
             plot_load.draupnir_sequences_predictions_msa, format='fasta')
def fasta_reader(file_name):
    "Parses my faulty fasta format (apparently the sequences are not formatted, they are just in 1 liner, I did it like this to make it faster) into a dictionary "
    fasta_dict = {}
    with open(file_name) as file_one:
        for line in file_one:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                active_sequence_name = line[1:]
                if active_sequence_name not in fasta_dict:
                    fasta_dict[active_sequence_name] = []
                continue
            sequence = line
            fasta_dict[active_sequence_name].append(sequence)
    return fasta_dict

def load(name, full_name, simulation_folder,dataset_number,root_sequence_name, sequence_input_type, test_mode,draupnir_alig_length):
    if use_GRU:
        test_folder, plot_only_sample, large_dataset = Draupnir_folders_results_GRU[name]
    else:
        pass

    if name.startswith("simulations"):

        plot_folder_name = os.path.join(script_dir_level_up,"Benchmark_Plots2/{}/Dataset{}".format(simulation_folder,dataset_number))
        short_name = re.split(r"_[0-9]+",name.split("simulations_")[1])[0]

        plot_load = PlotLoad(name=name,
                             dataset_number=dataset_number,
                             simulation_folder=simulation_folder,
                             root_sequence_name=root_sequence_name,
                             rst_file = os.path.join(script_dir,"CodeML/{}/Dataset{}/{}/rst".format(simulation_folder,dataset_number,sequence_input_type)),
                             true_ancestral_file=os.path.join(script_dir_2level_up,"Datasets_Simulations/{}/Dataset{}/{}_pep_Internal_Nodes_True_alignment.FASTA".format(simulation_folder, dataset_number, root_sequence_name)),
                             draupnir_incorrectly_predicted_sites=np.load("{}/{}_Plots/Incorrectly_Predicted_Sites.npy".format(test_folder,test_mode)), #[n_samples+1,incorrect_sites]
                             draupnir_incorrectly_predicted_sites_complete=np.load("{}/{}_Plots/Incorrectly_Predicted_Sites_Fast.npy".format(test_folder,test_mode)),#[n_samples,n_nodes,[node, incorrect]]
                             draupnir_sequences_predictions_msa=parseMSA("{}/{}_Plots/{}_sampled_nodes_seq.fasta".format(test_folder,test_mode,name)),
                             gap_positions=np.load("{}/Alignment_gap_positions.npy".format(test_folder)),
                             sites_count=pickle.load(open("{}/Sites_count.p".format(test_folder), 'rb')),
                             tree_by_levels=pickle.load(open("{}/Tree_by_levels_dict.p".format(test_folder), 'rb')),
                             children_dict=pickle.load(open("{}/Children_dict.p".format(test_folder), "rb")),
                             ancestor_info=pd.read_csv(os.path.join(script_dir_2level_up,"Datasets_Folder/Tree_LevelOrderInfo_Integers_simulations_{}_{}.csv".format(short_name, dataset_number)), sep="\t", index_col=False, low_memory=False),
                             paml_predictions_file = os.path.join(script_dir,"CodeML/{}/Dataset{}/{}/{}_{}_anc_seq.fasta".format(simulation_folder, dataset_number, sequence_input_type, simulation_folder, dataset_number))

        )

        if large_dataset:
            iqtree_dict, correspondence_iqtree_to_original, correspondence_original_to_iqtree = parse_IQTree(name,
                                                                                                             simulation_folder,
                                                                                                             dataset_number,
                                                                                                             root_sequence_name,
                                                                                                             sequence_input_type,
                                                                                                             plot_load.sites_count,
                                                                                                             script_dir_level_up,
                                                                                                             script_dir_2level_up,
                                                                                                             indel_calculation=False)
            plot_load2 = PlotLoad2(fastml_dict=None,
                                   phylobayes_dict=None,
                                   iqtree_dict=iqtree_dict,
                                   correspondence_dict_sim_to_paml=None,
                                   correspondence_dict_paml_to_sim=None,
                                   correspondence_fastml_to_original=None,
                                   correspondence_original_to_fastml=None,
                                   correspondence_iqtree_to_original=correspondence_iqtree_to_original,
                                   correspondence_original_to_iqtree=correspondence_original_to_iqtree,
                                   plot_folder_name=plot_folder_name,
                                   plot_full_name=full_name,
                                   name=name,
                                   sequence_input_type=sequence_input_type,
                                   test_mode=test_mode,
                                   draupnir_alig_length=draupnir_alig_length,
                                   plot_only_sample=plot_only_sample)
            plot_incorrectly_predicted_aa_simulations_800(plot_load, plot_load2)

        else:
            correspondence_dict_sim_to_paml, correspondence_dict_paml_to_sim = Load_Tree(plot_load.rst_file, name,
                                                                                             simulation_folder,
                                                                                             dataset_number,sequence_input_type)
            if not os.path.exists(plot_load.paml_predictions_file):
                Save_Sequences(plot_load.rst_file, plot_load.paml_predictions_file, plot_load.name)
            phylobayes_dict = parse_PhyloBayes(name, simulation_folder, dataset_number, root_sequence_name,sequence_input_type,script_dir_level_up, script_dir_2level_up)
            if name in ["simulations_PIGBOS_1","simulations_insulin_2"]: #could not run FastML on them
                fastml_dict = correspondence_fastml_to_original = correspondence_original_to_fastml = None
            else:
                fastml_dict, correspondence_fastml_to_original, correspondence_original_to_fastml = parse_FastML(name,
                                                                                                                 simulation_folder,
                                                                                                                 dataset_number,
                                                                                                                 root_sequence_name,sequence_input_type,script_dir_level_up, script_dir_2level_up)
            iqtree_dict,correspondence_iqtree_to_original,correspondence_original_to_iqtree = parse_IQTree(name, simulation_folder,dataset_number,root_sequence_name,
                                                                                                           sequence_input_type,plot_load.sites_count,script_dir_level_up, script_dir_2level_up,indel_calculation=False)
            plot_load2 = PlotLoad2(fastml_dict=fastml_dict,
                                   phylobayes_dict=phylobayes_dict,
                                   iqtree_dict= iqtree_dict,
                                   correspondence_dict_sim_to_paml=correspondence_dict_sim_to_paml,
                                   correspondence_dict_paml_to_sim=correspondence_dict_paml_to_sim,
                                   correspondence_fastml_to_original=correspondence_fastml_to_original,
                                   correspondence_original_to_fastml=correspondence_original_to_fastml,
                                   correspondence_iqtree_to_original = correspondence_iqtree_to_original,
                                   correspondence_original_to_iqtree = correspondence_original_to_iqtree,
                                   plot_folder_name=plot_folder_name,
                                   plot_full_name = full_name,
                                   name=name,
                                   sequence_input_type= sequence_input_type,
                                   test_mode=test_mode,
                                   draupnir_alig_length = draupnir_alig_length,
                                   plot_only_sample = plot_only_sample)
            plot_incorrectly_predicted_aa_simulations(plot_load, plot_load2)
    elif name == "benchmark_randall_original_naming":

        plot_folder_name = os.path.join(script_dir_level_up,"Benchmark_Plots2/AncestralResurrectionStandardDataset")
        plot_load = PlotLoad(name=name,
                             dataset_number=None,
                             simulation_folder=None,
                             root_sequence_name=None,
                             rst_file = os.path.join(script_dir,"CodeML/AncestralResurrectionStandardDataset/{}/rst".format(sequence_input_type)),
                             true_ancestral_file=os.path.join(script_dir_2level_up,"AncestralResurrectionStandardDataset/RandallExperimentalPhylogenyAASeqsINTERNAL.fasta"),
                             draupnir_incorrectly_predicted_sites=np.load("{}/{}_Plots/Incorrectly_Predicted_Sites.npy".format(test_folder,test_mode)),
                             draupnir_incorrectly_predicted_sites_complete=np.load("{}/{}_Plots/Incorrectly_Predicted_Sites_Fast.npy".format(test_folder, test_mode)),
                             draupnir_sequences_predictions_msa = parseMSA("{}/{}_Plots/{}_sampled_nodes_seq.fasta".format(test_folder, test_mode,name)),
                             gap_positions=np.load("{}/Alignment_gap_positions.npy".format(test_folder)),
                             sites_count=pickle.load(open("{}/Sites_count.p".format(test_folder), 'rb')),
                             tree_by_levels=pickle.load(open("{}/Tree_by_levels_dict.p".format(test_folder), 'rb')),
                             children_dict=pickle.load(open("{}/Children_dict.p".format(test_folder), "rb")),
                             ancestor_info=pd.read_csv(os.path.join(script_dir_2level_up,"Datasets_Folder/Tree_LevelOrderInfo_Integers_benchmark_randall_original_naming.csv"),sep="\t", index_col=False, low_memory=False),
                             paml_predictions_file = os.path.join(script_dir,"CodeML/AncestralResurrectionStandardDataset/{}/benchmark_randall_original_naming_anc_seq.fasta".format(sequence_input_type))
                             )

        correspondence_dict_sim_to_paml, correspondence_dict_paml_to_sim = Load_Tree(plot_load.rst_file,name,simulation_folder,dataset_number,sequence_input_type)

        #if not os.path.exists(plot_load.paml_predictions_file):
        Save_Sequences(plot_load.rst_file, plot_load.paml_predictions_file, plot_load.name)
        phylobayes_dict = parse_PhyloBayes(name,simulation_folder, dataset_number, root_sequence_name,sequence_input_type,script_dir_level_up,script_dir_2level_up)
        fastml_dict,correspondence_fastml_to_original,correspondence_original_to_fastml = parse_FastML(name,simulation_folder, dataset_number, root_sequence_name,sequence_input_type,script_dir_level_up,script_dir_2level_up)

        if sample_fastml:
            sample_FastML(name,plot_folder_name,
                            simulation_folder,
                            dataset_number,
                            root_sequence_name,
                            sequence_input_type,
                            correspondence_fastml_to_original,
                            script_dir_level_up,
                            script_dir_2level_up)
        iqtree_dict,correspondence_iqtree_to_original,correspondence_original_to_iqtree = parse_IQTree(name, simulation_folder, dataset_number, root_sequence_name,
                                                                                                       sequence_input_type,plot_load.gap_positions,script_dir_level_up,script_dir_2level_up,indel_calculation=False)
        plot_load2 = PlotLoad2(fastml_dict=fastml_dict,
                               phylobayes_dict=phylobayes_dict,
                               iqtree_dict=iqtree_dict,
                               correspondence_dict_sim_to_paml=correspondence_dict_sim_to_paml,
                               correspondence_dict_paml_to_sim=correspondence_dict_paml_to_sim,
                               correspondence_fastml_to_original=correspondence_fastml_to_original,
                               correspondence_original_to_fastml=correspondence_original_to_fastml,
                               correspondence_iqtree_to_original=correspondence_iqtree_to_original,
                               correspondence_original_to_iqtree=correspondence_original_to_iqtree,
                               plot_folder_name=plot_folder_name,
                               plot_full_name = full_name,
                               name = name,
                               sequence_input_type= sequence_input_type,
                               test_mode=test_mode,
                               draupnir_alig_length = draupnir_alig_length,
                               plot_only_sample = False)

        plot_incorrectly_predicted_aa_randall(plot_load,plot_load2)
    elif name.startswith("Coral"):

        plot_folder_name = os.path.join(script_dir_level_up,"Benchmark_Plots2/{}".format(name))
        plot_load = PlotLoad(name=name,
                             dataset_number=None,
                             simulation_folder=None,
                             root_sequence_name=None,
                             rst_file= os.path.join(script_dir,"CodeML/{}/{}/rst".format(name, sequence_input_type)),
                             true_ancestral_file=os.path.join(script_dir_2level_up,"GPFCoralDataset/Ancestral_Sequences.fasta"),
                             draupnir_incorrectly_predicted_sites=np.load("{}/{}_Plots/Incorrectly_Predicted_Sites.npy".format(test_folder, test_mode)),
                             draupnir_incorrectly_predicted_sites_complete=None,
                             draupnir_sequences_predictions_msa=parseMSA("{}/{}_Plots/test_samples_plus_train_true_aligned.fasta".format(test_folder, test_mode, name)),
                             gap_positions=np.load("{}/Alignment_gap_positions.npy".format(test_folder)),
                             sites_count=pickle.load(open("{}/Sites_count.p".format(test_folder), 'rb')),
                             tree_by_levels=pickle.load(open("{}/Tree_by_levels_dict.p".format(test_folder), 'rb')),
                             children_dict=pickle.load(open("{}/Children_dict.p".format(test_folder), "rb")),
                             ancestor_info=pd.read_csv(os.path.join(script_dir_2level_up,"Datasets_Folder/Tree_LevelOrderInfo_Integers_{}.csv".format(name)), sep="\t", index_col=False, low_memory=False),
                             paml_predictions_file = os.path.join(script_dir,"CodeML/{}/{}/{}_anc_seq.fasta".format(name, sequence_input_type, name)),

        )

        correspondence_dict_sim_to_paml, correspondence_dict_paml_to_sim = Load_Tree(plot_load.rst_file, name,
                                                                                     simulation_folder, dataset_number,sequence_input_type)

        # if not os.path.exists(plot_load.paml_predictions_file):
        Save_Sequences(plot_load.rst_file, plot_load.paml_predictions_file, plot_load.name)


        fastml_dict, correspondence_fastml_to_original, correspondence_original_to_fastml = parse_FastML(name,
                                                                                                         simulation_folder,
                                                                                                         dataset_number,
                                                                                                         root_sequence_name,
                                                                                                         sequence_input_type,script_dir_level_up,script_dir_2level_up)
        if sample_fastml:
            # sample_IQTree(name, simulation_folder, dataset_number, root_sequence_name, sequence_input_type,
            #               plot_load.sites_count,
            #               script_dir_level_up, script_dir_2level_up, plot_folder_name, indel_calculation=True)
            sample_FastML(name,plot_folder_name,
                            simulation_folder,
                            dataset_number,
                            root_sequence_name,
                            sequence_input_type,
                            correspondence_fastml_to_original,
                            script_dir_level_up,
                            script_dir_2level_up)

        phylobayes_dict = parse_PhyloBayes(name, simulation_folder, dataset_number, root_sequence_name,
                                           sequence_input_type,script_dir_level_up,script_dir_2level_up)
        iqtree_dict,correspondence_iqtree_to_original,correspondence_original_to_iqtree = parse_IQTree(name, simulation_folder, dataset_number,
                                                                                                       root_sequence_name, sequence_input_type,
                                                                                                       plot_load.gap_positions,script_dir_level_up,script_dir_2level_up,indel_calculation=False)
        plot_load2 = PlotLoad2(fastml_dict=fastml_dict,
                               phylobayes_dict=phylobayes_dict,
                               iqtree_dict=iqtree_dict,
                               correspondence_dict_sim_to_paml=correspondence_dict_sim_to_paml,
                               correspondence_dict_paml_to_sim=correspondence_dict_paml_to_sim,
                               correspondence_fastml_to_original=correspondence_fastml_to_original,
                               correspondence_original_to_fastml=correspondence_original_to_fastml,
                               correspondence_iqtree_to_original=correspondence_iqtree_to_original,
                               correspondence_original_to_iqtree=correspondence_original_to_iqtree,
                               plot_folder_name=plot_folder_name,
                               plot_full_name = full_name,
                               name=name,
                               sequence_input_type= sequence_input_type,
                               test_mode=test_mode,
                               draupnir_alig_length = [len(seq) for seq in SeqIO.parse("{}/{}_Plots/samples_plus_true_aligned.fasta".format(test_folder,test_mode),"fasta")][0],
                               plot_only_sample = False)

        plot_incorrectly_predicted_coral(plot_load, plot_load2)
    elif name.endswith("_subtree"):

        plot_folder_name = os.path.join(script_dir_level_up,"Benchmark_Plots2/{}".format(name))
        plot_load = PlotLoad(name=name,
                             dataset_number=None,
                             simulation_folder=None,
                             root_sequence_name=None,
                             #rst_file="/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_results/PAML/CodeML/{}/{}/rst".format(name,sequence_input_type),
                             rst_file=os.path.join(script_dir,"CodeML/{}/{}/rst".format(name, sequence_input_type)),
                             #true_ancestral_file="/home/lys/Dropbox/PhD/DRAUPNIR/Douglas_SRC_Dataset/Ancs.fasta",
                             true_ancestral_file= os.path.join(script_dir_2level_up,"Douglas_SRC_Dataset/Ancs.fasta"),
                             draupnir_incorrectly_predicted_sites=np.load("{}/{}_Plots/Incorrectly_Predicted_Sites.npy".format(test_folder, test_mode)),
                             draupnir_incorrectly_predicted_sites_complete=None,
                             draupnir_sequences_predictions_msa=parseMSA("{}/{}_Plots/{}_sampled_nodes_seq.fasta".format(test_folder, test_mode, name)),
                             gap_positions=np.load("{}/Alignment_gap_positions.npy".format(test_folder)),
                             sites_count=pickle.load(open("{}/Sites_count.p".format(test_folder), 'rb')),
                             tree_by_levels=pickle.load(open("{}/Tree_by_levels_dict.p".format(test_folder), 'rb')),
                             children_dict=pickle.load(open("{}/Children_dict.p".format(test_folder), "rb")),
                             #ancestor_info=pd.read_csv("/home/lys/Dropbox/PhD/DRAUPNIR/Datasets_Folder/Tree_LevelOrderInfo_Integers_{}.csv".format(name),sep="\t", index_col=False, low_memory=False),
                             ancestor_info=pd.read_csv(os.path.join(script_dir_2level_up,"Datasets_Folder/Tree_LevelOrderInfo_Integers_{}.csv".format(name)), sep="\t", index_col=False, low_memory=False),
                             #paml_predictions_file="/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_results/PAML/CodeML/{}/{}/{}_anc_seq.fasta".format(name,sequence_input_type,name)
                             paml_predictions_file = os.path.join(script_dir,"CodeML/{}/{}/{}_anc_seq.fasta".format(name, sequence_input_type, name))

        )

        correspondence_dict_sim_to_paml, correspondence_dict_paml_to_sim = Load_Tree(plot_load.rst_file, name,
                                                                                     simulation_folder, dataset_number,sequence_input_type)

        # if not os.path.exists(plot_load.paml_predictions_file):
        Save_Sequences(plot_load.rst_file, plot_load.paml_predictions_file, plot_load.name)
        phylobayes_dict = parse_PhyloBayes(name, simulation_folder, dataset_number, root_sequence_name,
                                           sequence_input_type,script_dir_level_up,script_dir_2level_up)

        fastml_dict, correspondence_fastml_to_original, correspondence_original_to_fastml = parse_FastML(name,
                                                                                                         simulation_folder,
                                                                                                         dataset_number,
                                                                                                         root_sequence_name,
                                                                                                         sequence_input_type,script_dir_level_up,script_dir_2level_up)
        if sample_fastml:
            sample_IQTree(name, simulation_folder, dataset_number, root_sequence_name, sequence_input_type, plot_load.sites_count,
                          script_dir_level_up, script_dir_2level_up, plot_folder_name, indel_calculation=False)
            sample_FastML(name,plot_folder_name,
                            simulation_folder,
                            dataset_number,
                            root_sequence_name,
                            sequence_input_type,
                            correspondence_fastml_to_original,
                            script_dir_level_up,
                            script_dir_2level_up)

        iqtree_dict,correspondence_iqtree_to_original,correspondence_original_to_iqtree = parse_IQTree(name, simulation_folder, dataset_number,
                                                                                                       root_sequence_name, sequence_input_type,
                                                                                                       plot_load.gap_positions,script_dir_level_up,script_dir_2level_up,indel_calculation=False)
        plot_load2 = PlotLoad2(fastml_dict=fastml_dict,
                               phylobayes_dict=phylobayes_dict,
                               iqtree_dict=iqtree_dict,
                               correspondence_dict_sim_to_paml=correspondence_dict_sim_to_paml,
                               correspondence_dict_paml_to_sim=correspondence_dict_paml_to_sim,
                               correspondence_fastml_to_original=correspondence_fastml_to_original,
                               correspondence_original_to_fastml=correspondence_original_to_fastml,
                               correspondence_iqtree_to_original=correspondence_iqtree_to_original,
                               correspondence_original_to_iqtree=correspondence_original_to_iqtree,
                               plot_folder_name=plot_folder_name,
                               plot_full_name = full_name,
                               name  = name,
                               sequence_input_type= sequence_input_type,
                               test_mode=test_mode,
                               draupnir_alig_length = [len(seq) for seq in SeqIO.parse("{}/{}_Plots/test_samples_plus_train_true_aligned.fasta".format(test_folder,test_mode),"fasta")][0],
                               plot_only_sample = False)

        plot_incorrectly_predicted_kinases(plot_load, plot_load2)
    elif name in ["PF00400","SH3_pf00018_larger_than_30aa","PF00096","PF00400_200","aminopeptidase","PKinase_PF07714","Douglas_SRC","Cnidarian"]: #datasets with NO ancestral sequences
        plot_folder_name = os.path.join(script_dir_level_up,"Benchmark_Plots2/{}".format(name))

        plot_load = PlotLoad(name=name,
                             dataset_number=dataset_number,
                             simulation_folder=simulation_folder,
                             root_sequence_name=root_sequence_name,
                             rst_file = os.path.join(script_dir,"CodeML/{}/{}/rst".format(name,sequence_input_type)),
                             true_ancestral_file=None,
                             draupnir_incorrectly_predicted_sites="{}/Test_argmax_Plots/{}_sampled_ancestors_seq.fasta".format(test_folder,name), #HIGHLIGHT: watch out, reusing this one!!
                             draupnir_incorrectly_predicted_sites_complete=None,#[n_samples,n_nodes,[node, incorrect]]
                             draupnir_sequences_predictions_msa=parseMSA("{}/{}_Plots/{}_sampled_ancestors_seq.fasta".format(test_folder,test_mode,name)),
                             gap_positions=np.load("{}/Alignment_gap_positions.npy".format(test_folder)),
                             sites_count=pickle.load(open("{}/Sites_count.p".format(test_folder), 'rb')),
                             tree_by_levels=pickle.load(open("{}/Tree_by_levels_dict.p".format(test_folder), 'rb')),
                             children_dict=pickle.load(open("{}/Children_dict.p".format(test_folder), "rb")),
                             ancestor_info=pd.read_csv(os.path.join(script_dir_2level_up,"Datasets_Folder/Tree_LevelOrderInfo_Integers_{}.csv".format(name)), sep="\t", index_col=False, low_memory=False),
                             paml_predictions_file = os.path.join(script_dir,"CodeML/{}/{}/{}_anc_seq.fasta".format(name,sequence_input_type,name)))
        correspondence_dict_sim_to_paml, correspondence_dict_paml_to_sim = Load_Tree(plot_load.rst_file, name,
                                                                                     simulation_folder, dataset_number,sequence_input_type)

        Save_Sequences(plot_load.rst_file, plot_load.paml_predictions_file, plot_load.name)
        fastml_dict, correspondence_fastml_to_original, correspondence_original_to_fastml = parse_FastML(name,
                                                                                                         simulation_folder,
                                                                                                         dataset_number,
                                                                                                         root_sequence_name,
                                                                                                  sequence_input_type,script_dir_level_up,script_dir_2level_up)

        if sample_fastml:
            #sample_IQTree(name, simulation_folder, dataset_number, root_sequence_name, sequence_input_type, plot_load.sites_count,script_dir_level_up, script_dir_2level_up, plot_folder_name, indel_calculation=True)
            sample_FastML(name,plot_folder_name,
                            simulation_folder,
                            dataset_number,
                            root_sequence_name,
                            sequence_input_type,
                            correspondence_fastml_to_original,
                            script_dir_level_up,
                            script_dir_2level_up)
        exit()
        iqtree_dict,correspondence_iqtree_to_original,correspondence_original_to_iqtree = parse_IQTree(name, simulation_folder, dataset_number,
                                                                                                       root_sequence_name, sequence_input_type,
                                                                                                       plot_load.sites_count,script_dir_level_up,script_dir_2level_up,indel_calculation=False)

        phylobayes_dict = parse_PhyloBayes(name, simulation_folder, dataset_number, root_sequence_name,sequence_input_type,script_dir_level_up,script_dir_2level_up)

        fastml_dict, correspondence_fastml_to_original, correspondence_original_to_fastml = parse_FastML(name,
                                                                                                         simulation_folder,
                                                                                                         dataset_number,
                                                                                                         root_sequence_name,
                                                                                                         sequence_input_type,script_dir_level_up,script_dir_2level_up)





        plot_load2 = PlotLoad2(fastml_dict=fastml_dict,
                               phylobayes_dict=phylobayes_dict,
                               iqtree_dict=iqtree_dict,
                               correspondence_dict_sim_to_paml=correspondence_dict_sim_to_paml,
                               correspondence_dict_paml_to_sim=correspondence_dict_paml_to_sim,
                               correspondence_fastml_to_original=correspondence_fastml_to_original,
                               correspondence_original_to_fastml=correspondence_original_to_fastml,
                               correspondence_iqtree_to_original=correspondence_iqtree_to_original,
                               correspondence_original_to_iqtree=correspondence_original_to_iqtree,
                               plot_folder_name=plot_folder_name,
                               plot_full_name=full_name,
                               name=name,
                               sequence_input_type=sequence_input_type,
                               test_mode=test_mode,
                               draupnir_alig_length=draupnir_alig_length,
                               plot_only_sample=plot_only_sample)
        save_sequences(plot_load,plot_load2)
    else:
        print("not in the options")


def main(datasets_to_analyze):
    input_types = ["PROTEIN","DNA"]
    test_modes = ["Test2_argmax","Test"] #Test2_argmax contains the Draupnir MAP, Test contains the Draupnir marginal samples

    for dataset_number in datasets_to_analyze:
        name,dataset_number,simulation_folder,root_sequence_name = datasets[dataset_number]
        full_name = datasets_full_names[name]
        print("Using dataset {}".format(name))
        dataset = np.load(os.path.join(script_dir_2level_up,"Datasets_Folder/Dataset_numpy_aligned_Integers_{}.npy".format(name)),allow_pickle=True)

        draupnir_alig_length = dataset.shape[1]-3
        if name.endswith("_subtree") or name in ["PF00400","SH3_pf00018_larger_than_30aa"]: #only protein sequences are available
            sequence_input_type = "PROTEIN"
            for test_mode in test_modes:
                load(name, full_name, simulation_folder, dataset_number, root_sequence_name, sequence_input_type,
                     test_mode, draupnir_alig_length)
        else:
            for sequence_input_type in input_types:
                for test_mode in test_modes:
                    load(name, full_name, simulation_folder, dataset_number, root_sequence_name, sequence_input_type, test_mode,draupnir_alig_length)


if __name__ == "__main__":
    datasets = {0: ["benchmark_randall", None, None, None],  # the tree is inferred
                1: ["benchmark_randall_original", None, None, None],
                # uses the original tree but changes the naming of the nodes (because the original tree was not rooted)
                2: ["benchmark_randall_original_naming", None, None, None],
                # uses the original tree and it's original node naming
                3: ["SH3_pf00018_larger_than_30aa", None, None, None],
                # SRC kinases domain SH3 ---> Leaves and angles testing
                4: ["simulations_blactamase_1", 1, "BLactamase", "BetaLactamase_seq"],
                # EvolveAGene4 Betalactamase simulation # 32 leaves
                5: ["simulations_src_sh3_1", 1, "SRC_simulations", "SRC_SH3"],
                # EvolveAGene4 SRC SH3 domain simulation 1 #100 leaves
                6: ["simulations_src_sh3_2", 2, "SRC_simulations", "SRC_SH3"],
                # EvolveAGene4 SRC SH3 domain simulation 2 #800 leaves
                7: ["simulations_src_sh3_3", 3, "SRC_simulations", "SRC_SH3"],
                # EvolveAGene4 SRC SH3 domain simulation 2 #200 leaves
                8: ["simulations_sirtuins_1", 1, "Sirtuin_simulations", "Sirtuin_seq"],
                # EvolveAGene4 Sirtuin simulation #150 leaves
                9: ["simulations_calcitonin_1", 1, "Calcitonin_simulations", "Calcitonin_seq"],
                # EvolveAGene4 Calcitonin simulation #50 leaves
                10: ["simulations_mciz_1", 1, "Mciz_simulations", "Mciz_seq"],
                # EvolveAGene4 MciZ simulation # 1600 leaves
                11: ["Douglas_SRC", None, None, None],
                # Douglas's Full SRC Kinases #Highlight: The tree is not similar to the one in the paper, therefore the sequences where splitted in subtrees according to the ancetral sequences in the paper
                12: ["ANC_A1_subtree", None, None, None],
                13: ["ANC_A2_subtree", None, None, None],  # highlight: 3D structure not available
                14: ["ANC_AS_subtree", None, None, None],
                15: ["ANC_S1_subtree", None, None, None],  # highlight: 3D structure not available
                16: ["Coral_Faviina", None, None, None],  # Faviina clade from coral sequences
                17: ["Coral_all", None, None, None],
                # All Coral sequences (includes Faviina clade and additional sequences)
                18: ["Cnidarian", None, None, None],
                # All Coral sequences plus other fluorescent cnidarians #Highlight: The tree is too different to certainly locate the all-coral / all-fav ancestors
                19: ["PKinase_PF07714", None, None, None],
                20: ["simulations_CNP_1", 1, "CNP_simulations", "CNP_seq"],  # EvolveAGene4 CNP simulation # 1000 leaves
                22: ["PF01038_msa", None, None, None],
                23: ["simulations_insulin_1", 1, "Insulin_simulations", "Insulin_seq"],
                24: ["simulations_insulin_2", 2, "Insulin_simulations", "Insulin_seq"],# EvolveAGene4 Insulin simulation #400 leaves
                25: ["simulations_PIGBOS_1", 1, "PIGBOS_simulations", "PIGBOS_seq"],# EvolveAGene4 PIGBOS simulation #300 leaves
                27: ["PF00400",None,None,None],
                28: ["aminopeptidase", None, None, None],
                29: ["PF01038_lipcti_msa_fungi", None, None, None],
                30: ["PF00096", None, None, None],
                31: ["PF00400_200", None, None, None]}


    datasets_full_names = {"benchmark_randall":"Randall's Coral fluorescent proteins (CFP) benchmark dataset",  # the tree is inferred
                "benchmark_randall_original":"Randall's Coral fluorescent proteins (CFP) benchmark dataset",
                "benchmark_randall_original_naming":"Randall's Coral fluorescent proteins (CFP) benchmark dataset",  # uses the original tree and it's original node naming
                "SH3_pf00018_larger_than_30aa":"PF00018 Pfam family of Protein Tyrosine Kinases SH3 domains",  # SRC kinases domain SH3 ---> Leaves and angles testing
                "simulations_blactamase_1":"32 leaves Simulation Beta-Lactamase",  # EvolveAGene4 Betalactamase simulation
                "simulations_src_sh3_1":"100 leaves Simulation SRC-Kinase SH3 domain",  # EvolveAGene4 SRC SH3 domain simulation
                "simulations_src_sh3_2": "800 leaves Simulation SRC-Kinase SH3 domain",
                "simulations_src_sh3_3": "200 leaves Simulation SRC-Kinase SH3 domain",
                "simulations_sirtuins_1": "150 leaves Simulation Sirtuin 1",
                "simulations_insulin_1": "50 leaves Simulation Insulin Growth Factor",
                "simulations_insulin_2": "400 leaves Simulation Insulin Growth Factor",
                "simulations_calcitonin_1": "50 leaves Simulation Calcitonin peptide",
                "simulations_mciz_1": "1600 leaves Simulation MciZ Factor",
                "simulations_CNP_1": "1000 leaves Simulation natriuretic peptide C",
                "simulations_PIGBOS_1": "300 leaves parser.add_argument('-use-cuda', type=str2bool, nargs='?',const=True, default=True, help='Use GPU')simulation PIGB Opposite Strand regulator",
                "Douglas_SRC":"Protein Tyrosin Kinases.",
                "ANC_A1_subtree":"Protein Tyrosin Kinases ANC-A1 clade",
                "ANC_A2_subtree":"Protein Tyrosin Kinases ANC-A2 clade",
                "ANC_AS_subtree":"Protein Tyrosin Kinases ANC-AS clade",
                "ANC_S1_subtree":"Protein Tyrosin Kinases ANC-S1 clade",
                "Coral_Faviina":"Coral fluorescent proteins (CFP) Faviina clade",  # Faviina clade from coral sequences
                "Coral_all":"Coral fluorescent proteins (CFP) clade",  # All Coral sequences (includes Faviina clade and additional sequences)
                "Cnidarian":"Cnidarian fluorescent proteins (CFP) clade",# All Coral sequences plus other fluorescent cnidarians #Highlight: The tree is too different to certainly locate the all-coral / all-fav ancestors
                "PKinase_PF07714":"PF07714 Pfam family of Protein Tyrosin Kinases",
                "PF01038_msa":"PF01038 Pfam family",
                "PF00271": "Helicase conserved C-terminal domain",
                "PF00400":"WD40 125 sequences",
                "aminopeptidase":"Amino Peptidase",
                "PF01038_lipcti_msa_fungi": "PF01038 Pfam lipcti fungi family ",
                "PF00096":"PF00096 protein kinases",
                "PF00400_200":"WD40 200 sequences"}


    #Highlight: Updated with Draupnir marginal results
    Draupnir_folders_results_GRU = {"simulations_src_sh3_2": ["/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_simulations_src_sh3_2_2022_01_04_04h23min36s229808ms_25000epochs_delta_map",False, True], #blosum, WITH plating
                                    "simulations_blactamase_1":["/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_simulations_blactamase_1_2022_01_03_19h49min49s379086ms_15000epochs_delta_map",False,False],
                                    "simulations_src_sh3_1":["/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_simulations_src_sh3_1_2022_01_03_21h10min28s830512ms_21600epochs_delta_map",False,False],
                                    "simulations_src_sh3_3":["/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_simulations_src_sh3_3_2022_01_04_15h07min22s126331ms_22000epochs_delta_map",False,False],
                                    "simulations_sirtuins_1":["/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_simulations_sirtuins_1_2022_01_03_21h52min21s591168ms_20000epochs_delta_map",False,False],
                                    "simulations_calcitonin_1":["/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_simulations_calcitonin_1_2022_01_03_20h35min01s171355ms_18700epochs_delta_map",False,False],
                                    "benchmark_randall_original_naming":["/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_benchmark_randall_original_naming_2022_01_03_17h03min50s119354ms_16600epochs_delta_map",False,False],
                                    "simulations_PIGBOS_1":["/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_simulations_PIGBOS_1_2022_01_04_00h09min52s442070ms_18000epochs_delta_map",False,False],
                                    "simulations_insulin_2":["/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_simulations_insulin_2_2022_01_04_00h52min27s593146ms_18400epochs_delta_map",False,False],
                                    "Coral_all":["/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_Coral_all_2022_01_04_16h03min41s511363ms_23000epochs_delta_map",False,False],
                                    "Coral_Faviina":["/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_Coral_Faviina_2022_01_04_16h03min06s615492ms_23000epochs_delta_map",False,False],
                                    "ANC_AS_subtree":["",False,False],
                                    "ANC_A1_subtree":["",False,False],
                                    "ANC_A2_subtree":["",False,False],
                                    "ANC_S1_subtree":["",False,False],
                                    "PF00400":["/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_PF00400_2021_12_14_16h19min09s878673ms_20000epochs_delta_map",False,False],
                                    "PF00096":["/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_PF00096_2021_12_16_20h02min05s417216ms_25000epochs_delta_map",False,False],
                                    "Douglas_SRC":["/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_Douglas_SRC_2021_12_15_16h07min27s740717ms_26000epochs_delta_map",False,False],
                                    "aminopeptidase":["/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_aminopeptidase_2021_12_14_21h31min01s188603ms_23000epochs_delta_map",False,False],
                                    "Cnidarian":["/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_Cnidarian_2021_12_16_14h02min28s832704ms_25000epochs_delta_map",False,False],
                                    "PF00400_200":["/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_PF00400_200_2021_12_14_20h36min23s108881ms_23000epochs_delta_map",False,False],
                                    "SH3_pf00018_larger_than_30aa":["/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_SH3_pf00018_larger_than_30aa_2021_12_15_18h01min46s419865ms_20000epochs_delta_map",False,False],
                                    "PKinase_PF07714":["/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_PKinase_PF07714_2021_12_16_06h53min55s804589ms_25000epochs_delta_map",False,False]}


    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir_level_up = script_dir.replace("/PAML","")
    script_dir_2level_up = script_dir_level_up.replace("/Benchmark_Results","")

    use_GRU = True
    sample_fastml=False
    save_folder = "GRU_map"
    #datasets_to_analyze = datasets.keys()
    datasets_to_analyze =[2,4,5,6,7,8,9,16,17,24,25]
    main(datasets_to_analyze)

