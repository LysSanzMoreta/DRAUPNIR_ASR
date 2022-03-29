"""
=======================
2022: Lys Sanz Moreta
Draupnir : Ancestral protein sequence reconstruction using a tree-structured Ornstein-Uhlenbeck variational autoencoder
=======================
"""
from prody import *
from pylab import *
from Bio import SeqIO
import os
now = datetime.datetime.now()
matplotlib.use('TkAgg')

def cal_coupling(fasta):
    """Calculates the DIC information criterion over a pylab alignment file
    :param fasta: fasta file"""
    print("calculating DIC coupling analysis")
    msa = parseMSA(fasta)
    #msa_refine = refineMSA(msa, label='RNAS2_HUMAN', rowocc=0.8, seqid=0.98)
    mi = buildDirectInfoMatrix(msa)
    return mi

def correlation_coefficient(T1, T2):
    """Correkation coefficient accross 2 matrices.
    :param numpy-matrix T1
    :param numpy-matrix T2"""
    numerator = np.mean((T1 - T1.mean()) * (T2 - T2.mean()))
    denominator = T1.std() * T2.std()
    if denominator == 0:
        return 0
    else:
        result = numerator / denominator
        return result
def create_root_samples_file(name,out_file,folder):
    """Extracts the root sequence from the predicted ancestral sequences fasta file.
    :param str name: dataset project name
    :param str out_file: file name where to output the sequences
    :param str folder: path to folder where the sampled sequences are found"""
    print("File does not exist, creating it")
    all_samples_file = "{}/{}_sampled_nodes_seq.fasta".format(folder,name)
    all_samples = SeqIO.parse(all_samples_file,"fasta")
    root_sequences = []
    root_node = None
    for idx,record in enumerate(all_samples):
        if idx == 0: #sequences are stored in tree level order. therefore the first node is always the root
            root_node = record.id.split("//")[0]
            root_sequences.append(record)
        else:
            if record.id.startswith(root_node):
                root_sequences.append(record)

    SeqIO.write(root_sequences,out_file,"fasta")

def plot_MI_matrices_variational(name,leaves_mi,draupnir_variational_mi,results_dir):
    """Plots the Mutual Information matrix using the DI criterion
    :param str name: dataset project name
    :param numpy-matrix leaves_mi: mutual information calculated among the leaf sequences with prody
    :param numpy-matrix draupnir_variational_mi: mutual information calculated among the sampled root sequences (obtained with draupnir variational)  with prody"""
    # 2 subplots in 1 row and 2 columns
    print("Plotting.............")
    fig, [ax1, ax2] = plt.subplots(2, figsize=(15, 7.5),constrained_layout=True)
    leaves_variational = correlation_coefficient(leaves_mi,draupnir_variational_mi)
    print("Correlation coefficient Leaves vs Variational: {}".format(leaves_variational))
    mean_variational = draupnir_variational_mi.mean()
    print("Mean Variational : {}".format(mean_variational) )
    std_variational = draupnir_variational_mi.std()
    print("Std Variational: {}".format(std_variational))

    # Title for subplots
    vmin = None
    l1 = ax1.imshow(leaves_mi,cmap = "hot",vmin=vmin) #hot_r
    l2 = ax2.imshow(draupnir_variational_mi,cmap = "hot",vmin=vmin)
    ticks_array = np.arange(0, len(leaves_mi) + 1, 40)
    ax1.set_title('Leaves',fontdict = {'fontsize':22})
    ax2.set_title('Variational',fontdict = {'fontsize':22})
    ax1.set_xticks(ticks_array)
    ax2.set_xticks(ticks_array)

    ax1.set_yticks(ticks_array)
    ax2.set_yticks(ticks_array)


    # pos1 = ax5.get_position()  # get the original position
    # pos2 = [pos1.x0 + 10, pos1.y0 , pos1.width / 2.0, pos1.height / 2.0]
    # ax5.set_position(pos2)
    cb_ax = fig.add_axes([0.75, 0.25, 0.02, 0.5])
    cbar = fig.colorbar(l2, cax=cb_ax)
    # #plt.tight_layout()
    plt.subplots_adjust(wspace=-0.5, hspace=0.5)
    plt.savefig("{}/Mutual_Information_{}_{}_PROTEIN_matrices_Variational_{}.png".format(results_dir, name,"root",now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms")),dpi=600)
    plt.clf()

def plot_MI_matrices(name,leaves_mi,draupnir_MAP_mi,draupnir_marginal_mi,draupnir_variational_mi,benchmark_folder):
    """Plots the Mutual Information matrix using the DI criterion
    :param str name: dataset project name
    :param numpy-matrix leaves_mi: mutual information calculated among the leaf sequences with prody
    :param numpy-matrix draupnir_MAP_mi: mutual information calculated among the sampled root sequences (obtainde with draupnir MAP) with prody
    :param numpy-matrix draupnir_marginal_mi: mutual information calculated among the sampled root sequences (obtainde with draupnir marginal)  with prody
    :param numpy-matrix draupnir_variational_mi: mutual information calculated among the sampled root sequences (obtained with draupnir variational)  with prody"""
    # 2 subplots in 1 row and 2 columns
    print("Plotting.............")
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(15, 7.5),constrained_layout=True)
    print("Correlation coefficient Leaves vs MAP:")
    leaves_MAP = correlation_coefficient(leaves_mi,draupnir_MAP_mi)
    print("Correlation coefficient Leaves vs MAP: {}".format(leaves_MAP))
    mean_MAP = draupnir_MAP_mi.mean()
    print("Mean MAP : {}".format(mean_MAP))
    std_MAP = draupnir_MAP_mi.std()
    print("Std MAP: {}".format(std_MAP))
    leaves_marginal = correlation_coefficient(leaves_mi,draupnir_marginal_mi)
    print("Correlation coefficient Leaves vs Marginal: {}".format(leaves_marginal))
    mean_marginal =draupnir_marginal_mi.mean()
    print("Mean Marginal : {}".format(mean_marginal))
    std_marginal = draupnir_marginal_mi.std()
    print("Std Marginal: {}".format(std_marginal))
    leaves_variational = correlation_coefficient(leaves_mi,draupnir_variational_mi)
    print("Correlation coefficient Leaves vs Variational: {}".format(leaves_variational))
    mean_variational = draupnir_variational_mi.mean()
    print("Mean Variational : {}".format(mean_variational) )
    std_variational = draupnir_variational_mi.std()
    print("Std Variational: {}".format(std_variational))

    # Title for subplots
    vmin = None
    l1 = ax1.imshow(leaves_mi,cmap = "hot",vmin=vmin) #hot_r
    l2 = ax2.imshow(draupnir_MAP_mi,cmap = "hot",vmin=vmin)
    l3 = ax3.imshow(draupnir_marginal_mi,cmap = "hot",vmin=vmin)
    l4 = ax4.imshow(draupnir_variational_mi,cmap = "hot",vmin=vmin)
    ticks_array = np.arange(0, len(leaves_mi) + 1, 40)
    ax1.set_title('Leaves',fontdict = {'fontsize':22})
    ax2.set_title('MAP',fontdict = {'fontsize':22})
    ax3.set_title('Marginal',fontdict = {'fontsize':22})
    ax4.set_title('Variational',fontdict = {'fontsize':22})
    ax1.set_xticks(ticks_array)
    ax2.set_xticks(ticks_array)
    ax3.set_xticks(ticks_array)
    ax4.set_xticks(ticks_array)


    ax1.set_yticks(ticks_array)
    ax2.set_yticks(ticks_array)
    ax3.set_yticks(ticks_array)
    ax4.set_yticks(ticks_array)

    # pos1 = ax5.get_position()  # get the original position
    # pos2 = [pos1.x0 + 10, pos1.y0 , pos1.width / 2.0, pos1.height / 2.0]
    # ax5.set_position(pos2)
    cb_ax = fig.add_axes([0.75, 0.25, 0.02, 0.5])
    cbar = fig.colorbar(l4, cax=cb_ax)
    # #plt.tight_layout()
    plt.subplots_adjust(wspace=-0.5, hspace=0.5)
    plt.savefig("{}/Mutual_Information_{}_{}_PROTEIN_matrices_MAP_Marginal_Variational_{}.png".format(benchmark_folder, name,"root",now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms")),dpi=600)
    plt.clf()

def plot_MI_matrices_delta_map(name,leaves_mi,draupnir_MAP_mi,draupnir_marginal_mi,benchmark_folder):
    """Plots the Mutual Information matrix using the DI criterion
    :param str name: dataset project name
    :param numpy-matrix leaves_mi: mutual information calculated among the leaf sequences with prody
    :param numpy-matrix draupnir_MAP_mi: mutual information calculated among the sampled root sequences (obtainde with draupnir MAP) with prody
    :param numpy-matrix draupnir_marginal_mi: mutual information calculated among the sampled root sequences (obtainde with draupnir marginal)  with prody"""
    # 2 subplots in 1 row and 2 columns
    print("Plotting.............")
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(15, 7.5),constrained_layout=True)
    print("Correlation coefficient Leaves vs MAP:")
    leaves_MAP = correlation_coefficient(leaves_mi,draupnir_MAP_mi)
    print("Correlation coefficient Leaves vs MAP: {}".format(leaves_MAP))
    mean_MAP = draupnir_MAP_mi.mean()
    print("Mean MAP : {}".format(mean_MAP))
    std_MAP = draupnir_MAP_mi.std()
    print("Std MAP: {}".format(std_MAP))
    leaves_marginal = correlation_coefficient(leaves_mi,draupnir_marginal_mi)
    print("Correlation coefficient Leaves vs Marginal: {}".format(leaves_marginal))
    mean_marginal =draupnir_marginal_mi.mean()
    print("Mean Marginal : {}".format(mean_marginal))
    std_marginal = draupnir_marginal_mi.std()
    print("Std Marginal: {}".format(std_marginal))


    # Title for subplots
    vmin = None
    l1 = ax1.imshow(leaves_mi,cmap = "hot",vmin=vmin) #hot_r
    l2 = ax2.imshow(draupnir_MAP_mi,cmap = "hot",vmin=vmin)
    l3 = ax3.imshow(draupnir_marginal_mi,cmap = "hot",vmin=vmin)
    ticks_array = np.arange(0, len(leaves_mi) + 1, 40)
    ax1.set_title('Leaves',fontdict = {'fontsize':22})
    ax2.set_title('MAP',fontdict = {'fontsize':22})
    ax3.set_title('Marginal',fontdict = {'fontsize':22})
    ax1.set_xticks(ticks_array)
    ax2.set_xticks(ticks_array)
    ax3.set_xticks(ticks_array)


    ax1.set_yticks(ticks_array)
    ax2.set_yticks(ticks_array)
    ax3.set_yticks(ticks_array)

    # pos1 = ax5.get_position()  # get the original position
    # pos2 = [pos1.x0 + 10, pos1.y0 , pos1.width / 2.0, pos1.height / 2.0]
    # ax5.set_position(pos2)
    cb_ax = fig.add_axes([0.75, 0.25, 0.02, 0.5])
    cbar = fig.colorbar(l3, cax=cb_ax)
    # #plt.tight_layout()
    plt.subplots_adjust(wspace=-0.5, hspace=0.5)
    plt.savefig("{}/Mutual_Information_{}_{}_PROTEIN_matrices_MAP_Marginal_{}.png".format(benchmark_folder, name,"root",now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms")),dpi=600)
    plt.clf()

def MI_root(name,draupnir_folder_MAP, draupnir_folder_marginal, draupnir_folder_variational, benchmark_folder):
    """Reads or creates the root files necessary for MI calculation.
    :param str name: dataset project name
    :param str draupnir_folder_MAP: path to result of draupnir on a dataset using guide=delta_map, the results in <Test2> folders are used
    :param str draupnir_folder_marginal: path to result of draupnir on a dataset using guide=delta_map, the results in <Test> folders are used
    :param str draupnir_folder_variational: path to result of draupnir on a dataset using guide=variational
    :param str benchmark_folder
    :param only_root: If True it calculates MI only among the samples from the root, otherwise among all nodes"""

    leaves_fasta = "{}/{}_training_aligned.fasta".format(draupnir_folder_MAP,name)

    draupnir_fasta_MAP = "{}/Test2_Plots/{}_root_node_sampled.fasta".format(draupnir_folder_MAP,name)
    if not os.path.exists(draupnir_fasta_MAP):
        create_root_samples_file(name,draupnir_fasta_MAP,"{}/Test2_Plots".format(draupnir_folder_MAP))

    draupnir_fasta_marginal = "{}/Test_Plots/{}_root_node_sampled.fasta".format(draupnir_folder_marginal,name)
    if not os.path.exists(draupnir_fasta_marginal):
        create_root_samples_file(name,draupnir_fasta_marginal,"{}/Test_Plots".format(draupnir_folder_marginal))

    draupnir_fasta_variational = "{}/Test_Plots/{}_root_node_sampled.fasta".format(draupnir_folder_variational,name)
    if not os.path.exists(draupnir_fasta_variational):
        create_root_samples_file(name,draupnir_fasta_variational,"{}/Test_Plots".format(draupnir_folder_variational))

    print("leaves")
    leaves_mi = cal_coupling(leaves_fasta)
    print("MAP")
    draupnir_MAP_mi = cal_coupling(draupnir_fasta_MAP)
    print("Marginal")
    draupnir_marginal_mi = cal_coupling(draupnir_fasta_marginal)
    print("Variational")
    draupnir_variational_mi = cal_coupling(draupnir_fasta_variational)
    plot_MI_matrices(name,leaves_mi, draupnir_MAP_mi, draupnir_marginal_mi, draupnir_variational_mi,benchmark_folder)

def MI_root_variational(name, draupnir_folder_variational, results_dir):
    """Reads or creates the root files necessary for MI calculation.
    :param str name: dataset project name
    :param str draupnir_folder_MAP: path to result of draupnir on a dataset using guide=delta_map, the results in <Test2> folders are used
    :param str draupnir_folder_marginal: path to result of draupnir on a dataset using guide=delta_map, the results in <Test> folders are used
    :param str draupnir_folder_variational: path to result of draupnir on a dataset using guide=variational
    :param str benchmark_folder
    :param only_root: If True it calculates MI only among the samples from the root, otherwise among all nodes"""

    leaves_fasta = "{}/{}_training_aligned.fasta".format(draupnir_folder_variational,name)

    draupnir_fasta_variational = "{}/Test_Plots/{}_root_node_sampled.fasta".format(draupnir_folder_variational,name)

    if not os.path.exists(draupnir_fasta_variational):
        create_root_samples_file(name,draupnir_fasta_variational,"{}/Test_Plots".format(draupnir_folder_variational))

    print("leaves")
    leaves_mi = cal_coupling(leaves_fasta)
    print("Variational")
    draupnir_variational_mi = cal_coupling(draupnir_fasta_variational)
    plot_MI_matrices_variational(name,leaves_mi, draupnir_variational_mi,results_dir)

def MI_root_delta_map(name,draupnir_folder_MAP, draupnir_folder_marginal,benchmark_folder):
    """Reads or creates the root files necessary for MI calculation.
    :param str name: dataset project name
    :param str draupnir_folder_MAP: path to result of draupnir on a dataset using guide=delta_map, the results in <Test2> folders are used
    :param str draupnir_folder_marginal: path to result of draupnir on a dataset using guide=delta_map, the results in <Test> folders are used
    :param str draupnir_folder_variational: path to result of draupnir on a dataset using guide=variational
    :param str benchmark_folder
    :param only_root: If True it calculates MI only among the samples from the root, otherwise among all nodes"""

    leaves_fasta = "{}/{}_training_aligned.fasta".format(draupnir_folder_MAP,name)

    draupnir_fasta_MAP = "{}/Test2_Plots/{}_root_node_sampled.fasta".format(draupnir_folder_MAP,name)
    if not os.path.exists(draupnir_fasta_MAP):
        create_root_samples_file(name,draupnir_fasta_MAP,"{}/Test2_Plots".format(draupnir_folder_MAP))

    draupnir_fasta_marginal = "{}/Test_Plots/{}_root_node_sampled.fasta".format(draupnir_folder_marginal,name)
    if not os.path.exists(draupnir_fasta_marginal):
        create_root_samples_file(name,draupnir_fasta_marginal,"{}/Test_Plots".format(draupnir_folder_marginal))


    print("leaves")
    leaves_mi = cal_coupling(leaves_fasta)
    print("MAP")
    draupnir_MAP_mi = cal_coupling(draupnir_fasta_MAP)
    print("Marginal")
    draupnir_marginal_mi = cal_coupling(draupnir_fasta_marginal)
    plot_MI_matrices_delta_map(name,leaves_mi, draupnir_MAP_mi, draupnir_marginal_mi,benchmark_folder)



def calculate_mutual_information(args,results_dir,draupnir_folder_variational=None,draupnir_folder_MAP=None, draupnir_folder_marginal=None):
    """Calculates Direct Information criterion or Multual information and plots the MI matrices.
    :param namedtuple args
    :param str results_dir: path to folder where to store the results
    :param str or None draupnir_folder_variational: path to result of draupnir on a dataset using guide=variational
    :param str or None draupnir_folder_MAP: path to result of draupnir on a dataset using guide=delta_map, the results in <Test2> folders are used
    :param str or None draupnir_folder_marginal: path to result of draupnir on a dataset using guide=delta_map, the results in <Test> folders are used
    :param str benchmark_folder
    :param only_root: If True it calculates MI only among the samples from the root, otherwise among all nodes
    :param only_variational: If True it will calculate DCA with a run only from the variational model, otherwise it requires to compare MAP, marginal and variational"""
    if all(v is None for v in [draupnir_folder_variational,draupnir_folder_MAP,draupnir_folder_marginal]):
        raise ValueError("Please provide at least the results for the variational guide")
    elif (draupnir_folder_MAP,draupnir_folder_variational) == (None,None) and draupnir_folder_marginal is not None:
        warnings.warn("You have assigned draupnir_folder_MAP to None, therefore I am using the results from draupnir_folder_marginal. There are not variational guide results")
        draupnir_folder_MAP = draupnir_folder_marginal
        MI_root_delta_map(args.name,draupnir_folder_MAP,draupnir_folder_marginal,results_dir)
    elif (draupnir_folder_marginal, draupnir_folder_variational) == (None, None) and draupnir_folder_MAP is not None:
        warnings.warn("You have assigned draupnir_folder_marginal to None, therefore I am using the results from draupnir_folder_marginal. There are not variational guide results")
        draupnir_folder_marginal = draupnir_folder_MAP
        MI_root_delta_map(args.name, draupnir_folder_MAP, draupnir_folder_marginal, results_dir)
    elif (draupnir_folder_MAP,draupnir_folder_marginal) == (None,None) and draupnir_folder_variational is not None:
        warnings.warn("You have assigned draupnir_folder_marginal and draupnir_folder_MAP to None. Only using variational guide results")
        MI_root_variational(args.name,draupnir_folder_variational,results_dir)
    else:
        MI_root(args.dataset_name,draupnir_folder_MAP, draupnir_folder_marginal, draupnir_folder_variational, results_dir)






