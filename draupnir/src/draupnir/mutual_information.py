import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from prody import *
from pylab import *
import pandas as pd
import seaborn
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import Draupnir_utils as DraupnirUtils
import warnings
import os
now = datetime.datetime.now()
matplotlib.use('TkAgg')

def cal_coupling(fasta):
    """Calculates the DIC information criterio over a pylab alignment file
    :param fasta: fasta file"""
    print("calculating DIC coupling analysis")
    msa = parseMSA(fasta)
    #msa_refine = refineMSA(msa, label='RNAS2_HUMAN', rowocc=0.8, seqid=0.98)

    mi = buildDirectInfoMatrix(msa)
    return mi

def correlation_coefficient(T1, T2):
    """Correkation coefficient accross 2 matrices"""
    numerator = np.mean((T1 - T1.mean()) * (T2 - T2.mean()))
    denominator = T1.std() * T2.std()
    if denominator == 0:
        return 0
    else:
        result = numerator / denominator
        return result
def create_root_samples_file(file,folder):
    """Extracts the root sequence from the predicted ancestral sequences fasta file"""
    print("File does not exist, creating it")
    all_samples_file = "{}/{}_sampled_ancestors_seq.fasta".format(folder,name)
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

    SeqIO.write(root_sequences,file,"fasta")

def plot_MI_matrices(leaves_mi,draupnir_MAP_mi,draupnir_marginal_mi,draupnir_variational_mi):
    """Plots the Mutual Information matrix using the DIC criterior"""
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

def MI_root():
    """Reads or creates the root files necessary for MI calculation"""

    leaves_fasta = "{}/{}_training_aligned.fasta".format(draupnir_folder_MAP,name)
    draupnir_fasta_MAP = "{}/Test2_Plots/{}_root_node_sampled.fasta".format(draupnir_folder_MAP,name)

    if not os.path.exists(draupnir_fasta_MAP):
        create_root_samples_file(draupnir_fasta_MAP,"{}/Test2_Plots".format(draupnir_folder_MAP))

    draupnir_fasta_marginal = "{}/Test_Plots/{}_root_node_sampled.fasta".format(draupnir_folder_marginal,name)

    if not os.path.exists(draupnir_fasta_marginal):
        create_root_samples_file(draupnir_fasta_marginal,"{}/Test_Plots".format(draupnir_folder_MAP))

    draupnir_fasta_variational = "{}/Test_Plots/{}_root_node_sampled.fasta".format(draupnir_folder_variational,name)

    if not os.path.exists(draupnir_fasta_variational):
        create_root_samples_file(draupnir_fasta_variational,"{}/Test_Plots".format(draupnir_folder_variational))

    print("leaves")
    leaves_mi = cal_coupling(leaves_fasta)
    print("MAP")
    draupnir_MAP_mi = cal_coupling(draupnir_fasta_MAP)
    print("Marginal")
    draupnir_marginal_mi = cal_coupling(draupnir_fasta_marginal)
    print("Variational")
    draupnir_variational_mi = cal_coupling(draupnir_fasta_variational)
    plot_MI_matrices(leaves_mi, draupnir_MAP_mi, draupnir_marginal_mi, draupnir_variational_mi)


if __name__ == "__main__":
    datasets = {
        3: ["SH3_pf00018_larger_than_30aa",  # new sampling
             "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_SH3_pf00018_larger_than_30aa_2021_12_15_18h01min46s419865ms_20000epochs_delta_map",
             "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_SH3_pf00018_larger_than_30aa_2021_12_15_18h01min46s419865ms_20000epochs_delta_map",
             # marginal
             "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_SH3_pf00018_larger_than_30aa_2021_12_15_19h21min58s336062ms_20000epochs_variational",
             # variational
             "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/Benchmark_Plots2/SH3_pf00018_larger_than_30aa"],
        11: ["Douglas_SRC",  # new sampling
             "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_Douglas_SRC_2021_12_15_16h07min27s740717ms_26000epochs_delta_map",
             "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_Douglas_SRC_2021_12_15_16h07min27s740717ms_26000epochs_delta_map",
             # marginal
             "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_Douglas_SRC_2021_12_15_01h39min56s116999ms_26000epochs_variational",
             # variational
             "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/Benchmark_Plots2/Douglas_SRC"],
        16: ["Coral_Faviina",  # new sampling
             "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_Coral_Faviina_2021_12_15_16h52min11s969164ms_23000epochs_delta_map",
             "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_Coral_Faviina_2021_12_15_16h52min11s969164ms_23000epochs_delta_map",
             # marginal
             "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_Coral_Faviina_2021_12_16_15h37min16s986174ms_25000epochs_variational",
             # variational
             "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/Benchmark_Plots2/Coral_Faviina"],
        17: ["Coral_all", #new sampling
             "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_Coral_all_2021_12_14_19h21min01s147060ms_23000epochs_delta_map",
             "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_Coral_all_2021_12_14_19h21min01s147060ms_23000epochs_delta_map",# marginal
             "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_Coral_all_2021_12_14_19h16min18s942525ms_23000epochs_variational",# variational
             "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/Benchmark_Plots2/Coral_all"],
        18: ["Cnidarian",  # new sampling
             "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_Cnidarian_2021_12_16_14h02min28s832704ms_25000epochs_delta_map",
             "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_Cnidarian_2021_12_16_14h02min28s832704ms_25000epochs_delta_map",
             # marginal
             "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_Cnidarian_2021_12_17_09h41min36s405387ms_30000epochs_variational",
             # variational
             "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/Benchmark_Plots2/Cnidarian"],
        19: ["PKinase_PF07714",  # new sampling
             "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_PKinase_PF07714_2021_12_16_06h53min55s804589ms_25000epochs_delta_map",
             "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_PKinase_PF07714_2021_12_16_06h53min55s804589ms_25000epochs_delta_map",
             # marginal
             "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_PKinase_PF07714_2021_12_15_23h19min41s395833ms_25000epochs_variational",
             # variational
             "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/Benchmark_Plots2/PKinase_PF07714"],
        27: ["PF00400",  # Highlight: 125 samples, new new sampling
             "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_PF00400_2021_12_20_23h09min22s072191ms_3epochs_delta_map",
             # MAP Test2 folder
             "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_PF00400_2021_12_20_23h09min22s072191ms_3epochs_delta_map",
             # marginal Test folder
             # "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_PF00400_2021_12_14_00h00min02s264900ms_15000epochs_variational",# variational, with same DCA score
             "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_PF00400_2021_12_20_23h12min06s365176ms_3epochs_variational",
             "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/Benchmark_Plots2/PF00400"],
        28: ["aminopeptidase",  # new sampling
             #MAP
             "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_aminopeptidase_2021_12_14_21h31min01s188603ms_23000epochs_delta_map",
             # Marginal
             "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_aminopeptidase_2021_12_14_21h31min01s188603ms_23000epochs_delta_map",
             # variational
             "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_aminopeptidase_2021_12_14_22h41min20s232547ms_23000epochs_variational",
             "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/Benchmark_Plots2/aminopeptidase"],
        30: ["PF00096",  # new sampling
             # MAP
             "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_PF00096_2021_12_16_20h02min05s417216ms_25000epochs_delta_map",
             # Marginal
             "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_PF00096_2021_12_16_20h02min05s417216ms_25000epochs_delta_map",
             # variational
             "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_PF00096_2021_12_16_18h56min33s908882ms_25000epochs_variational",
             "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/Benchmark_Plots2/PF00096"],
        31: ["PF00400_200",  # 200 samples, new sampling
             #MAP , Test2 folder
             "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_PF00400_200_2021_12_14_20h36min23s108881ms_23000epochs_delta_map",
             #Marginal, Test folder
             "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_PF00400_200_2021_12_14_21h21min13s145074ms_23000epochs_variational",
             #Variational Test folder
             "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_PF00400_200_2021_12_14_21h21min13s145074ms_23000epochs_variational",
             "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/Benchmark_Plots2/PF00400_200"]
        }
    #datasets_list = [3,11,16,17,18,19,27,28,30]
    datasets_list = [27]

    dataframes_list = []
    for d in datasets_list:
        name,draupnir_folder_MAP,draupnir_folder_marginal,draupnir_folder_variational,benchmark_folder = datasets[d]
        MI_root()





