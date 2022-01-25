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
def cal_mi_corrected(fasta):
    print("Using the corrected MI")
    msa = parseMSA(fasta)
    entropy = calcShannonEntropy(msa)
    mi = applyMutinfoNorm(buildMutinfoMatrix(msa),entropy,norm='sument')
    #mi = (mi - mi.mean())/mi.std()
    return mi
def cal_mi_corrected_CORR(fasta):
    print("Using Multi info corr")

    msa = parseMSA(fasta)
    mi = applyMutinfoCorr(buildMutinfoMatrix(msa))
    #mi = (mi - mi.mean())/mi.std()
    return mi
def cal_mi(fasta):
    print("using standard MI")
    msa = parseMSA(fasta)
    return buildMutinfoMatrix(msa)

def cal_coupling(fasta):
    print("calculating coupling analysis")
    msa = parseMSA(fasta)
    #msa_refine = refineMSA(msa, label='RNAS2_HUMAN', rowocc=0.8, seqid=0.98)

    mi = buildDirectInfoMatrix(msa)
    return mi

def plot_histograms(leaves_mi,draupnir_delta_mi,draupnir_custom_mi):
    fig, ax = plt.subplots(nrows=1, ncols=3,figsize=(15,7.5))
    min_val = min(leaves_mi.min(),draupnir_delta_mi.min(),draupnir_custom_mi.min())
    max_val = max(leaves_mi.max(),draupnir_delta_mi.max(),draupnir_custom_mi.max())
    l1 = ax[0].hist(leaves_mi.reshape(-1), bins=50, density=True, label = "Leaves")
    l2 = ax[1].hist(draupnir_delta_mi.reshape(-1), bins=50, density=True, label = "Draupnir MAP")
    l3 = ax[2].hist(draupnir_custom_mi.reshape(-1), bins=50, density=True, label = "Draupnir Variational")
    #plt.legend([l1, l2, l3], ["Leaves", "Draupnir MAP", "Draupnir Variational"])
    plt.suptitle("Mutual Information (PROTEIN): \n {}".format(name))
    plt.tight_layout()
    plt.savefig("{}/Mutual_Information_{}_PROTEIN_histograms.png".format(benchmark_folder,name))
    plt.clf()

def plot_overlapping_densities(leaves_mi,draupnir_delta_mi,draupnir_custom_mi):
    fig, ax = plt.subplots(nrows=1, ncols=3,figsize=(15,7.5))
    plt.hist(leaves_mi.reshape(-1), bins=50, density=True,label="Leaves")
    plt.hist(draupnir_delta_mi.reshape(-1), bins=50, density=True,label="Draupnir MAP")
    plt.hist(draupnir_custom_mi.reshape(-1), bins=50, density=True, label = "Draupnir Variational")

    plt.tight_layout()

    plt.savefig("{}/Mutual_Information_{}_PROTEIN_overlapping_densities.png".format(benchmark_folder,name),dpi=400)
    plt.clf()

def plot_mutual_information(leaves_mi,draupnir_delta_mi,draupnir_custom_mi):

    plt.scatter(leaves_mi,draupnir_custom_mi,color="orange",label="Draupnir variational",s=2)
    plt.scatter(leaves_mi,draupnir_delta_mi,color="green",label="Draupnir MAP",s=2)
    plt.xlabel("Leaves MI")
    plt.ylabel("Samples MI")
    plt.legend()
    plt.savefig("{}/Mutual_Information_{}_PROTEIN.png".format(benchmark_folder, name),dpi=400)
    plt.clf()

def plot_errobar(leaves_mi,draupnir_delta_mi,draupnir_custom_mi):
    def calculate_params(data_mi):
        x = np.arange(0,len(data_mi))
        y = np.mean(data_mi,axis=0)
        e = np.std(data_mi,axis=0)
        return x,y,e
    leaves_x,leaves_y,leaves_e = calculate_params(leaves_mi)
    draupnir_delta_x, draupnir_delta_y, draupnir_delta_e = calculate_params(draupnir_delta_mi)
    draupnir_custom_x, draupnir_custom_y, draupnir_custom_e = calculate_params(draupnir_custom_mi)

    plt.errorbar(leaves_x, leaves_y, leaves_e, linestyle='None', marker='o', elinewidth=0.1,markersize=1,color="blue",label="Leaves")
    plt.errorbar(draupnir_delta_x, draupnir_delta_y, draupnir_delta_e, linestyle='None', marker='o', elinewidth=0.1,markersize=1,color="green",label="Draupnir MAP")
    plt.errorbar(draupnir_custom_x, draupnir_custom_y, draupnir_custom_e, linestyle='None', marker='o', elinewidth=0.1,markersize=1,color="orange",label="Draupnir Variational")
    plt.legend()
    plt.savefig("{}/Mutual_Information_{}_PROTEIN_errorbar.png".format(benchmark_folder, name),dpi=400)
    plt.clf()

def correlation_coefficient(T1, T2):
    numerator = np.mean((T1 - T1.mean()) * (T2 - T2.mean()))
    denominator = T1.std() * T2.std()
    if denominator == 0:
        return 0
    else:
        result = numerator / denominator
        return result


def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]
def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list.

        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = matplotlib.colors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp

def plot_matrices1(leaves_mi,draupnir_MAP_mi,draupnir_marginal_mi,draupnir_variational_mi,fastml_mi):
    # 2 subplots in 1 row and 2 columns
    print("Plotting.............")
    fig, [[ax1, ax2], [ax3, ax4],[ax5,ax6]] = plt.subplots(3, 2, figsize=(15, 7.5),constrained_layout=True)

    leaves_median = np.median(leaves_mi)
    leaves_mean = leaves_mi.mean()
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
    leaves_fastml = correlation_coefficient(leaves_mi, fastml_mi)
    print("Correlation coefficient Leaves vs FastML: {}".format(leaves_fastml))
    fastml_mean = fastml_mi.mean()
    print("Mean FastML : {}".format(fastml_mean))
    fastml_std = fastml_mi.std()
    print("Std FastML: {}".format(fastml_std))
    df = pd.DataFrame([[name,leaves_median,leaves_mean,leaves_MAP,mean_MAP,std_MAP,leaves_marginal,mean_marginal,std_marginal,leaves_variational,mean_variational,std_variational,leaves_fastml,fastml_mean,fastml_std]],columns=["Dataset","Leaves median","Leaves mean",
                                                                                                                                              "CC MAPvsLeaves","Mean MAP","Std MAP",
                                                                                                                                              "CC MarginalvsLeaves","Mean Marginal","Std Marginal",
                                                                                                                                              "CC VariationalvsLeaves","Mean Variational","Std Marginal",
                                                                                                                                              "CC FastMLvsLeaves","Mean FastML","Std FastML"])

    df.to_csv("{}/Summary.csv".format(benchmark_folder),sep="\t")

    # Title for subplots
    vmin = None
    l1 = ax1.imshow(leaves_mi,cmap = "hot",vmin=vmin) #hot_r
    l2 = ax2.imshow(draupnir_MAP_mi,cmap = "hot",vmin=vmin)
    l3 = ax3.imshow(draupnir_marginal_mi,cmap = "hot",vmin=vmin)
    l4 = ax4.imshow(draupnir_variational_mi,cmap = "hot",vmin=vmin)
    l5 = ax5.imshow(fastml_mi, cmap="hot", vmin=vmin)
    ax6.axis("off")
    ticks_array = np.arange(0, len(leaves_mi) + 1, 40)
    ax1.set_title('Leaves',fontdict = {'fontsize':22})
    ax2.set_title('MAP',fontdict = {'fontsize':22})
    ax3.set_title('Marginal',fontdict = {'fontsize':22})
    ax4.set_title('Variational',fontdict = {'fontsize':22})
    ax5.set_title('FastML', fontdict={'fontsize': 22})
    ax1.set_xticks(ticks_array)
    ax2.set_xticks(ticks_array)
    ax3.set_xticks(ticks_array)
    ax4.set_xticks(ticks_array)
    ax5.set_xticks(ticks_array)


    ax1.set_yticks(ticks_array)
    ax2.set_yticks(ticks_array)
    ax3.set_yticks(ticks_array)
    ax4.set_yticks(ticks_array)
    ax5.set_yticks(ticks_array)

    # pos1 = ax5.get_position()  # get the original position
    # pos2 = [pos1.x0 + 10, pos1.y0 , pos1.width / 2.0, pos1.height / 2.0]
    # ax5.set_position(pos2)
    cb_ax = fig.add_axes([0.75, 0.25, 0.02, 0.5])
    cbar = fig.colorbar(l4, cax=cb_ax)
    # #plt.tight_layout()
    plt.subplots_adjust(wspace=-0.5, hspace=0.5)
    comparison_type = ["root" if only_root else "allnodes"][0]
    plt.savefig("{}/Mutual_Information_{}_{}_PROTEIN_matrices_MAP_Marginal_Variational_FastML_{}.png".format(benchmark_folder, name,comparison_type,now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms")),dpi=600)
    plt.clf()

def plot_matrices3(leaves_mi,draupnir_MAP_mi,draupnir_marginal_mi,draupnir_variational_mi):
    # 2 subplots in 1 row and 2 columns
    print("Plotting.............")
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(15, 7.5),constrained_layout=True)



    leaves_median = np.median(leaves_mi)
    leaves_mean = leaves_mi.mean()
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
    # df = pd.DataFrame([[name,leaves_median,leaves_mean,leaves_MAP,mean_MAP,std_MAP,leaves_marginal,mean_marginal,std_marginal,leaves_variational,mean_variational,std_variational,None,None,None]],columns=["Dataset","Leaves median","Leaves mean",
    #                                                                                                                                           "CC MAPvsLeaves","Mean MAP","Std MAP",
    #                                                                                                                                           "CC MarginalvsLeaves","Mean Marginal","Std Marginal",
    #                                                                                                                                           "CC VariationalvsLeaves","Mean Variational","Std Marginal",
    #                                                                                                                                           "CC FastMLvsLeaves","Mean FastML","Std FastML"])
    #
    #
    # df.to_csv("{}/Summary.csv".format(benchmark_folder),sep="\t")


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
    comparison_type = ["root" if only_root else "allnodes"][0]
    plt.savefig("{}/Mutual_Information_{}_{}_PROTEIN_matrices_MAP_Marginal_Variational_{}.png".format(benchmark_folder, name,comparison_type,now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms")),dpi=600)
    plt.clf()

def MI_root():

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

    if use_indels:
        fastml_samples = "{}/FastML_{}_root_sampled_ancestors_seq_PROTEIN.fasta".format(benchmark_folder,name)
    else:
        fastml_samples = "{}/FastML_{}_root_sampled_ancestors_seq_PROTEIN_noindels.fasta".format(benchmark_folder,name)
    if not os.path.exists(fastml_samples):
        fastml_samples = None
    #iqtree_samples = "{}/IQTree_{}_root_sampled_ancestors_seq_PROTEIN.fasta".format(benchmark_folder,name)
    # print(draupnir_fasta_MAP)
    # print(draupnir_fasta_marginal)
    # print(draupnir_fasta_variational)
    # exit()
    print("1")
    leaves_mi = cal_coupling(leaves_fasta)
    print("2")
    draupnir_MAP_mi = cal_coupling(draupnir_fasta_MAP)
    print("3")
    draupnir_marginal_mi = cal_coupling(draupnir_fasta_marginal)
    print("4")
    draupnir_variational_mi = cal_coupling(draupnir_fasta_variational)
    print("Not plotting FastML")
    plot_matrices3(leaves_mi, draupnir_MAP_mi, draupnir_marginal_mi, draupnir_variational_mi)
    exit()
    #iqtree_mi = cal_coupling(iqtree_samples)
    if fastml_samples is not None:
        print("5")
        fastml_mi = cal_coupling(fastml_samples)
        plot_matrices1(leaves_mi,draupnir_MAP_mi,draupnir_marginal_mi,draupnir_variational_mi,fastml_mi)
    else:
        print("FastML is not available")
        plot_matrices3(leaves_mi,draupnir_MAP_mi,draupnir_marginal_mi,draupnir_variational_mi)

    #plot_matrices2(leaves_mi,draupnir_MAP_mi,draupnir_marginal_mi,draupnir_variational_mi,fastml_mi,iqtree_mi)

def MI_all():

    leaves_fasta = "{}/{}_training_aligned.fasta".format(draupnir_folder_MAP,name)
    draupnir_fasta_MAP = "{}/Test2_Plots/{}_sampled_ancestors_seq.fasta".format(draupnir_folder_MAP,name)

    if not os.path.exists(draupnir_fasta_MAP):
        create_root_samples_file(draupnir_fasta_MAP,"{}/Test2_Plots".format(draupnir_folder_MAP))
    draupnir_fasta_marginal = "{}/Test_Plots/{}_sampled_ancestors_seq.fasta".format(draupnir_folder_marginal,name)
    if not os.path.exists(draupnir_fasta_marginal):
        create_root_samples_file(draupnir_fasta_marginal,"{}/Test_Plots".format(draupnir_folder_MAP))
    draupnir_fasta_variational = "{}/Test2_Plots/{}_sampled_ancestors_seq.fasta".format(draupnir_folder_variational,name)
    if not os.path.exists(draupnir_fasta_variational):
        create_root_samples_file(draupnir_fasta_variational,"{}/Test_Plots".format(draupnir_folder_MAP))

    if use_indels:
        fastml_samples = "{}/FastML_{}_sampled_ancestors_seq_PROTEIN.fasta".format(benchmark_folder,name)
    else:
        fastml_samples = "{}/FastML_{}_sampled_ancestors_seq_PROTEIN_not_sampled_gaps.fasta".format(benchmark_folder,name)

    #iqtree_samples = "{}/IQTree_{}_root_sampled_ancestors_seq_PROTEIN.fasta".format(benchmark_folder,name)


    leaves_mi = cal_coupling(leaves_fasta)
    draupnir_MAP_mi = cal_coupling(draupnir_fasta_MAP)
    draupnir_marginal_mi = cal_coupling(draupnir_fasta_marginal)
    draupnir_variational_mi = cal_coupling(draupnir_fasta_variational)
    fastml_mi = cal_coupling(fastml_samples)
    #iqtree_mi = cal_coupling(iqtree_samples)

    plot_matrices1(leaves_mi,draupnir_MAP_mi,draupnir_marginal_mi,draupnir_variational_mi,fastml_mi)
    #plot_matrices2(leaves_mi,draupnir_MAP_mi,draupnir_marginal_mi,draupnir_variational_mi,fastml_mi,iqtree_mi)


def create_root_samples_file(file,folder):
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


def calculatemi():
    ""
    if only_root:
        MI_root()
    else:
        MI_all()


def summaryplot(merged_df):
    "Plot median MI of the leaves vs each CC across all datasets"
    #["Dataset","Leaves median","Leaves mean",
                                                                                                                                              # "CC MAPvsLeaves","Mean MAP","Std MAP",
                                                                                                                                              # "CC MarginalvsLeaves","Mean Marginal","Std Marginal",
    #Highlight: Sort dataframe by leaves with most intensity                                                                                                                                   # "CC VariationalvsLeaves","Mean Variational","Std Marginal",
    fig, ax = plt.subplots(figsize=(25, 15))                                                                                                                                   # "CC FastMLvsLeaves","Mean FastML","Std FastML"]
    #merged_df = merged_df.sort_values("Leaves median")
    xlabels = merged_df["Dataset"].tolist()
    merged_df.index = xlabels

    colors = ["skyblue","ForestGreen","GreenYellow","mediumspringgreen","Firebrick"]
    labels = ["Leaves Median","Draupnir MAP","Draupnir Marginal/\n Variational","Draupnir Marginal/Variational","FastML"]

    # for idx,dataset in enumerate(xlabels):
    #     plt.plot(idx,merged_df.loc[dataset,"Leaves median"],'-o',color = colors[0],label=labels[0] if idx == 0 else "",markersize=15)
    #     plt.plot(idx,merged_df.loc[dataset,"CC MAPvsLeaves"],'-o',color = colors[1],label=labels[1] if idx == 0 else "",markersize=15)
    #     plt.plot(idx,merged_df.loc[dataset,"CC MarginalvsLeaves"],'-o',color = colors[2],label=labels[2] if idx == 0 else "",markersize=15)
    #     plt.plot(idx,merged_df.loc[dataset,"CC VariationalvsLeaves"],'-o',color = colors[3],label=labels[3] if idx == 0 else "",markersize=15)
    #     plt.plot(idx,merged_df.loc[dataset,"CC FastMLvsLeaves"],'-o',color = colors[4],label=labels[4] if idx == 0 else "",markersize=15)
    positions = list(range(len(xlabels)))
    #plt.plot(positions,merged_df["Leaves median"],'-o',color = colors[0],label=labels[0] ,markersize=15,linewidth=2)
    plt.plot(positions,merged_df["CC MAPvsLeaves"],'-o',color = colors[1],label=labels[1],markersize=15,linewidth=2)
    plt.plot(positions,merged_df["CC MarginalvsLeaves"],'-o',color = colors[2],label=labels[2],markersize=15,linewidth=2)
    #plt.plot(positions,merged_df["CC VariationalvsLeaves"],'-o',color = colors[3],label=labels[2] ,markersize=15,linewidth=2)
    plt.plot(positions,merged_df["CC FastMLvsLeaves"],'-o',color = colors[4],label=labels[4],markersize=15,linewidth=2)


    ax.set_xticks(positions)
    new_labels = ["PF00018","Protein Kinases","Faviina subclade","Coral subclade","Cnidarian","PF07714","PF00400","Aminopeptidase","PF00096"]
    ax.set_xticklabels(new_labels, rotation=90, fontsize=50)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 27}, framealpha=0.8, ncol=1)
    plt.savefig("/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/Benchmark_Plots2/MI_CC.png",dpi=600,bbox_inches='tight')


def summaryplot2(merged_df):
    "Plot median MI of the leaves vs each CC across all datasets"
    # ["Dataset","Leaves median","Leaves mean",
    # "CC MAPvsLeaves","Mean MAP","Std MAP",
    # "CC MarginalvsLeaves","Mean Marginal","Std Marginal",
    # Highlight: Sort dataframe by leaves with most intensity                                                                                                                                   # "CC VariationalvsLeaves","Mean Variational","Std Marginal",
    fig, ax = plt.subplots(figsize=(25, 15))  # "CC FastMLvsLeaves","Mean FastML","Std FastML"]
    # merged_df = merged_df.sort_values("Leaves median")
    xlabels = merged_df["Dataset"].tolist()
    merged_df.index = xlabels

    colors = ["skyblue", "ForestGreen", "GreenYellow", "mediumspringgreen", "Firebrick"]
    labels = ["Leaves Median", "Draupnir MAP", "Draupnir Marginal/Variational", "Draupnir Marginal/Variational",
              "FastML"]

    # for idx,dataset in enumerate(xlabels):
    #     plt.plot(idx,merged_df.loc[dataset,"Leaves median"],'-o',color = colors[0],label=labels[0] if idx == 0 else "",markersize=15)
    #     plt.plot(idx,merged_df.loc[dataset,"CC MAPvsLeaves"],'-o',color = colors[1],label=labels[1] if idx == 0 else "",markersize=15)
    #     plt.plot(idx,merged_df.loc[dataset,"CC MarginalvsLeaves"],'-o',color = colors[2],label=labels[2] if idx == 0 else "",markersize=15)
    #     plt.plot(idx,merged_df.loc[dataset,"CC VariationalvsLeaves"],'-o',color = colors[3],label=labels[3] if idx == 0 else "",markersize=15)
    #     plt.plot(idx,merged_df.loc[dataset,"CC FastMLvsLeaves"],'-o',color = colors[4],label=labels[4] if idx == 0 else "",markersize=15)
    positions = list(range(len(xlabels)))
    plt.scatter(positions, merged_df["Leaves median"], '-o', color=colors[0], label=labels[0], markersize=15, linewidth=2)
    plt.scatter(positions, merged_df["CC MAPvsLeaves"], '-o', color=colors[1], label=labels[1], markersize=15, linewidth=2)
    plt.scatter(positions, merged_df["CC MarginalvsLeaves"], '-o', color=colors[2], label=labels[2], markersize=15,linewidth=2)
    plt.scatter(positions, merged_df["CC VariationalvsLeaves"], '-o', color=colors[3], label=labels[2], markersize=15,linewidth=2)
    plt.scatter(positions, merged_df["CC FastMLvsLeaves"], '-o', color=colors[4], label=labels[4], markersize=15,linewidth=2)

    ax.set_xticks(positions)
    new_labels = ["PF00018", "Protein Kinases", "Faviina subclade", "Coral subclade", "Cnidarian", "PF07714", "PF00400",
                  "Aminopeptidase", "PF00096"]
    ax.set_xticklabels(new_labels, rotation=90, fontsize=50)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 27}, framealpha=0.8, ncol=1)
    plt.savefig("/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/Benchmark_Plots2/MI_CC.png", dpi=600,
                bbox_inches='tight')

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
    #test_folders = {0:"Test_argmax_Plots",1:"Test_Plots"}
    only_root = True
    use_indels = True # use FastML with the indels reconstructions

    #datasets_list = [3,11,16,17,18,19,27,28,30]
    datasets_list = [27]

    dataframes_list = []
    for d in datasets_list:
        name,draupnir_folder_MAP,draupnir_folder_marginal,draupnir_folder_variational,benchmark_folder = datasets[d]
        calculatemi()


    # dataframes_list = []
    # for d in datasets_list:
    #     name,draupnir_folder_MAP,draupnir_folder_marginal,draupnir_folder_variational,benchmark_folder = datasets[d]
    #     df = pd.read_csv("{}/Summary.csv".format(benchmark_folder),sep="\t")
    #     #calculatemi()
    #     dataframes_list.append(df)
    #
    # merged_df = pd.concat(dataframes_list,axis=0)
    # summaryplot(merged_df)




