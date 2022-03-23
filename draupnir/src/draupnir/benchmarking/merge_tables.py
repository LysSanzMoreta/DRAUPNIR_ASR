import pandas as pd
import numpy as np
import datetime,os,glob
from functools import partial
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib
def merge_dataframes(input_type,dataframes_path):
    "Merges the PercentID dataframes"
    dataframes_dict = dict.fromkeys(["MAP","samples"])

    for fname in os.listdir(dataframes_path):
        if "PercentID" in fname and input_type in fname:
            test_type = ["MAP" if fname.split("_")[-1] == "argmax.tex" else "samples"][0]
            headers = ["Draupnir {}".format(test_type), "PAML-CodeML", "PhyloBayes", "FastML","IQTree"]
            df = pd.read_csv(os.path.join(dataframes_path,fname),
                                                 sep='&',
                                                 header=None,
                                                 skiprows=4,
                                                 skipfooter=2,
                                                 engine='python',
                                                 index_col=0)
            df.columns = headers
            dataframes_dict[test_type] = df


    merged_df  = pd.merge(dataframes_dict["samples"],dataframes_dict["MAP"], on=["PAML-CodeML", "PhyloBayes", "FastML","IQTree"],left_index=True, right_index=True)
    merged_df = merged_df[["Draupnir MAP","Draupnir samples","PAML-CodeML", "PhyloBayes", "FastML","IQTree"]]
    for column in merged_df.columns:
        merged_df[column] = merged_df[column].str.replace("\\","",regex=True).str.replace("textbf","\textbf")

    merged_df.to_latex(os.path.join(dataframes_path,"Merged_{}.tex".format(input_type)), escape=False,
                                    na_rep=np.nan, column_format="lcccccc")  # escape false to avoid probles with \

def large_dataset_accuracy():
    "Grabbing the %ID for the leaves datasets and the larger datasets---NOT used anymore "
    plot_folder = "PLOTS_GP_VAE_simulations_src_sh3_2_2021_09_21_21h07min21s172926ms_12000epochs_SRU"
    folders = {"Test_argmax_Plots":["Draupnir MAP($\mu$)","Draupnir MAP($\sigma$)"],"Test_Plots":["Draupnir samples($\mu$)","Draupnir samples($\sigma$)"]}
    dataset_info= ["800","99", "Simulation SRC-Kinase SH3 domain, 800"]


    average_dict = dict.fromkeys([value[0] for value in folders.values()])
    std_dict = dict.fromkeys([value[1] for value in folders.values()])
    for folder in folders.keys():
        folder_path = "/home/lys/Dropbox/PhD/DRAUPNIR/{}/{}/PercentID_df.csv".format(plot_folder,folder)
        df = pd.read_csv(folder_path,sep="\t", index_col=0)
        average_dict[folders[folder][0]] =  round(df["Average"].mean(0),2)
        std_dict[folders[folder][1]] = round(df["Average"].std(0), 2)

    dataframe = pd.DataFrame([*average_dict.values(),*std_dict.values()]).T
    dataframe.columns =  [*average_dict.keys(),*std_dict.keys()]
    dataframe.index = [dataset_info[2]]
    dataframe["Number of leaves"] = dataset_info[0]
    dataframe["Alignment length"] = dataset_info[1]
    dataframe = dataframe[["Number of leaves","Alignment length","Draupnir MAP($\mu$)","Draupnir MAP($\sigma$)","Draupnir samples($\mu$)","Draupnir samples($\sigma$)"]]
    dataframe["PAML-CodeML($\mu$)"] = ""
    dataframe["PAML-CodeML($\sigma$)"] = ""
    dataframe["PhyloBayes($\mu$)"] = ""
    dataframe["PhyloBayes($\sigma$)"] = ""
    dataframe["FastML($\mu$)"] = ""
    dataframe["FastML($\sigma$)"] = ""
    dataframe = dataframe.astype(str)
    return dataframe
def combine(input_type,datasets_folder_names):
    "Combines the Merged dataframes"

    datasets_names = {"AncestralResurrectionStandardDataset":["19","225", "Randall's Coral fluorescent proteins (CFP)"], #[number of leaves, alignment length, dataset name]
                      "Coral_all":["71","272" ,"Coral fluorescent proteins (CFP) clade"],
                      "Coral_Faviina":["35","261" ,"Coral fluorescent proteins (CFP) Faviina subclade"],
                      "BLactamase/Dataset1":["32","314" ,"Simulation Beta-Lactamase"],
                      "Calcitonin_simulations/Dataset1": ["50", "71", "Simulation Calcitonin, 50"],
                      "SRC_simulations/Dataset1":["100","63","Simulation SRC-Kinase SH3 domain, 100"],
                      "Sirtuin_simulations/Dataset1": ["150", "477", "Simulation Sirtuin SIRT1, 150"],
                      "SRC_simulations/Dataset3":["200","128" ,"Simulation SRC-Kinase SH3 domain, 200"],
                      "PIGBOS_simulations/Dataset1": ["300", "77", "Simulation PIGB Opposite Strand regulator"],
                      "Insulin_simulations/Dataset2": ["400", "558", "Simulation Insulin Factor like"],
                      "SRC_simulations/Dataset2":["800","99" ,"Simulation SRC-Kinase SH3 domain, 800"],
                      }
    headers = [r"Draupnir MAP($\mu$)", r"Draupnir samples($\mu$)", r"PAML-CodeML($\mu$)", r"PhyloBayes($\mu$)", r"FastML($\mu$)",r"IQTree($\mu$)"]
    datasets_average_dict = dict.fromkeys([info[2] for info in datasets_names.values()]) #TODO: check that this is correct
    datasets_std_dict = dict.fromkeys([info[2] for info in datasets_names.values()])
    for dataset in datasets_folder_names:
        dataset_path = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/Benchmark_Plots2/{}/{}".format(dataset,rnn_type)
        for fname in os.listdir(dataset_path):
            if "Merged_{}".format(input_type) in fname :
                df = pd.read_csv(os.path.join(dataset_path, fname),
                                 sep='&',
                                 header=None,
                                 skiprows=5,
                                 skipfooter=2,
                                 engine='python',
                                 index_col=0)
                df = df.replace({r"textbf{":""}, regex=True)
                df = df.replace(r'\\','', regex=True)
                df = df.replace({"}":""},regex=True)
                df = df.astype(float)
                df.columns = headers
                df_average_series = df.mean(axis=0)
                df_std_series = df.std(axis=0).round(2)
                full_name = datasets_names[dataset][2]
                df_average = pd.DataFrame(df_average_series,columns=[full_name])
                df_std = pd.DataFrame(df_std_series, columns=[full_name]) #index=[name.replace("\mu","\sigma") for name in df_std_series.index]
                datasets_average_dict[full_name] = df_average.transpose()
                datasets_std_dict[full_name] = df_std.transpose()

    merged_df_average = pd.concat(datasets_average_dict.values())
    merged_df_std = pd.concat(datasets_std_dict.values())
    merged_df_std.columns = [column.replace("\mu","\sigma") for column in merged_df_std]
    #Highlight: Create table with hightlighted top results
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
    merged_df_average = merged_df_average.transpose()
    column_names = [column[2] for column in datasets_names.values()]
    format_dict = {column: partial(bold_formatter, value=merged_df_average[column].max(), num_decimals=2) for column in column_names}
    merged_df_average.to_latex(buf="/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/Benchmark_Plots2/Combined_average_{}.tex".format(input_type),
                          bold_rows=True,
                          escape=False,
                          formatters= format_dict,
                          na_rep=np.nan,
                          index_names=datasets_names.values())
    # Highlight: weird trick to transpose the table
    from astropy.table import Table
    merged_df = Table.read('/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/Benchmark_Plots2/Combined_average_{}.tex'.format(input_type)).to_pandas(index='col0')
    merged_df.index = headers
    percendid_df_transpose = merged_df.transpose()
    percendid_df_transpose["Number of leaves"] = [info[0] for info in datasets_names.values()]
    percendid_df_transpose["Alignment length"] = [info[1] for info in datasets_names.values()]
    percendid_df_transpose = percendid_df_transpose[["Number of leaves","Alignment length",r"Draupnir MAP($\mu$)", r"Draupnir samples($\mu$)", r"PAML-CodeML($\mu$)", r"PhyloBayes($\mu$)", r"FastML($\mu$)",r"IQTree($\mu$)"]]
    percendid_df_transpose = pd.concat([percendid_df_transpose,merged_df_std],axis=1)
    percendid_df_transpose = percendid_df_transpose[
        ["Number of leaves", "Alignment length", r"Draupnir MAP($\mu$)",r"Draupnir MAP($\sigma$)", r"Draupnir samples($\mu$)",r"Draupnir samples($\sigma$)",
         r"PAML-CodeML($\mu$)",r"PAML-CodeML($\sigma$)", r"PhyloBayes($\mu$)",r"PhyloBayes($\sigma$)", r"FastML($\mu$)",r"FastML($\sigma$)",r"IQTree($\mu$)",r"IQTree($\sigma$)"]]

    #Highlight: Append the larger datasets and leaf datasets
    # large_dataset = large_dataset_accuracy()
    # percendid_df_transpose = percendid_df_transpose.append(large_dataset)
    #TODO: Concat the columns as mu +- std
    percendid_df_transpose.to_latex("/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/Benchmark_Plots2/Combined_average_{}.tex".format(input_type),escape=False,column_format="lccccccccccccc")
    for program in ["Draupnir MAP","Draupnir samples","PAML-CodeML","PhyloBayes","FastML","IQTree"]:
        percendid_df_transpose[program] = percendid_df_transpose[program+r"($\mu$)"].astype(str)+ "$\pm$" + percendid_df_transpose[program +r"($\sigma$)"].astype(str)
        percendid_df_transpose.pop("{}($\mu$)".format(program))
        percendid_df_transpose.pop("{}($\sigma$)".format(program))
    percendid_df_transpose.to_latex(
        "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/Benchmark_Plots2/Combined_average_{}_table.tex".format(input_type),
        escape=False, na_rep=np.nan, column_format="lcccccccc")


def plot_performance():
    "Plot the combined tables, which sum up the average accuracy"
    print("Plotting performance")
    combined_averages = {"DNA":"/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/Benchmark_Plots2/Combined_average_DNA.tex",
                         "Protein":"/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/Benchmark_Plots2/Combined_average_PROTEIN.tex"}
    headers = ["Number of leaves","Alignment length","Draupnir MAP($\mu$)","Draupnir MAP($\sigma$)", "Draupnir samples($\mu$)",
        "Draupnir samples($\sigma$)", "PAML-CodeML($\mu$)","PAML-CodeML($\sigma$)", "PhyloBayes($\mu$)","PhyloBayes($\sigma$)", "FastML($\mu$)","FastML($\sigma$)",r"IQTree($\mu$)",r"IQTree($\sigma$)"]
    #colors = plt.get_cmap("gray", 5)
    colors_cmap  = matplotlib.cm.gray(np.linspace(0, 1, 6))
    fig, (ax1, ax2) = plt.subplots(1, 2,sharex=True,sharey=True,figsize=(25,10))
    axes = [ax1,ax2]
    #colors = ["yellowgreen", "green", "darkturquoise", "orange", "deeppink"]
    for ax,(name,df_name) in zip(axes,combined_averages.items()):
        df = pd.read_csv(df_name,
                         sep='&',
                         header=None,
                         skiprows=4,
                         skipfooter=2,
                         engine='python',
                         index_col=0)

        df= df.replace({r"textbf{": ""}, regex=True)
        df = df.replace(r'\\', '', regex=True)
        df= df.replace({"}": ""}, regex=True)
        df = df.apply(pd.to_numeric, args=('coerce',))
        df.columns = headers
        df = df.sort_values("Number of leaves")
        #fig, ax= plt.subplots(figsize=(18, 13))
        #xlabels = [r"$\mathbf{19^{*}}$",'$\mathbf{32}$      ',r"$\mathbf{35^{*}}$",r"$\mathbf{50}$",r"$\mathbf{71^{*}}$",r"$\mathbf{100}$",r"$\mathbf{150}$",r"$\mathbf{200}$",r"$\mathbf{800}$"] #TODO: needs to be updated manually
        #xlabels = [r"$19^{*}$",'$32$      ',r"$35^{*}$",r"$50$",r"$71^{*}$",r"$100$",r"$150$",r"$200$",r"$800$"] #TODO: needs to be updated manually, automatize
        xlabels = [r"$19^{*}$",'$32$      ',r"$35^{*}$",r"$50$",r"$71^{*}$",r"$100$",r"$150$",r"$200$",r"$300$",r"$400$",r"$800$"] #TODO: needs to be updated manually, automatize

        programs = [r"IQTree","PAML-CodeML", "PhyloBayes","FastML","Draupnir MAP", "Draupnir marginal samples"]
        for idx,program,color,marker,linestyle,color in zip(list(range(len(programs))),
                    [ r"IQTree($\mu$)","PAML-CodeML($\mu$)", "PhyloBayes($\mu$)",
                     "FastML($\mu$)","Draupnir MAP($\mu$)", "Draupnir samples($\mu$)"],colors_cmap,["s","D","h","p","o","X"],["dotted","dashed","dashdot", (0, (3, 1, 1, 1)),"solid","solid"],[ "darkturquoise", "orange", "deeppink","firebrick","yellowgreen", "green"]):
            values = df[program].tolist() #percentage of identity

            # if np.isnan(values).any(): #remove single datapoints, due to missing data in between
            #     index_nan = np.argwhere(np.isnan(values))[0][0]
            #     if index_nan < len(values)-1:
            #         values[index_nan:] = [np.nan]*(len(values)-index_nan)
            ##p = program.replace("($\mu$)","")

            ax.plot(df["Number of leaves"].tolist(),values,color=color,linestyle="solid", marker=marker,label=programs[idx],alpha=1,markersize=15,linewidth=2)
            #ax.errorbar(df["Number of leaves"].tolist(),values,yerr=df[program.replace("\mu","\sigma")].tolist(),fmt='-',color="black",linestyle=linestyle, marker=marker,label=program.replace("($\mu$)",""),alpha=1,markersize=15,linewidth=2)

        ax.set_xscale('log')
        ax.set_xticks(df["Number of leaves"].tolist())
        ax.set_xticklabels(xlabels, rotation=90,fontsize=40)
        #ax.tick_params(axis='x', which='major', pad=15)


        trans = ax.get_xaxis_transform()
        for n_leaves,alig_len in zip(df["Number of leaves"].tolist(),df["Alignment length"].tolist()):
            ax.vlines(n_leaves,linestyles="dashed", ymax=100, ymin=0, alpha=0.4,color="grey")
            if n_leaves == 32:
                t = ax.text(n_leaves, .17, alig_len, transform=trans,ha='center', va='center',rotation='vertical',fontsize=30) #backgroundcolor='white'
            else:
                t = ax.text(n_leaves, .32, alig_len, transform=trans, ha='center', va='center', rotation='vertical',fontsize=30)  # backgroundcolor='white'
            t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='white'))

        ax.set_ylim(0,100)
        ax.tick_params(axis="y", labelsize=40)
        ax.set_title(name,fontdict = {'fontsize':25})

    # handles, labels = axes[-1].get_legend_handles_labels()
    # fig.legend(handles, labels, loc = "upper center",bbox_to_anchor=(0.5, 1),bbox_transform=plt.gcf().transFigure,borderaxespad=0.)

    h, l = axes[-1].get_legend_handles_labels()
    kw = dict(ncol=3, loc="lower center", frameon=False,fontsize=26)
    leg1 = axes[0].legend(h[:3], l[:3], bbox_to_anchor=[0.27, 1.08], **kw)
    leg2 = axes[1].legend(h[3:], l[3:], bbox_to_anchor=[0.5, 1.08], **kw)
    fig.add_artist(leg1)


    fig.text(0.47, 0.07, 'Number of leaves', ha='center',fontsize=35)
    fig.text(0.04, 0.6, 'Percentage of Identity', va='center', rotation='vertical',fontsize=35)
    #fig.supxlabel("Number of leaf nodes in the dataset", fontsize=50)
    plt.subplots_adjust(wspace=0.1, hspace=0,bottom=0.25)
    #fig.tight_layout()
    plt.savefig("/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/Benchmark_Plots2/Performance_comparison_combined_{}.pdf".format(rnn_type),dpi=600)
    print("......................................")
def main(datasets_to_analyze,datasets_folders_name):
    input_types = ["DNA","PROTEIN"]
    for dataset_number in datasets_to_analyze:
        name, dataset_number, simulation_folder, root_sequence_name,folder = datasets[dataset_number]
        print("Merging {}".format(name))
        dataframes_path = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/Benchmark_Plots2/{}/{}".format(folder,rnn_type)
        if name.endswith("_subtree"):
            input_type = "PROTEIN"
            merge_dataframes(input_type, dataframes_path)
        else:
            for input_type in input_types:
                merge_dataframes(input_type,dataframes_path)
        print("Done merging {}".format(name))

    print("Final combination")
    for input_type in input_types:
        print("Using {}".format(input_type))
        combine(input_type,datasets_folders_name)

if __name__ == "__main__":
    now = datetime.datetime.now()
    datasets = {0: ["benchmark_randall_original_naming", None, None, None,"AncestralResurrectionStandardDataset"],# uses the original tree and it's original node naming
                1: ["simulations_blactamase_1", 1, "BLactamase", "BetaLactamase_seq","BLactamase/Dataset1"],# EvolveAGene4 Betalactamase simulation # 32 leaves
                2: ["simulations_src_sh3_1", 1, "SRC_simulations", "SRC_SH3","SRC_simulations/Dataset1"],# EvolveAGene4 SRC SH3 domain simulation 1 #100 leaves# 6:["simulations_src_sh3_2",2, "SRC_simulations","SRC_SH3"],  #EvolveAGene4 SRC SH3 domain simulation 2 #800 leaves
                3: ["simulations_src_sh3_3", 3, "SRC_simulations", "SRC_SH3","SRC_simulations/Dataset3"],# EvolveAGene4 SRC SH3 domain simulation 2 #200 leaves
                4: ["simulations_src_sh3_2", 2, "SRC_simulations", "SRC_SH3", "SRC_simulations/Dataset2"],#800 leaves
                5: ["simulations_sirtuins_1", 1, "Sirtuin_simulations", "Sirtuin_seq","Sirtuin_simulations/Dataset1"],# EvolveAGene4 Sirtuin simulation #150 leaves
                6: ["simulations_insulin_1", 1, "Insulin_simulations", "Insulin_seq","Insulin_simulations/Dataset1"],# EvolveAGene4 Insulin simulation #50 leaves
                7: ["simulations_calcitonin_1", 1, "Calcitonin_simulations", "Calcitonin_seq","Calcitonin_simulations/Dataset1"],# EvolveAGene4 Calcitonin simulation #50 leaves
                # 10: ["simulations_mciz_1",1, "Mciz_simulations","Mciz_seq"],  # EvolveAGene4 MciZ simulation # 1600 leaves
                8: ["ANC_A1_subtree", None, None, None,"ANC_A1_subtree"],
                9: ["ANC_A2_subtree", None, None, None,"ANC_A2_subtree"],  # highlight: 3D structure not available
                10: ["ANC_AS_subtree", None, None, None,"ANC_AS_subtree"],
                11: ["ANC_S1_subtree", None, None, None,"ANC_S1_subtree"],  # highlight: 3D structure not available
                12: ["Coral_Faviina", None, None, None,"Coral_Faviina"],  # Faviina clade from coral sequences
                13: ["Coral_all", None, None, None,"Coral_all"],# All Coral sequences (includes Faviina clade and additional sequences
                14: ["simulations_insulin_2", 2, "Insulin_simulations", "Insulin_seq","Insulin_simulations/Dataset2"],# EvolveAGene4 Insulin simulation #400 leaves
                15: ["simulations_PIGBOS_1", 1, "PIGBOS_simulations", "PIGBOS_seq","PIGBOS_simulations/Dataset1"],# EvolveAGene4 PIGBOS simulation #300 leaves
                }

    datasets_to_analyze = [0, 12, 13, 1, 7, 5, 2, 3, 15,14,4] #new added datasets
    rnn_type = "GRU_map"
    datasets_folders_name = [datasets[number][-1] for number in datasets_to_analyze]
    main(datasets_to_analyze,datasets_folders_name)
    plot_performance()