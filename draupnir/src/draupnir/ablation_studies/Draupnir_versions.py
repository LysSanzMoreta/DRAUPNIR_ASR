import json
import os.path
from collections import defaultdict

import pandas as pd
import torch
import sys
import numpy as np
local_repository=True
if local_repository:
    sys.path.insert(1,"/home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src")
    import draupnir
else:#pip installed module
    import draupnir

import draupnir.models_utils as DraupnirModelsUtils
import draupnir.utils as DraupnirsUtils
import draupnir.main as DraupnirMain
import draupnir.datasets as DraupnirDatasets
import re
from typing import Union
from sklearn.metrics.pairwise import cosine_similarity
import dromi
import dromi.similarities as DromiSimilarities
import dataframe_image as dfi
from pandas._config import get_option
import imgkit
from ete3 import Tree as TreeEte3
from pprint import pprint
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats

storage_metrics_folder = "/home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src/draupnir/ablation_studies/draupnir_models/metrics"


def calculate_descendants_consensus_sequence(tree,model_output,nodes_dict):
    """Only calculate for the internal nodes"""

    rename_fx = lambda node: nodes_dict[f"I{node.name}"] if node.name.isdigit() else nodes_dict[node.name]

    root_int_name = rename_fx(tree)
    root_descendants = [rename_fx(descendant) for descendant in tree.get_descendants() if descendant.is_leaf()]
    descendants_dict = {root_int_name:root_descendants}
    true_dataset = model_output["dataset"].detach().cpu().numpy()

    descendants_consensus_dict = {}
    for node in tree.iter_descendants('preorder'):
        if not node.is_leaf():  # for internal nodes
            int_name = nodes_dict[f"I{node.name}"] if node.name.isdigit() else nodes_dict[node.name]
            descendants_nodes = [rename_fx(descendant)  for descendant in node.get_descendants() if descendant.is_leaf()]
            descendants_dict[int_name] = descendants_nodes
            descendants_sequences_idx = (true_dataset[:,0,1][..., None] == np.array(descendants_nodes)).any(-1)
            descendants_sequences = true_dataset[descendants_sequences_idx, 2:,0]
            descendants_consensus_sequence =  stats.mode(descendants_sequences, axis=0, keepdims=True).mode
            descendants_consensus_dict[int_name] = descendants_consensus_sequence

    return descendants_dict, descendants_consensus_dict


def calculate_tree_nodes_depth(storage_folder,dataset_name,mode_name,model_output,lowass_dict=None):

    root_sequence_name = DraupnirDatasets.available_datasets(print_dict = False)[0][dataset_name]
    tree_file = "{}/{}/{}_True_Rooted_tree_node_labels.tre".format(storage_folder, dataset_name, root_sequence_name)
    tree = TreeEte3(tree_file, format=1, quoted_node_names=True) #not in tree level order and internal nodes are missing the "I"

    ancestor_info = pd.read_csv("{}/{}/{}_tree_levelorder_info.csv".format(storage_folder,dataset_name,dataset_name), sep="\t",index_col=False,low_memory=False)
    ancestor_info["0"] = ancestor_info["0"].astype(str)
    ancestor_info.drop('Unnamed: 0', inplace=True, axis=1)
    nodes_names = ancestor_info["0"].tolist()

    leaves_nodes_dict = dict((node,i) for i,node in enumerate(nodes_names) if re.search('^A{1}[0-9]+(?![A-Z])+', str(node)))
    internal_nodes_dict = dict((node, i) for i, node in enumerate(nodes_names) if re.search('^(?!^A{1}[0-9]+(?![A-Z])+)', str(node))) #tree level order names
    nodes_dict = {**leaves_nodes_dict,**internal_nodes_dict}

    if mode_name.endswith("train") and lowass_dict is None:
        descendants_dict, descendants_consensus_dict = calculate_descendants_consensus_sequence(tree,model_output,nodes_dict) #todo: we need to calculate the consensus sequence from the train, but use it with the test
    else: #todo:compare the consensus sequence to the prediction of the internal node
        descendants_dict = lowass_dict["descendants_dict"] #todo: inherit the dicts from the train and then calculate pid with the test
        descendants_consensus_dict = lowass_dict["descendants_consensus_dict"]



    node2rootdist = {tree: 1} #root is deepest
    depth_dict = {}
    for node in tree.iter_descendants('preorder'):
        int_name = nodes_dict[f"I{node.name}"] if node.name.isdigit() else nodes_dict[node.name]
        depth = node.dist + node2rootdist[node.up]
        node2rootdist[node] = depth
        depth_dict[int_name] = 1-depth #not sure if to reverse it like this or not

    return dict(depth_dict = depth_dict,
                descendants_dict=descendants_dict,
                descendants_consensus_dict=descendants_consensus_dict
                )


def lowass_scores(storage_folder:str,dataset_name:str, mode_name:str, metrics_dict:dict,model_output:dict,lowass_dict:Union[dict | None]):
    """Calculation of the LOWESS (Locally Weighted Scatterplot Smoothing) between the """

    results_dict = calculate_tree_nodes_depth(storage_folder,dataset_name,mode_name,model_output,lowass_dict)
    depth_array = np.array([results_dict["depth_dict"][node] for node in metrics_dict["nodes_ids_order"]])
    pid = metrics_dict["equal_aminoacids"]
    cosine_similarity = metrics_dict["cosine_similarity"]
    lowass_pid = sm.nonparametric.lowess(endog=pid, exog=depth_array, frac=1. / 3)
    lowass_cosine = sm.nonparametric.lowess(endog=cosine_similarity, exog=depth_array, frac=1. / 3)


    return {"lowass_pid":lowass_pid,
            "lowass_cosine":lowass_cosine}


def metrics(predictions_dict:dict,dataset_name:str,results_folder:str,mode_name:str="") -> dict:

    """
    Specific epistasis—in which one mutation influences the phenotypic effect of few other mutations—is caused by direct and indirect physical interactions between mutations,
    which nonadditively change the protein's physical properties, such as conformation, stability, or affinity for ligands. In contrast, nonspecific epistasis describes
    mutations that modify the effect of many others; these typically behave additively with respect to the physical properties of a protein but exhibit epistasis because
    of a nonlinear relationship between the physical properties and their biological effects, such as function or fitness
    """

    DraupnirsUtils.folders(mode_name,storage_metrics_folder,overwrite=False)
    #mi_dict = draupnir.MI_root_variational(dataset_name, results_folder, f"{storage_folder}/{name}")

    aa_predictions = predictions_dict["aa_predictions"].cpu()
    _,_,blosum_dict = DraupnirsUtils.create_blosum(21,"BLOSUM62")

    nodes_ids_order = predictions_dict["dataset"][:,0,1].cpu().detach().numpy()
    dataset_int = predictions_dict["dataset"][:,2:,0].cpu().detach().numpy()

    #Highlight: transform to blosum
    dataset_blosum = np.vectorize(blosum_dict.get,signature='()->(n)')(dataset_int)
    aa_predictions_blosum = np.vectorize(blosum_dict.get,signature='()->(n)')(aa_predictions[0]) #we take the first sample

    n_seqs, align_lenght = predictions_dict["aa_predictions"].shape[1], predictions_dict["aa_predictions"].shape[2]
    percent_id_out = DraupnirMain.calculate_percent_id(predictions_dict["dataset"].cpu().detach(), aa_predictions, align_lenght) # the function slices inside the amino acids

    #metrics dataframe
    dataset_train_nodes = predictions_dict["dataset"].cpu().long()[:, 0, 1]

    logits = predictions_dict["logits"].cpu()
    entropies, probabilities_softmax = DraupnirModelsUtils.compute_sites_entropies(logits= logits,node_names = dataset_train_nodes)

    average_entropy = torch.mean(entropies[:,1:])
    average_probabilities = torch.mean(probabilities_softmax[:,1:])

    concat_predictions = np.concatenate([dataset_blosum,aa_predictions_blosum],axis=0)
    total_seqs = concat_predictions.shape[0]
    array_mask = np.ones_like(concat_predictions[:,:,0]).astype(bool) #we have to take into account the gaps
    overwrite=False
    filepath = f"{storage_metrics_folder}/{mode_name}/cosine_similarity_mean.npy"
    if not os.path.exists(filepath) or overwrite:
        if total_seqs < 1000: #todo: how to only compute the similarities between true and predicted, and not true vs true or predicted vs predicted --> some flattening technique
            results, final_time = DromiSimilarities.calculate_similarities(concat_predictions, align_lenght, array_mask, f"{storage_metrics_folder}/{mode_name}",
                                                                           batch_size=100,
                                                                           ksize=3,
                                                                           neighbours=1,
                                                                           metric="cosine",
                                                                           calculate_kmers=False,
                                                                           calculate_positional_weights=False) #calculate_similarities_ondisk

            cosine_similarity = np.load(filepath) #contains cosine similarity stacked [nseqs + nseqs, nseqs + nseqs]
        else:
            cosine_similarity = np.ones((n_seqs,n_seqs))*np.nan
    else:

        cosine_similarity = np.load(filepath)


    cosine_similarity = np.diagonal(cosine_similarity[:n_seqs,n_seqs:]) #we get the similarities of true vs predicted only, in this case top right corner


    metrics_dict = {
                    # "probabilities_softmax": probabilities_softmax,
                    # "entropies": entropies,
                    "nodes_ids_order": nodes_ids_order,
                    "cosine_similarity": cosine_similarity,
                    "pid": percent_id_out["equal_aminoacids"],
                    "average_pid": percent_id_out["average_pid"],
                    "average_std": percent_id_out["std_pid"],
                    "average_entropy": average_entropy,
                    "average_probabilities": average_probabilities,
                    "average_cosine_similarity": cosine_similarity.mean()*100,
                    #"correlation_leaves_samples_mi": mi_dict["correlation"]
                    }



    return metrics_dict


folders_dict = {"simulations_src_sh3_3":{
                "draupnir_classic":"/home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src/draupnir/ablation_studies/draupnir_models/PLOTS_Draupnir_simulations_src_sh3_3_2025_12_03_14h24min11s451285ms_3000epochs_variational",
                # "draupnir_z_esm":"/home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src/draupnir/ablation_studies/draupnir_models/PLOTS_Draupnir_simulations_src_sh3_3_2025_12_03_15h14min18s836930ms_3000epochs_variational_z_esm",
                # "draupnir_hid_esm":"/home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src/draupnir/ablation_studies/draupnir_models/PLOTS_Draupnir_simulations_src_sh3_3_2025_12_03_14h48min20s543620ms_3000epochs_variational_rnn_esm_embeddings",
                # "draupnir_classic_2":"/home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src/draupnir/ablation_studies/draupnir_models/PLOTS_Draupnir_simulations_src_sh3_3_2025_12_08_18h31min09s305015ms_3000epochs_variational",
                # "draupnir_z_esm_2":"/home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src/draupnir/ablation_studies/draupnir_models/PLOTS_Draupnir_simulations_src_sh3_3_2025_12_08_20h06min16s422135ms_3000epochs_variational_z_esm",
                # "draupnir_hid_esm_2":"/home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src/draupnir/ablation_studies/draupnir_models/PLOTS_Draupnir_simulations_src_sh3_3_2025_12_09_14h33min09s636948ms_3000epochs_variational_rnn_esm_embeddings",
                # "draupnir_classic_3" : "/home/lys/Dropbox/PhD/DRAUPNIR_ASR/PLOTS_Draupnir_simulations_src_sh3_3_2025_12_12_14h04min29s168767ms_3000epochs_variational",
                # "draupnir_z_esm_3" : "/home/lys/Dropbox/PhD/DRAUPNIR_ASR/PLOTS_Draupnir_simulations_src_sh3_3_2025_12_12_16h26min18s402748ms_3000epochs_variational_z_esm",
                # "draupnir_hid_esm_3" : "/home/lys/Dropbox/PhD/DRAUPNIR_ASR/PLOTS_Draupnir_simulations_src_sh3_3_2025_12_12_17h10min23s288319ms_3000epochs_variational_rnn_esm_embeddings",
                },
    #             "simulations_1GMM":{
    #             "draupnir_classic_60_samples": "/home/lys/Dropbox/PhD/DRAUPNIR_ASR/PLOTS_Draupnir_simulations_1GMM_2025_12_19_14h40min42s087801ms_0epochs_variational",
    #             "draupnir_classic_200_samples": "/home/lys/Dropbox/PhD/DRAUPNIR_ASR/PLOTS_Draupnir_simulations_1GMM_2026_01_07_12h09min45s765532ms_0epochs_variational",
    #             "draupnir_z_esm": "/home/lys/Dropbox/PhD/DRAUPNIR_ASR/PLOTS_Draupnir_simulations_1GMM_2025_12_26_14h52min35s209209ms_0epochs_variational",
    #             "draupnir_hid_esm": "/home/lys/Dropbox/PhD/DRAUPNIR_ASR/PLOTS_Draupnir_simulations_1GMM_2025_12_27_18h16min44s519045ms_0epochs_variational"
    #             }
    # ,
}


def analyze(results_dict=None):

    storage_folder = "/home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src/draupnir/data"
    if not results_dict is not None:
        results_dict = defaultdict(dict)

    for dataset_name in folders_dict.keys():
        for mode in folders_dict[dataset_name].keys():
            if f"{mode}_train" not in results_dict[dataset_name].keys():
                print(f"Analyzing {mode}")

                results_folder = folders_dict[dataset_name][mode]
                train_argmax_dict = torch.load("{}/Train_argmax_Plots/train_argmax_info_dict.torch".format(results_folder),
                                               weights_only=False)
                test_argmax_dict = torch.load("{}/Test_argmax_Plots/test_argmax_info_dict.torch".format(results_folder),
                                              weights_only=False)

                print("Metrics train")
                metrics_dict = metrics(train_argmax_dict, dataset_name, results_folder, f"{mode}_train")
                lowass_dict_train = lowass_scores(storage_folder, dataset_name,"train",metrics_dict,train_argmax_dict,lowass_dict=None)



                #del metrics_dict["cosine_similarity"] #to big to save it?
                metrics_dict = {**metrics_dict,**lowass_dict_train}
                results_dict[dataset_name][f"{mode}_train"] = metrics_dict

                print("Metrics test")
                metrics_dict = metrics(test_argmax_dict, dataset_name, results_folder, f"{mode}_test")
                lowass_dict_test = lowass_scores(storage_folder, dataset_name,"test", metrics_dict,test_argmax_dict,lowass_dict=lowass_dict_train)
                #del metrics_dict["cosine_similarity"]  # to big to save it
                metrics_dict = {**metrics_dict, **lowass_dict_test}

                results_dict[dataset_name][f"{mode}_test"] = metrics_dict

    torch.save(results_dict,"/home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src/draupnir/ablation_studies/draupnir_models/metrics/results_dict_new.torch")
    return results_dict


#results_dict = torch.load("/home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src/draupnir/ablation_studies/draupnir_models/metrics/results_dict_new.torch",weights_only=False)
results_dict=None
analyze(results_dict)


results_dict_reoriented = defaultdict(dict)
for i in results_dict.keys():
    for j in results_dict[i].keys():
        vals_dict = results_dict[i][j]
        for key,val in vals_dict.items():
            if isinstance(val,(np.float16,np.float64,torch.Tensor,np.ndarray)):
                vals_dict[key] = round(val.item(),4)
            else:
                vals_dict[key] = round(val, 4)
        results_dict_reoriented[(i,j)] = vals_dict

results_df = pd.DataFrame.from_dict(results_dict_reoriented,orient="index")

print(results_df)

results_df_styled = results_df.style.background_gradient(axis=0, cmap='YlOrRd').format(precision=4)

sparse_index = get_option("styler.sparse.index")
sparse_columns = get_option("styler.sparse.columns")
html = results_df_styled._render_html(sparse_index, sparse_columns, None, None)

with open("/home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src/draupnir/ablation_studies/draupnir_models/metrics/temp.html", "w") as f:
    f.write(html)

# Convert HTML → PNG
options = {"format": "png", "encoding": "UTF-8"}
imgkit.from_file("/home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src/draupnir/ablation_studies/draupnir_models/metrics/temp.html",
                 "/home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src/draupnir/ablation_studies/draupnir_models/metrics/results_dict.png", options=options)




















#/home/lys/Dropbox/PhD/DRAUPNIR_ASR/PLOTS_Draupnir_simulations_1GMM_2026_01_07_12h09min45s765532ms_0epochs_variational/Train_argmax_Plots/train_argmax_info_dict.torch
#/home/lys/Dropbox/PhD/DRAUPNIR_ASR/PLOTS_Draupnir_simulations_1GMM_2025_12_19_14h40min42s087801ms_0epochs_variational
#/home/lys/Dropbox/PhD/DRAUPNIR_ASR/PLOTS_Draupnir_simulations_1GMM_2025_12_26_14h52min35s209209ms_0epochs_variational/Train_argmax_Plots/train_argmax_info_dict.torch



