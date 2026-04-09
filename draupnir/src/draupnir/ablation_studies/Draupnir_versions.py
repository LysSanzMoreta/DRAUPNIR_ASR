import json
import os.path
from collections import defaultdict
import matplotlib.colors
import pandas as pd
import torch
import sys
import numpy as np


script_dir = os.path.dirname(os.path.abspath(__file__))




local_repository=True
if local_repository:
    sys.path.insert(1,"/home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src")
    sys.path.insert(1,f"/opt/project/draupnir/src")
    sys.path.insert(1,f"/media/lys/0c4a2be6-0148-4ef1-8df3-b89418dfece3/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src")
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


if "opt" in script_dir:
    storage_metrics_folder = "/opt/project/draupnir/src/draupnir/ablation_studies/draupnir_models/metrics"
    storage_folder = "/opt/project/draupnir/src/draupnir/data"
elif "media" in script_dir:
    storage_metrics_folder = "/media/lys/0c4a2be6-0148-4ef1-8df3-b89418dfece3/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src/draupnir/ablation_studies/draupnir_models/metrics"
    storage_folder = "/media/lys/0c4a2be6-0148-4ef1-8df3-b89418dfece3/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src/draupnir/data"
else:
    storage_metrics_folder = "/home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src/draupnir/ablation_studies/draupnir_models/metrics"
    storage_folder = "/home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src/draupnir/data"


def calculate_descendants_consensus_sequence(tree,model_output,nodes_dict):
    """Only calculate for the internal nodes"""

    rename_fx = lambda node: nodes_dict[f"I{node.name}"] if node.name.isdigit() else nodes_dict[node.name]

    root_int_name = rename_fx(tree)
    root_descendants = [rename_fx(descendant) for descendant in tree.get_descendants() if descendant.is_leaf()]
    descendants_dict = {root_int_name:root_descendants}
    true_dataset = model_output["dataset"].detach().cpu().numpy()


    root_sequence = stats.mode(true_dataset[:,2:,0], axis=0, keepdims=True).mode

    descendants_consensus_dict = {root_int_name:root_sequence}
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

def calculate_tree_nodes_depth(storage_folder,dataset_name,mode_name,predictions_dict,lowass_dict=None):

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
        descendants_dict, descendants_consensus_dict = calculate_descendants_consensus_sequence(tree,predictions_dict,nodes_dict)
        consensus_metrics = {}

    else: #todo:compare the consensus sequence to the prediction of the internal node

        descendants_dict = lowass_dict["descendants_dict"]
        descendants_consensus_dict = lowass_dict["descendants_consensus_dict"]
        nodes_order_idx = predictions_dict["dataset"][:,0,1].cpu().detach().numpy().astype(int)
        descendants_consensus =np.concatenate([descendants_consensus_dict[node.item()] for node in nodes_order_idx],axis=0) #if this fails is because some internal node consensus sequence is missing

        predictions_dict = {"aa_predictions":descendants_consensus[None,:,:],
                            "dataset":predictions_dict["dataset"], #true sequences, the descendant consensus need to be ordered according to the node ordering here
                            "dataset_name":predictions_dict["dataset_name"],
                            "folder_path":predictions_dict["folder_path"]
                            }
        consensus_metrics = metrics(predictions_dict, mode_name.replace("_test","_consensus"))
        lowass_dict_consensus = {}
        lowass_dict_consensus = lowass_calculation(lowass_dict,consensus_metrics, consensus_metrics["cosine_similarity"], "lowass_cosine", lowass_dict_consensus)
        lowass_dict_consensus = lowass_calculation(lowass_dict,consensus_metrics, consensus_metrics["pid"].view(-1), "lowass_pid", lowass_dict_consensus)

        consensus_metrics = {**consensus_metrics,**lowass_dict_consensus}



    rename_fx = lambda node: nodes_dict[f"I{node.name}"] if node.name.isdigit() else nodes_dict[node.name]
    node2rootdist = {tree: 0}
    depth_dict = {rename_fx(tree): 1} #root is deepest
    for node in tree.iter_descendants('preorder'):
        int_name = rename_fx(node)
        depth = node.dist + node2rootdist[node.up]
        node2rootdist[node] = depth
        depth_dict[int_name] = 1-depth #not sure if to reverse it like this or not

    return dict(depth_dict = depth_dict,
                descendants_dict=descendants_dict,
                descendants_consensus_dict=descendants_consensus_dict,
                consensus_metrics = consensus_metrics
                )

def lowass_calculation(tree_results_dict,metrics_dict,metric_array,metric_name,lowass_dict):

    depth_array = np.array([tree_results_dict["depth_dict"][node] for node in metrics_dict["nodes_ids_order"]])
    if metric_array.size != 0:
        lowass_score = sm.nonparametric.lowess(endog=metric_array, exog=depth_array, frac=1. / 3)
        lowass_dict[metric_name] = lowass_score
    else:
        lowass_dict[metric_name] = []

    return lowass_dict


def lowass_scores(storage_folder:str,dataset_name:str, mode_name:str, metrics_dict:dict,model_output:dict,lowass_dict:Union[dict | None]):
    """Calculation of the LOWESS (Locally Weighted Scatterplot Smoothing) between the """


    tree_results_dict = calculate_tree_nodes_depth(storage_folder,dataset_name,mode_name,model_output,lowass_dict)

    lowass_dict = {}

    #depth_array = np.array([results_dict["depth_dict"][node] for node in metrics_dict["nodes_ids_order"]])
    pid = metrics_dict["pid"].view(-1)
    lowass_dict = lowass_calculation(tree_results_dict,metrics_dict,pid,"lowass_pid",lowass_dict)
    cosine_similarity = metrics_dict["cosine_similarity"]
    lowass_dict = lowass_calculation(tree_results_dict,metrics_dict,cosine_similarity,"lowass_cosine",lowass_dict)

    #lowass_dict["lowass_pid"] = lowass_pid
    lowass_dict["depth_dict"] = tree_results_dict["depth_dict"]
    lowass_dict["descendants_dict"] = tree_results_dict["descendants_dict"]
    lowass_dict["descendants_consensus_dict"] = tree_results_dict["descendants_consensus_dict"]
    lowass_dict["consensus_metrics"] = tree_results_dict["consensus_metrics"]

    return lowass_dict

def metrics(predictions_dict:dict,mode_name:str="") -> dict:

    """
    Specific epistasis—in which one mutation influences the phenotypic effect of few other mutations—is caused by direct and indirect physical interactions between mutations,
    which nonadditively change the protein's physical properties, such as conformation, stability, or affinity for ligands. In contrast, nonspecific epistasis describes
    mutations that modify the effect of many others; these typically behave additively with respect to the physical properties of a protein but exhibit epistasis because
    of a nonlinear relationship between the physical properties and their biological effects, such as function or fitness
    """

    metrics_dict = defaultdict()
    DraupnirsUtils.folders(mode_name,storage_metrics_folder,overwrite=False)
    mi_dict = draupnir.MI_root_variational(predictions_dict["dataset_name"], predictions_dict["folder_path"], f"{storage_metrics_folder}/{mode_name}")

    aa_predictions = predictions_dict["aa_predictions"].cpu() if isinstance(predictions_dict["aa_predictions"],torch.Tensor) else predictions_dict["aa_predictions"] #should have shape [n_samples, L, feats]
    _,_,blosum_dict = DraupnirsUtils.create_blosum(21,"BLOSUM62")

    nodes_ids_order = predictions_dict["dataset"][:,0,1].cpu().detach().numpy().astype(int)
    dataset_int = predictions_dict["dataset"][:,2:,0].cpu().detach().numpy()

    #Highlight: transform to blosum
    dataset_blosum = np.vectorize(blosum_dict.get,signature='()->(n)')(dataset_int)
    aa_predictions_blosum = np.vectorize(blosum_dict.get,signature='()->(n)')(aa_predictions[0]) #we take the first sample, which is the most likely sequence

    n_seqs, align_lenght = aa_predictions.shape[1], aa_predictions.shape[2]
    percent_id_out = DraupnirMain.calculate_percent_id(predictions_dict["dataset"].cpu().detach(), aa_predictions, align_lenght) # the function slices inside the amino acids

    #metrics dataframe
    dataset_train_nodes = predictions_dict["dataset"].cpu().long()[:, 0, 1]

    if "logits" in predictions_dict.items():
        logits = predictions_dict["logits"].cpu()
        entropies, probabilities_softmax = DraupnirModelsUtils.compute_sites_entropies(logits= logits,node_names = dataset_train_nodes)

        average_entropy = torch.mean(entropies[:,1:])
        average_probabilities = torch.mean(probabilities_softmax[:,1:])

        metrics_dict["average_entropy"] = average_entropy
        metrics_dict["average_probabilities"] = average_probabilities

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

    metrics_dict["nodes_ids_order"] = nodes_ids_order
    metrics_dict["cosine_similarity"] = cosine_similarity
    metrics_dict["pid"] = percent_id_out["equal_aminoacids"]
    metrics_dict["average_pid"] = percent_id_out["average_pid"]
    metrics_dict["average_pid_std"] = percent_id_out["std_pid"]
    metrics_dict["average_cosine_similarity"] = cosine_similarity.mean()*100
    metrics_dict["correlations_leaves_samples_mi"] = mi_dict["correlation"]



    return metrics_dict


#todo: include batched vs no batched
#classic_blosum uses the blosum weighted
folders_dict = {
    "simulations_src_sh3_1": {
        "draupnir_classic_blosum": "draupnir_models/PLOTS_Draupnir_simulations_src_sh3_1_2026_01_20_21h08min29s151422ms_10000epochs_variational",
        "draupnir_classic_no_blosum": "draupnir_models/PLOTS_Draupnir_simulations_src_sh3_1_2026_01_20_21h31min54s951820ms_10000epochs_variational",
        "draupnir_z_esm": "draupnir_models/PLOTS_Draupnir_simulations_src_sh3_1_2026_01_19_20h23min58s667738ms_10000epochs_variational",
        "draupnir_hidden_esm": "draupnir_models/PLOTS_Draupnir_simulations_src_sh3_1_2026_01_19_20h40min49s587822ms_10000epochs_variational",
    },
    "simulations_src_sh3_2": {
        "draupnir_classic_blosum": "draupnir_models/PLOTS_Draupnir_simulations_src_sh3_2_2026_01_16_19h25min29s965361ms_10000epochs_variational",
        "draupnir_classic_no_blosum": "draupnir_models/PLOTS_Draupnir_simulations_src_sh3_2_2026_01_16_23h26min19s155055ms_10000epochs_variational",
        "draupnir_z_esm": "draupnir_models/PLOTS_Draupnir_simulations_src_sh3_2_2026_01_19_12h14min37s844603ms_10000epochs_variational",
        "draupnir_hidden_esm": "draupnir_models/PLOTS_Draupnir_simulations_src_sh3_2_2026_01_19_14h44min01s395934ms_10000epochs_variational",
    },
    "simulations_src_sh3_3": {
        "draupnir_classic_blosum": "",
        "draupnir_classic_no_blosum": "draupnir_models/PLOTS_Draupnir_simulations_src_sh3_3_2026_01_16_16h29min59s341296ms_10000epochs_variational",
        "draupnir_z_esm": "draupnir_models/PLOTS_Draupnir_simulations_src_sh3_3_2026_01_16_17h35min55s614522ms_10000epochs_variational",
        "draupnir_hidden_esm": "draupnir_models/PLOTS_Draupnir_simulations_src_sh3_3_2026_01_16_18h19min00s229714ms_10000epochs_variational",
    },
    "simulations_blactamase_1": {
        "draupnir_classic_blosum": "draupnir_models/PLOTS_Draupnir_simulations_blactamase_1_2026_01_20_21h53min12s628284ms_10000epochs_variational",
        "draupnir_classic_no_blosum": "draupnir_models/PLOTS_Draupnir_simulations_blactamase_1_2026_01_20_23h10min50s249395ms_10000epochs_variational",
        "draupnir_z_esm": "draupnir_models/PLOTS_Draupnir_simulations_blactamase_1_2026_01_21_00h23min24s640140ms_10000epochs_variational",
        "draupnir_hidden_esm": "draupnir_models/PLOTS_Draupnir_simulations_blactamase_1_2026_01_21_01h08min11s963248ms_10000epochs_variational",
    },
    "simulations_calcitonin_1": {
        "draupnir_classic_blosum":"draupnir_models/PLOTS_Draupnir_simulations_calcitonin_1_2026_01_29_18h43min49s404503ms_10000epochs_variational",
        "draupnir_classic_no_blosum":"draupnir_models/PLOTS_Draupnir_simulations_calcitonin_1_2026_01_29_19h01min07s450467ms_10000epochs_variational",
        "draupnir_z_esm":"draupnir_models/PLOTS_Draupnir_simulations_calcitonin_1_2026_01_29_19h17min09s563094ms_10000epochs_variational",
        "draupnir_hidden_esm":"draupnir_models/PLOTS_Draupnir_simulations_calcitonin_1_2026_01_29_19h29min23s604870ms_10000epochs_variational",
                                 },
    "simulations_sirtuins_1": {
        "draupnir_classic_blosum": "draupnir_models/PLOTS_Draupnir_simulations_sirtuins_1_2026_01_19_21h01min46s029776ms_10000epochs_variational",
        "draupnir_classic_no_blosum": "draupnir_models/PLOTS_Draupnir_simulations_sirtuins_1_2026_01_19_23h39min47s797048ms_10000epochs_variational",
        "draupnir_z_esm": "draupnir_models/PLOTS_Draupnir_simulations_sirtuins_1_2026_01_20_01h59min03s763175ms_10000epochs_variational",
        "draupnir_hidden_esm": "draupnir_models/PLOTS_Draupnir_simulations_sirtuins_1_2026_01_20_03h18min33s750688ms_10000epochs_variational",
    },
    "simulations_1GMM": {
        "draupnir_classic_blosum": "draupnir_models/PLOTS_Draupnir_simulations_1GMM_2026_01_15_18h11min01s898288ms_10000epochs_variational",
        "draupnir_classic_no_blosum": "draupnir_models/PLOTS_Draupnir_simulations_1GMM_2026_01_16_00h59min19s996824ms_10000epochs_variational",
        "draupnir_z_esm": "draupnir_models/PLOTS_Draupnir_simulations_1GMM_2026_01_16_06h38min12s618111ms_10000epochs_variational",
        "draupnir_hidden_esm": "draupnir_models/PLOTS_Draupnir_simulations_1GMM_2026_01_15_18h11min01s898288ms_10000epochs_variational",
    },

}


folders_dict = {  "simulations_src_sh3_3": {
        "draupnir_classic_blosum": "",
        "draupnir_classic_no_blosum": "",
        "draupnir_z_esm": "",
        "draupnir_hidden_esm": "",
        "draupnir_whitening": "",
        "draupnir_no_ou_params": "",
    },
    "simulations_1GMM": {
        "draupnir_classic_blosum": "",
        "draupnir_classic_no_blosum": "",
        "draupnir_z_esm": "",
        "draupnir_hidden_esm": "",
        "draupnir_no_ou_params": ""
    },




}


def analyze(results_dict=None):


    if not results_dict is not None:
        results_dict = defaultdict(dict)

    for dataset_name in folders_dict.keys():
        print(f"Dataset: {dataset_name}")
        for mode in folders_dict[dataset_name].keys():
            if f"{mode}_train" not in results_dict[dataset_name].keys():
                print(f"Analyzing {mode}")

                results_folder = folders_dict[dataset_name][mode]
                train_argmax_dict = torch.load("{}/Train_argmax_Plots/train_argmax_info_dict.torch".format(results_folder),weights_only=False)
                train_argmax_dict["folder_path"] = results_folder
                train_argmax_dict["dataset_name"] = dataset_name
                test_argmax_dict = torch.load("{}/Test_argmax_Plots/test_argmax_info_dict.torch".format(results_folder),weights_only=False)
                test_argmax_dict["folder_path"] = results_folder
                test_argmax_dict["dataset_name"] = dataset_name

                print("#########  Metrics train ##########")
                metrics_dict_train = metrics(train_argmax_dict, f"{mode}_train")
                lowass_dict_train = lowass_scores(storage_folder, dataset_name,f"{mode}_train",metrics_dict_train,train_argmax_dict,lowass_dict=None)

                #del metrics_dict["cosine_similarity"] #too big to save it?
                metrics_dict = {**metrics_dict_train,**lowass_dict_train}
                results_dict[dataset_name][f"{mode}_train"] = metrics_dict

                print("########  Metrics test & consensus #######")
                metrics_dict_test = metrics(test_argmax_dict, f"{mode}_test")
                lowass_dict_test = lowass_scores(storage_folder, dataset_name,f"{mode}_test", metrics_dict_test,test_argmax_dict,lowass_dict=lowass_dict_train)

                #del metrics_dict["cosine_similarity"]  # too big to save it
                metrics_dict = {**metrics_dict,**metrics_dict_test, **lowass_dict_test}

                results_dict[dataset_name][f"{mode}_test"] = metrics_dict
                results_dict[dataset_name][f"{mode}_consensus"] = metrics_dict["consensus_metrics"]
                del results_dict[dataset_name][f"{mode}_test"]["consensus_metrics"] #otherwise it is saved twice
                torch.save(results_dict,"draupnir_models/metrics/results_dict2.torch")
            else:
                print(f"{mode} found")


    torch.save(results_dict,"draupnir_models/metrics/results_dict2.torch")
    print("Finished analysis and saved results")
    return results_dict


#results_dict = torch.load("draupnir_models/metrics/results_dict.torch",weights_only=False)
analyze(None)
#analyze(results_dict)

exit()


def build_metrics_table(results_dict,name):
    table_keys = ["average_pid","average_pid_std","average_cosine_similarity","correlations_leaves_samples_mi"]
    skip = ["draupnir_classic_no_blosum_consensus","draupnir_z_esm_consensus","draupnir_hidden_esm_consensus"]
    results_dict_reoriented = defaultdict(dict)
    for dataset in results_dict.keys():
        for model in results_dict[dataset].keys():
            if model not in skip:
                vals_dict = results_dict[dataset][model]
                vals_dict = {key:vals_dict[key] for key in table_keys}

                for key,val in vals_dict.items():
                    if isinstance(val,(np.float16,np.float64,torch.Tensor,np.ndarray)):
                        vals_dict[key] = round(val.item(),4)
                    else:
                        vals_dict[key] = round(val, 4)
                results_dict_reoriented[(dataset,model)] = vals_dict




    colors = {
        "draupnir_classic_blosum_train": matplotlib.colors.to_hex("forestgreen"),
        "draupnir_classic_blosum_test": matplotlib.colors.to_hex("greenyellow"),
        "draupnir_classic_blosum_consensus": matplotlib.colors.to_hex("turquoise"),
        "draupnir_classic_no_blosum_train": matplotlib.colors.to_hex("orangered"),
        "draupnir_classic_no_blosum_test": matplotlib.colors.to_hex("darkorange"),
        "draupnir_z_esm_train": matplotlib.colors.to_hex("mediumorchid"),
        "draupnir_z_esm_test": matplotlib.colors.to_hex("violet"),
        "draupnir_hidden_esm_train": matplotlib.colors.to_hex("dodgerblue"),
        "draupnir_hidden_esm_test": matplotlib.colors.to_hex("royalblue"),

              }

    results_df = pd.DataFrame.from_dict(results_dict_reoriented,orient="index")
    idx = results_df.index.get_level_values(1)

    css1 = [{'selector': f'.row{i}.level1', 'props': [('background-color', colors[v])]} for i, v in enumerate(idx)]

    style = results_df.style

    # for i, _ in results_df.iterrows():
    #     style.set_table_styles({i: [{'selector': f'.row{i}.level0', 'props': 'border-top: 3px solid black;'}]}, overwrite=False, axis=1)
    css2 = [{'selector': f'.row{i}.level0', 'props': 'border-top: 3px solid black;'} for i, v in enumerate(idx)]

    style.background_gradient(axis=0, cmap='YlOrRd').format(precision=4).set_table_styles(css1,overwrite=False).set_table_styles(css2,overwrite=False)

    #todo: put different colours for each of the model types


    sparse_index = get_option("styler.sparse.index")
    sparse_columns = get_option("styler.sparse.columns")
    html = style._render_html(sparse_index, sparse_columns, None, None)

    with open("draupnir_models/metrics/temp.html", "w") as f:
        f.write(html)

    # Convert HTML → PNG
    options = {"format": "png", "encoding": "UTF-8"}
    imgkit.from_file("draupnir_models/metrics/temp.html",
                     f"draupnir_models/metrics/{name}.png", options=options)


def compute_lowass_curves(results_dict):

    # colors_dict = {
    #             "hidden_esm": [matplotlib.colors.to_hex("dodgerblue"),matplotlib.colors.to_hex("royalblue")],
    #             "z_esm": [matplotlib.colors.to_hex("mediumorchid"),matplotlib.colors.to_hex("violet")],
    #             "classic": [matplotlib.colors.to_hex("forestgreen"),matplotlib.colors.to_hex("greenyellow")],
    #             "no": [matplotlib.colors.to_hex("orangered"),matplotlib.colors.to_hex("darkorange")],
    #                }

    colors_dict2 = {
                "hidden_esm": ["dodgerblue","royalblue"],
                "z_esm": ["mediumorchid","violet"],
                "classic_blosum_consensus": ["turquoise", "turquoise"], #keep in this order
                "classic_blosum": ["forestgreen","greenyellow"],
                "no_blosum": ["orangered","darkorange"],

                   }
    #skip = ["draupnir_classic_no_blosum_consensus", "draupnir_classic_z_esm_consensus","draupnir_classic_hidden_esm_consensus"]

    skip=[]
    for dataset in results_dict.keys():
        fig, axs = plt.subplots(nrows=2,ncols=2,figsize = (16,12))
        for mode in results_dict[dataset].keys():
            if mode not in skip:
                vals_dict = results_dict[dataset][mode]
                #color_names = [val for key,val in colors_dict.items() if key in mode][0]
                color_names = [val for key,val in colors_dict2.items() if re.search(key,mode)][0]

                if mode.endswith("_train"):
                    row_idx = 0
                    color_name = color_names[0]
                elif mode.endswith("_test"):
                    row_idx = 1
                    color_name = color_names[0]
                else: #consensus
                    row_idx = 1
                    color_name = color_names[1]

                lowass_cosine = vals_dict["lowass_cosine"] if "lowass_cosine" in vals_dict.keys() else []
                lowass_pid = vals_dict["lowass_pid"]
                if len(lowass_cosine) > 0:#i do not have the cosine sim for all datasets
                    axs[row_idx,0].plot(lowass_cosine[:,0],lowass_cosine[:,1],label=mode,color=color_name)
                axs[row_idx,1].plot(lowass_pid[:,0],lowass_pid[:,1],label=mode,color = color_name)

        axs[0,0].set_title("Train. Lowass cosine")
        axs[0,1].set_title("Train. Lowass pid")
        axs[1,0].set_title("Test. Lowass cosine")
        axs[1,1].set_title("Test. Lowass pid")

        fig.supxlabel("Node depth (1 = root)")
        fig.supylabel("Node reconstruction accuracy")

        axs[0,1].legend(loc="upper right",bbox_to_anchor=(1.7, 1.05))
        axs[1,1].legend(bbox_to_anchor=(1.2, 1))
        fig.suptitle(dataset)
        fig.tight_layout(pad=4.0)
        plt.savefig(f"{storage_metrics_folder}/Lowass_{dataset}.png")
        plt.clf()




results_dict = torch.load("draupnir_models/metrics/results_dict.torch",weights_only=False)
build_metrics_table(results_dict,"results_dict_metrics")

compute_lowass_curves(results_dict)






