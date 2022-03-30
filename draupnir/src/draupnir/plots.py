"""
=======================
2022: Lys Sanz Moreta
Draupnir : Ancestral protein sequence reconstruction using a tree-structured Ornstein-Uhlenbeck variational autoencoder
=======================
"""
import os, itertools
from collections import defaultdict
import torch
import numpy as np
import draupnir.utils as DraupnirUtils
import draupnir.models_utils as DraupnirModelUtils
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib
from Bio import SeqRecord, SeqIO
from Bio.Seq import Seq
import seaborn as sns
from sklearn.decomposition import PCA
from Bio.SeqUtils import seq3
from sklearn.manifold import TSNE
import statistics
import umap
from scipy import stats,spatial
import matplotlib
matplotlib.use('Agg')

def plot_ELBO(train_elbo,results_dict):
    """Plots the model's error loss
    :param list train_elbo: list of accumulated error losses
    :param str results_dict: path to results directory
    """
    train_elbo = np.array(train_elbo)
    #list_epochs = np.arange(0,len(train_elbo))
    if np.isnan(train_elbo).any():
        print("Error loss contains nan")
        pass
    else:
        #plt.plot(list_epochs,train_elbo,color="dodgerblue")
        plt.plot(train_elbo, color="dodgerblue")
        plt.xlabel("List of epochs")
        plt.ylabel("-ELBO")
        plt.title("Training Error Loss (max -ELBO, min KL)")
        plt.savefig("{}/ELBO_error.png".format(results_dict))
        plt.close()
    plt.clf()
def plot_entropy(train_entropy,results_dict):
    """Plots the model's entropy
    :param list train_entropy: list of accumulated entropies
    :param str results_dict: path to results directory
    """
    train_entropy = np.array(train_entropy)
    list_epochs = list(range(0, len(train_entropy)))
    if np.isnan(train_entropy).any():
        print("Entropy contains nan")
        pass
    else:
        plt.plot(list_epochs, train_entropy, color="tomato")
        plt.title("Shanon Entropy Convergence")
        plt.savefig("{}/Entropy_convergence.png".format(results_dict))
        plt.close()
    plt.clf()
def plot_z(latent_space, children_dict, results_dir):
    """Plots the latent space vectors
    :param tensor latent_space
    :param dict children_dict
    :param str results_dir
    """

    fig, ax = plt.subplots(figsize=(50, 30))  # 18 for 1 col
    colors = plt.get_cmap('cool', len(children_dict.keys()))
    markers = itertools.cycle(('s', '+', 'd', 'o', '*'))
    for idx, (ancestor, children) in enumerate(children_dict.items()):
        if not pd.isnull(ancestor):
            sequences = [ancestor] + children
            indexes = (latent_space[:, 0][..., None] == sequences).any(-1)
            sequences_latent_space = latent_space[indexes,1:]
            for seq in range(len(sequences)):
                ax.plot(sequences_latent_space[seq,1:],color=colors(idx),marker=next(markers))
    plt.ylabel(r"Latent space ($\mathcal{Z}$) vector",fontsize=40)
    plt.xlabel(r"$\mathcal{Z}$ dimensions",fontsize=40)
    plt.title(r"Latent space ($\mathcal{Z}$) vector coloured by ancestor and respective children nodes",fontsize=40)
    plt.savefig("{}/z_vector_plot".format(results_dir))
def plot_angles(samples_out,dataset_test,results_dir,additional_load,additional_info,n_samples,test_ordered_nodes):
    """Plot Ramachandran plots of the predicted angles, both per individual amino acid and all amino acids combined
    :param namedtuple samples_out: contains the output from Draupnir.samples
    :param tensor dataset_test
    :param str results_dir
    :param namedtuple additional_load
    :param namedtuple additional_info
    :param int n_samples
    :param tensor test_ordered_nodes: tensor containing the associated nodes indexes"""

    phi_angles = samples_out.phis
    psi_angles = samples_out.psis
    phi_mean = samples_out.mean_phi
    psi_mean = samples_out.mean_psi
    phi_kappa = samples_out.kappa_phi
    psi_kappa = samples_out.kappa_psi
    #phi_angles = dataset_test[:,2:,1].repeat(n_samples,1,1)
    #psi_angles = dataset_test[:, 2:, 2].repeat(n_samples, 1, 1)
    Dataset_mask = DraupnirModelUtils.masking(dataset_test[:,2:,0]) #where there is a GAP, aka 0, put a 0 in the mask
    DraupnirUtils.ramachandran_plot_sampled(phi_angles, psi_angles,"{}/Angles_Predictions.png".format(results_dir), r"Sampled angles (φ,ψ); {}".format(additional_load.full_name))
    DraupnirUtils.ramachandran_plot_sampled(phi_mean, psi_mean, "{}/Angles_MEANS_Predictions.png".format(results_dir),
                                            r"Sampled angles's means (φ,ψ); {}".format(additional_load.full_name))
    DraupnirUtils.ramachandran_plot_sampled(phi_kappa, psi_kappa, "{}/Angles_KAPPAS_Predictions.png".format(results_dir),
                                            r"Sampled angles's kappas (φ,ψ); {}".format(additional_load.full_name),plot_kappas=True)

    phi_angles_list = []
    psi_angles_list = []
    phi_mean_list= []
    psi_mean_list=[]
    phi_kappa_list = []
    psi_kappa_list = []
    for i in range(n_samples): #TODO: Merge with angles per aa
        for row_mask,row_phi,row_psi in zip(Dataset_mask,phi_angles[i],psi_angles[i]):
            phi_angles_list.append(row_phi[row_mask !=0].tolist())
            psi_angles_list.append(row_psi[row_mask != 0].tolist())

    for row_mask,row_phi_mean, row_psi_mean, row_phi_kappa, row_psi_kappa in zip(Dataset_mask, phi_mean, psi_mean, phi_kappa, psi_kappa):
            phi_mean_list.append(row_phi_mean[row_mask != 0].tolist())
            psi_mean_list.append(row_psi_mean[row_mask != 0].tolist())
            phi_kappa_list.append(row_phi_kappa[row_mask != 0].tolist())
            psi_kappa_list.append(row_psi_kappa[row_mask != 0].tolist())

    DraupnirUtils.ramachandran_plot_sampled(sum(phi_angles_list,[]),sum(psi_angles_list,[]),"{}/Angles_Predictions_NoGAPS".format(results_dir),r"Sampled angles (φ,ψ) (without GAPS); {}".format(additional_load.full_name))
    DraupnirUtils.ramachandran_plot_sampled(sum(phi_mean_list,[]),sum(psi_mean_list,[]),"{}/Angles_MEANS_Predictions_NoGAPS".format(results_dir),r"Sampled angles's means (φ,ψ) (without GAPS); {}".format(additional_load.full_name))
    DraupnirUtils.ramachandran_plot_sampled(sum(phi_kappa_list,[]),sum(psi_kappa_list,[]),"{}/Angles_KAPPAS_Predictions_NoGAPS".format(results_dir),r"Sampled angles's kappas (φ,ψ) (without GAPS); {}".format(additional_load.full_name),plot_kappas=True)
def plot_angles_per_aa(samples_out,dataset_test,results_dir,build_config,additional_load,additional_info,n_samples,test_ordered_nodes):
    """Plot Ramachandran plots of the predicted angles, both per individual amino acid and all amino acids combined
    :param namedtuple samples_out: contains the output from Draupnir.samples
    :param tensor dataset_test
    :param str results_dir
    :param namedtuple additional_load
    :param namedtuple additional_info
    :param int n_samples
    :param tensor test_ordered_nodes: tensor containing the associated nodes indexes"""

    phi_angles = samples_out.phis
    psi_angles = samples_out.psis
    amino_acid_dict = DraupnirUtils.aminoacid_names_dict(build_config.aa_prob)
    amino_acids_data = dataset_test[:,2:,0]
    for aa_name, aa_number in amino_acid_dict.items():
        if aa_name == "-": continue
        else:
            phi_angles_list = []
            psi_angles_list = []
            for i in range(n_samples):
                for row_data,row_phi,row_psi in zip(amino_acids_data,phi_angles[i],psi_angles[i]):
                    phi_angles_list.append(row_phi[(row_data !=0)&(row_data == aa_number)].tolist())
                    psi_angles_list.append(row_psi[(row_data !=0)&(row_data == aa_number)].tolist())

            DraupnirUtils.ramachandran_plot_sampled(sum(phi_angles_list,[]),sum(psi_angles_list,[]),"{}/Angles_plots_per_aa/Angles_Predictions_{}".format(results_dir,aa_name),r"Sampled angles (φ,ψ) {} (without GAPS); {}".format(seq3(aa_name),additional_load.full_name))
    # DraupnirUtils.ramachandran_plot_sampled(sum(phi_mean_list,[]),sum(psi_mean_list,[]),"{}/Angles_MEANS_Predictions_NoGAPS".format(results_dir),"Sampled mean angles (φ,ψ) (without GAPS)")
    # DraupnirUtils.ramachandran_plot_sampled(sum(phi_kappa_list,[]),sum(psi_kappa_list,[]),"{}/Angles_KAPPAS_Predictions_NoGAPS".format(results_dir),"Sampled kappas angles (φ,ψ) (without GAPS)",plot_kappas=True)
def save_ancestors_predictions(name,dataset_test,aa_sequences_predictions,n_samples,results_directory,correspondence_dict,aa_prob):
    """Saves the ancestors sequences to a fasta format
    :param str name: data set project name
    :param tensor dataset_test: tensor containing the true aminoacid sequences in integer form [n_test,align_length] (gap or aa are referred from numbers 0 to 20)
    :param tensor aa_sequences_predictions: tensor containing the predicted aminoacid sequences in integer form [n_samples,n_test,align_length] (gap or aa are referred from numbers 0 to 20)
    :param int n_samples: number of samples
    :param str results_directory: path to results directory
    :param dict correspondence_dict: dictionary that contains the correspondence upon the node indexes (integers) and the tree nodes names (ordered in tree level order)
    :param aa_prob: amino acid probabilities
    """
    print("Saving ancestor's nodes sequences ...")
    node_info = dataset_test[:, 0, 1].repeat(n_samples).unsqueeze(-1).reshape(n_samples, len(dataset_test), 1)
    node_names = ["{}//{}".format(correspondence_dict[index], index) for index in dataset_test[:, 0, 1].tolist()]
    aa_sequences_predictions = torch.cat(( node_info, aa_sequences_predictions), dim=2)
    aa_dict = DraupnirUtils.aminoacid_names_dict(aa_prob)
    aa_dict = {float(v): k for k, v in aa_dict.items()}
    aa_sequences = aa_sequences_predictions[:, :, 1:].permute(1, 0,2)  # [n_nodes,n_samples,L] #TODO: Review this is correct, the order, seems correct
    aa_sequences = np.vectorize(aa_dict.get)(aa_sequences.cpu().numpy())
    file_name = "{}/{}_sampled_nodes_seq.fasta".format(results_directory, name)
    with open(file_name, "a+") as f:
        for node_samples, node_name in zip(aa_sequences, node_names):
            for idx, sample in enumerate(node_samples):
                f.write(">Node_{}_sample_{}\n".format(node_name, idx))
                full_seq = "".join(sample)
                n = 50
                splitted_seq = [full_seq[i:i + n] for i in range(0, len(full_seq), n)]
                for segment in splitted_seq:
                    f.write("{}\n".format(segment))
def save_ancestors_predictions_coral(name,test_ordered_nodes,aa_sequences_predictions,n_samples,results_directory,correspondence_dict,aa_prob):
    """Saves the ancestors sequences to a fasta format
     :param str name: data set project name
     :param tensor dataset_test: tensor containing the true aminoacid sequences in integer form [n_test,align_length] (gap or aa are referred from numbers 0 to 20)
     :param tensor aa_sequences_predictions: tensor containing the predicted aminoacid sequences in integer form [n_samples,n_test,align_length] (gap or aa are referred from numbers 0 to 20)
     :param int n_samples: number of samples
     :param str results_directory: path to results directory
     :param dict correspondence_dict: dictionary that contains the correspondence upon the node indexes (integers) and the tree nodes names (ordered in tree level order)
     :param aa_prob: amino acid probabilities
     """
    print("Saving ancestor's nodes sequences ...")
    node_info = test_ordered_nodes.repeat(n_samples).unsqueeze(-1).reshape(n_samples, len(test_ordered_nodes), 1)
    node_names = ["{}//{}".format(correspondence_dict[index], index) for index in test_ordered_nodes.tolist()]
    aa_sequences_predictions = torch.cat(( node_info, aa_sequences_predictions), dim=2)
    aa_dict = DraupnirUtils.aminoacid_names_dict(aa_prob)
    aa_dict = {float(v): k for k, v in aa_dict.items()}
    aa_sequences = aa_sequences_predictions[:, :, 1:].permute(1, 0,2)  # [n_nodes,n_samples,L] #TODO: Review this is correct, the order, seems correct
    aa_sequences = np.vectorize(aa_dict.get)(aa_sequences.cpu().numpy())
    file_name = "{}/{}_sampled_nodes_seq.fasta".format(results_directory, name)
    with open(file_name, "a+") as f:
        for node_samples, node_name in zip(aa_sequences, node_names):
            for idx, sample in enumerate(node_samples):
                f.write(">Node_{}_sample_{}\n".format(node_name, idx))
                full_seq = "".join(sample)
                n = 50
                splitted_seq = [full_seq[i:i + n] for i in range(0, len(full_seq), n)]
                for segment in splitted_seq:
                    f.write("{}\n".format(segment))
def percent_id_sampled_observed(dataset_test,aa_sequences_predictions,node_info,n_samples,node_names,results_directory):
    """Fast version to calculate %ID among predictions and observed data. It also calculates the number of incorrectly predicted amino acids
    :param tensor dataset_test
    :param tensor aa_sequences_predictions
    :param tensor node_info
    :param int n_samples
    :param node_names
    :param str results_directory
    """
    align_lenght = dataset_test[:,2:,0].shape[1]
    #node_names = ["{}//{}".format(correspondence_dict[index], index) for index in Dataset_test[:, 0, 1].tolist()]
    samples_names =  ["sample_{}".format(index) for index in range(n_samples)]
    equal_aminoacids = (aa_sequences_predictions[:,:,3:]== dataset_test[:, 2:, 0]).float() #is correct #[n_samples,n_nodes,L]
    #Highlight: Incorrectly predicted sites
    incorrectly_predicted_sites = (~equal_aminoacids.bool()).float().sum(-1)
    incorrectly_predicted_sites_per_sample = np.concatenate([node_info.cpu().detach().numpy(),incorrectly_predicted_sites.cpu().detach().numpy()[:,:,np.newaxis]],axis=-1)
    np.save("{}/Incorrectly_Predicted_Sites_Fast".format(results_directory), incorrectly_predicted_sites_per_sample)
    incorrectly_predicted_sites_df = pd.DataFrame(incorrectly_predicted_sites.T.cpu().detach().numpy(),index=node_names)
    incorrectly_predicted_sites_df.columns = samples_names
    incorrectly_predicted_sites_df["Average"] =incorrectly_predicted_sites_df.mean(1).values.tolist()
    incorrectly_predicted_sites_df["Std"] = incorrectly_predicted_sites_df.std(1).values.tolist()
    incorrectly_predicted_sites_df.to_csv("{}/Incorrectly_predicted_sites_df.csv".format(results_directory), sep="\t")
    #Highlight: PERCENT ID
    equal_aminoacids = equal_aminoacids.sum(-1)/align_lenght#equal_aminoacids.sum(-1)
    percent_id_df = pd.DataFrame(equal_aminoacids.T.cpu().detach().numpy()*100, index=node_names ) #[n_nodes, n_samples]
    percent_id_df.columns = samples_names
    percent_id_df["Average"] = percent_id_df.mean(1).values.tolist()
    percent_id_df["Std"] = percent_id_df.std(1).values.tolist()
    percent_id_df.to_csv("{}/PercentID_df.csv".format(results_directory),sep="\t")
    return percent_id_df, incorrectly_predicted_sites_df, align_lenght
def save_predictions_to_fasta(aa_sequences_predictions,aa_prob,name,node_names,results_directory):
    """Transforms via vectorization the integer encoded aminoacids into the amino acid letters. Saves the results into a fasta file
    :param aa_sequences_predictions: tensor dataframe with the predictions
    :param int aa_prob
    :param name: data set project name
    :param node_names: list of the real nodes names in the same order as in the dataset
    :param str results_directory"""
    aa_dict = DraupnirUtils.aminoacid_names_dict(aa_prob)
    aa_dict  = {float(v): k for k, v in aa_dict.items()}

    aa_sequences = aa_sequences_predictions[:,:,3:].permute(1,0,2) #[n_nodes,n_samples,L] #TODO: Review this is correct, the order, seems correct
    aa_sequences = np.vectorize(aa_dict.get)(aa_sequences.cpu().numpy())
    n_samples = aa_sequences.shape[1]
    file_name = "{}/{}_sampled_nodes_seq.fasta".format(results_directory,name)
    with open(file_name, "a+") as f:
        for node_samples,node_name in zip(aa_sequences,node_names):
            for idx, sample in enumerate(node_samples):
                f.write(">Node_{}_sample_{}\n".format(node_name, idx))
                full_seq = "".join(sample)
                n=50
                #splitted_seq = list(map(''.join, zip(*[iter(full_seq)]*50))) #if the seq is smaller it leaves it out
                splitted_seq = [full_seq[i:i+n] for i in range(0, len(full_seq), n)]
                for segment in splitted_seq:
                    f.write("{}\n".format(segment))
def plot_heatmap_and_incorrect_aminoacids(name,dataset_test,aa_sequences_predictions,n_samples,results_directory,correspondence_dict,aa_probs,additional_load,additional_info,replacement_plots=True):
    """Prepares the output from Draupnir for plotting by re-adding the nodes names to the data sets. It also detects large data sets and selects only
     a few nodes to plot
     :param str name: data set project name
     :param tensor dataset_test: true data [n_nodes,align_len + 2, 30]
     :param tensor aa_sequences_predictions: predictions [n_samples,n_nodes,align_len]
     :param int n_samples
     :param str results_directory
     :param dict correspondence_dict: contains the information about which node index corresponds to which node name
     :param int aa_probs
     :param namedtuple additional_load
     :param namedtuple additional_info
     :param bool replacement_plots: True --> plots the amino acid replacements occurred per site and per sequence"""
    #TODO: By simple permutation it rearranges to [n_seq,n_samples], preserving the right order?
    len_info = dataset_test[:, 0, 0].repeat(n_samples).unsqueeze(-1).reshape(n_samples, len(dataset_test), 1)
    node_info = dataset_test[:, 0, 1].repeat(n_samples).unsqueeze(-1).reshape(n_samples, len(dataset_test), 1)
    distance_info = dataset_test[:, 0, 2].repeat(n_samples).unsqueeze(-1).reshape(n_samples, len(dataset_test), 1)
    node_names = ["{}//{}".format(correspondence_dict[index], index) for index in dataset_test[:, 0, 1].tolist()]
    aa_sequences_predictions = torch.cat((len_info, node_info, distance_info, aa_sequences_predictions), dim=2)

    percent_id_df,incorrectly_predicted_sites_df, alignment_length =percent_id_sampled_observed(dataset_test,aa_sequences_predictions,node_info,n_samples,node_names,results_directory)
    save_predictions_to_fasta(aa_sequences_predictions, aa_probs, name, node_names, results_directory)

    aa_sequences_predictions = aa_sequences_predictions.cpu().numpy()
    dataset_test = dataset_test.cpu()
    n_test = dataset_test.shape[0]
    if n_test > 100:
        torch.manual_seed(0)
        percentage_test = 50/n_test #get 50 seq only
        test_indx = (torch.rand(size=(n_test,)) < percentage_test).int().bool() #create n  (n= n internal nodes) random True and False, Supposedly 45% of the values will be 1(True)
        dataset_test = dataset_test[test_indx]
        nodes_indexes = dataset_test[:, 0, 1]
        dataset_children_predicted = aa_sequences_predictions[:,np.isin(aa_sequences_predictions[0, :, 1], nodes_indexes)]
        children_indexes_str = nodes_indexes.numpy().astype(str).tolist()
        incorrectly_predicted_sites_df = incorrectly_predicted_sites_df[incorrectly_predicted_sites_df.index.str.split("//").str[1].isin(children_indexes_str)]


    else:
        nodes_indexes = dataset_test[:, 0, 1]  #test indexes
        dataset_children_predicted = aa_sequences_predictions[:,np.isin(aa_sequences_predictions[0, :, 1], nodes_indexes)]

    plot_incorrectly_predicted_aa(incorrectly_predicted_sites_df,results_directory,alignment_length,additional_load)

    DraupnirUtils.heatmaps(dataset_children_predicted,
                           dataset_test,
                           name,
                           n_samples,
                           nodes_indexes,
                           results_directory,
                           aa_probs,
                           additional_load,
                           additional_info,
                           correspondence_dict)
    if replacement_plots:
        barplot_aa_replacement(dataset_children_predicted[:,:,3:],
                                            dataset_test[:, 2:, 0],
                                            additional_load.full_name,
                                            n_samples,
                                            nodes_indexes,
                                            results_directory,
                                            aa_probs,
                                            correspondence_dict)
def plot_entropies(name,seq_entropies,results_directory,correspondence_dict):
    """Plot amino acid entropies per alignment site
    :param str name
    :param tensor seq_entropies
    :param str results_directory
    :param dict correspondence_dict"""
    max_entropy_per_site = -((1 / 21) * np.log(1/21)) * 21  # maximum overall possible entropy per site (not sequence!)

    min_value = np.min(seq_entropies[:,1:])
    max_value = np.max(seq_entropies[:,1:])
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    folder_name = os.path.basename(results_directory)
    data_name = ["Internal nodes/Test" if folder_name.startswith("Test")  else "Leaves nodes/Train"][0]
    node_indexes = seq_entropies[:,0].tolist()
    colors = pl.cm.jet(np.linspace(0, 1, len(node_indexes)))
    ax.hlines(max_entropy_per_site, linestyles="dashed", xmin=0, xmax=seq_entropies.shape[1],color="black",label="max allowed \n entropy")
    for idx,node in enumerate(node_indexes):
        pl.plot(seq_entropies[idx,1:], color=colors[idx],label="Node {}".format(node_indexes[idx]))
    plt.legend(loc = "upper right",bbox_to_anchor=(1.14, 0.96))
    plt.ylim((0,max_entropy_per_site + 0.3))
    plt.ylabel("Shannon entropy per site (Σ(-Pi.log(Pi))")
    plt.xlabel("Sequence Sites")
    plt.title("Shannon entropy: {}".format(data_name))
    plt.savefig("{}/Entropy.png".format(results_directory))
    plt.close()
def plot_overlapping_histogram(name,dataset_train,dataset_test, aa_sequences_predictions,n_samples,results_directory,correspondence_dict,aa_prob):
    """Plot histogram whose bars represent the percentage identity between the true ancestors vs true leaves and the predicted ancestors vs true leaves
    :param str name: dataset name project
    :param tensor dataset_train
    :param tensor dataset_test
    :param tensor aa_sequences_predictions
    :param int n_samples
    :param str results_directory
    :param dict correspondence_dict: correspondence between the tree level order indexes and the --> only useful for benchmark_randall_original_naming
    :param int aa_prob
    """
    #TODO: Fix the device problem if they should be used to see the distances between internal and leaves nodes
    children_indexes = dataset_test[:, 0, 1].cpu()
    len_info = dataset_test[:, 0, 0].repeat(n_samples).unsqueeze(-1).reshape(n_samples, len(dataset_test), 1)
    node_info = dataset_test[:, 0, 1].repeat(n_samples).unsqueeze(-1).reshape(n_samples, len(dataset_test), 1)
    distance_info = dataset_test[:, 0, 2].repeat(n_samples).unsqueeze(-1).reshape(n_samples, len(dataset_test), 1)
    aa_sequences_predictions = aa_sequences_predictions.cpu().detach()
    aa_sequences_predictions = torch.cat((len_info, node_info, distance_info, aa_sequences_predictions), dim=2)
    aa_sequences_predictions = aa_sequences_predictions.cpu().detach().numpy()
    dataset_children_predicted = aa_sequences_predictions[:,np.isin(aa_sequences_predictions[0, :, 1], children_indexes)]
    DraupnirUtils.build_dataframes_overlapping_histograms(dataset_children_predicted,
                                                         dataset_train.cpu(),
                                                         dataset_test.cpu(),
                                                         name,
                                                         n_samples,
                                                         children_indexes,
                                                         results_directory,
                                                         aa_prob,
                                                         correspondence_dict)
def plot_latent_space_tsne_by_clade(latent_space, additional_load, epoch, results_dir, triTSNE):
    """t-SNE projection of a z-dimensional latent space onto a 2D space. The latent space represents the sequences in the tree. The latent space is coloured according to
    the clade membership
    https://towardsdatascience.com/visualizing-feature-vectors-embeddings-using-pca-and-t-sne-ef157cea3a42
    :param tensor latent_space: [n_leaves + n_internal,1+ z_dim], first column contains the nodes indexes
    :param namedtuple additional_load
    :param int epoch
    :param str results_dir
    :param bool triTSNE: boolean paramter to indicate whether to perform a 3D latent space projection, Not used right now
    """
    # Create a two dimensional t-SNE projection of the z dim latent space
    print("Building T-SNE plot by clades")
    #annotate = [True if latent_space.shape[0] < 100 else False][0]
    annotate = False
    stripped = False
    if stripped: #no legend, no axis, nothing
        clades_dict_all = additional_load.clades_dict_all
        #n_cols = DraupnirUtils.Define_batch_size(latent_space.shape[0], batch_size=False,benchmarking=True)

        tsne_proj = TSNE(n_components=2).fit_transform(latent_space[:, 1:])
        color_map21 = matplotlib.colors.ListedColormap(
            ["plum", "navy", "turquoise", "peachpuff", "palevioletred", "red", "darkorange", "yellow", "lime", "green",
             "dodgerblue", "blue", "purple", "magenta", "grey", "maroon", "lightcoral", "olive", "teal", "goldenrod",
             "black"])

        color_map_name = [color_map21 if len(clades_dict_all) <= 21 else 'nipy_spectral'][0] #nipy_spectral, gist_rainbow
        clrs = plt.get_cmap(color_map_name, len(clades_dict_all))
        #n_cols = int(len(clades_dict_all.keys()) / 30) + 1

        fig, ax = plt.subplots(figsize=(22, 15),dpi=200)

        for idx,(clade, nodes) in enumerate(clades_dict_all.items()):
                sequences = nodes["internal"] + nodes["leaves"]
                indexes = (latent_space[:, 0][..., None] == sequences).any(-1)
                ax.scatter(tsne_proj[indexes, 0], tsne_proj[indexes, 1], color=clrs(idx), label=clade.replace("_"," "), alpha=1,s=700)
                if annotate:
                    for name, point in zip(sequences, tsne_proj[indexes]):
                        ax.annotate(name, xy=(point[0], point[1]), size=7)  # xytext=(1,1)
        #plt.legend(loc='upper left', prop={'size': 25},ncol=1, shadow=True)
        #plt.legend( bbox_to_anchor=(1.01, 1), loc='upper left', prop={'size': 25}, ncol=1, shadow=True)
        plt.tight_layout()
        plt.axis("off")
    else:
        clades_dict_all = additional_load.clades_dict_all
        # n_cols = DraupnirUtils.Define_batch_size(latent_space.shape[0], batch_size=False,benchmarking=True)

        tsne_proj = TSNE(n_components=2).fit_transform(latent_space[:, 1:])
        color_map21 = matplotlib.colors.ListedColormap(
            ["plum", "navy", "turquoise", "peachpuff", "palevioletred", "red", "darkorange", "yellow", "lime", "green",
             "dodgerblue", "blue", "purple", "magenta", "grey", "maroon", "lightcoral", "olive", "teal", "goldenrod",
             "black"])

        color_map_name = [color_map21 if len(clades_dict_all) <= 21 else 'nipy_spectral'][0]
        clrs = plt.get_cmap(color_map_name, len(clades_dict_all))
        # n_cols = int(len(clades_dict_all.keys()) / 30) + 1

        fig, ax = plt.subplots(figsize=(22, 15), dpi=200)

        for idx, (clade, nodes) in enumerate(clades_dict_all.items()):
            sequences = nodes["internal"] + nodes["leaves"]
            indexes = (latent_space[:, 0][..., None] == sequences).any(-1)
            ax.scatter(tsne_proj[indexes, 0], tsne_proj[indexes, 1], color=clrs(idx), label=clade.replace("_", " "),
                       alpha=1, s=200)
            if annotate:
                for name, point in zip(sequences, tsne_proj[indexes]):
                    ax.annotate(name, xy=(point[0], point[1]), size=7)  # xytext=(1,1)
        plt.legend(title='Clades', bbox_to_anchor=(1.01, 1), loc='upper left', prop={'size': 15},ncol=1,shadow=True,fontsize=15)
        plt.title("T-SNE projection of the tree's latent space; \n" + "{}".format(additional_load.full_name),fontsize=20)
        plt.axis("off")
    plt.savefig("{}/t_SNE_z_space_by_clade_epoch_{}.png".format(results_dir, epoch))
def plot_latent_space_tsne_by_clade_leaves(latent_space, additional_load, epoch, results_dir, triTSNE):
    """t-SNE projection of a z-dimensional latent space onto a 2D space.
    The latent space represents the sequences in the tree. The latent space is coloured according to
    the clade membership
    https://towardsdatascience.com/visualizing-feature-vectors-embeddings-using-pca-and-t-sne-ef157cea3a42
    :param tensor latent_space [n_leaves, 1 + z_dim]
    :param namedtuple additional_load
    :param int epoch
    :param str results_dir
    :param bool triTSNE: boolean paramter to indicate whether to perform a 3D latent space projection, Not used right now
    """
    # Create a two dimensional t-SNE projection of the z dim latent space
    print("Building t-SNE plot COLOURED by clades (only leaves)")
    #annotate = [True if latent_space.shape[0] < 100 else False][0]
    annotate = False
    stripped = False
    clades_dict_all = additional_load.clades_dict_all
    #n_cols = DraupnirUtils.Define_batch_size(latent_space.shape[0], batch_size=False,benchmarking=True)
    tsne_proj = TSNE(n_components=2).fit_transform(latent_space[:, 1:])
    color_map21 = matplotlib.colors.ListedColormap(
        ["plum", "navy", "turquoise", "peachpuff", "palevioletred", "red", "darkorange", "yellow", "lime", "green",
         "dodgerblue", "blue", "purple", "magenta", "grey", "maroon", "lightcoral", "olive", "teal", "goldenrod",
         "black"])

    color_map_name = [color_map21 if len(clades_dict_all) <= 21 else 'nipy_spectral'][0]
    clrs = plt.get_cmap(color_map_name, len(clades_dict_all))
    #n_cols = int(len(clades_dict_all.keys()) / 30) + 1
    if stripped:
        fig, ax = plt.subplots(figsize=(22, 15))

        for idx, (clade, nodes) in enumerate(clades_dict_all.items()):
            sequences = nodes["leaves"]
            indexes = (latent_space[:, 0][..., None] == sequences).any(-1)
            ax.scatter(tsne_proj[indexes, 0], tsne_proj[indexes, 1], color=clrs(idx), label=clade, alpha=1, s=700)
            if annotate:
                for name, point in zip(sequences, tsne_proj[indexes]):
                    ax.annotate(name, xy=(point[0], point[1]), size=7)  # xytext=(1,1)
        #plt.legend(loc='upper left', prop={'size': 25},ncol=1, shadow=True)
        plt.tight_layout()
        plt.axis("off")
    else:
        fig, ax = plt.subplots(figsize=(22, 15))

        for idx,(clade, nodes) in enumerate(clades_dict_all.items()):
                sequences = nodes["leaves"]
                indexes = (latent_space[:, 0][..., None] == sequences).any(-1)
                ax.scatter(tsne_proj[indexes, 0], tsne_proj[indexes, 1], color=clrs(idx), label=clade, alpha=1)
                if annotate:
                    for name, point in zip(sequences, tsne_proj[indexes]):
                        ax.annotate(name, xy=(point[0], point[1]), size=7)  # xytext=(1,1)
        plt.legend(title='Clades', bbox_to_anchor=(1.01, 1), loc='upper left', prop={'size': 10},ncol=1, shadow=True,fontsize=10)
        plt.title("UMAP projection of the tree's latent space; \n" + "{}".format(additional_load.full_name),fontsize=20)
    plt.savefig("{}/UMAP_z_space_by_clade__only_leaves_epoch_{}.png".format(results_dir, epoch))
    #plt.savefig("{}/Tree latent space representation (T-SNE projection); SH3 domain 200 leaves simulation".format(results_dir))
def plot_latent_space_umap_by_clade(latent_space, additional_load, epoch, results_dir, triTSNE):
    """UMAP projection of a z-dimensional latent space onto a 2D space. The latent space represents the sequences in the tree. The latent space is coloured according to
    the clade membership
    :param tensor latent_space: [n_leaves + n_internal,1+ z_dim], first column contains the nodes indexes
    :param namedtuple additional_load
    :param int epoch
    :param str results_dir
    :param bool triTSNE: boolean paramter to indicate whether to perform a 3D latent space projection, Not used right now
    """
    # Create a two dimensional t-SNE projection of the z dim latent space
    print("Building UMAP plot by clades (both internal and leaves)")
    #annotate = [True if latent_space.shape[0] < 100 else False][0]
    annotate = False
    stripped = False
    reducer = umap.UMAP()
    if stripped:
        clades_dict_all = additional_load.clades_dict_all
        #n_cols = DraupnirUtils.Define_batch_size(latent_space.shape[0], batch_size=False,benchmarking=True)
        tsne_proj = reducer.fit_transform(latent_space[:, 1:])

        color_map21 = matplotlib.colors.ListedColormap(
            ["plum", "navy", "turquoise", "peachpuff", "palevioletred", "red", "darkorange", "yellow", "lime", "green",
             "dodgerblue", "blue", "purple", "magenta", "grey", "maroon", "lightcoral", "olive", "teal", "goldenrod",
             "black"])

        color_map_name = [color_map21 if len(clades_dict_all) <= 21 else 'nipy_spectral'][0] #nipy_spectral, gist_rainbow
        clrs = plt.get_cmap(color_map_name, len(clades_dict_all))
        #n_cols = int(len(clades_dict_all.keys()) / 30) + 1

        fig, ax = plt.subplots(figsize=(22, 15),dpi=200)

        for idx,(clade, nodes) in enumerate(clades_dict_all.items()):
                sequences = nodes["internal"] + nodes["leaves"]
                indexes = (latent_space[:, 0][..., None] == sequences).any(-1)
                ax.scatter(tsne_proj[indexes, 0], tsne_proj[indexes, 1], color=clrs(idx), label=clade.replace("_"," "), alpha=1,s=700)
                if annotate:
                    for name, point in zip(sequences, tsne_proj[indexes]):
                        ax.annotate(name, xy=(point[0], point[1]), size=7)  # xytext=(1,1)
        #plt.legend(loc='upper left', prop={'size': 25},ncol=1, shadow=True)
        #plt.legend( bbox_to_anchor=(1.01, 1), loc='upper left', prop={'size': 25}, ncol=1, shadow=True)
        plt.tight_layout()
        plt.axis("off")
    else:
        clades_dict_all = additional_load.clades_dict_all
        # n_cols = DraupnirUtils.Define_batch_size(latent_space.shape[0], batch_size=False,benchmarking=True)

        tsne_proj = reducer.fit_transform(latent_space[:, 1:])
        color_map21 = matplotlib.colors.ListedColormap(
            ["plum", "navy", "turquoise", "peachpuff", "palevioletred", "red", "darkorange", "yellow", "lime", "green",
             "dodgerblue", "blue", "purple", "magenta", "grey", "maroon", "lightcoral", "olive", "teal", "goldenrod",
             "black"])

        color_map_name = [color_map21 if len(clades_dict_all) <= 21 else 'nipy_spectral'][0]
        clrs = plt.get_cmap(color_map_name, len(clades_dict_all))
        # n_cols = int(len(clades_dict_all.keys()) / 30) + 1

        fig, ax = plt.subplots(figsize=(22, 15), dpi=200)

        for idx, (clade, nodes) in enumerate(clades_dict_all.items()):
            sequences = nodes["internal"] + nodes["leaves"]
            indexes = (latent_space[:, 0][..., None] == sequences).any(-1)
            ax.scatter(tsne_proj[indexes, 0], tsne_proj[indexes, 1], color=clrs(idx), label=clade.replace("_", " "),
                       alpha=1, s=200)
            if annotate:
                for name, point in zip(sequences, tsne_proj[indexes]):
                    ax.annotate(name, xy=(point[0], point[1]), size=7)  # xytext=(1,1)
        plt.legend(title='Clades', bbox_to_anchor=(1.01, 1), loc='upper left', prop={'size': 15},ncol=1,shadow=True,fontsize=15)
        plt.title("UMAP projection of the tree's latent space; \n" + "{}".format(additional_load.full_name),fontsize=20)
        plt.axis("off")
    plt.savefig("{}/UMAP_z_space_by_clade_epoch_{}.png".format(results_dir, epoch))
def plot_latent_space_umap_by_clade_leaves(latent_space, additional_load, epoch, results_dir, triTSNE):
    """UMAP projection of a z-dimensional latent space onto a 2D space. The latent space represents the sequences in the tree. The latent space is coloured according to
    the clade membership
    :param tensor latent_space: [n_leaves,1+ z_dim], first column contains the nodes indexes
    :param namedtuple additional_load
    :param int epoch
    :param str results_dir
    :param bool triTSNE: boolean paramter to indicate whether to perform a 3D latent space projection, Not used right now
    """
    # Create a two dimensional t-SNE projection of the z dim latent space
    print("Building UMAP plot COLOURED by clades")
    #annotate = [True if latent_space.shape[0] < 100 else False][0]
    annotate = False
    stripped = True
    clades_dict_all = additional_load.clades_dict_all
    #n_cols = DraupnirUtils.Define_batch_size(latent_space.shape[0], batch_size=False,benchmarking=True)
    reducer = umap.UMAP()
    umap_proj =  reducer.fit_transform(latent_space[:,1:])
    #tsne_proj = TSNE(n_components=2).fit_transform(latent_space[:, 1:])
    color_map21 = matplotlib.colors.ListedColormap(
        ["plum", "navy", "turquoise", "peachpuff", "palevioletred", "red", "darkorange", "yellow", "lime", "green",
         "dodgerblue", "blue", "purple", "magenta", "grey", "maroon", "lightcoral", "olive", "teal", "goldenrod",
         "black"])

    color_map_name = [color_map21 if len(clades_dict_all) <= 21 else 'nipy_spectral'][0]
    clrs = plt.get_cmap(color_map_name, len(clades_dict_all))
    #n_cols = int(len(clades_dict_all.keys()) / 30) + 1
    if stripped:
        fig, ax = plt.subplots(figsize=(22, 15))

        for idx, (clade, nodes) in enumerate(clades_dict_all.items()):
            sequences = nodes["leaves"]
            indexes = (latent_space[:, 0][..., None] == sequences).any(-1)
            ax.scatter(umap_proj[indexes, 0], umap_proj[indexes, 1], color=clrs(idx), label=clade, alpha=1, s=700)
            if annotate:
                for name, point in zip(sequences, umap_proj[indexes]):
                    ax.annotate(name, xy=(point[0], point[1]), size=7)  # xytext=(1,1)
        #plt.legend(loc='upper left', prop={'size': 25},ncol=1, shadow=True)
        plt.tight_layout()
        plt.axis("off")
    else:
        fig, ax = plt.subplots(figsize=(22, 15))

        for idx,(clade, nodes) in enumerate(clades_dict_all.items()):
                sequences = nodes["leaves"]
                indexes = (latent_space[:, 0][..., None] == sequences).any(-1)
                ax.scatter(umap_proj[indexes, 0], umap_proj[indexes, 1], color=clrs(idx), label=clade, alpha=1)
                if annotate:
                    for name, point in zip(sequences, umap_proj[indexes]):
                        ax.annotate(name, xy=(point[0], point[1]), size=7)  # xytext=(1,1)
        plt.legend(title='Clades', bbox_to_anchor=(1.01, 1), loc='upper left', prop={'size': 10},ncol=1, shadow=True,fontsize=10)
        plt.title("UMAP projection of the tree's latent space; \n" + "{}".format(additional_load.full_name),fontsize=20)
    plt.savefig("{}/UMAP_z_space_by_clade__only_leaves_epoch_{}.png".format(results_dir, epoch))
    #plt.savefig("{}/Tree latent space representation (T-SNE projection); SH3 domain 200 leaves simulation".format(results_dir))
def plot_latent_space_pca_by_clade(latent_space,additional_load,num_epochs, results_dir):
    """PCA projection of a z-dimensional latent space onto a 2D space. The latent space represents the sequences in the tree. The latent space is coloured according to
    the clade membership
    :param tensor latent_space: [n_leaves + n_internal,1+ z_dim], first column contains the nodes indexes
    :param namedtuple additional_load
    :param int epoch
    :param str results_dir
    :param bool triTSNE: boolean paramter to indicate whether to perform a 3D latent space projection, Not used right now
    """
    print("Building PCA plot")
    #annotate = [True if latent_space.shape[0] < 100 else False][0]
    annotate = False
    clades_dict_all = additional_load.clades_dict_all
    pca = PCA(n_components=2)
    pca.fit(latent_space[:,1:])
    latent_space_pca = pca.transform(latent_space[:,1:])
    color_map21 = matplotlib.colors.ListedColormap(
        ["plum", "navy", "turquoise", "peachpuff", "palevioletred", "red", "darkorange", "yellow", "lime", "green",
         "dodgerblue", "blue", "purple", "magenta", "grey", "maroon", "lightcoral", "olive", "teal", "goldenrod",
         "black"])

    color_map_name = [color_map21 if len(clades_dict_all) <= 21 else 'nipy_spectral'][0]
    clrs = plt.get_cmap(color_map_name, len(clades_dict_all))
    #n_cols = int(len(clades_dict_all.keys()) / 30) + 1
    stripped = True
    if stripped:
        fig, ax = plt.subplots(figsize=(22, 15))

        for idx, (clade, nodes) in enumerate(clades_dict_all.items()):
            sequences = nodes["internal"] + nodes["leaves"]
            indexes = (latent_space[:, 0][..., None] == sequences).any(-1)
            ax.scatter(latent_space_pca[indexes, 0], latent_space_pca[indexes, 1], color=clrs(idx), label=clade, alpha=1, s=200)
            if annotate:
                for name, point in zip(sequences, latent_space_pca[indexes]):
                    ax.annotate(name, xy=(point[0], point[1]), size=7)  # xytext=(1,1)
        #plt.legend(loc='upper left', prop={'size': 25},ncol=1, shadow=True)
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', prop={'size': 25}, ncol=1, shadow=True)

        plt.tight_layout()
        plt.axis("off")
    else:
        fig, ax = plt.subplots(figsize=(22, 15))
        for idx, (clade, nodes) in enumerate(clades_dict_all.items()):
            sequences = nodes["internal"] + nodes["leaves"]
            indexes = (latent_space[:, 0][..., None] == sequences).any(-1)
            ax.scatter(latent_space_pca[indexes, 0], latent_space_pca[indexes, 1], color=clrs(idx), label=clade, alpha=0.5)
            if annotate:
                for name, point in zip(sequences, latent_space_pca[indexes]):
                    ax.annotate(name, xy=(point[0], point[1]))  # xytext=(1,1)
        plt.xlabel("PC 1", fontsize = 20)
        plt.ylabel("PC 2", fontsize = 20)
        plt.legend(title='Clades', bbox_to_anchor=(1.01, 1), loc='upper left', prop={'size': 10},ncol=1,shadow=True,fontsize=10)
        plt.title("PCA projection of the tree's latent space;\n" + r"{}".format(additional_load.full_name),fontsize=20)
    plt.savefig("{}/PCA_z_space_epoch_{}.png".format(results_dir, num_epochs))
def plot_latent_space_pca_by_clade_leaves(latent_space,additional_load,num_epochs, results_dir):
    """PCA projection of a z-dimensional latent space onto a 2D space. The latent space represents the sequences in the tree. The latent space is coloured according to
    the clade membership
    https://towardsdatascience.com/visualizing-feature-vectors-embeddings-using-pca-and-t-sne-ef157cea3a42
    :param tensor latent_space: [n_leaves,1+ z_dim], first column contains the nodes indexes
    :param namedtuple additional_load
    :param int epoch
    :param str results_dir
    :param bool triTSNE: boolean paramter to indicate whether to perform a 3D latent space projection, Not used right now
    """
    """"""
    print("Building PCA plot")
    #annotate = [True if latent_space.shape[0] < 100 else False][0]
    annotate = False
    clades_dict_all = additional_load.clades_dict_all
    pca = PCA(n_components=2)
    pca.fit(latent_space[:,1:])
    latent_space_pca = pca.transform(latent_space[:,1:])
    color_map21 = matplotlib.colors.ListedColormap(
        ["plum", "navy", "turquoise", "peachpuff", "palevioletred", "red", "darkorange", "yellow", "lime", "green",
         "dodgerblue", "blue", "purple", "magenta", "grey", "maroon", "lightcoral", "olive", "teal", "goldenrod",
         "black"])

    color_map_name = [color_map21 if len(clades_dict_all) <= 21 else 'nipy_spectral'][0]
    clrs = plt.get_cmap(color_map_name, len(clades_dict_all))
    #n_cols = int(len(clades_dict_all.keys()) / 30) + 1


    fig, ax = plt.subplots(figsize=(22, 15))

    for idx, (clade, nodes) in enumerate(clades_dict_all.items()):
        sequences =  nodes["leaves"]
        indexes = (latent_space[:, 0][..., None] == sequences).any(-1)
        ax.scatter(latent_space_pca[indexes, 0], latent_space_pca[indexes, 1], color=clrs(idx), label=clade, alpha=0.5)
        if annotate:
            for name, point in zip(sequences, latent_space_pca[indexes]):
                ax.annotate(name, xy=(point[0], point[1]))  # xytext=(1,1)
    plt.xlabel("PC 1", fontsize = 20)
    plt.ylabel("PC 2", fontsize = 20)
    plt.legend(title='Clades', bbox_to_anchor=(1.01, 1), loc='upper left', prop={'size': 10},ncol=1,shadow=True,fontsize=10)
    plt.title("PCA projection of the tree's latent space;\n" + r"{}".format(additional_load.full_name),fontsize=20)
    plt.savefig("{}/PCA_z_space_epoch_{}.png".format(results_dir, num_epochs))
def plot_pairwise_distances(latent_space,additional_load,num_epochs, results_dir):
    """Plots the pairwise distance between the latent space vectors of 2 nodes in the tree vs the branch length/patristic distance between them
    :param tensor latent_space
    :param namedtuple additional_load
    :param int num_epochs
    :param str results_dir"""
    print("Plotting z distances vs branch lengths...")
    clades_dict_all = additional_load.clades_dict_all

    color_map21 = matplotlib.colors.ListedColormap(
        ["plum", "navy", "turquoise", "peachpuff", "palevioletred", "red", "darkorange", "yellow", "lime", "green",
         "dodgerblue", "blue", "purple", "magenta", "grey", "maroon", "lightcoral", "olive", "teal", "goldenrod",
         "black"])
    color_map_name = [color_map21 if len(clades_dict_all) <= 21 else 'nipy_spectral'][0]  # nipy_spectral, gist_rainbow
    clrs = plt.get_cmap(color_map_name, len(clades_dict_all))
    linked_nodes_dict = additional_load.linked_nodes_dict
    patristic_matrix_full = additional_load.patristic_matrix_full
    patristic_matrix_full_no_indexes = patristic_matrix_full[1:,1:]
    latent_space_no_indexes = latent_space[:,1:]
    use_cosine_similarity = True
    def cosine_similarity(a,b):
        return spatial.distance.cosine(a, b)
    def pairwise_dist(a,b):
        return np.linalg.norm(a-b)
    if use_cosine_similarity:
        distance_type = "Cosine_similarity"
        y_label = "Cosine similarity"
        distance_fn = cosine_similarity
    else:
        distance_type = "Pairwise_distance"
        y_label = "Euclidean distance"
        distance_fn = pairwise_dist
    fig, ax = plt.subplots(figsize=(22, 15), dpi=200)
    distances_list = []
    for node, children in linked_nodes_dict.items():
        if children: #if it has descendants
            # Highlight: find to which clade does this node belong and use its color---> List comprehesion does not work
            for idx, (clades, nodes) in enumerate(clades_dict_all.items()):
                if node in nodes["internal"] or node in nodes["leaves"]:
                    color_clade = clrs(idx)
            for child in children:
                nodes_pairs = torch.tensor([float(node),float(child)])
                pair_indexes_branch_lengths = (patristic_matrix_full[1:, 0][..., None] == nodes_pairs.cpu()).any(-1)
                pair_branch_lengths = patristic_matrix_full_no_indexes[pair_indexes_branch_lengths]
                pair_branch_lengths = pair_branch_lengths[:,pair_indexes_branch_lengths][0,1].detach().cpu().numpy()
                pair_indexes_latent_space = (latent_space[:,0][..., None] == nodes_pairs).any(-1)
                vector_pair_latent_space = latent_space_no_indexes[pair_indexes_latent_space].detach().cpu().numpy()
                latent_space_distance = distance_fn(vector_pair_latent_space[0], vector_pair_latent_space[1])
                distances_list.append(np.array([pair_branch_lengths,latent_space_distance]).T)
                ax.scatter(pair_branch_lengths,latent_space_distance,color = color_clade, s=200)
    distances_array = np.vstack(distances_list)
    pearson_correlation_coefficient = np.corrcoef(distances_array,rowvar=False)
    print("Correlation coefficient: {}".format(pearson_correlation_coefficient))
    spearman_correlation_coefficient = stats.spearmanr(distances_array[:,0],distances_array[:,1])[0]
    print("Spearman correlation coefficient {}".format(spearman_correlation_coefficient))
    plt.ylabel("{}".format(y_label), fontsize=20)
    plt.xlabel("Branch length between contiguous nodes",fontsize=20)
    plt.title("GP-VAE: Z {} vs Branch lengths for linked nodes (ancestors and leaves) \n Pearson correlation: {} \n Spearman correlation Coefficient: {}".format(distance_type,pearson_correlation_coefficient,spearman_correlation_coefficient),fontsize=20)
    plt.savefig("{}/Distances_GP_VAE_z_vs_branch_lengths_{}_INTERNAL_and_LEAVES.png".format(results_dir,distance_type))
def plot_pairwise_distances_only_leaves(latent_space,additional_load,num_epochs, results_dir,patristic_matrix_train):
    """Plots the distance between the latent space vectors of 2 leaf nodes in the tree and the branch length between them
    :param tensor latent_space [n_leaves, 1 + z_dim]
    :param namedtuple additional_load
    :param int num_epochs
    :param str results_dir
    :param patristic_matrix_train : [n_leaves + 1, n_leaves + 1]"""
    print("Plotting z distances vs branch lengths...")
    clades_dict_all = additional_load.clades_dict_all

    color_map21 = matplotlib.colors.ListedColormap(
        ["plum", "navy", "turquoise", "peachpuff", "palevioletred", "red", "darkorange", "yellow", "lime", "green",
         "dodgerblue", "blue", "purple", "magenta", "grey", "maroon", "lightcoral", "olive", "teal", "goldenrod",
         "black"])
    color_map_name = [color_map21 if len(clades_dict_all) <= 21 else 'nipy_spectral'][0]  # nipy_spectral, gist_rainbow
    clrs = plt.get_cmap(color_map_name, len(clades_dict_all))
    patristic_matrix_train = patristic_matrix_train.detach().cpu()
    patristic_matrix_train_no_indexes = patristic_matrix_train[1:,1:]
    patristic_matrix_train_no_top_indexes = patristic_matrix_train[1:,:].detach().cpu().numpy()
    latent_space_no_indexes = latent_space[:,1:].detach().cpu().numpy()
    latent_space_indexes = latent_space[:,0]
    latent_space = latent_space.detach().cpu()
    #proj = TSNE(n_components=2).fit_transform(latent_space[:, 1:])
    #reducer = umap.UMAP()
    #proj = reducer.fit_transform(latent_space[:, 1:])
    fig, ax = plt.subplots(figsize=(22, 15), dpi=200)
    distances_list = []
    use_cosine_similarity = False
    def cosine_similarity(a,b):
        return spatial.distance.cosine(a, b)

    def pairwise_dist(a,b):
        return np.linalg.norm(a-b)

    if use_cosine_similarity:
        distance_type = "Cosine_similarity"
        y_label = "Cosine similarity"
        distance_fn = cosine_similarity
    else:
        distance_type = "Pairwise_distance"
        y_label = "Euclidean distance"
        distance_fn = pairwise_dist

    assert latent_space[:,0].all() == patristic_matrix_train[1:,0].all()
    nodes_indexes = latent_space[:,0]
    for leaf_i_z, leaf_j_z,leaf_i_idx,leaf_j_idx in zip(latent_space_no_indexes,latent_space_no_indexes[1:],nodes_indexes,nodes_indexes[1:]):
            for idx, (clade, nodes) in enumerate(clades_dict_all.items()):
                if leaf_i_idx in nodes["leaves"]:
                    color_clade = clrs(idx)
            latent_space_distance = distance_fn(leaf_i_z, leaf_j_z)
            nodes_pairs = torch.tensor([leaf_i_idx,leaf_j_idx]).cpu()
            print("node pairs")
            print(nodes_pairs)
            print("patristic matrix train")
            print(patristic_matrix_train[1:,0])
            branch_length_idx = (patristic_matrix_train[1:,0][..., None] == nodes_pairs).any(-1)
            print("branch length idx")
            print(branch_length_idx)
            branch_length = patristic_matrix_train_no_indexes[branch_length_idx]
            branch_length = branch_length[:,branch_length_idx]
            print("branch length")
            print(branch_length)
            distances_list.append(np.array([branch_length[0,1],latent_space_distance]).T)
            ax.scatter(branch_length[0,1],latent_space_distance,color = color_clade, s=200)
    distances_array = np.vstack(distances_list)
    pearson_correlation_coefficient = np.corrcoef(distances_array,rowvar=False)
    print("Correlation coefficient: {}".format(pearson_correlation_coefficient[0][1]))
    spearman_correlation_coefficient = stats.spearmanr(distances_array[:,0],distances_array[:,1])[0]
    print("Spearman correlation coefficient {}".format(spearman_correlation_coefficient))
    plt.ylabel("{}".format(y_label), fontsize=50)
    plt.xlabel("Branch length",fontsize=50)
    plt.title("GP-VAE: Z {} vs Branch lengths for linked nodes (ancestors and leaves) \n Spearman correlation Coefficient: {}".format(distance_type,spearman_correlation_coefficient),fontsize=20)

    #plt.title("TOU-VAE: Z {} vs Branch lengths between leaves. \n Correlation coefficient : {} \n Spearman correlation : {}".format(distance_type,pearson_correlation_coefficient[0,1], spearman_correlation_coefficient),fontsize=40)
    #plt.title("Standard VAE",fontsize=50)
    #plt.title("Draupnir marginal", fontsize=50)
    plt.savefig("{}/Distances_GP_VAE_z_vs_branch_lengths_{}_ONLY_LEAVES.png".format(results_dir,distance_type))
def align_true_predicted(records,results_dir): #TODO: can be faster, although is mainly used with the coral sequences, which only have 5 sequences
    """Writes the list of records (sampled and true), transforms into fasta file
    :param list records: list of biopython Seq records with the sequences in the alignment
    :param str results_dir"""
    samples_out_file = os.path.join(results_dir,"samples_plus_true_to_align.fasta")
    SeqIO.write(records, samples_out_file, "fasta")
    alignment_out_file = os.path.join(results_dir,"samples_plus_true_aligned.fasta")
    dict_alignment, alignment = DraupnirUtils.infer_alignment(None, samples_out_file, alignment_out_file)
    alignment_length = len(alignment[0].seq)
    sampled_sequences = defaultdict()
    true_sequences = defaultdict()
    for key, value in dict_alignment.items():
        if key.startswith("Sample"):
            sampled_sequences[key] = value
        elif key.startswith("True"):
            true_sequences[key] = value
    return true_sequences,sampled_sequences,alignment_length
def clean_and_realign_test(name,dataset_test,aa_sequences_predictions,test_ordered_nodes,n_samples,aa_probs,results_dir,additional_load,additional_info):
    """
     a)Select from the predictions only the sequences in the dataset_test.
     b) Convert to sequences
     c) Remove gaps
     d) Save to file both samples and the true observed nodes (for coral, there is only 1 node)
     c) Align to the 'observed' sequences to determine percent identity
     :param str name
     :param tensor dataset_test
     :param tensor aa_sequences_predictions
     :param test_ordered_nodes
     :param int n_samples
     :param int aa_probs
     :param str results_dir
     :param namedtuple additional_load
     :param namedtuple additional_info
      """
    node_info = test_ordered_nodes.repeat(n_samples).unsqueeze(-1).reshape(n_samples, len(test_ordered_nodes), 1)
    aa_sequences_predictions = torch.cat((node_info,aa_sequences_predictions),dim=2)
    #aa_sequences_predictions = aa_sequences_predictions.permute(1,0,2)
    test_indx_patristic = (aa_sequences_predictions[:,:, 0][..., None] == dataset_test[:,0,1]).any(-1) #[..., None]
    aa_sequences_predictions = aa_sequences_predictions[test_indx_patristic].unsqueeze(1) #unsqueeze to keep the the size = [n_samples, n_nodes, seq-len]
    n_unique_nodes = len(list(set(dataset_test[:,0,1].tolist()))) #for the coral faviina sequences we have 5 times the root node
    tree_levelorder_dict = dict(zip(list(range(0,len(additional_load.tree_levelorder_names))),additional_load.tree_levelorder_names))
    special_nodes_list = list(additional_load.special_nodes_dict.keys()) #assuming it preserves the order    exit()

    records = []
    for sample_idx in range(n_samples):
        for node_idx in range(n_unique_nodes):
            predicted_seq = aa_sequences_predictions[sample_idx,node_idx]
            predicted_seq_letters = DraupnirUtils.convert_to_letters(predicted_seq,aa_probs)
            record = SeqRecord.SeqRecord(Seq(''.join(predicted_seq_letters).replace("-", "")),
                                         annotations={"molecule_type": "protein"},
                                         id="Sampled_node_{}_{}_sample_index_{}".format(node_idx,tree_levelorder_dict[node_idx],sample_idx),
                                         description="")
            records.append(record)
    # Highlight: Append also the true internal/test sequences (from the paper)
    for true_index, true_seq in enumerate(dataset_test[:, 2:, 0]):
        true_seq_letters = DraupnirUtils.convert_to_letters(true_seq, aa_probs)
        node_number = dataset_test[true_index, 0, 1].item()  # tree levelorder number
        node_name = tree_levelorder_dict[node_number]  # tree name
        node_true_name = special_nodes_list[true_index]  # sequence name in the paper
        record = SeqRecord.SeqRecord(Seq(''.join(true_seq_letters).replace("-", "")),
                                     annotations={"molecule_type": "protein"},
                                     id="True_node_{}_{}".format(node_name, node_true_name), description="")
        records.append(record)

    # def Align():
    #     samples_out_file = os.path.join(results_dir,"samples_plus_true_to_align.fasta")
    #     SeqIO.write(records, samples_out_file, "fasta")
    #     alignment_out_file = os.path.join(results_dir,"samples_plus_true_aligned.fasta")
    #     dict_alignment, alignment = DraupnirUtils.infer_alignment(None, samples_out_file, alignment_out_file)
    #     alignment_length = len(alignment[0].seq)
    #     sampled_sequences = defaultdict()
    #     true_sequences = defaultdict()
    #     for key, value in dict_alignment.items():
    #         if key.startswith("Sample"):
    #             sampled_sequences[key] = value
    #         elif key.startswith("True"):
    #             true_sequences[key] = value
    #     return true_sequences,sampled_sequences,alignment_length
    true_sequences, sampled_sequences, alignment_length = align_true_predicted(records,results_dir)


    def plot_results():
        correlations_dataframe_TEST = pd.DataFrame(index=["Sample_{}".format(n) for n in range(n_samples)] + ["Average", "Std"])
        #HIGHLIGHT : WE HAVE several true sequences for the same node, therefore, we will check all samples for the same node to each of the true nodes
        incorrectly_predicted_sites = []
        percentage_identity_dict = defaultdict()
        blosum_score_dict = defaultdict()
        blosum_score_dict_true = defaultdict()
        for true_name, true_seq in true_sequences.items():
            blosum_true = DraupnirUtils.score_pairwise(true_seq,true_seq, additional_info.blosum_dict, gap_s=11, gap_e=1)
            blosum_score_dict_true[true_name] = blosum_true
            incorrectly_predicted_sites_sample = []
            percentage_identity_samples=[]
            blosum_score_samples = []
            for sample_name, sample_seq in sampled_sequences.items():
                ips = DraupnirUtils.incorrectly_predicted_aa(sample_seq, true_seq)
                pid = DraupnirUtils.perc_identity_pair_seq(sample_seq,true_seq) #TODO: is this correctly done?
                blosum = DraupnirUtils.score_pairwise(sample_seq,true_seq, additional_info.blosum_dict, gap_s=11, gap_e=1)
                incorrectly_predicted_sites_sample.append(ips)
                percentage_identity_samples.append(pid)
                blosum_score_samples.append(blosum)
                correlations_dataframe_TEST.loc[["Sample_{}".format(sample_name.split("_")[-1])], "PercentageID_Sampled_{}_Observed_{}".format(sample_name.split("_")[-1],true_name)] = DraupnirUtils.perc_identity_pair_seq(sample_seq,true_seq)
            incorrectly_predicted_sites.append(incorrectly_predicted_sites_sample)
            percentage_identity_dict[true_name] = sum(percentage_identity_samples)/len(percentage_identity_samples)
            blosum_score_dict[true_name] = sum(blosum_score_samples)/len(blosum_score_samples)

        incorrectly_predicted_sites_array = np.array(incorrectly_predicted_sites)
        np.save("{}/Incorrectly_Predicted_Sites".format(results_dir), incorrectly_predicted_sites)
        average_incorrectly_predicted_sites = np.mean(incorrectly_predicted_sites_array, axis=1)
        std_incorrectly_predicted_sites = np.std(incorrectly_predicted_sites_array, axis=1)
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111)
        ## the data
        N = average_incorrectly_predicted_sites.shape[0]
        ## necessary variables
        ind = np.arange(N)  # the x locations for the groups
        width = 0.35  # the width of the bars
        blue, = sns.color_palette("muted", 1)
        ## the bars
        rects1 = ax.bar(ind, average_incorrectly_predicted_sites,
                        width,
                        color=blue,
                        alpha=0.75,
                        edgecolor="blue",
                        yerr=std_incorrectly_predicted_sites,
                        error_kw=dict(elinewidth=2, ecolor='red'))

        DraupnirUtils.autolabel(rects1, ax,blosum_dict=blosum_score_dict,percent_id_dict=percentage_identity_dict,blosum_true_dict = blosum_score_dict_true)
        # axes and labels
        ax.set_xlim(-width, N * + width)
        ax.set_ylim(0, alignment_length)
        ax.set_ylabel('Number of incorrect sites')
        ax.set_title('{} : Incorrectly predicted aa sites (%ID)'.format(name))
        xTickMarks = list(true_sequences.keys())
        ax.set_xticks(ind + width)
        xtickNames = ax.set_xticklabels(xTickMarks)
        plt.setp(xtickNames, rotation=10, fontsize=8)

        ## add a legend--> Not working (inside or outside loop)
        # ax.legend((rects1[0]), ('Draupnir')) #changed from rects1[0], rects2[0]
        plt.title(r"Incorrectly predicted amino acids; {}".format(additional_load.full_name))
        plt.savefig("{}/IncorrectlyPredictedAA_BarPlot.png".format(results_dir))
        plt.close()

        columns_test = sorted(list(correlations_dataframe_TEST.columns.values))

        correlations_dataframe_TEST = correlations_dataframe_TEST.round(1)
        correlations_dataframe_TEST = correlations_dataframe_TEST[columns_test]
        correlations_dataframe_TEST = correlations_dataframe_TEST.transpose()

        correlations_dataframe_TEST.to_csv("{}/Correlations_samples_vs_Observed_INCORRECTLY_predicted_AA.csv".format(results_dir), sep="\t")


    plot_results()
def clean_and_realign_train(name,dataset_test,dataset_train,aa_sequences_predictions,test_ordered_nodes,n_samples,aa_probs,results_dir,additional_load,additional_info):
    """Function designed to deal with data sets whose ancestral sequences aree not aligned to the leaves.
     :param str name
     :param dataset_test
     :param dataset_train
     :param aa_sequences_predictions
     :param test_ordered_nodes
     :param n_samples
     :param aa_probs
     :param results_dir
     :param additional_load
     :param additional_info"""
    # Highlight: Heatmap of the test sequences against the train sequences
    node_info = test_ordered_nodes.repeat(n_samples).unsqueeze(-1).reshape(n_samples, len(test_ordered_nodes), 1)
    aa_sequences_predictions = torch.cat((node_info,aa_sequences_predictions),dim=2)
    #aa_sequences_predictions = aa_sequences_predictions.permute(1,0,2)
    test_indx_patristic = (aa_sequences_predictions[:,:, 0][..., None] == dataset_test[:,0,1]).any(-1) #[..., None]
    aa_sequences_predictions = aa_sequences_predictions[test_indx_patristic].unsqueeze(1) #unsqueeze to keep the the size = [n_samples, n_nodes, seq-len]
    n_unique_nodes = len(list(set(dataset_test[:,0,1].tolist()))) #for the coral faviina sequences we have 5 times the root node
    tree_levelorder_dict = dict(zip(list(range(0,len(additional_load.tree_levelorder_names))),additional_load.tree_levelorder_names))
    special_nodes_list = list(additional_load.special_nodes_dict.keys()) #assuming it preserves the order    exit()

    records=[]
    #Append the test samples
    for sample_idx in range(n_samples):
        for node_idx in range(n_unique_nodes):
            predicted_seq = aa_sequences_predictions[sample_idx, node_idx]
            predicted_seq_letters = DraupnirUtils.convert_to_letters(predicted_seq, aa_probs)
            record = SeqRecord.SeqRecord(Seq(''.join(predicted_seq_letters).replace("-", "")),
                                         annotations={"molecule_type": "protein"},
                                         id="Sampled_node_{}_{}_sample_index_{}".format(node_idx,
                                                                                        tree_levelorder_dict[node_idx],
                                                                                        sample_idx),
                                         description="")
            records.append(record)

    #Append the train sequences
    for train_index, true_train_seq in enumerate(dataset_train[:, 2:, 0]):
        true_seq_letters = DraupnirUtils.convert_to_letters(true_train_seq, aa_probs)
        node_number = dataset_train[train_index, 0, 1].item()  # tree levelorder number
        node_name = tree_levelorder_dict[node_number]  # tree name
        #node_true_name = special_nodes_list[train_index]  # sequence name in the paper
        record = SeqRecord.SeqRecord(Seq(''.join(true_seq_letters).replace("-", "")),
                                     annotations={"molecule_type": "protein"},
                                     id="Train_True_node_{}_{}".format(node_name, train_index), description="")
        records.append(record)

    samples_out_file = os.path.join(results_dir, "test_samples_plus_train_true_to_align.fasta")
    SeqIO.write(records, samples_out_file, "fasta")
    alignment_out_file = os.path.join(results_dir, "test_samples_plus_train_true_aligned.fasta")
    dict_alignment, alignment = DraupnirUtils.infer_alignment(None, samples_out_file, alignment_out_file)
    alignment_length = len(alignment[0].seq)
    test_sampled_sequences = defaultdict()
    train_true_sequences = defaultdict()
    for key, value in dict_alignment.items():
        if key.startswith("Sample"):
            test_sampled_sequences[key] = value
        elif key.startswith("Train_True"):
            train_true_sequences[key] = value

    correlations_dataframe_PID = defaultdict()
    for node_idx in range(n_unique_nodes):
        nodes_seq_dict = {key: value for key, value in test_sampled_sequences.items() if "node_{}".format(node_idx) in key}
        correlations_dataframe_PID_node = pd.DataFrame(index=[name for name in nodes_seq_dict.keys()] + ["Average", "Std"])
        percentage_identity_dict = defaultdict()
        blosum_score_dict = defaultdict()
        for train_true_name, train_true_seq in train_true_sequences.items():
            percentage_identity_samples = []
            blosum_score_samples = []
            for test_sample_name, sample_seq in nodes_seq_dict.items():
                pid = DraupnirUtils.perc_identity_pair_seq(sample_seq, train_true_seq)
                blosum = DraupnirUtils.score_pairwise(sample_seq, train_true_seq, additional_info.blosum_dict, gap_s=11, gap_e=1)
                percentage_identity_samples.append(pid)
                blosum_score_samples.append(blosum)
                correlations_dataframe_PID_node.loc[[test_sample_name], "PercentageID_vs_Train_Observed_{}".format( train_true_name)] = DraupnirUtils.perc_identity_pair_seq(sample_seq,train_true_seq)

            average_pid = sum(percentage_identity_samples) / len(percentage_identity_samples)
            if n_samples > 1:
                std_pid = statistics.stdev(percentage_identity_samples)
            else:
                std_pid = [0]
            percentage_identity_dict[train_true_name] = average_pid
            blosum_score_dict[train_true_name] = sum(blosum_score_samples) / len(blosum_score_samples)
            correlations_dataframe_PID_node.loc[["Average"], "PercentageID_vs_Train_Observed_{}".format(train_true_name)] = average_pid
            correlations_dataframe_PID_node.loc[["Std"], "PercentageID_vs_Train_Observed_{}".format(train_true_name)] = std_pid
        correlations_dataframe_PID["node_{}".format(node_idx)] = correlations_dataframe_PID_node


    train_nodes_names = [name.replace("Train_True_node_","").split("_")[0] for name in train_true_sequences.keys()]

    for node_idx in range(n_unique_nodes):
            correlations_dataframe_PID_node = correlations_dataframe_PID["node_{}".format(node_idx)]
            average_pid = np.array(correlations_dataframe_PID_node.loc[["Average"]].values.tolist()).squeeze(0)
            std_pid = np.array(correlations_dataframe_PID_node.loc[["Std"]].values.tolist()).squeeze(0)
            fig = plt.figure(figsize=(14, 8))
            ax = fig.add_subplot(111)
            ## the data
            N = average_pid.shape[0]
            ## necessary variables
            ind = np.arange(N)  # the x locations for the groups
            width = 0.35  # the width of the bars
            blue, = sns.color_palette("muted", 1)
            ## the bars
            rects1 = ax.bar(ind, average_pid,
                            width,
                            color=blue,
                            alpha=0.75,
                            edgecolor="blue",
                            yerr=std_pid,
                            error_kw=dict(elinewidth=2, ecolor='red'))

            DraupnirUtils.autolabel(rects1, ax)
            # axes and labels
            ax.set_xlim(-width, N * + width)
            ax.set_ylim(0, 100)
            ax.set_ylabel('% ID ')
            ax.set_title('{} :% ID of {} against the train sequences'.format(name,node_idx))
            xTickMarks = train_nodes_names
            ax.set_xticks(ind + width)
            xtickNames = ax.set_xticklabels(xTickMarks)
            plt.setp(xtickNames, rotation=180, fontsize=8)

            ## add a legend--> Not working (inside or outside loop)
            # ax.legend((rects1[0]), ('Draupnir')) #changed from rects1[0], rects2[0]
            plt.title(r"% ID of the predicted internal node (root) against the leaves; {}".format(additional_load.full_name))
            plt.savefig("{}/PercentID_node_{}_against_train_seq.pdf".format(results_dir,node_idx))
            plt.close()
def plot_incorrectly_predicted_aa(incorrectly_predicted_sites_df,results_directory,align_len, additional_load):
    """Plots a histogram with the number of incorrectly predicted sites per sequence
    :param pandas-dataframe incorrectly_predicted_sites_df
    :param str results_directory
    :param align_len: alignment length
    :param namedtuple additional_load
    """
    print("Building Fast Incorrectly Predicted aa...")
    average_incorrectly_predicted_sites = incorrectly_predicted_sites_df["Average"].values
    try:
        accuracy = len(np.where(average_incorrectly_predicted_sites < 75)[0]) / average_incorrectly_predicted_sites.shape[0]
        print("Accuracy (% incorrect sites < 75)  : {}".format(accuracy))
    except:
        accuracy = 0.
    folder_name =os.path.basename(results_directory)
    data_name = ["Internal nodes" if folder_name.startswith("Test") else "Leaves nodes"][0]
    results_directory_abs = os.path.dirname(results_directory)

    text_file = open("{}/Hyperparameters_{}.txt".format(results_directory_abs, "_".join(results_directory_abs.split("_")[-5:])),"a")
    text_file.write("accuracy {} (% incorrect sites < 75) : {}\n".format(data_name,accuracy))
    std_incorrectly_predicted_sites = incorrectly_predicted_sites_df["Std"].values

    nodes_indexes = np.array(incorrectly_predicted_sites_df.index.str.split("//").str[1]).astype(float).tolist()
    incorrectly_predicted_sites_df = incorrectly_predicted_sites_df.drop(["Average","Std"],axis=1)
    incorrectly_predicted_sites = np.vstack([nodes_indexes,incorrectly_predicted_sites_df.values.T])
    np.save("{}/Incorrectly_Predicted_Sites".format(results_directory), incorrectly_predicted_sites) #Highlight: Hopefully this is the same as before


    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)

    ## the data
    N = average_incorrectly_predicted_sites.shape[0]
    ## necessary variables
    ind = np.arange(N) # the x locations for the groups
    width = 0.35  # the width of the bars
    blue, = sns.color_palette("muted", 1)
    ## the bars
    rects1 = ax.bar(ind, average_incorrectly_predicted_sites,
                    width,
                    color=blue,
                    alpha=0.75,
                    edgecolor="blue",
                    yerr=std_incorrectly_predicted_sites,
                    error_kw=dict(elinewidth=2, ecolor='red'))


    DraupnirUtils.autolabel(rects1,ax)
    # axes and labels
    ax.set_xlim(-width, N* + width)
    ax.set_ylim(0, align_len)
    ax.set_ylabel('Number of incorrect sites')
    ax.set_title('Incorrectly predicted aa sites (%ID); {} \n'.format(data_name) + r'{}'.format(additional_load.full_name))
    xTickMarks = incorrectly_predicted_sites_df.index.tolist()
    ax.set_xticks(ind + width)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=45, fontsize=8)

    plt.savefig("{}/IncorrectlyPredictedAA_BarPlot".format(results_directory))
def build_dataframes_overlapping_histograms(predictions_samples, dataset_train,dataset_test, name, num_samples,children_indexes,results_directory,aa_probs,correspondence_dict=None):
    #TODO: remove????
    print("Building Overlapping Histogram...")
    n_children = len(children_indexes)
    def Percent_ID_OBS_OBS():
        "Generate the  %ID of the OBS TEST sequences against the OBS train"
        percent_id_OBS = dict.fromkeys(dataset_train[:,0,1].tolist(), dict.fromkeys(dataset_test[:,0,1].tolist(),[]))
        for i,trainseq in enumerate(dataset_train):
            seq_TRAIN_obs_letters_i = DraupnirUtils.convert_to_letters(dataset_train[i, 2:,0],aa_probs)
            for j,testseq in enumerate(dataset_test):
                seq_TEST_obs_letters_i = DraupnirUtils.convert_to_letters(dataset_test[j, 2:,0],aa_probs)
                pid =DraupnirUtils.perc_identity_pair_seq(seq_TRAIN_obs_letters_i,seq_TEST_obs_letters_i)
                percent_id_OBS[dataset_train[i,0,1].item()][dataset_test[j,0,1].item()] = pid
        return percent_id_OBS
    def Percent_ID_PRED_OBS():
        "Generate the Average and STD %ID of the sampled sequences against the OBS train"
        #percent_id_PRED = dict.fromkeys(Dataset_train[:,0,1].tolist(), dict.fromkeys(predictions_samples[0,:,1].tolist(),{"Average":[],"Std":[]})) #{"Average":[],"Std":[]}
        percent_id_PRED = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for i, seq in enumerate(dataset_train):
            seq_TRAIN_obs_letters_i = DraupnirUtils.convert_to_letters(dataset_train[i, 2:,0],aa_probs)
            for j in range(n_children):#for test node #n_children
                all_sampled_index = predictions_samples[:,j,3:]  # All samples for same test seq
                percent_id_i=[] #all samples for the same test sequence
                for w in range(num_samples):
                    seq_sampled_test = DraupnirUtils.convert_to_letters(all_sampled_index[w],aa_probs)
                    pid = DraupnirUtils.perc_identity_pair_seq(seq_TRAIN_obs_letters_i, seq_sampled_test)
                    percent_id_i.append(pid)
                percent_id_PRED[dataset_train[i,0,1].item()][predictions_samples[0,j,1]]["Average"] = np.mean(np.array(percent_id_i),axis=0)
                percent_id_PRED[dataset_train[i,0,1].item()][predictions_samples[0,j,1]]["Std"] = np.std(np.array(percent_id_i), axis=0)
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
def barplot_aa_replacement(predictions_samples, dataset_test,full_name, num_samples,nodes_indexes,results_directory,aa_probs,correspondence_dict=None):
    """Plots the amino acid replacements occurred during the evolutionary process. It visualizes the frequency of the amino acids have been
    predicted to replace the true amino acid
    :param predictions_samples  [n_nodes,n_samples,align_len]
    :param dataset_test
    :param full_name: complete name of the data set project
    :param num_samples
    :param nodes_indexes
    :param results_directory
    :param aa_probs
    :param correspondence_dict: contains the correspondence among the nodes indexes and their true names"""
    print("Building aa replacement plots...this might take a while")
    results_directory_path = os.path.dirname(results_directory)
    folder_type =os.path.basename(results_directory).split("_")[0]
    aa_types_dict = DraupnirUtils.aminoacid_names_dict(aa_probs) #apply
    for index,aa in enumerate(list(aa_types_dict.keys())):
        for seq_idx,seq_name in zip(range(dataset_test.shape[0]),nodes_indexes):
            seq_name = int(seq_name.item())
            positions = np.where(dataset_test[seq_idx] == index)[0]
            prediction_for_seq = predictions_samples[:,seq_idx,positions] #predictions for positions where there should be aa == aa_type
            if prediction_for_seq.size !=0:#if it's not empty (for example in the benchmark there are no gaps and then it becomes empty)
                aa_count = np.apply_along_axis(lambda x: np.bincount(x, minlength=aa_probs), axis=0,arr=prediction_for_seq.astype("int64")).T
                aa_count = pd.DataFrame(aa_count,index=positions,columns=list(aa_types_dict.keys()))
                fig = plt.figure(figsize=(16, 12))
                #color_map = matplotlib.colors.ListedColormap(plt.cm.tab20c(np.linspace(0, 1,aa_probs)))
                color_map = matplotlib.colors.ListedColormap(["plum","navy","turquoise","peachpuff","palevioletred","red","darkorange","yellow","lime","green","dodgerblue","blue","purple","magenta","grey","maroon","lightcoral","olive","teal","goldenrod","black"])
                aa_count.plot.bar(stacked=True,colormap=color_map)
                plt.legend( prop={'size': 6},bbox_to_anchor = (1.02, 0.9))
                plt.ylabel("aa count")
                plt.xlabel("True positions of {}".format(aa))
                plt.title("Aa replacement for {} in seq {}; \n".format(aa,correspondence_dict[seq_name]) +r"{}".format(full_name))
                plt.savefig("{}/ReplacementPlots_{}/Replacement_{}_seq_{}.png".format(results_directory_path,folder_type,aa,correspondence_dict[seq_name]))
                plt.clf()
                plt.close(fig)








