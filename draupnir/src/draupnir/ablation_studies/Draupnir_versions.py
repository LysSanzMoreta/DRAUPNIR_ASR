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
import dromi

#Draupnir-classic-batched
folder = "/home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src/draupnir/ablation_studies/draupnir_models/PLOTS_Draupnir_simulations_src_sh3_3_2025_12_03_14h24min11s451285ms_3000epochs_variational"
train_argmax_dict = torch.load("{}/Train_argmax_Plots/train_argmax_info_dict.torch".format(folder), weights_only=False)
test_argmax_dict = torch.load("{}/Test_argmax_Plots/test_argmax_info_dict.torch".format(folder), weights_only=False)

#Draupnir-Zesm
folder = "/home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src/draupnir/ablation_studies/draupnir_models/PLOTS_Draupnir_simulations_src_sh3_3_2025_12_03_15h14min18s836930ms_3000epochs_variational_z_esm"
train_argmax_dict = torch.load("{}/Train_argmax_Plots/train_argmax_info_dict.torch".format(folder), weights_only=False)
test_argmax_dict = torch.load("{}/Test_argmax_Plots/test_argmax_info_dict.torch".format(folder), weights_only=False)

#Draupnir-ESMhidden
folder = "/home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src/draupnir/ablation_studies/draupnir_models/PLOTS_Draupnir_simulations_src_sh3_3_2025_12_03_14h48min20s543620ms_3000epochs_variational_rnn_esm_embeddings"
train_argmax_dict = torch.load("{}/Train_argmax_Plots/train_argmax_info_dict.torch".format(folder), weights_only=False)
test_argmax_dict = torch.load("{}/Test_argmax_Plots/test_argmax_info_dict.torch".format(folder), weights_only=False)



def metrics_train(predictions_dict):


    aa_predictions = predictions_dict["aa_predictions"].cpu()

    _,_,blosum_dict = DraupnirsUtils.create_blosum(21,"BLOSUM62")

    dataset_int = predictions_dict["dataset"][:,2:,0].cpu().detach().numpy()

    #Highlight: transform to blosum

    dataset_blosum = np.vectorize(blosum_dict.get,signature='()->(n)')(dataset_int)
    aa_predictions_blosum = np.vectorize(blosum_dict.get,signature='()->(n)')(aa_predictions)




    n_seqs, align_lenght = predictions_dict["aa_predictions"].shape
    average_pid, std_pid = DraupnirMain.calculate_percent_id(predictions_dict["dataset"].cpu(), aa_predictions, align_lenght)

    #metrics dataframe
    dataset_train_nodes = predictions_dict["dataset"].cpu().long()[:, 0, 1]

    logits = predictions_dict["logits"].cpu()
    entropies, probabilities_softmax = DraupnirModelsUtils.compute_sites_entropies(logits= logits,node_names = dataset_train_nodes)


    average_entropy = torch.mean(entropies[:,1:])
    average_probabilities = torch.mean(probabilities_softmax[:,1:])

    for i in range(n_seqs):
        true = dataset_blosum[i]
        pred = aa_predictions_blosum[i]







    metrics_dict = {"probabilities_softmax": probabilities_softmax,
                    "entropies": entropies,
                    "average_entropy": average_entropy,
                    "average_probabilities": average_probabilities
                    }



metrics_dict = metrics_train(train_argmax_dict)




























