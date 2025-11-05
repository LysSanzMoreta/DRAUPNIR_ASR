"""
=======================
2022: Lys Sanz Moreta
Draupnir : Ancestral protein sequence reconstruction using a tree-structured Ornstein-Uhlenbeck variational autoencoder
=======================
"""
from collections import defaultdict

import torch
import numpy as np
from dataclasses import dataclass
import draupnir
import draupnir.utils as DraupnirUtils
@dataclass
class TrainConfig:
    beta_a: float = 1
    beta_b : float = 2.5


def index_generator(indexes):
    """Call method to subsample in order.
    model (args, iter_num = index_generator())
    plate(..., next(iter_num))"""
    i=0
    while True:
        yield indexes[i]
        i = (i + 1) % len(indexes)
        return i ##?

def fill_estimates(guide_map_estimates,map_estimates):
    for key, val in guide_map_estimates.items():
        if key in ["latent_z"]:
            guide_map_estimates[key] = DraupnirUtils.squeeze_tensor(required_ndims=2, tensor=val)
            if key not in map_estimates:
                map_estimates[key] = val
            else:
                #map_estimates[key] = torch.concat([map_estimates[key], guide_map_estimates[key]], dim=1)
                map_estimates[key] = val
        elif key in ["alpha", "sigma_n", "sigma_f", "lambd"]:
            guide_map_estimates[key] = DraupnirUtils.squeeze_tensor(required_ndims=1, tensor=val)
            map_estimates[key] = val
        elif key in ["rnn_final_hidden_state", "rnn_hidden_states"]:
            if key not in map_estimates:
                map_estimates[key] = val
            else:
                #map_estimates[key] = torch.concat([map_estimates[key], guide_map_estimates[key]], dim=0)
                map_estimates[key] = val
        elif key in ["z_scale","z_loc"]:
            map_estimates[key] = val

        elif key in ["rnn_final_bidirectional"]:
            if key not in map_estimates:
                map_estimates[key] = val
            else:
                #map_estimates[key] = torch.concat([map_estimates[key], guide_map_estimates[key]], dim=1)
                map_estimates[key] = val
        elif key in ["context_vector","attention_scores","attention_logits","hidden_states"]:
            print(f"{key}", val.shape)


            if key not in map_estimates:
                map_estimates[key] = val
            else:
                map_estimates[key] = val
                map_estimates[key] = torch.concat([map_estimates[key], guide_map_estimates[key]], dim=0)
    return  guide_map_estimates, map_estimates

def train_batch(svi,training_function_input):
    """Regular batch training without shuffling datatasets
    :param svi: pyro infer engine
    :param cladistic_matrix
    :param patristic_matrix
    :param dataloader train_loader: Pytorch dataloader
    :param namedtuple args
    """
    patristic_matrix_model = training_function_input["patristic_matrix_model"]
    cladistic_matrix_full = training_function_input["cladistic_matrix_full"]
    cladistic_matrix_train = training_function_input["cladistic_matrix_train"]
    dataset_train_blosum = training_function_input["dataset_train_blosum"]
    train_loader = training_function_input["train_loader"]
    guide = training_function_input["guide"]
    args = training_function_input["args"]
    train_loss = 0.0
    seq_lens = []
    map_estimates = defaultdict()
    for batch_number, dataset in enumerate(train_loader):
        for batch_name, batch_dataset, batch_patristic, batch_blosum_weighted, batch_data_blosum in zip(
                dataset["batch_name"],
                dataset["batch_data"],
                dataset["batch_patristic"],
                dataset["batch_blosum_weighted"],
                dataset["batch_data_blosum"]):
            if args.use_cuda:
                batch_dataset = batch_dataset.cuda()
                batch_blosum_weighted = batch_blosum_weighted.cuda()
                batch_patristic = batch_patristic.cuda()
                batch_data_blosum = batch_data_blosum.cuda()

            batch_datasets = {"int":batch_dataset,
                              "blosum":batch_data_blosum,
                              "onehot":torch.ones(batch_dataset.shape[0]),
                              "mask":torch.ones_like(batch_dataset)} #todo: investigate mask
            seq_lens += batch_dataset[:, 0, 0].tolist()
            guide_map_estimates = guide(batch_datasets,
                                  batch_patristic, #recall that the patristic is n_seqs + 1 to re-add the node names
                                  cladistic_matrix_train,
                                  dataset_train_blosum,
                                  batch_blosum=None,
                                  map_estimates=None)  # only saving 1 sample

            #YOU ARE IN THE WRONG FUNCTION


            guide_map_estimates,map_estimates = fill_estimates(guide_map_estimates,map_estimates)



            train_loss += svi.step(batch_datasets,
                                   batch_patristic,
                                   cladistic_matrix_full,
                                   batch_data_blosum,
                                   batch_blosum_weighted,
                                   map_estimates)

            # Normalize loss
            # torch.cuda.reset_max_memory_allocated() #necessary?
    normalizer_train = sum(seq_lens)
    total_epoch_loss_train = train_loss / normalizer_train
    return total_epoch_loss_train, map_estimates

def train(svi,training_function_input):
    """Non batched training
    :param svi: pyro infer engine
    :param cladistic_matrix
    :param patristic_matrix
    :param dataloader train_loader: Pytorch dataloader
    """

    patristic_matrix = training_function_input["patristic_matrix_model"]
    cladistic_matrix = training_function_input["cladistic_matrix_full"]
    dataset_blosum = training_function_input["dataset_train_blosum"]
    train_loader = training_function_input["train_loader"]
    map_estimates = training_function_input["map_estimates"]
    args = training_function_input["args"]
    train_loss = 0.0
    seq_lens = []

    for batch_number, datasets in enumerate(train_loader):
            data_int = datasets["int"]
            seq_lens += data_int[:, 0, 0].tolist()
            if args.use_cuda:
                datasets = {key:val.cuda() for key,val in datasets.items()}
            train_loss += svi.step(datasets,patristic_matrix,cladistic_matrix,dataset_blosum,None,map_estimates) #None is the clade blosum, it's None because here we do not do clade batching
    # Normalize loss
    #normalizer_train = sum(seq_lens)
    total_epoch_loss_train = train_loss #/ normalizer_train
    return total_epoch_loss_train

def train_batch_clade(svi,training_function_input):
    """Batch by clade training
    :param svi: pyro infer engine
    :param cladistic_matrix
    :param patristic_matrix
    :param dataloader train_loader: Pytorch dataloader
    :param namedtuple args
    """
    #patristic_matrix = training_function_input["patristic_matrix_model"]
    cladistic_matrix = training_function_input["cladistic_matrix_full"]
    #dataset_blosum = training_function_input["dataset_train_blosum"]
    train_loader = training_function_input["train_loader"]
    map_estimates = training_function_input["map_estimates"]
    args = training_function_input["args"]
    train_loss = 0.0
    seq_lens = []
    for batch_number, dataset in enumerate(train_loader):
        for clade_name, clade_dataset, clade_patristic, clade_blosum_weighted, clade_data_blosum in zip(dataset["clade_name"],
                                                                                                        dataset["clade_data"],
                                                                                                        dataset["clade_patristic"],
                                                                                                        dataset["clade_blosum_weighted"],
                                                                                                        dataset["clade_data_blosum"]):
            if args.use_cuda:
                clade_dataset = clade_dataset.cuda()
                clade_blosum_weighted = clade_blosum_weighted.cuda()
                clade_patristic = clade_patristic.cuda()  # cannot be used like this, we cannot have a variable size latent space
                clade_data_blosum = clade_data_blosum.cuda()
            seq_lens += clade_dataset[:, 0, 0].tolist()
            train_loss += svi.step(clade_dataset, clade_patristic, cladistic_matrix, clade_data_blosum,clade_blosum_weighted,map_estimates)  # Highlight: if we want to use this for plating, input the entire patristic distance
            # Normalize loss
    normalizer_train = sum(seq_lens)
    total_epoch_loss_train = train_loss / normalizer_train
    return total_epoch_loss_train

def random_masking(data):

    dim0,dim1,dim2 = data.shape
    random_mask = torch.rand(dim0)
    random_mask = (random_mask > 0.5)

    random_mask = random_mask[:,None,None].repeat(1,dim1,dim2)

    return random_mask


def train_transformer(svi,training_function_input):
    """Masked diffusion trasformer
    https://github.com/apapiu/transformer_latent_diffusion/blob/main/tld/train.py
    https://medium.com/@mickael.boillaud/denoising-diffusion-model-from-scratch-using-pytorch-658805d293b4
    https://www.youtube.com/watch?v=zc5NTeJbk-k
    TODO: Likelihood annealing if it does not work

    https://medium.com/@luvverma2011/demystifying-attention-mechanisms-in-sequence-to-sequence-models-transformers-part-1-98e2962408f0
    Flash cosine sim attention: https://github.com/lucidrains/flash-cosine-sim-attention
    Fast attention: https://github.com/kyegomez/MegaVIT/blob/main/mega_vit/main.py
        Einops+atention : https://medium.com/@kyeg/einops-in-30-seconds-377a5f4d641a

    Evoformer: https://github.com/hpcaitech/FastFold/blob/main/fastfold/model/fastnn/evoformer.py
    Differentiable MSA: https://academic.oup.com/bioinformatics/article/39/1/btac724/6820925
        smith waterman: local alignment
        Needleman–Wunsch: global alignment
    Benchmark MSA methods: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1633746/

    SNAIL: https://lilianweng.github.io/posts/2018-06-24-attention/

    Other positional encodings: https://dongkwan-kim.github.io/blogs/a-short-history-of-positional-encoding/


    """
    patristic_matrix_model = training_function_input["patristic_matrix_model"]
    cladistic_matrix_full = training_function_input["cladistic_matrix_full"]
    cladistic_matrix_train = training_function_input["cladistic_matrix_train"]
    dataset_train_blosum = training_function_input["dataset_train_blosum"]
    train_loader = training_function_input["train_loader"]
    guide = training_function_input["guide"]
    args = training_function_input["args"]
    train_loss = 0.0
    seq_lens = []
    map_estimates = defaultdict()
    for batch_number, dataset in enumerate(train_loader):
        for batch_name, batch_dataset, batch_patristic, batch_blosum_weighted, batch_data_blosum in zip(
                dataset["batch_name"],
                dataset["batch_data"],
                dataset["batch_patristic"],
                dataset["batch_blosum_weighted"],
                dataset["batch_data_blosum"]):
            if args.use_cuda:
                batch_dataset = batch_dataset.cuda()
                batch_blosum_weighted = batch_blosum_weighted.cuda()
                batch_patristic = batch_patristic.cuda()
                batch_data_blosum = batch_data_blosum.cuda()

            batch_datasets = {"int": batch_dataset,
                              "blosum": batch_data_blosum,
                              "onehot": torch.ones(batch_dataset.shape[0])}


            seq_lens += batch_dataset[:, 0, 0].tolist()
            guide_map_estimates = guide(batch_datasets,
                                        batch_patristic,
                                        # recall that the patristic is n_seqs + 1 to re-add the node names
                                        cladistic_matrix_train,
                                        dataset_train_blosum,
                                        batch_blosum=None,
                                        map_estimates=None)  # only saving 1 sample



            #guide_map_estimates, map_estimates = fill_estimates(guide_map_estimates, map_estimates) #todo: why did i do this? remove? #todo : try without the guide estimates to fix the shapes
            #todo: the error is in the guide or in the guide_map estimates or in the fill_estimates
            map_estimates = None

            train_loss += svi.step(batch_datasets,
                                   batch_patristic,
                                   cladistic_matrix_full,
                                   batch_data_blosum,
                                   batch_blosum_weighted,
                                   map_estimates)  # TODO: Check trainng loop for vegvisir, something is off here, why the guide is separated?


            # Normalize loss
            # torch.cuda.reset_max_memory_allocated() #necessary?
    normalizer_train = sum(seq_lens)
    total_epoch_loss_train = train_loss / normalizer_train
    return total_epoch_loss_train, map_estimates

def select_training_function(clades_dict,svi, training_function_input):
    """Selects a training function
    :param : Stochastic variational inference engine

    """
    args = training_function_input["args"]
    training_method= lambda f, svi, training_function_input: lambda svi, training_function_input: f(svi, training_function_input)


    if args.draupnir_version == "2":
        print("Using Draupnir 2.0")
        training_function = training_method(train_transformer,
                                            svi,
                                            training_function_input)

    else: #first draupnir version

        if args.batch_by_clade and clades_dict:
            training_function = training_method(train_batch_clade,
                                                svi,
                                                training_function_input
                                                )
        elif args.batch_size == 1:#no batching or plating

            training_function = training_method(train,
                                                svi,
                                                training_function_input
                                                )


        else:#batching
            training_function = training_method(train_batch,
                                                svi,
                                                training_function_input
                                                )



    return training_function
