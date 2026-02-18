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
from pyro import poutine

# @dataclass
# class TrainConfig:
#     beta_a: float = 1
#     beta_b : float = 2.5

def index_generator(indexes):
    """Call method to subsample in order.
    model (args, iter_num = index_generator())
    plate(..., next(iter_num))"""
    i=0
    while True:
        yield indexes[i]
        i = (i + 1) % len(indexes)
        return i ##?

# def fill_estimates(guide_map_estimates,map_estimates,batching=False): #old which did not work
#     for key, val in guide_map_estimates.items():
#         if key in ["latent_z"]:
#             guide_map_estimates[key] = DraupnirUtils.squeeze_tensor(required_ndims=2, tensor=val)
#             if key not in map_estimates:
#                 map_estimates[key] = val
#             else:
#                 if batching:
#                     map_estimates[key] = val
#                 else:
#                     map_estimates[key] = torch.concat([map_estimates[key], guide_map_estimates[key]], dim=1)
#         elif key in ["alpha", "sigma_n", "sigma_f", "lambd"]:
#             guide_map_estimates[key] = DraupnirUtils.squeeze_tensor(required_ndims=1, tensor=val)
#             map_estimates[key] = val
#         elif key in ["rnn_final_hidden_state", "rnn_hidden_states","rnn_final_forward_backward_sum"]:
#             if key not in map_estimates:
#                 map_estimates[key] = val
#             else:
#                 if batching:
#                     map_estimates[key] = val
#                 else:
#                     map_estimates[key] = torch.concat([map_estimates[key], guide_map_estimates[key]], dim=0)
#         elif key in ["z_scale","z_loc"]:
#             map_estimates[key] = val
#
#         elif key in ["rnn_final_bidirectional"]:
#             if key not in map_estimates:
#                 map_estimates[key] = val
#             else:
#                 if batching:
#                     map_estimates[key] = val
#                 else:
#                     map_estimates[key] = torch.concat([map_estimates[key], guide_map_estimates[key]], dim=1)
#         elif key in ["context_vector","attention_scores","attention_logits","hidden_states"]:
#             if key not in map_estimates:
#                 map_estimates[key] = val
#             else:
#                 map_estimates[key] = val
#                 #map_estimates[key] = torch.concat([map_estimates[key], guide_map_estimates[key]], dim=0)
#     return  guide_map_estimates, map_estimates

def fill_estimates(guide_map_estimates,map_estimates,batching=True): #todo: examine one by one and see which one breaks it
    for key, val in guide_map_estimates.items():
        if key in ["latent_z"]:
            guide_map_estimates[key] = DraupnirUtils.squeeze_tensor(required_ndims=2, tensor=val)
            if key not in map_estimates:
                map_estimates[key] = val
            else:
                map_estimates[key] = torch.concat([map_estimates[key], guide_map_estimates[key]], dim=1)
        elif key in ["alpha", "sigma_n", "sigma_f", "lambd"]:
            guide_map_estimates[key] = DraupnirUtils.squeeze_tensor(required_ndims=1, tensor=val)
            map_estimates[key] = val
        elif key in ["rnn_final_hidden_state", "z_scale", "z_loc"]:
            if key not in map_estimates:
                map_estimates[key] = val
            else:
                map_estimates[key] = torch.concat([map_estimates[key], guide_map_estimates[key]], dim=0)
        elif key in [ "rnn_hidden_states","rnn_final_bidirectional","rnn_final_forward_backward_sum"]:
            map_estimates[key] = val
        # elif key in ["rnn_final_bidirectional","rnn_final_forward_backward_sum"]:
        #     if key not in map_estimates:
        #         map_estimates[key] = val
        #     else:
        #         map_estimates[key] = torch.concat([map_estimates[key], guide_map_estimates[key]], dim=1)
        elif key in ["context_vector","attention_scores","attention_logits","hidden_states"]:
            if key not in map_estimates:
                map_estimates[key] = val
            else:
                map_estimates[key] = torch.concat([map_estimates[key], guide_map_estimates[key]], dim=0)
        elif key in ["embeddings","batch_nodes"]:
            map_estimates[key] = val
    return  guide_map_estimates, map_estimates

def train_batch_new(svi,training_function_input):
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

        if args.use_cuda:
            dataset_batch = defaultdict()
            for batch_name, *extra  in zip(*dataset.values()) : #we have to this trick to unpack, otherwise one extra dimension is added
                for key, val in zip(list(dataset.keys())[1:],extra): #we skip the batch_name key
                    dataset_batch[key] = val.cuda() if isinstance(val,torch.Tensor) else val

            batch_datasets = {"int":dataset_batch["batch_data_int"],
                              "blosum":dataset_batch["batch_data_blosum"],
                              "onehot":torch.ones(dataset_batch["batch_data_int"].shape[0]),
                              "mask":torch.ones_like(dataset_batch["batch_data_int"]),
                              "embedding": dataset_batch["batch_embedding"],
                              "sequences_representations": dataset_batch["batch_sequence_representation"],
                              }
            seq_lens += dataset_batch["batch_data_int"][:, 0, 0].tolist()


            guide_map_estimates = guide(batch_datasets,
                                  dataset_batch["batch_patristic"], #recall that the patristic is n_seqs + 1 to re-add the node names
                                  cladistic_matrix_train,
                                  dataset_train_blosum,
                                  batch_blosum=None,
                                  map_estimates=None)  # only saving 1 sample


            guide_map_estimates,map_estimates = fill_estimates(guide_map_estimates,map_estimates,batching=True)
            #map_estimates["annealing_factor"] = torch.Tensor([min(1,0.1 + ((training_function_input["epoch"] +1) / 50))]).to(args.device) #until epoch 49 rampage
            map_estimates["annealing_factor"] = torch.Tensor([1.]).to(args.device)

            train_loss += svi.step(batch_datasets,
                                   dataset_batch["batch_patristic"],
                                   cladistic_matrix_full,
                                   dataset_batch["batch_data_blosum"],
                                   dataset_batch["batch_blosum_weighted"],
                                   map_estimates)

            # Normalize loss
            # torch.cuda.reset_max_memory_allocated() #necessary?
    normalizer_train = sum(seq_lens)
    total_epoch_loss_train = train_loss / normalizer_train
    return total_epoch_loss_train, map_estimates

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
    epoch = training_function_input["epoch"] + 1
    train_loss = 0.0
    seq_lens = []
    map_estimates = defaultdict()
    #from torch.profiler import profile, ProfilerActivity, record_function

    for batch_number, dataset in enumerate(train_loader):
        # for batch_name, batch_dataset_int, batch_patristic, batch_blosum_weighted, batch_data_blosum , batch_dataset_embedding, batch_sequences_representation in zip(
        #         dataset["batch_name"],
        #         dataset["batch_data_int"],
        #         dataset["batch_patristic"],
        #         dataset["batch_blosum_weighted"],
        #         dataset["batch_data_blosum"],
        #         dataset["batch_embedding"],
        #         dataset["batch_sequence_representation"],
        # ):
        #
        #         if args.use_cuda:
        #             batch_dataset_int = batch_dataset_int.cuda()
        #             batch_blosum_weighted = batch_blosum_weighted.cuda()
        #             batch_patristic = batch_patristic.cuda()
        #             batch_data_blosum = batch_data_blosum.cuda()
        #             batch_dataset_embedding = batch_dataset_embedding.cuda()
        #             batch_sequences_representation = batch_sequences_representation.cuda()

                if args.use_cuda:
                    batch_dataset_int = dataset["batch_data_int"].squeeze(0).to('cuda', non_blocking=True)
                    batch_blosum_weighted = dataset["batch_blosum_weighted"].squeeze(0).to('cuda', non_blocking=True)
                    batch_patristic = dataset["batch_patristic"].squeeze(0).to('cuda', non_blocking=True)
                    batch_data_blosum = dataset["batch_data_blosum"].squeeze(0).to('cuda', non_blocking=True)
                    batch_dataset_embedding = dataset["batch_embedding"].squeeze(0).to('cuda', non_blocking=True)
                    batch_sequences_representation = dataset["batch_sequence_representation"].squeeze(0).to('cuda', non_blocking=True)

                batch_datasets = {"int":batch_dataset_int,
                                  "blosum":batch_data_blosum,
                                  "onehot":torch.ones(batch_dataset_int.shape[0]),
                                  "mask":torch.ones_like(batch_dataset_int),
                                  "embedding": batch_dataset_embedding,
                                  "sequences_representations": batch_sequences_representation,
                                  }
                seq_lens += batch_dataset_int[:, 0, 0].tolist()

                # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,
                #              with_stack=True) as prof:



                guide_map_estimates = guide(batch_datasets,
                                      batch_patristic, #recall that the patristic is n_seqs + 1 to re-add the node names
                                      cladistic_matrix_train,
                                      dataset_train_blosum,
                                      batch_blosum=None,
                                      map_estimates=None)  # only saving 1 sample

                guide_map_estimates,map_estimates = fill_estimates(guide_map_estimates,map_estimates,batching=True)
                torch.cuda.synchronize()

                #map_estimates["annealing_factor"] = torch.Tensor([min(1,training_function_input["step"]/training_function_input["temp_anneal"])]).to(args.device)
                #map_estimates["annealing_factor"] = torch.Tensor([0.3]).to(args.device) if epoch < 500 else torch.Tensor([1.]).to(args.device)
                map_estimates["annealing_factor"] = torch.Tensor([1.]).to(args.device)
                #print("step:",training_function_input["step"],"annealing factor",map_estimates["annealing_factor"])

                train_loss += svi.step(batch_datasets,
                                       batch_patristic,
                                       cladistic_matrix_full,
                                       batch_data_blosum,
                                       batch_blosum_weighted,
                                       map_estimates)

                training_function_input["step"] += 1
                torch.cuda.synchronize()

                #
                # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
                # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
                # #
                # #prof.export_chrome_trace("trace.json")
                # #
                # exit()

                    # Normalize loss
    torch.cuda.reset_max_memory_allocated() #necessary?
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
    map_estimates["annealing_factor"] = torch.Tensor([1.]).to(args.device)

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

        if args.use_cuda:
            dataset_batch = defaultdict()
            for batch_name, *extra in zip(*dataset.values()):  # we have to this trick to unpack, otherwise one extra dimension is added
                for key, val in zip(list(dataset.keys())[1:], extra):  # we skip the batch_name key
                    dataset_batch[key] = val.to('cuda', non_blocking=True) if isinstance(val, torch.Tensor) else val
        # for clade_name, clade_dataset, clade_patristic, clade_blosum_weighted, clade_data_blosum, clade_embedding, clade_seq_representation in zip(datasets["clade_name"],
        #                                                                                                 datasets["clade_data"],
        #                                                                                                 datasets["clade_patristic"],
        #                                                                                                 datasets["clade_blosum_weighted"],
        #                                                                                                 datasets["clade_data_blosum"],
        #                                                                                                 datasets["clade_embedding"],
        #                                                                                                 datasets["clade_seq_representation"],
        #                                                                                                                  ):
        #
        #     if args.use_cuda:
        #         clade_dataset = clade_dataset.cuda()
        #         clade_blosum_weighted = clade_blosum_weighted.cuda()
        #         clade_patristic = clade_patristic.cuda()  # cannot be used like this, we cannot have a variable size latent space
        #         clade_data_blosum = clade_data_blosum.cuda()
        #         clade_embedding = clade_embedding.cuda()
        #         clade_seq_representation = clade_seq_representation.cuda()


            # clade_datasets = {"int":clade_dataset,
            #                   "blosum":clade_data_blosum,
            #                   "embedding":clade_embedding,
            #                   "seq_representation": clade_seq_representation
            #                   }
            clade_datasets = {"int":dataset_batch["clade_data"],
                              "blosum":dataset_batch["clade_data_blosum"],
                              "embedding":dataset_batch["clade_embedding"],
                              "sequences_representations": dataset_batch["clade_seq_representation"]
                              }



            seq_lens += clade_datasets["int"][:, 0, 0].tolist()
            train_loss += svi.step(clade_datasets, dataset_batch["clade_patristic"], cladistic_matrix,dataset_batch["clade_blosum_weighted"],map_estimates)  # Highlight: if we want to use this for plating, input the entire patristic distance


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
                batch_dataset = batch_dataset.to('cuda', non_blocking=True)
                batch_blosum_weighted = batch_blosum_weighted.to('cuda', non_blocking=True)
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



            guide_map_estimates, map_estimates = fill_estimates(guide_map_estimates, map_estimates) #todo: why did i do this? remove? #todo : try without the guide estimates to fix the shapes
            #todo: the error is in the guide or in the guide_map estimates or in the fill_estimates
            map_estimates = guide_map_estimates

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
