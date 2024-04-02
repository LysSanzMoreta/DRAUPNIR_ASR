"""
=======================
2022: Lys Sanz Moreta
Draupnir : Ancestral protein sequence reconstruction using a tree-structured Ornstein-Uhlenbeck variational autoencoder
=======================
"""
def index_generator(indexes):
    """Call method to subsample in order.
    model (args, iter_num = index_generator())
    plate(..., next(iter_num))"""
    i=0
    while True:
        yield indexes[i]
        i = (i + 1) % len(indexes)
        return i ##?
def train_batch(svi,training_function_input):
    """Regular batch training without shuffling datatasets
    :param svi: pyro infer engine
    :param cladistic_matrix
    :param patristic_matrix
    :param dataloader train_loader: Pytorch dataloader
    :param namedtuple args
    """
    patristic_matrix = training_function_input["patristic_matrix_model"]
    cladistic_matrix = training_function_input["cladistic_matrix_full"]
    dataset_blosum = training_function_input["dataset_train_blosum"]
    train_loader = training_function_input["train_loader"]
    map_estimates = training_function_input["map_estimates"]
    args = training_function_input["args"]
    train_loss = 0.0
    seq_lens = []
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
            seq_lens += batch_dataset[:, 0, 0].tolist()
            train_loss += svi.step(batch_dataset,
                                   batch_patristic,
                                   cladistic_matrix,
                                   batch_data_blosum,
                                   batch_blosum_weighted,
                                   map_estimates)
            # Normalize loss
            # torch.cuda.reset_max_memory_allocated() #necessary?
    normalizer_train = sum(seq_lens)
    total_epoch_loss_train = train_loss / normalizer_train
    return total_epoch_loss_train


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
    train_loss = 0.0
    seq_lens = []
    for batch_number, dataset in enumerate(train_loader):
            seq_lens += dataset[:,0,0].tolist()
            train_loss += svi.step(dataset,patristic_matrix,cladistic_matrix,dataset_blosum,None,map_estimates) #None is the clade blosum, it's None because here we do not do clade batching
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



def select_training_function(clades_dict,svi, training_function_input):
    """Selects a training function
    :param : Stochastic variational inference engine

    """
    args = training_function_input["args"]
    training_method= lambda f, svi, training_function_input: lambda svi, training_function_input: f(svi, training_function_input)

    print(training_method.__code__.co_varnames)
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
