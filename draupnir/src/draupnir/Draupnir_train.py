
def index_generator(indexes):
    """Call method to subsample in order.
    model (args, iter_num = index_generator())
    plate(..., next(iter_num))"""
    i=0
    while True:
        yield indexes[i]
        i = (i + 1) % len(indexes)
        return i ##?
def train_batch(svi,patristic_matrix,cladistic_matrix,train_loader,args):
    "Regular batch training without shuffleing dats"
    train_loss = 0.0
    seq_lens = []
    for batch_number, dataset in enumerate(train_loader):
        for batch_name, batch_dataset, batch_patristic, batch_blosum in zip(dataset["batch_name"],
                                                                            dataset["batch_data"],
                                                                            dataset["batch_patristic"],
                                                                            dataset["batch_blosum_weighted"]):
            if args.use_cuda:
                batch_dataset = batch_dataset.cuda()
                batch_blosum = batch_blosum.cuda()
                batch_patristic = batch_patristic.cuda()  # cannot be used like this, we cannot have a variable size latent space
            seq_lens += batch_dataset[:, 0, 0].tolist()
            train_loss += svi.step(batch_dataset, batch_patristic, cladistic_matrix, batch_blosum)
            # Normalize loss
            # torch.cuda.reset_max_memory_allocated() #necessary?
    normalizer_train = sum(seq_lens)
    total_epoch_loss_train = train_loss / normalizer_train
    return total_epoch_loss_train


def train(svi,patristic_matrix,cladistic_matrix,train_loader,args):
    "Non batched training"

    train_loss = 0.0
    seq_lens = []
    for batch_number, dataset in enumerate(train_loader):
            seq_lens += dataset[:,0,0].tolist()
            train_loss += svi.step(dataset,patristic_matrix,cladistic_matrix,None) #None is the clade blosum, it's None because here we do not do clade batching
    # Normalize loss
    normalizer_train = sum(seq_lens)
    total_epoch_loss_train = train_loss #/ normalizer_train
    return total_epoch_loss_train
def train_plating(svi,patristic_matrix,cladistic_matrix,train_loader,args):
    "Plated/subsample training"
    #TODO: make the index_generator() work

    train_loss = 0.0
    seq_lens = []
    for batch_number, dataset in enumerate(train_loader):
            seq_lens += dataset[:,0,0].tolist()
            train_loss += svi.step(dataset,patristic_matrix,cladistic_matrix,index_generator(dataset[:,0,1].tolist())) #None is the clade blosum, it's None because here we do not do clade batching
    # Normalize loss
    normalizer_train = sum(seq_lens)
    total_epoch_loss_train = train_loss #/ normalizer_train
    return total_epoch_loss_train
def train_batch_clade(svi,patristic_matrix,cladistic_matrix,train_loader,args):
    "Batch by clade training"
    train_loss = 0.0
    seq_lens = []
    for batch_number, dataset in enumerate(train_loader):
        for clade_name,clade_dataset,clade_patristic,clade_blosum in zip(dataset["clade_name"], dataset["clade_data"],dataset["clade_patristic"],dataset["clade_blosum"]):
            if args.use_cuda:
                clade_dataset = clade_dataset.cuda()
                clade_blosum = clade_blosum.cuda()
                clade_patristic = clade_patristic.cuda() #cannot be used like this, we cannot have a variable size latent space
            seq_lens += clade_dataset[:,0,0].tolist()
            train_loss += svi.step(clade_dataset, patristic_matrix,cladistic_matrix,clade_blosum)
            # Normalize loss
            #torch.cuda.reset_max_memory_allocated() #necessary?
    normalizer_train = sum(seq_lens)
    total_epoch_loss_train = train_loss / normalizer_train
    return total_epoch_loss_train

