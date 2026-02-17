

for batch_number, dataset in enumerate (train_loader):

    batch_dataset = dataset["batch_data_int"].squeeze(0).to('cuda', non_blocking=True)




internal_indexes = (matrix[1:, 0][..., None] == internal_nodes_batch).any(-1)

other_matrix = other_matrix[internal_indexes]