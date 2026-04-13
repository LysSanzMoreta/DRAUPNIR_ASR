def prediction_batching_preprocessing_ATTEMPT2(self, map_estimates, patristic_matrix_full, patristic_matrix_test,
                                               batch_idx, use_test, use_test2):
    """Correction of a few parameters to be able to carry on with the batched sampling"""
    if use_test or use_test2:  # internal nodes. Only Marginal posterior available when batching
        assert patristic_matrix_full[1:, 1:].shape == (self.n_all, self.n_all)
        # Highlight: Slice out the train sequences and only a batch from the test sequences
        if batch_idx[1] is None:
            self.internal_nodes_batch = patristic_matrix_test[int(batch_idx[0]) + 1:, 0]
        else:
            self.internal_nodes_batch = patristic_matrix_test[int(batch_idx[0]) + 1:int(batch_idx[1]) + 1, 0]
        self.n_internal_batch = len(self.internal_nodes_batch)
        self.leaves_nodes = map_estimates[
            "train_leaves_nodes"] if "train_leaves_nodes" in map_estimates.keys() else self.leaves_nodes
        self.n_leaves = len(self.leaves_nodes)
        nodes_batch = torch.cat(
            (self.leaves_nodes, self.internal_nodes_batch))  # this needs to contain only the leave nodes on
        self.n_leaves_internal_batch = len(nodes_batch)  # leave nodes + internal nodes
        indexes = (patristic_matrix_full[:, 0][..., None] == nodes_batch).any(-1)
        indexes[0] = True  # re-add the nodes names
        # patristic_matrix = patristic_matrix_full[indexes]
        # patristic_matrix = patristic_matrix[:,indexes]
        # cond_samp_out_dict = self.conditional_sampling_batch(map_estimates,patristic_matrix)
        patristic_matrix_test_batch = patristic_matrix_full[indexes]
        patristic_matrix_test_batch = patristic_matrix_test_batch[:, indexes]
        cond_samp_out_dict = self.conditional_sampling_batch(map_estimates, patristic_matrix_test_batch)
        latent_space = cond_samp_out_dict["latent_space"]

        covariance = cond_samp_out_dict["covariance"]
        if covariance.ndim == 2:
            covariance = covariance[cond_samp_out_dict["internal_idx"]]  # [n_test_batch, n_test_batch+n_train]
            covariance = covariance[:, cond_samp_out_dict["internal_idx"]]  # [n_test_batch, n_test_batch]
        else:
            covariance = covariance[:, cond_samp_out_dict["internal_idx"]]  # [n_test_batch, n_test_batch+n_train]
            covariance = covariance[:, :, cond_samp_out_dict["internal_idx"]]  # [n_test_batch, n_test_batch]

        n_nodes = self.n_internal_batch

    else:  # training/leaves

        n_nodes = self.n_leaves_batch  # here n_leaves has been overloaded by the batch size
        # latent_space = map_estimates["latent_z"].T #the map estimates have been pre-concatenated, that is why we have to index them out
        # latent_space = latent_space[int(batch_idx[0]):int(batch_idx[1])] if batch_idx is not None else latent_space
        indexes = (patristic_matrix_full[:, 0][..., None] == self.leaves_nodes).any(-1)
        indexes[0] = True
        patristic_matrix_train = patristic_matrix_full[indexes]
        patristic_matrix_train = patristic_matrix_train[:, indexes]
        self.leaves_nodes = patristic_matrix_train[
            1:, 0]  # just to make sure we have exactly the same order, i overwrite these

        batch_leaves_nodes = self.leaves_nodes[int(batch_idx[0]):int(batch_idx[
                                                                         1])]  # we only care about the amount of leaves on the row axis (should be the same on the col axis)
        self.leaves_nodes_batch = batch_leaves_nodes
        self.n_leaves_batch = len(batch_leaves_nodes)

        indexes = (patristic_matrix_train[:, 0][..., None] == self.leaves_nodes_batch).any(-1)
        indexes[0] = True
        patristic_matrix_train_batch = patristic_matrix_train[indexes]
        patristic_matrix_train_batch = patristic_matrix_train_batch[:, indexes]

        out_dict = self.gp_prior_batched(patristic_matrix_train_batch, map_estimates)
        # if batch_idx is not None:  # if it is None then the shape should be correct already (for the test_batched_train_batched approach)
        #     if self.covariance.ndim == 2:
        #         covariance = self.covariance[batch_idx[0]:batch_idx[1]] if batch_idx[1] is not None else self.covariance[batch_idx[0]:] # this should be in the same order as the predicted dataset (we override the self.covariance when we predict)
        #     else:
        #         print("BEFORE !prediction batching preprocessing ", self.covariance.shape)
        #
        #         print(batch_idx[0],batch_idx[1])
        #
        #         covariance = self.covariance[:,batch_idx[0]:batch_idx[1]] if batch_idx[1] is not None else self.covariance[:,batch_idx[0]:batch_idx[1]]
        #
        #         print("AFTER !prediction batching preprocessing ", covariance.shape)
        # else:
        covariance = out_dict["covariance"]
        latent_space = out_dict["latent_space"]

        print("batch idx", batch_idx)

        print("covariance", covariance.shape)

        print("latent space", latent_space.shape)

        assert latent_space.shape == (n_nodes, self.z_dim)

    return {"latent_space": latent_space, "n_nodes": n_nodes, "covariance": covariance}


def prediction_batching_preprocessing_ATTEMPT1(self, map_estimates, patristic_matrix_full, patristic_matrix_test,
                                               batch_idx, use_test, use_test2):
    """Correction of a few parameters to be able to carry on with the batched sampling"""
    if use_test or use_test2:  # internal nodes. Only Marginal posterior available when batching
        assert patristic_matrix_full[1:, 1:].shape == (self.n_all, self.n_all)
        # Highlight: Slice out the train sequences and only a batch from the test sequences
        if batch_idx[1] is None:
            self.internal_nodes_batch = patristic_matrix_test[int(batch_idx[0]) + 1:, 0]

        else:
            self.internal_nodes_batch = patristic_matrix_test[int(batch_idx[0]) + 1:int(batch_idx[1]) + 1, 0]
        self.n_internal_batch = len(self.internal_nodes_batch)
        self.leaves_nodes = map_estimates[
            "train_leaves_nodes"] if "train_leaves_nodes" in map_estimates.keys() else self.leaves_nodes
        self.n_leaves = len(self.leaves_nodes)
        nodes_batch = torch.cat(
            (self.leaves_nodes, self.internal_nodes_batch))  # this needs to contain only the leave nodes on
        self.n_leaves_internal_batch = len(nodes_batch)  # leave nodes + internal nodes
        indexes = (patristic_matrix_full[:, 0][..., None] == nodes_batch).any(-1)
        indexes[0] = True  # re-add the nodes names
        # patristic_matrix = patristic_matrix_full[indexes]
        # patristic_matrix = patristic_matrix[:,indexes]
        # cond_samp_out_dict = self.conditional_sampling_batch(map_estimates,patristic_matrix)
        patristic_matrix_test_batch = patristic_matrix_full[indexes]
        patristic_matrix_test_batch = patristic_matrix_test_batch[:, indexes]
        cond_samp_out_dict = self.conditional_sampling_batch(map_estimates, patristic_matrix_test_batch)
        latent_space = cond_samp_out_dict["latent_space"]

        covariance = cond_samp_out_dict["covariance"]
        if covariance.ndim == 2:
            covariance = covariance[cond_samp_out_dict["internal_idx"]]  # [n_test_batch, n_test_batch+n_train]
            covariance = covariance[:, cond_samp_out_dict["internal_idx"]]  # [n_test_batch, n_test_batch]
        else:
            covariance = covariance[:, cond_samp_out_dict["internal_idx"]]  # [n_test_batch, n_test_batch+n_train]
            covariance = covariance[:, :, cond_samp_out_dict["internal_idx"]]  # [n_test_batch, n_test_batch]

        n_nodes = self.n_internal_batch

    else:  # training/leaves

        n_nodes = self.n_leaves_batch  # here n_leaves has been overloaded by the batch size
        indexes = (patristic_matrix_full[:, 0][..., None] == self.leaves_nodes).any(-1)
        indexes[0] = True
        patristic_matrix_train = patristic_matrix_full[indexes]
        patristic_matrix_train = patristic_matrix_train[:, indexes]

        self.leaves_nodes = patristic_matrix_train[
            1:, 0]  # just to make sure we have exactly the same order, i overwrite these
        rows_idx, cols_idx = (batch_idx, batch_idx) if isinstance(batch_idx[0], int) else batch_idx

        # batch_leaves_nodes =  self.leaves_nodes[int(batch_idx[0]):int(batch_idx[1])]
        batch_leaves_nodes = self.leaves_nodes[int(rows_idx[0]):int(rows_idx[
                                                                        1])]  # we only care about the amount of leaves on the row axis (should be the same on the col axis)
        self.leaves_nodes_batch = batch_leaves_nodes

        if sorted(rows_idx) == sorted(cols_idx):
            row_indexes = (patristic_matrix_train[:, 0][..., None] == self.leaves_nodes[rows_idx[0]:rows_idx[1]]).any(
                -1)
            row_indexes[0] = True  # re-add the nodes names
            col_indexes = row_indexes

        else:
            row_indexes = (patristic_matrix_train[:, 0][..., None] == self.leaves_nodes[rows_idx[0]:rows_idx[1]]).any(
                -1)
            row_indexes[0] = True  # re-add the nodes names
            col_indexes = (patristic_matrix_train[:, 0][..., None] == self.leaves_nodes[cols_idx[0]:cols_idx[1]]).any(
                -1)
            col_indexes[0] = True  # re-add the nodes names

        patristic_matrix_train_batch = patristic_matrix_train[row_indexes]
        patristic_matrix_train_batch = patristic_matrix_train_batch[:, col_indexes]

        # this turns into a non symmetric matrix because we build a matrix of n_train_batch_0 vs n_train_batch_1

        patristic_matrix_train_batch[1:, 1:] = patristic_matrix_train_batch[1:, 1:] @ patristic_matrix_train_batch[
            1:, 1:].T

        patristic_matrix_train_batch[1:, 1:] = (patristic_matrix_train_batch[1:, 1:] + patristic_matrix_train_batch[
            1:, 1:].T) / 2

        print(patristic_matrix_train_batch)
        print("patristic matrix train batch", patristic_matrix_train_batch.shape)

        tmp_n_leaves_batch = self.n_leaves_batch
        self.n_leaves_batch = len(self.leaves_nodes_batch)

        out_dict = self.gp_prior_batched(patristic_matrix_train_batch, map_estimates)
        # latent_space = map_estimates["latent_z"].T #the map estimates have been pre-concatenated, that is why we have to index them out
        # latent_space = latent_space[int(batch_idx[0]):int(batch_idx[1])] if batch_idx is not None else latent_space

        latent_space = out_dict["latent_space"]
        covariance = out_dict["covariance"]

        assert latent_space.shape == (n_nodes, self.z_dim)
        self.n_leaves_batch = tmp_n_leaves_batch  # re-store in case

    return {"latent_space": latent_space, "n_nodes": n_nodes, "covariance": covariance}