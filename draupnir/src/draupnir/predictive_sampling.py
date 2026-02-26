import torch
from collections import defaultdict, namedtuple
import draupnir.models_utils as DraupnirModelsUtils
import dill
import warnings
import pickle
import gc


SamplingOutput = namedtuple("SamplingOutput",["aa_sequences","latent_space","logits","phis","psis","mean_phi","mean_psi","kappa_phi","kappa_psi","covariance"])

def predictive_test_full_train_full_delta_map(args,
                                    train_load,
                                    test_load,
                                    additional_load,
                                    params_config,
                                    build_config,
                                    guide, Draupnir,
                                    datasets_train,
                                    datasets_test,
                                    dataset_train_blosum,
                                    dataset_test_blosum,
                                    blocks_train=None):
    """Direct for the non-amortised model, no batching involved.
    """

    samples_names = ["sample_{}".format(i) for i in range(args.n_samples)]
    print("Loading map estimates")
    map_estimates_dict = defaultdict()
    # Highlight: Train storage
    # aa_sequences_train_samples = torch.zeros((n_samples, dataset_train.shape[0], dataset_train.shape[1] - 2)).detach()
    # latent_space_train_samples = torch.zeros((n_samples, dataset_train.shape[0], int(config["z_dim"]))).detach()
    # logits_train_samples = torch.zeros((n_samples, dataset_train.shape[0], dataset_train.shape[1] - 2, build_config.aa_probs)).detach()

    with torch.no_grad():
        map_estimates = guide(datasets_train, train_load.patristic_matrix_train,
                              train_load.patristic_matrix_train, dataset_train_blosum,
                              batch_blosum=None,
                              map_estimates=None)

        map_estimates_dict["sample_0"] = map_estimates

        # Highlight: Test storage: Marginal

        n_seq_test, max_len = test_load.patristic_matrix_test[1:].shape[0], train_load.dataset_train.shape[1] - 2
        # aa_sequences_test_samples = torch.zeros((args.n_samples, test_load.patristic_matrix_test[1:].shape[0], train_load.dataset_train.shape[1] - 2)).detach().cpu()
        # latent_space_test_samples = torch.zeros((args.n_samples, test_load.patristic_matrix_test[1:].shape[0], int(params_config["z_dim"]))).detach().cpu()
        # logits_test_samples = torch.zeros((args.n_samples, test_load.patristic_matrix_test[1:].shape[0], train_load.dataset_train.shape[1] - 2, build_config.aa_probs)).detach().cpu()

        aa_sequences_test_samples = torch.zeros((args.n_samples, n_seq_test, max_len)).detach().cpu()
        latent_space_test_samples = torch.zeros((args.n_samples, n_seq_test, int(params_config["z_dim"]))).detach().cpu()
        logits_test_samples = torch.zeros((args.n_samples, n_seq_test, max_len, build_config.aa_probs)).detach().cpu()

        if args.prior_experiment in ["3", "4", "5"] and args.draupnir_version in ["1bB","1nbA"]:
            covariance_test_samples = torch.zeros((args.n_samples, n_seq_test, n_seq_test))
        else:
            covariance_test_samples = torch.zeros((args.n_samples, args.z_dim, n_seq_test, n_seq_test))


        for sample_idx, sample in enumerate(samples_names):
            print(f"## Sample {sample_idx} ###")
            # Highlight: Sample one test sequence (from Marginal)
            test_sample = Draupnir.sample(map_estimates,
                                          1,
                                          test_load.dataset_test,
                                          additional_load.patristic_matrix_full,
                                          test_load.patristic_matrix_test,
                                          use_argmax=False,
                                          use_test=True,
                                          use_test2=False)
            aa_sequences_test_samples[sample_idx] = test_sample.aa_sequences.detach()
            latent_space_test_samples[sample_idx] = test_sample.latent_space.detach()
            logits_test_samples[sample_idx] = test_sample.logits.detach()
            covariance_test_samples[sample_idx] = test_sample.covariance.detach()
            del test_sample

    sample_out_train = Draupnir.sample(map_estimates,
                                       args.n_samples,
                                       train_load.dataset_train,
                                       additional_load.patristic_matrix_full,
                                       train_load.patristic_matrix_train,
                                       use_argmax=False,# <----ATTENTION, not using most likely sequence, cause not using conditional sampling
                                       use_test=False,
                                       use_test2=False)

    sample_out_train = SamplingOutput(aa_sequences=sample_out_train.aa_sequences.detach().cpu(),
                                      latent_space=sample_out_train.latent_space.detach().cpu(),
                                      logits = sample_out_train.logits.detach().cpu(),
                                      phis=None,
                                      psis=None,
                                      mean_phi=None,
                                      mean_psi=None,
                                      kappa_phi=None,
                                      kappa_psi=None,
                                      covariance= sample_out_train.covariance.detach().cpu()
                                      )

    sample_out_test = SamplingOutput(aa_sequences=aa_sequences_test_samples.detach().cpu(),
                                     latent_space=latent_space_test_samples.detach().cpu(),
                                     logits=logits_test_samples.detach().cpu(),
                                     phis=None,
                                     psis=None,
                                     mean_phi=None,
                                     mean_psi=None,
                                     kappa_phi=None,
                                     kappa_psi=None,
                                     covariance= covariance_test_samples.detach().cpu()
                                     )
    # Highlight: Sample MAP sequences
    sample_out_test2 = Draupnir.sample(map_estimates,
                                       args.n_samples,
                                       test_load.dataset_test,
                                       additional_load.patristic_matrix_full,
                                       test_load.patristic_matrix_test,
                                       use_argmax=False,
                                       use_test=False,
                                       use_test2=True)

    sample_out_test2 = SamplingOutput(aa_sequences= sample_out_test2.aa_sequences.detach().cpu(),
                                     latent_space= sample_out_test2.latent_space.detach().cpu(),
                                     logits= sample_out_test2.logits.detach().cpu(),
                                     phis=None,
                                     psis=None,
                                     mean_phi=None,
                                     mean_psi=None,
                                     kappa_phi=None,
                                     kappa_psi=None,
                                     covariance= sample_out_test2.covariance.detach().cpu()
                                     )

    # Highlight: compute majority vote for "most likely sequence"
    sample_out_train_argmax = SamplingOutput(
        aa_sequences=torch.mode(sample_out_train.aa_sequences, dim=0)[0].unsqueeze(0).detach().cpu(),
        latent_space=sample_out_train.latent_space.detach().cpu(),
        logits=sample_out_train.logits.detach().cpu(),
        phis=None,
        psis=None,
        mean_phi=None,
        mean_psi=None,
        kappa_phi=None,
        kappa_psi=None,
        covariance= sample_out_train.covariance[0].detach().cpu().unsqueeze(0) #unsqueeze to simulate 1 sample
    )
    # Highlight: compute majority vote for "most likely sequence"
    sample_out_test_argmax = SamplingOutput(
        aa_sequences=torch.mode(sample_out_test.aa_sequences, dim=0)[0].unsqueeze(0).detach().cpu(),
        latent_space=sample_out_test.latent_space[0].detach().cpu(),
        logits=sample_out_test.logits[0].detach().cpu(),
        phis=None,
        psis=None,
        mean_phi=None,
        mean_psi=None,
        kappa_phi=None,
        kappa_psi=None,
        covariance=sample_out_test.covariance[0].detach().cpu().unsqueeze(0) #unsqueeze to simulate 1 sample
    )

    # Highlight = Sample MAP sequences
    sample_out_test_argmax2 = Draupnir.sample(map_estimates,
                                              args.n_samples,
                                              test_load.dataset_test,
                                              additional_load.patristic_matrix_full,
                                              test_load.patristic_matrix_test,
                                              use_argmax=True,  # Attention!
                                              use_test2=True,
                                              use_test=False)


    sample_out_test_argmax2 = SamplingOutput(aa_sequences= sample_out_test_argmax2.aa_sequences.detach().cpu(),
                                     latent_space= sample_out_test_argmax2.latent_space.detach().cpu(),
                                     logits= sample_out_test_argmax2.logits.detach().cpu(),
                                     phis=None,
                                     psis=None,
                                     mean_phi=None,
                                     mean_psi=None,
                                     kappa_phi=None,
                                     kappa_psi=None,
                                     covariance= sample_out_test_argmax2.covariance[0].detach().cpu().unsqueeze(0) #unsqueeze to simulate 1 sample
                                     )

    # # Highlight: Compute sequences Shannon entropies per site
    train_entropies, train_probs = DraupnirModelsUtils.compute_sites_entropies(sample_out_train_argmax.logits.cpu(),train_load.dataset_train.cpu().long()[:, 0, 1])
    test_entropies, test_probs = DraupnirModelsUtils.compute_sites_entropies(sample_out_test_argmax.logits.cpu(),test_load.patristic_matrix_test.cpu().long()[1:, 0])
    test_entropies2, test_probs2 = DraupnirModelsUtils.compute_sites_entropies(sample_out_test_argmax2.logits.cpu(),test_load.patristic_matrix_test.cpu().long()[1:, 0])

    return (sample_out_train,
            sample_out_train_argmax,
            sample_out_test,
            sample_out_test_argmax,
            sample_out_test2,
            sample_out_test_argmax2,
            train_entropies,
            test_entropies,
            test_entropies2,
            map_estimates_dict)

def predictive_test_full_train_full_variational(args,
                                    train_load,
                                    test_load,
                                    additional_load,
                                    params_config,
                                    build_config,
                                    guide, Draupnir,
                                    datasets_train,
                                    datasets_test,
                                    dataset_train_blosum,
                                    dataset_test_blosum,
                                    blocks_train=None):
    map_estimates_dict = defaultdict()
    samples_names = ["sample_{}".format(i) for i in range(args.n_samples)]
    # Highlight: Train storage
    n_seq_train, max_len = train_load.dataset_train.shape[0], train_load.dataset_train.shape[1] - 2
    aa_sequences_train_samples = torch.zeros((args.n_samples, n_seq_train, max_len)).detach().cpu()
    latent_space_train_samples = torch.zeros((args.n_samples, n_seq_train, int(params_config["z_dim"]))).detach().cpu()
    logits_train_samples = torch.zeros((args.n_samples, n_seq_train, max_len, build_config.aa_probs)).detach().cpu()

    # Highlight: Test storage
    n_seq_test = test_load.patristic_matrix_test[1:].shape[0]
    aa_sequences_test_samples = torch.zeros((args.n_samples, n_seq_test, max_len)).detach().cpu()
    latent_space_test_samples = torch.zeros((args.n_samples, n_seq_test, int(params_config["z_dim"]))).detach().cpu()
    logits_test_samples = torch.zeros((args.n_samples, n_seq_test, max_len, build_config.aa_probs)).detach().cpu()

    if args.prior_experiment in ["3", "4", "5"] and args.draupnir_version in ["1bB","1nbA"]:
        covariance_train_samples = torch.zeros((args.n_samples,n_seq_train,n_seq_train)).detach().cpu()
        covariance_test_samples = torch.zeros((args.n_samples,n_seq_test,n_seq_test)).detach().cpu()
    else:
        covariance_train_samples = torch.zeros((args.n_samples,args.z_dim, n_seq_train, n_seq_train)).detach().cpu()
        covariance_test_samples = torch.zeros((args.n_samples,args.z_dim, n_seq_test, n_seq_test)).detach().cpu()

    with torch.no_grad():
        for sample_idx, sample in enumerate(samples_names):
            print(f" ##  Sample {sample_idx} ##")
            map_estimates = guide(datasets_train,
                                  train_load.patristic_matrix_train,
                                  train_load.patristic_matrix_train,
                                  dataset_train_blosum,
                                  batch_blosum=None,
                                  map_estimates=None)  # only saving 1 sample
            map_estimates_dict[sample] = {val: key.detach() for val, key in map_estimates.items()}
            # Highlight: Sample one train sequence
            train_sample = Draupnir.sample(map_estimates,
                                           1,
                                           train_load.dataset_train,
                                           additional_load.patristic_matrix_full,
                                           train_load.patristic_matrix_train,
                                           use_argmax=False,
                                           use_test=False,
                                           use_test2=False)

            aa_sequences_train_samples[sample_idx] = train_sample.aa_sequences.detach().cpu()
            latent_space_train_samples[sample_idx] = train_sample.latent_space.detach().cpu()
            logits_train_samples[sample_idx] = train_sample.logits.detach().cpu()
            covariance_train_samples[sample_idx] = train_sample.covariance.detach().cpu()

            del train_sample
            # Highlight: Sample one test sequence
            test_sample = Draupnir.sample(map_estimates,
                                          1,
                                          test_load.dataset_test,
                                          additional_load.patristic_matrix_full,
                                          test_load.patristic_matrix_test,
                                          use_argmax=False,
                                          use_test=True,
                                          use_test2=False)
            aa_sequences_test_samples[sample_idx] = test_sample.aa_sequences.detach().cpu()
            latent_space_test_samples[sample_idx] = test_sample.latent_space.detach().cpu()
            logits_test_samples[sample_idx] = test_sample.logits.detach().cpu()
            covariance_test_samples[sample_idx] = test_sample.covariance.detach().cpu()

            del test_sample
            del map_estimates
            torch.cuda.empty_cache()

        dill.dump(map_estimates_dict, open('{}/Draupnir_Checkpoints/Map_estimates.p'.format(args.results_dir), 'wb'))
        sample_out_train = SamplingOutput(aa_sequences=aa_sequences_train_samples,
                                          latent_space=latent_space_train_samples,
                                          logits=logits_train_samples,
                                          phis=None,
                                          psis=None,
                                          mean_phi=None,
                                          mean_psi=None,
                                          kappa_phi=None,
                                          kappa_psi=None,
                                          covariance= covariance_train_samples
                                          )
        sample_out_test = SamplingOutput(aa_sequences=aa_sequences_test_samples,
                                         latent_space=latent_space_test_samples,
                                         logits=logits_test_samples,
                                         phis=None,
                                         psis=None,
                                         mean_phi=None,
                                         mean_psi=None,
                                         kappa_phi=None,
                                         kappa_psi=None,
                                         covariance = covariance_test_samples
                                         )
        sample_out_test2 = sample_out_test
        # Highlight: compute majority vote/ Argmax
        sample_out_train_argmax = SamplingOutput(
            aa_sequences=torch.mode(sample_out_train.aa_sequences, dim=0)[0].unsqueeze(0).cpu(),  # I think is correct
            latent_space=sample_out_train.latent_space[0].cpu(),  # TODO:Average?
            logits=sample_out_train.logits[0].cpu(),
            phis=None,
            psis=None,
            mean_phi=None,
            mean_psi=None,
            kappa_phi=None,
            kappa_psi=None,
            covariance = sample_out_train.covariance[0].cpu().unsqueeze(0) #unsqueeze to simulate 1 sample
        )
        sample_out_test_argmax = SamplingOutput(
            aa_sequences=torch.mode(sample_out_test.aa_sequences, dim=0)[0].unsqueeze(0).cpu(),
            latent_space=sample_out_test.latent_space[0].cpu(),
            logits=sample_out_test.logits[0].cpu(),
            phis=None,
            psis=None,
            mean_phi=None,
            mean_psi=None,
            kappa_phi=None,
            kappa_psi=None,
            covariance=sample_out_test.covariance[0].cpu().unsqueeze(0) #unsqueeze to simulate 1 sample
        )
        sample_out_test_argmax2 = sample_out_test_argmax
        # # Highlight: Compute sequences Shannon entropies per site
        train_entropies, train_probs = DraupnirModelsUtils.compute_sites_entropies(sample_out_train_argmax.logits.cpu(),train_load.dataset_train.cpu().long()[:, 0, 1])
        test_entropies, test_probs = DraupnirModelsUtils.compute_sites_entropies(sample_out_test_argmax.logits.cpu(),test_load.patristic_matrix_test.cpu().long()[1:, 0])
        test_entropies2, test_probs2 = DraupnirModelsUtils.compute_sites_entropies(sample_out_test_argmax.logits.cpu(),test_load.patristic_matrix_test.cpu().long()[1:, 0])

    return (sample_out_train,
            sample_out_train_argmax,
            sample_out_test,
            sample_out_test_argmax,
            sample_out_test2,
            sample_out_test_argmax2,
            train_entropies,
            test_entropies,
            test_entropies2,
            map_estimates_dict)


def predictive_test_batched_train_full(args,
                   train_load,
                   test_load,
                   additional_load,
                   params_config,
                   build_config,
                   guide,Draupnir,
                   datasets_train,
                   datasets_test,
                   dataset_train_blosum,
                   dataset_test_blosum,
                   blocks_train):

    """Simple batched approach for the batched variational model, we calculate the latent variables for all the train/leaves sequences and then, each test batch is predicted conditioned on the entire set of train leaves latent estimates.
    The decoding of the train sequences from Z -> Sequence also takes place in a batch setting. Only the guide output is estimated with the entire set of leaf nodes.
    """

    print("Variational approach: Re-sampling from the guide")
    # map_estimates_dict = dill.load(open('{}/Draupnir_Checkpoints/Map_estimates.p'.format(args.load_pretrained_path), "rb"))
    map_estimates_dict = defaultdict()
    samples_names = ["sample_{}".format(i) for i in range(args.n_samples)]
    # Highlight: Train storage
    n_seq_train, max_len = train_load.dataset_train.shape[0], train_load.dataset_train.shape[1] - 2
    aa_sequences_train_samples = torch.zeros((args.n_samples, n_seq_train, max_len)).detach().cpu()
    latent_space_train_samples = torch.zeros((args.n_samples, n_seq_train, int(params_config["z_dim"]))).detach().cpu()
    logits_train_samples = torch.zeros((args.n_samples, n_seq_train, max_len,build_config.aa_probs)).detach().cpu()
    # Highlight: Test storage
    n_seq_test = test_load.patristic_matrix_test[1:].shape[0]
    aa_sequences_test_samples = torch.zeros((args.n_samples, n_seq_test, max_len)).detach().cpu()
    latent_space_test_samples = torch.zeros((args.n_samples, n_seq_test, int(params_config["z_dim"]))).detach().cpu()
    logits_test_samples = torch.zeros((args.n_samples, n_seq_test, max_len,build_config.aa_probs)).detach().cpu()

    if args.prior_experiment in ["3", "4", "5"] and args.draupnir_version in ["1bB","1nbA"]: #TODO: right now we are saving over and over again the covariance matrix from the last batch, needs to be fixed
        covariance_train_samples = torch.zeros((args.n_samples,n_seq_train,n_seq_train)).detach().cpu()
        covariance_test_samples = torch.zeros((args.n_samples,n_seq_test,n_seq_test)).detach().cpu()
    else:
        covariance_train_samples = torch.zeros((args.n_samples,args.z_dim, n_seq_train, n_seq_train)).detach().cpu()
        covariance_test_samples = torch.zeros((args.n_samples,args.z_dim, n_seq_test, n_seq_test)).detach().cpu()

    assert blocks_train is not None, "this is batched sampling there should always be batched indexes"
    with torch.no_grad():
        #if blocks_train is not None:  # batched sampling
            blocks_test = blocks_train.copy()
            blocks_test[-1] = (blocks_test[-1][0],None)  # correcting the indexes of the test, this trick works by re-using blocks train, but this approach is more flexible
            for sample_idx, sample in enumerate(samples_names):
                #print("Recalculating train map estimates")
                map_estimates = guide(datasets_train, train_load.patristic_matrix_train,
                                      train_load.patristic_matrix_train, dataset_train_blosum,
                                      batch_blosum=None,
                                      map_estimates=None)  # todo: ideally we we would load the pre.learnt map estimates, i need to make sure the right ones are saved

                map_estimates = {val: key.detach() for val, key in map_estimates.items() if key is not None}
                print("sample idx {}".format(sample_idx))
                if args.draupnir_version in ["1bB","1nbA","2", "4","5"]:
                    map_estimates_test = guide(datasets_test, test_load.patristic_matrix_test,test_load.patristic_matrix_test, dataset_test_blosum,batch_blosum=None)  # i extracted the "test" estimates here for some experiment
                    map_estimates["test"] = {val: key.detach() for val, key in map_estimates_test.items() if key is not None}

                map_estimates_dict[sample] = map_estimates

                for batch_idx, batch_idx_test in zip(blocks_train, blocks_test):
                    batch_train_sample = Draupnir.sample_batched(map_estimates,
                                                                 1,
                                                                 train_load.dataset_train,
                                                                 additional_load.patristic_matrix_full,
                                                                 train_load.patristic_matrix_train,
                                                                 # substitute with something else
                                                                 batch_idx=batch_idx,
                                                                 use_argmax=False,
                                                                 use_test=False,
                                                                 use_test2=False)
                    aa_sequences_train_samples[sample_idx, int(batch_idx[0]):int(batch_idx[1])] = batch_train_sample.aa_sequences.detach().cpu()
                    latent_space_train_samples[sample_idx, int(batch_idx[0]):int(batch_idx[1])] = batch_train_sample.latent_space.detach().cpu()
                    logits_train_samples[sample_idx, int(batch_idx[0]):int(batch_idx[1])] = batch_train_sample.logits.detach().cpu()

                    #TODO: why this does not work when loading from pre-trained model


                    if covariance_train_samples.ndim == 4:
                        covariance_train_samples[sample_idx, :, int(batch_idx[0]):int(batch_idx[1]),int(batch_idx[0]):int(batch_idx[1])] = batch_train_sample.covariance.detach().cpu()  # we only subset once, because it is [n_train_batch,n_train]
                    else:
                        covariance_train_samples[sample_idx, int(batch_idx[0]):int(batch_idx[1])] = batch_train_sample.covariance.detach().cpu() #we only subset once, because it is [n_train_batch,n_train]

                    test_sample = Draupnir.sample_batched(map_estimates,
                                                          1,
                                                          test_load.dataset_test,
                                                          additional_load.patristic_matrix_full,
                                                          test_load.patristic_matrix_test,
                                                          batch_idx=batch_idx_test,
                                                          use_argmax=False,
                                                          use_test=True,
                                                          use_test2=False)

                    if batch_idx[1] is None:  # last batch
                        aa_sequences_test_samples[sample_idx, int(batch_idx[0]):] = test_sample.aa_sequences.detach().cpu()
                        latent_space_test_samples[sample_idx, int(batch_idx[0]):] = test_sample.latent_space.detach().cpu()
                        logits_test_samples[sample_idx, int(batch_idx[0]):] = test_sample.logits.detach().cpu()
                        if covariance_test_samples.ndim == 4:
                            covariance_test_samples[sample_idx,:, int(batch_idx[0]):, int(batch_idx[0]):] = test_sample.covariance.detach().cpu()  # we subset twice because we are batching the test sequences
                        else:
                            covariance_test_samples[sample_idx, int(batch_idx[0]):, int(batch_idx[0]):] = test_sample.covariance.detach().cpu() #we subset twice because we are batching the test sequences

                    else:
                        aa_sequences_test_samples[sample_idx, int(batch_idx[0]):int(batch_idx[1])] = test_sample.aa_sequences.detach().cpu()
                        latent_space_test_samples[sample_idx, int(batch_idx[0]):int(batch_idx[1])] = test_sample.latent_space.detach().cpu()
                        logits_test_samples[sample_idx, int(batch_idx[0]):int(batch_idx[1])] = test_sample.logits.detach().cpu()
                        if covariance_test_samples.ndim == 4:
                            covariance_test_samples[sample_idx, :, int(batch_idx[0]):int(batch_idx[1]), int(batch_idx[0]):int(batch_idx[1])] = test_sample.covariance.detach().cpu()
                        else:
                            covariance_test_samples[sample_idx, int(batch_idx[0]):int(batch_idx[1]),int(batch_idx[0]):int(batch_idx[1])] = test_sample.covariance.detach().cpu()

                    del test_sample,batch_train_sample
                torch.cuda.empty_cache()

    dill.dump(map_estimates_dict, open('{}/Draupnir_Checkpoints/Map_estimates.p'.format(args.results_dir), 'wb'))
    sample_out_train = SamplingOutput(aa_sequences=aa_sequences_train_samples.detach().cpu(),
                                      latent_space=latent_space_train_samples.detach().cpu(),
                                      logits=logits_train_samples.detach().cpu(),
                                      phis=None,
                                      psis=None,
                                      mean_phi=None,
                                      mean_psi=None,
                                      kappa_phi=None,
                                      kappa_psi=None,
                                      covariance= covariance_train_samples.detach().cpu()
                                      )
    sample_out_test = SamplingOutput(aa_sequences=aa_sequences_test_samples.detach().cpu(),
                                     latent_space=latent_space_test_samples.detach().cpu(),
                                     logits=logits_test_samples,
                                     phis=None,
                                     psis=None,
                                     mean_phi=None,
                                     mean_psi=None,
                                     kappa_phi=None,
                                     kappa_psi=None,
                                     covariance=covariance_test_samples.detach().cpu()
                                     )
    warnings.warn("In variational method Test folder results = Test2 folder results")
    sample_out_test2 = sample_out_test
    # Highlight: compute majority vote
    sample_out_train_argmax = SamplingOutput(
        aa_sequences=torch.mode(sample_out_train.aa_sequences, dim=0)[0].unsqueeze(0).detach().cpu(),  # I think is correct
        latent_space=sample_out_train.latent_space[0].detach().cpu(),  # TODO:Average?
        logits=sample_out_train.logits[0].detach().cpu(),
        phis=None,
        psis=None,
        mean_phi=None,
        mean_psi=None,
        kappa_phi=None,
        kappa_psi=None,
        covariance=sample_out_train.covariance[0].detach().cpu().unsqueeze(0) #unsqueeze to simulate 1 sample
    )
    sample_out_test_argmax = SamplingOutput(
        aa_sequences=torch.mode(sample_out_test.aa_sequences, dim=0)[0].unsqueeze(0).detach().cpu(),
        latent_space=sample_out_test.latent_space[0].detach().cpu(),
        logits=sample_out_test.logits[0].detach().cpu(),
        phis=None,
        psis=None,
        mean_phi=None,
        mean_psi=None,
        kappa_phi=None,
        kappa_psi=None,
        covariance=sample_out_test.covariance[0].detach().cpu().unsqueeze(0) #unsqueeze to simulate 1 sample
    )
    sample_out_test_argmax2 = sample_out_test_argmax
    # # Highlight: Compute sequences Shannon entropies per site
    train_entropies, train_probs = DraupnirModelsUtils.compute_sites_entropies(sample_out_train_argmax.logits.cpu(),train_load.dataset_train.cpu().long()[:, 0, 1])
    test_entropies, test_probs = DraupnirModelsUtils.compute_sites_entropies(sample_out_test_argmax.logits.cpu(),test_load.patristic_matrix_test.cpu().long()[1:, 0])
    test_entropies2, test_probs2 = DraupnirModelsUtils.compute_sites_entropies(sample_out_test_argmax.logits.cpu(),test_load.patristic_matrix_test.cpu().long()[1:, 0])

    return (sample_out_train,
            sample_out_train_argmax,
            sample_out_test,
            sample_out_test_argmax,
            sample_out_test2,
            sample_out_test_argmax2,
            train_entropies,
            test_entropies,
            test_entropies2,
            map_estimates_dict)

def predictive_test_batched_train_batched(args,
                   train_load,
                   test_load,
                   additional_load,
                   params_config,
                   build_config,
                   guide,Draupnir,
                   datasets_train,
                   datasets_test,
                   dataset_train_blosum,
                   dataset_test_blosum,
                   blocks_train):

    """Simple batched approach for the batched variational model, we calculate the latent variables for all the train/leaves sequences and then, each test batch is predicted conditioned on the entire set of train leaves latent estimates.
    The decoding of the train sequences from Z -> Sequence also takes place in a batch setting. Only the guide output is estimated with the entire set of leaf nodes.
    """

    # map_estimates_dict = defaultdict()
    print("Variational approach: Re-sampling from the guide")
    n_train_batches = len(blocks_train)
    # todo: make work for arbitrary number of samples not multiple of n_train_batches

    args.__dict__["n_samples_batched"] = int(
        args.n_samples / n_train_batches)  # we can always generate all the samples, however, since we will generate n_batches conditionally sampled test sequences, it would be a lot

    print("NUMBER OF SAMPLES", args.n_samples)
    print("NUMBER OF SAMPLES BATCHED", args.n_samples_batched)

    warnings.warn("This batched sampling procedure has not been fully tested, latents seem well predicted though. Finish testing with the appending of the nodes")

    map_estimates_dict = defaultdict()
    samples_names = ["sample_{}".format(i) for i in range(args.n_samples_batched)]
    # Highlight: Train storage
    n_train_leaves, train_dim1 = train_load.dataset_train.shape[0], train_load.dataset_train.shape[1]
    aa_sequences_train_samples = torch.zeros((args.n_samples, n_train_leaves, train_dim1 - 2)).detach().cpu()
    latent_space_train_samples = torch.zeros((args.n_samples, n_train_leaves, int(params_config["z_dim"]))).detach().cpu()
    logits_train_samples = torch.zeros((args.n_samples, n_train_leaves, train_dim1 - 2, build_config.aa_probs)).detach().cpu()

    # train_nodes_storage = torch.zeros((args.n_samples,n_train_leaves,n_train_leaves)).detach().cpu() #todo: finish, to check that the sampling idx are correct

    # Highlight: Test storage
    n_test_internal = test_load.patristic_matrix_test[1:].shape[0]
    aa_sequences_test_samples = torch.zeros((args.n_samples, n_test_internal, train_dim1 - 2)).detach().cpu()
    latent_space_test_samples = torch.zeros((args.n_samples, n_test_internal, int(params_config["z_dim"]))).detach().cpu()
    logits_test_samples = torch.zeros((args.n_samples, n_test_internal, train_dim1 - 2, build_config.aa_probs)).detach().cpu()
    # batch_test = test_load.dataset_test[batch_idx_test[0]:batch_idx_test[1]] if batch_idx_test[1] is not None else test_load.dataset_test[batch_idx_test[0]:]


    if args.prior_experiment in ["3", "4", "5"] and args.draupnir_version in ["1bB","1nbA"]:
        covariance_train_samples = torch.zeros((args.n_samples,n_train_leaves,n_train_leaves)).detach().cpu()
        covariance_test_samples = torch.zeros((args.n_samples,n_test_internal,n_test_internal)).detach().cpu()
    else:
        covariance_train_samples = torch.zeros((args.n_samples,args.z_dim, n_train_leaves, n_train_leaves)).detach().cpu()
        covariance_test_samples = torch.zeros((args.n_samples,args.z_dim, n_test_internal, n_test_internal)).detach().cpu()

    with torch.no_grad():
        if blocks_train is not None:  # batched sampling
            blocks_test = blocks_train.copy()
            blocks_test[-1] = (blocks_test[-1][0],None)  # correcting the indexes of the test, this trick works by re-using blocks train, but this approach is more flexible
            #print("Recalculating train map estimates")
            train_sample_idx = 0
            test_sample_idx = 0
            for sample_idx, sample in enumerate(samples_names):
                print("###### sample idx {} #######".format(sample_idx))
                for batch_idx_test in blocks_test:  # each test batch is conditionally sampled on each train batch, so we generate n_test*n_train_batches(*n_samples)
                    # Highlight: in this second loop, for the train sequences we obtain 1 sample for each train block/batch
                    # Highlight: whereas for the test sequences we have obtained -n_train_batches- number of samples for 1 batch block of the test sequences
                    for train_block_idx, batch_idx_train in enumerate(blocks_train):

                        datasets_train_batch = {key: data[batch_idx_train[0]:batch_idx_train[1]] for key, data in
                                                datasets_train.items()}
                        map_estimates_batch_train = guide(datasets_train_batch,
                                                          train_load.patristic_matrix_train,
                                                          train_load.patristic_matrix_train,
                                                          dataset_train_blosum,
                                                          batch_blosum=None,
                                                          map_estimates=None)

                        map_estimates_batch_train = {val: key.detach() for val, key in map_estimates_batch_train.items() if key is not None}

                        if train_block_idx == 0: #we only store it once
                            map_estimates_dict[sample] = map_estimates_batch_train
                        map_estimates_batch_train["train_leaves_nodes"] = datasets_train["int"][batch_idx_train[0]:batch_idx_train[1]][:, 0, 1]  # todo: if batch_idx_train is out of range thsi needs to be corrected

                        if args.draupnir_version in ["1bB","1nbA","2", "4","5"]:
                            datasets_test_batch = {key: data[batch_idx_train[0]:batch_idx_train[1]] for key, data in
                                                   datasets_test}
                            map_estimates_test_batch = guide(datasets_test_batch, test_load.patristic_matrix_test,
                                                             test_load.patristic_matrix_test, dataset_test_blosum,
                                                             batch_blosum=None)
                            map_estimates_batch_train["test"] = {val: key.detach() for val, key in map_estimates_test_batch.items() if key is not None}

                        batch_train_sample = Draupnir.sample_batched(map_estimates_batch_train,
                                                                     1,
                                                                     train_load.dataset_train[
                                                                         batch_idx_train[0]:batch_idx_train[1]],
                                                                     additional_load.patristic_matrix_full,
                                                                     train_load.patristic_matrix_train,# batch_idx=batch_idx_train, #we do not need to index the train here, because it is already subsampled
                                                                     batch_idx=None,# we do not need to index the train here, because it is already subsampled
                                                                     use_argmax=False,
                                                                     use_test=False,
                                                                     use_test2=False)


                        batch_test_sample = Draupnir.sample_batched(map_estimates_batch_train,
                                                                    1,
                                                                    test_load.dataset_test[batch_idx_test[0]:batch_idx_test[1]] if batch_idx_test[1] is not None else test_load.dataset_test[batch_idx_test[0]:],
                                                                    additional_load.patristic_matrix_full,
                                                                    test_load.patristic_matrix_test,
                                                                    batch_idx=batch_idx_test,# we need the batch idx here (unlike the train) because we provide with the full patristic matrix test which needs to be subsampled
                                                                    use_argmax=False,
                                                                    use_test=True,
                                                                    use_test2=False)


                        # Highlight: for the train sequences we obtain 1 sample for this train batch
                        aa_sequences_train_samples[train_sample_idx, batch_idx_train[0]:batch_idx_train[1]] = batch_train_sample.aa_sequences.detach().cpu()
                        latent_space_train_samples[train_sample_idx, batch_idx_train[0]:batch_idx_train[1]] = batch_train_sample.latent_space.detach().cpu()
                        logits_train_samples[train_sample_idx, batch_idx_train[0]:batch_idx_train[1]] = batch_train_sample.logits.detach().cpu()

                        if covariance_train_samples.ndim == 4:
                            covariance_train_samples[train_sample_idx, :, batch_idx_train[0]:batch_idx_train[1], batch_idx_train[0]:batch_idx_train[1]] = batch_train_sample.covariance.detach().cpu()  # this time the train is also batche
                        else:
                            covariance_train_samples[train_sample_idx,batch_idx_train[0]:batch_idx_train[1],batch_idx_train[0]:batch_idx_train[1]] = batch_train_sample.covariance.detach().cpu() #this time the train is also batched

                        # train_nodes_storage[train_sample_idx,batch_idx_train[0]:batch_idx_train[1]] = map_estimates_batch_train["train_leaves_nodes"][batch_idx_train[0]:batch_idx_train[1]] #todo: finish, to check that the sampling idx are correct

                        if batch_idx_test[1] is None:  # last batch
                            aa_sequences_test_samples[test_sample_idx + train_block_idx, batch_idx_test[0]:] = batch_test_sample.aa_sequences.detach().cpu()
                            latent_space_test_samples[test_sample_idx + train_block_idx, batch_idx_test[0]:] = batch_test_sample.latent_space.detach().cpu()[None, :]
                            logits_test_samples[test_sample_idx + train_block_idx, batch_idx_test[0]:] = batch_test_sample.logits.detach().cpu()[None, :]
                            if covariance_test_samples.ndim == 4:
                                covariance_test_samples[test_sample_idx + train_block_idx,:, batch_idx_test[0]:, batch_idx_test[0]:] = batch_test_sample.covariance.detach().cpu()
                            else:
                                covariance_test_samples[test_sample_idx + train_block_idx, batch_idx_test[0]:, batch_idx_test[0]:] = batch_test_sample.covariance.detach().cpu()

                        else:
                            aa_sequences_test_samples[test_sample_idx + train_block_idx, batch_idx_test[0]:batch_idx_test[1]] = batch_test_sample.aa_sequences.detach().cpu()
                            latent_space_test_samples[test_sample_idx + train_block_idx, batch_idx_test[0]:batch_idx_test[1]] = batch_test_sample.latent_space.detach().cpu()[None, :]
                            logits_test_samples[test_sample_idx + train_block_idx, batch_idx_test[0]:batch_idx_test[1]] = batch_test_sample.logits.detach().cpu()[None, :]
                            if covariance_test_samples.ndim == 4:
                                covariance_test_samples[test_sample_idx + train_block_idx,: ,batch_idx_test[0]:batch_idx_test[1],batch_idx_test[0]:batch_idx_test[1]] = batch_test_sample.covariance.detach().cpu()
                            else:
                                covariance_test_samples[test_sample_idx + train_block_idx, batch_idx_test[0]:batch_idx_test[1],batch_idx_test[0]:batch_idx_test[1]] = batch_test_sample.covariance.detach().cpu()

                        del batch_test_sample, batch_train_sample
                        gc.collect()

                    train_sample_idx += 1

                test_sample_idx += n_train_batches

                torch.cuda.empty_cache()
                gc.collect()

    dill.dump(map_estimates_dict, open('{}/Draupnir_Checkpoints/Map_estimates.p'.format(args.results_dir), 'wb'))
    sample_out_train = SamplingOutput(aa_sequences=aa_sequences_train_samples.detach().cpu(),
                                      latent_space=latent_space_train_samples.detach().cpu(),
                                      logits=logits_train_samples.detach().cpu(),
                                      phis=None,
                                      psis=None,
                                      mean_phi=None,
                                      mean_psi=None,
                                      kappa_phi=None,
                                      kappa_psi=None,
                                      covariance= covariance_train_samples
                                      )
    sample_out_test = SamplingOutput(aa_sequences=aa_sequences_test_samples.detach().cpu(),
                                     latent_space=latent_space_test_samples.detach().cpu(),
                                     logits=logits_test_samples,
                                     phis=None,
                                     psis=None,
                                     mean_phi=None,
                                     mean_psi=None,
                                     kappa_phi=None,
                                     kappa_psi=None,
                                     covariance= covariance_test_samples
                                     )
    warnings.warn("In variational method Test folder results = Test2 folder results")
    sample_out_test2 = sample_out_test
    # Highlight: compute majority vote
    sample_out_train_argmax = SamplingOutput(
        aa_sequences=torch.mode(sample_out_train.aa_sequences, dim=0)[0].unsqueeze(0).detach().cpu(),  # I think is correct
        latent_space=sample_out_train.latent_space[0].detach().cpu(),  # TODO:Average?
        logits=sample_out_train.logits[0].detach().cpu(),
        phis=None,
        psis=None,
        mean_phi=None,
        mean_psi=None,
        kappa_phi=None,
        kappa_psi=None,
        covariance= sample_out_train.covariance[0].unsqueeze(0) #unsqueeze to simulate 1 sample
    )
    sample_out_test_argmax = SamplingOutput(
        aa_sequences=torch.mode(sample_out_test.aa_sequences, dim=0)[0].unsqueeze(0).detach().cpu(),
        latent_space=sample_out_test.latent_space[0].detach().cpu(),
        logits=sample_out_test.logits[0].detach().cpu(),
        phis=None,
        psis=None,
        mean_phi=None,
        mean_psi=None,
        kappa_phi=None,
        kappa_psi=None,
        covariance= sample_out_test.covariance[0].unsqueeze(0) #unsqueeze to simulate 1 sample
    )
    sample_out_test_argmax2 = sample_out_test_argmax
    # # Highlight: Compute sequences Shannon entropies per site
    train_entropies, train_probs = DraupnirModelsUtils.compute_sites_entropies(sample_out_train_argmax.logits.cpu(),
                                                                               train_load.dataset_train.cpu().long()[
                                                                                   :, 0, 1])
    test_entropies, test_probs = DraupnirModelsUtils.compute_sites_entropies(sample_out_test_argmax.logits.cpu(),
                                                                             test_load.patristic_matrix_test.cpu().long()[
                                                                                 1:, 0])
    test_entropies2, test_probs2 = DraupnirModelsUtils.compute_sites_entropies(sample_out_test_argmax.logits.cpu(),
                                                                               test_load.patristic_matrix_test.cpu().long()[
                                                                                   1:, 0])

    return (sample_out_train,
            sample_out_train_argmax,
            sample_out_test,
            sample_out_test_argmax,
            sample_out_test2,
            sample_out_test_argmax2,
            train_entropies,
            test_entropies,
            test_entropies2,
            map_estimates_dict)



