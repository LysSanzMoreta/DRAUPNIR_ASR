import sys,os
import pytest
import torch
import datetime
import logging
local_repository=True
if local_repository:
    sys.path.insert(1,"/home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src")
    import draupnir
else:#pip installed module
    import draupnir

from test_load import Args
now = datetime.datetime.now()
logger = logging.getLogger(__name__)

@pytest.fixture(params=['simulations_blactamase_1', 'aminopeptidase', 'benchmark_randall_original_naming',"Coral_all"])
def prepare_dataset(request):
    """request is an object required to access the parameters"""
    print(request.param)
    logger.info("Running:\ndataset/{}".format(request.param))
    def _prepare_dataset(request):
        args = Args(name=request.param,
                    use_custom=False,
                    alignment_file=None,
                    tree_file=None,
                    fasta_file=None,
                    build=False,
                    pdb_folder=None,
                    one_hot_encoded=False,
                    num_epochs=5,
                    batch_size=1,
                    select_guide="delta_map",
                    batch_by_clade=False,
                    infer_angles=False,
                    plating=False,
                    plating_size=None,
                    plate_unordered=False,
                    aa_probs=21,
                    n_samples=30,
                    kappa_addition=5,
                    use_blosum=True,
                    subs_matrix="BLOSUM62",
                    generate_samples=False,
                    load_pretrained_path="",
                    embedding_dim=50,
                    use_cuda=True,
                    use_scheduler=False,
                    activate_elbo_convergence=False,
                    activate_entropy_convergence=False,
                    test_frequency=100,
                    config_dict="",
                    parameter_search=False)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        build_config,settings_config, root_sequence_name = draupnir.create_draupnir_dataset(args.name,
                                                                                             args.use_custom,
                                                                                             script_dir,
                                                                                             args,
                                                                                             args.build,
                                                                                             args.fasta_file,args.tree_file,args.alignment_file)
        device = torch.device("cuda" if args.use_cuda else "cpu")
        param_config = draupnir.config_build(args)
        results_dir = "{}/PLOTS_Draupnir_{}_{}_{}epochs_{}".format(script_dir, args.name,
                                                                   now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),
                                                                   args.num_epochs, args.select_guide)

        train_load, test_load, additional_load, build_config = draupnir.load_data(args.name, settings_config, build_config,param_config, results_dir, script_dir, args)
        train_load, test_load, additional_load = draupnir.datasets_pretreatment(args.name, root_sequence_name,
                                                                                         train_load, test_load,
                                                                                         additional_load, build_config,
                                                                                         device, settings_config,
                                                                                         script_dir)
        n_samples = 3
        dataset_test = test_load.dataset_test
        dataset_train = train_load.dataset_train
        if not additional_load.correspondence_dict:
            correspondence_dict = dict(zip(list(range(len(additional_load.tree_levelorder_names))), additional_load.tree_levelorder_names))
        else:
            correspondence_dict = additional_load.correspondence_dict
        def cal_pid(dataset,n_samples):
            aa_sequences_predictions = dataset[:, 2:, 0].repeat(n_samples, 1, 1)
            node_info = dataset[:, 0, 1].repeat(n_samples).unsqueeze(-1).reshape(n_samples, len(dataset),1)
            len_info = dataset[:, 0, 0].repeat(n_samples).unsqueeze(-1).reshape(n_samples, len(dataset), 1)
            distance_info = dataset[:, 0, 2].repeat(n_samples).unsqueeze(-1).reshape(n_samples, len(dataset),1)
            aa_sequences_predictions = torch.cat((len_info, node_info, distance_info, aa_sequences_predictions), dim=2)
            node_names = ["{}//{}".format(correspondence_dict[index], index) for index in dataset[:, 0, 1].tolist()]
            percent_id_df, incorrectly_predicted_sites_df, alignment_length = draupnir.percent_id_sampled_observed(dataset,
                                                                                                          aa_sequences_predictions,
                                                                                                          node_info,
                                                                                                          n_samples,
                                                                                                          node_names,
                                                                                                          results_dir)
            return percent_id_df
        percent_id_df_test = cal_pid(dataset_test,n_samples)
        percent_id_df_train = cal_pid(dataset_train,n_samples)
        print("Done calculating accuracy")
        return percent_id_df_test, percent_id_df_train

    return _prepare_dataset(request)

def test_accuracy(prepare_dataset):
    """Tests that the %ID comparison technique between the samples and the real dataset is correct. """
    percent_id_df_test, percent_id_df_train = prepare_dataset
    print(".....................................")
    assert percent_id_df_test["Average"].mean() == 100.
    assert percent_id_df_train["Average"].mean() == 100.
    print("Done testing accuracy")

if __name__ == '__main__':
    pytest.main(["-s","test_accuracy.py"]) #-s is to run the statements after yield?