import unittest,sys,os
from argparse import Namespace
import pytest
local_repository=True
if local_repository:
    sys.path.insert(1,"/home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src")
    import draupnir
else:#pip installed module
    import draupnir
sys.path.insert(1,"/home/lys/Dropbox/PhD/DRAUPNIR_ASR")
from Draupnir_example import main as DraupnirMain

test_cases = [
        {"dataset_name": "simulations_src_sh3_3", "use_blosum": True, "batch_size": 1,"draupnir_version": "1","covariance_prior": "og"},
        {"dataset_name": "simulations_src_sh3_3", "use_blosum": True, "batch_size": 50,"draupnir_version": "1","covariance_prior": "og"},
        {"dataset_name": "simulations_src_sh3_3", "use_blosum": False, "batch_size": 50,"draupnir_version": "1","covariance_prior": "og"},
        {"dataset_name": "simulations_src_sh3_3", "use_blosum": False, "batch_size": 1,"draupnir_version": "1","covariance_prior": "og"},
        ##prior 0
        {"dataset_name": "simulations_src_sh3_3", "use_blosum": True, "batch_size": 1,"draupnir_version": "1","covariance_prior": "0"},
        {"dataset_name": "simulations_src_sh3_3", "use_blosum": True, "batch_size": 50,"draupnir_version": "1","covariance_prior": "0"},
        {"dataset_name": "simulations_src_sh3_3", "use_blosum": False, "batch_size": 1,"draupnir_version": "1","covariance_prior": "0"},
        {"dataset_name": "simulations_src_sh3_3", "use_blosum": False, "batch_size": 50,"draupnir_version": "1","covariance_prior": "0"},
        #prior 5
        {"dataset_name": "simulations_src_sh3_3", "use_blosum": True, "batch_size": 1, "draupnir_version": "1","covariance_prior": "5"},
        {"dataset_name": "simulations_src_sh3_3", "use_blosum": True, "batch_size": 50, "draupnir_version": "1", "covariance_prior": "5"},
        {"dataset_name": "simulations_src_sh3_3", "use_blosum": False, "batch_size": 1,"draupnir_version": "1","covariance_prior": "5"},
        {"dataset_name": "simulations_src_sh3_3", "use_blosum": False, "batch_size": 50,"draupnir_version": "1","covariance_prior": "5"},
        # sampling from checkpoint

        # different likelihood
]

@pytest.mark.parametrize("case", test_cases)
def test_start(case):
    """request is an object required to access the parameters"""

    args_dict = dict(dataset_name=case["dataset_name"],
                     output_path="",
                     use_custom=False,
                     num_epochs=3,
                     alignment_file=None,
                     tree_file=None,
                     fasta_file=None,
                     embeddings=None,
                     build_dataset=False,
                     batch_size=case["batch_size"],
                     aa_probs=21,
                     z_dim=30,
                     n_samples=10,
                     prediction_method="test_batched_train_full",
                     use_blosum=case["use_blosum"],
                     subs_matrix="BLOSUM62",
                     embedding_dim=50,
                     use_cuda=True,
                     use_scheduler=False,
                     scheduler_type="reduce_on_plateau",
                     test_frequency=10,
                     select_guide="variational",
                     load_pretrained_path="",
                     generate_samples=False,
                     leaf_embeddings=None,
                     draupnir_version=case["draupnir_version"],
                     covariance_prior=case["covariance_prior"],
                     one_hot_encoded=False,
                     use_align_seq=True,
                     batch_by_clade=False,
                     pdb_folder=None,
                     infer_angles=False,
                     kappa_addition=5,
                     plating=False,
                     plating_size=None,
                     plate_unordered=False,
                     activate_elbo_convergence=False,
                     activate_entropy_convergence=False,
                     config_dict=None,
                     parameter_search=False,
                     device="cuda",
                     results_dir="",
                     make_plots=False
                     )

    args = Namespace(**args_dict)
    #script_dir = os.path.dirname(os.path.abspath(__file__))


    check = DraupnirMain(args)

    assert check  == True, "Test passed"

