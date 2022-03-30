import unittest,sys,os
import pytest
from collections import namedtuple
local_repository=True
if local_repository:
    sys.path.insert(1,"/home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src")
    import draupnir
else:#pip installed module
    import draupnir


Args = namedtuple("Args",["name","use_custom","alignment_file","tree_file","fasta_file","build","pdb_folder","one_hot_encoded",
                          "num_epochs","batch_size","select_guide","batch_by_clade","infer_angles","plating","plating_size",
                          "plate_unordered","aa_probs","n_samples","kappa_addition","use_blosum","subs_matrix","generate_samples",
                          "load_pretrained_path","embedding_dim","use_cuda","use_scheduler","activate_elbo_convergence",
                          "activate_entropy_convergence","test_frequency","config_dict","parameter_search"])

@pytest.fixture(params=['simulations_blactamase_1', 'aminopeptidase', 'benchmark_randall_original_naming'])
def prepare_dataset(request):
    """request is an object required to access the parameters"""
    print(request.param)
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


        #yield  #---> before yield setup activities, after cleanup activities

        #draupnir.run?
        return build_config,settings_config,root_sequence_name
    return _prepare_dataset(request)

#@pytest.mark.usefixtures("prepare_dataset") #TODO: How the heck does this work
def test_load_build_config(prepare_dataset):
    build_config, settings_config, root_sequence_name = prepare_dataset
    print(build_config)
    isbuildconfig = [True if build_config.__module__ =="build_config" else False][0]
    assert isbuildconfig == True
    print("Done testing build_config")

if __name__ == '__main__':
    pytest.main(["-s","test_load.py"]) #-s is to run the statements after yield?