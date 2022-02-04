"""
2021: aleatoryscience
Lys Sanz Moreta
Draupnir : GP prior VAE for Ancestral Sequence Resurrection
=======================
"""
import Draupnir_utils as DraupnirUtils
import warnings
from collections import namedtuple
import pprint
import os
def available_datasets():
    datasets = {0: ["simulations_blactamase_1", "BetaLactamase_seq"],# EvolveAGene4 Betalactamase simulation # 32 leaves
                1: ["simulations_calcitonin_1", "Calcitonin_seq"],# EvolveAGene4 Calcitonin simulation #50 leaves
                2: ["simulations_src_sh3_1", "SRC_SH3"],# EvolveAGene4 SRC SH3 domain simulation 1 #100 leaves
                3: ["simulations_sirtuins_1", "Sirtuin_seq"],# EvolveAGene4 Sirtuin simulation #150 leaves
                4: ["simulations_src_sh3_3", "SRC_SH3"],# EvolveAGene4 SRC SH3 domain simulation 2 #200 leaves
                5: ["simulations_PIGBOS_1", "PIGBOS_seq"],# EvolveAGene4 PIGBOS simulation #300 leaves
                6: ["simulations_insulin_2", "Insulin_seq"],# EvolveAGene4 Insulin simulation #400 leaves
                7: ["simulations_src_sh3_2","SRC_SH3"],# EvolveAGene4 SRC SH3 domain simulation 2 #800 leaves
                8: ["benchmark_randall_original_naming", None],# uses the original tree and it's original node naming
                9: ["SH3_pf00018_larger_than_30aa",  None],# SRC kinases domain SH3 ---> Leaves and angles testing
                10: ["Coral_Faviina",  None],  # Faviina clade from coral sequences # 35 leaves
                11: ["Coral_all", None],# All Coral sequences (includes Faviina clade and additional sequences) #71 leaves
                12: ["PF00400",  None], # 125 real sequences
                13: ["aminopeptidase",  None], #another real sequences example
                14: ["PF00096", None]} #another real sequences example
    pprint.pprint(datasets)
    datasets_full_names = {"benchmark_randall_original_naming":"Randall's Coral fluorescent proteins (CFP) benchmark dataset",  # uses the original tree and it's original node naming
                "SH3_pf00018_larger_than_30aa":"PF00018 Pfam family of Protein Tyrosine Kinases SH3 domains",  # SRC kinases domain SH3 ---> Leaves and angles testing
                "simulations_blactamase_1":"32 leaves Simulation Beta-Lactamase",  # EvolveAGene4 Betalactamase simulation
                "simulations_src_sh3_1":"100 leaves Simulation SRC-Kinase SH3 domain",  # EvolveAGene4 SRC SH3 domain simulation
                "simulations_src_sh3_2": "800 leaves Simulation SRC-Kinase SH3 domain",
                "simulations_src_sh3_3": "200 leaves Simulation SRC-Kinase SH3 domain",
                "simulations_sirtuins_1": "150 leaves Simulation Sirtuin 1",
                "simulations_insulin_2": "400 leaves Simulation Insulin Growth Factor",
                "simulations_calcitonin_1": "50 leaves Simulation Calcitonin peptide",
                "simulations_PIGBOS_1": "300 leaves parser.add_argument('-use-cuda', type=str2bool, nargs='?',const=True, default=True, help='Use GPU')simulation PIGB Opposite Strand regulator",
                "Coral_Faviina":"Coral fluorescent proteins (CFP) Faviina clade",  # Faviina clade from coral sequences
                "Coral_all":"Coral fluorescent proteins (CFP) clade",  # All Coral sequences (includes Faviina clade and additional sequences)
                "PF00400":"WD40 125 sequences",
                "aminopeptidase":"Amino Peptidase",
                "PF00096":"PF00096 protein kinases"}
    return datasets,datasets_full_names



def create_draupnir_dataset(name,use_custom,build=False,fasta_file=None,tree_file=None,alignment_file=None):
    """In:
    :param str name: Dataset name
    :param bool use_custom: True (uses a custom dataset, located in datasets/custom/"folder_name" ) or False (uses a Draupnir default dataset (used in the publication))
    :param str script_dir: Working directory of Draupnir #TODO: remove
    :param bool build: Activates the construction of the dataset, might take a while if it requires to build tree, so it's recommended to use the pre-saved files
    :param str fasta_file: Path to NOT aligned sequences
    :param str tree_file: Path to Newick tree, format 1 in ete3
    :param str alignment_file: Path to pre-aligned sequences
    :returns namedtuple build_config:
        :str alignment-file:
        :bool use_ancestral: True (patristic_matrix_train = patristic_matrix_full (leaves + ancestors)), False (patristic_matrix_train = patristic_matrix) otherwise we remove the ancestral nodes from patristic_matrix_train. Necessary for some datasets
        :int n_test: percentage of train/leaves sequences to be used as test, i-e n_test = 20 ---> 20% leaves will be th etest datasets
        build_graph: make a graph for CNN #TODO: remove?
        :int aa_prob: Number of amino acid probabilities (21 or 24), depends on the different types of amino acids in the sequence alignment
        triTSNE: Whether to plot TSNE in 3D (True) or not #TODO: Remove
        :bool leaves_testing: True (uses all the leaf's evolutionary distances for training, it only observes (n-n_test) leafsequences. USE WITH n_test), False (uses all the leaf's evolutionary distances for training
                            and observes all the leaf sequences. Use with datasets without ancestors for testing, only generate sequences).
        """
    BuildConfig = namedtuple('BuildConfig',['alignment_file','use_ancestral','n_test','build_graph',"aa_prob","triTSNE","leaves_testing","script_dir","no_testing"])
    SettingsConfig = namedtuple("SettingsConfig", ["one_hot_encoding", "model_design", "aligned_seq","data_folder"])
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not use_custom:
        name, root_sequence_name = available_datasets()[0][name]
        full_name = available_datasets()[1][name]
        storage_folder = "datasets/default"
        warnings.warn("You have selected a pre-defined dataset")
        if name == "benchmark_randall_original_naming":
            build_config = BuildConfig(alignment_file=None, use_ancestral=True, n_test=0, build_graph=True,aa_prob=21,triTSNE=False,leaves_testing=False,script_dir=script_dir,no_testing=False)
            if build:
                DraupnirUtils.benchmark_dataset(name, aa_prob=21, inferred=False, original_naming=True) #TODO: fix
        # Highlight: Simulation datasets, Simulations might produce stop codons---Use probabilities == 21
        if name.startswith("simulations"):
            # DraupnirUtils.Remove_Stop_Codons("Datasets_Simulations/{}/Dataset{}/{}.txt".format(simulation_folder,dataset_number,root_sequence_name))
            alignment_file = "{}/datasets/default/{}/{}_True_Pep_alignment.FASTA".format(script_dir,name,root_sequence_name)
            build_config = BuildConfig(alignment_file=alignment_file, use_ancestral=True, n_test=0, build_graph=True,aa_prob=21,triTSNE=False,leaves_testing=False,script_dir=script_dir,no_testing=False)
            if build:
                DraupnirUtils.create_dataset(name,
                               one_hot_encoding=False,
                               tree_file="{}/datasets/default/{}/{}_True_Rooted_tree_node_labels.tre".format(script_dir,name,root_sequence_name),
                               alignment_file="{}/datasets/default/{}/{}_True_Pep_alignment.FASTA".format(script_dir,name,root_sequence_name),
                               aa_probs=21,
                               rename_internal_nodes=True,
                               storage_folder=storage_folder)

        if name == "SH3_pf00018_larger_than_30aa":# Highlight: SRC Kinases, SH3 domain with PDB structures
            alignment_file = "{}/datasets/default/SH3_pf00018_larger_than_30aa/SH3_pf00018_larger_than_30aa.mafft".format(script_dir) #I hope it's the correct one
            build_config = BuildConfig(alignment_file=alignment_file,
                                       use_ancestral=False,
                                       n_test=10, #i.e n_test = 20 --> 20% sequences for testing
                                       build_graph=False,
                                       aa_prob=21,
                                       triTSNE=False,
                                       leaves_testing=False,#turn to true to split the leaves into a train and test dataset
                                       script_dir=script_dir,
                                       no_testing=True) #leave testing activated to use ModelK(full latent space prediction) if False it uses Modeli
            warnings.warn("Remember with this dataset we use part of the leaves (default 10%) as test and part as training")
            if build:
                family_name = "PF00018;" #protein family name, src-sh3 domain
                pfam_dict, pdb_list = DraupnirUtils.Pfam_parser(family_name,first_match=True,update_pfam=False)
                #DraupnirUtils.Download_PDB_Lists(family_name.strip(";"),pdb_list) #Highlight: Reactivate if needed to download PDB files again
                DraupnirUtils.create_dataset(name_file=name,
                                             one_hot_encoding=False,
                                             method="iqtree",
                                             PDB_folder="{}/datasets/default/PDB_files_Draupnir_{}_{}".format(script_dir,family_name.strip(";"),len(pdb_list)),
                                             alignment_file=alignment_file,
                                             tree_file="{}/datasets/default/SH3_pf00018_larger_than_30aa/SH3_pf00018_larger_than_30aa.mafft.treefile".format(script_dir),
                                             pfam_dict=pfam_dict,
                                             rename_internal_nodes=True,
                                             storage_folder=storage_folder)
        if name == "PF00096":#PKKinases
            alignment_file = "{}/datasets/default/PF00096/PF00096.fasta".format(script_dir)
            build_config = BuildConfig(alignment_file=alignment_file,
                                       use_ancestral=False,
                                       n_test=0, #Indicates the percentage of the leaves sequences for testing
                                       build_graph=False,
                                       aa_prob=21,
                                       triTSNE=False,
                                       leaves_testing=False,
                                       no_testing=True,
                                       script_dir=script_dir)
            if build:
                DraupnirUtils.create_dataset(name_file=name,
                                             one_hot_encoding=False,
                                             method="iqtree",
                                             alignment_file=alignment_file,
                                             tree_file="{}/datasets/default/PF00096/PF00096.fasta.treefile".format(script_dir),
                                             rename_internal_nodes=True,
                                             min_len=0,
                                             storage_folder=storage_folder)
        if name == "aminopeptidase":#
            alignment_file = "{}/datasets/default/aminopeptidase/2MAT_BLAST90.fasta".format(script_dir)
            build_config = BuildConfig(alignment_file=alignment_file,
                                       use_ancestral=False,
                                       n_test=0, #Indicates the percentage of the leaves sequences for testing
                                       build_graph=False,
                                       aa_prob=21,
                                       triTSNE=False,
                                       leaves_testing=False,
                                       no_testing=True,
                                       script_dir=script_dir)
            if build:
                #DraupnirUtils.Download_PDB_Lists(family_name.strip(";"), pdb_list)
                DraupnirUtils.create_dataset(name_file=name,
                                             one_hot_encoding=False,
                                             method="iqtree",
                                             alignment_file=alignment_file,
                                             tree_file="{}/datasets/default/aminopeptidase/2MAT_BLAST90.fasta.treefile".format(script_dir),
                                             rename_internal_nodes=True,
                                             storage_folder=storage_folder)
        if name == "PF00400":
            alignment_file = "{}/datasets/default/PF00400/PF00400.mafft".format(script_dir)
            build_config = BuildConfig(alignment_file=alignment_file,
                                       use_ancestral=False,
                                       n_test=0,  # Indicates the percentage of the leaves sequences for testing
                                       build_graph=False,
                                       aa_prob=21,
                                       triTSNE=False,
                                       leaves_testing=False,
                                       script_dir=script_dir,
                                       no_testing=True)
            if build:
                family_name = "PF00400;"  # protein family name
                pfam_dict, pdb_list = DraupnirUtils.Pfam_parser(family_name, first_match=True, update_pfam=False)
                #DraupnirUtils.Download_PDB_Lists(family_name.strip(";"), pdb_list)
                DraupnirUtils.create_dataset(name_file=name,
                                             one_hot_encoding=False,
                                             method="iqtree",
                                             PDB_folder="{}/datasets/default/PDB_files_Draupnir_{}_{}".format(script_dir,
                                                                                             family_name.strip(";"),
                                                                                             len(pdb_list)),
                                             alignment_file=alignment_file,
                                             tree_file="{}/datasets/default/PF00400/PF00400.mafft.treefile".format(script_dir),
                                             pfam_dict=pfam_dict,
                                             rename_internal_nodes=True,
                                             min_len=30,
                                             storage_folder=storage_folder)
        if name == "Coral_Faviina":
            alignment_file = "{}/datasets/default/Coral_Faviina/Coral_Faviina_Aligned_Protein.fasta".format(script_dir)
            build_config = BuildConfig(alignment_file=alignment_file,use_ancestral=False,n_test=0,build_graph=False,aa_prob=21,triTSNE=False,leaves_testing=False,script_dir=script_dir,no_testing=False)
            if build:
                DraupnirUtils.create_dataset(name_file=name,
                                             one_hot_encoding=False,
                                             tree_file="{}/datasets/default/Coral_Faviina/c90.PP.tree".format(script_dir),
                                             alignment_file=alignment_file,
                                             aa_probs=21,
                                             rename_internal_nodes=True,
                                             storage_folder=storage_folder)
        if name == "Coral_all":
            alignment_file = "{}/datasets/default/Coral_all/Coral_all_Aligned_Protein.fasta".format(script_dir)
            build_config = BuildConfig(alignment_file=alignment_file,
                                       use_ancestral=False,
                                       n_test=0,
                                       build_graph=False,
                                       aa_prob=21,
                                       triTSNE=False,
                                       leaves_testing=False,script_dir=script_dir,
                                       no_testing=False)
            if build_config.no_testing: warnings.warn("You have chose NOT to test the Coral sequences")
            if build:
                DraupnirUtils.create_dataset(name_file=name,
                                             one_hot_encoding=False,
                                             tree_file="{}/datasets/default/Coral_all/c90.PP.tree".format(script_dir),
                                             alignment_file=alignment_file,
                                             aa_probs=23,
                                             rename_internal_nodes=True,
                                             storage_folder=storage_folder)
        else:
             raise NameError("Name not in the available datasets")

    else:
        warnings.warn("You have selected to use a custom dataset")
        root_sequence_name = None
        assert any(v is not None for v in [fasta_file,alignment_file,tree_file]) != False,"Provide at least 1 file path: fasta_file or alignment_file or alignment_file + tree_file"
        build_config = BuildConfig(alignment_file=alignment_file,
                                   use_ancestral=False,
                                   n_test=0,
                                   build_graph=False,
                                   aa_prob=21,
                                   triTSNE=False,
                                   leaves_testing=False,
                                   script_dir=script_dir,
                                   no_testing=False)
        if build:
            if fasta_file is not None and alignment_file is None:
                alignment_file = tree_file = None
                storage_folder = os.path.dirname(fasta_file)
            else:
                assert alignment_file is not None, "Provide at least 1 file path: alignment_file or alignment_file + tree_file"
                storage_folder = os.path.dirname(alignment_file)

            DraupnirUtils.create_dataset(name_file=name,
                                             one_hot_encoding=False,
                                             tree_file=tree_file,
                                             alignment_file=alignment_file,
                                             fasta_file=fasta_file,
                                             aa_probs=21,
                                             rename_internal_nodes=True,
                                             storage_folder=storage_folder)
        else:
            assert tree_file is not None, "Please provide a tree file in datasets/custom or first build the tree"
            assert alignment_file is not None, "Please provide a alignment file in datasets/custom or first build the alignment"

    settings_config = SettingsConfig(one_hot_encoding=False,
                             model_design="GP_VAE",
                             aligned_seq=True,
                             data_folder=["datasets/custom" if use_custom else "datasets/default"][0])


    return build_config,settings_config, root_sequence_name