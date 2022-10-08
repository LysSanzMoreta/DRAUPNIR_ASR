"""
=======================
2022: Lys Sanz Moreta
Draupnir : Ancestral protein sequence reconstruction using a tree-structured Ornstein-Uhlenbeck variational autoencoder
=======================
"""
import draupnir.utils as DraupnirUtils
import warnings
from collections import namedtuple
import pprint
import os
import numpy as np
import torch
from Bio import SeqIO
import gdown

def available_datasets(print_dict = False):
    """Displays the available default data sets shown in the paper"""
    datasets = {"simulations_blactamase_1": "BetaLactamase_seq",# EvolveAGene4 Betalactamase simulation # 32 leaves
                "simulations_calcitonin_1": "Calcitonin_seq",# EvolveAGene4 Calcitonin simulation #50 leaves
                "simulations_src_sh3_1": "SRC_SH3",# EvolveAGene4 SRC SH3 domain simulation 1 #100 leaves
                "simulations_sirtuins_1": "Sirtuin_seq",# EvolveAGene4 Sirtuin simulation #150 leaves
                "simulations_src_sh3_3": "SRC_SH3",# EvolveAGene4 SRC SH3 domain simulation 2 #200 leaves
                "simulations_PIGBOS_1": "PIGBOS_seq",# EvolveAGene4 PIGBOS simulation #300 leaves
                "simulations_insulin_2": "Insulin_seq",# EvolveAGene4 Insulin simulation #400 leaves
                "simulations_src_sh3_2":"SRC_SH3",# EvolveAGene4 SRC SH3 domain simulation 2 #800 leaves
                "simulations_jj_1": "jj1",
                "simulations_jj_2": "jj2",
                "benchmark_randall_original_naming": None,# uses the original tree and it's original node naming
                "SH3_pf00018_larger_than_30aa":  None,# SRC kinases domain SH3 ---> Leaves and angles testing
                "Coral_Faviina":  None,  # Faviina clade from coral sequences # 35 leaves
                "Coral_all": None,# All Coral sequences (includes Faviina clade and additional sequences) #71 leaves
                "PF00400": None, # 125 real sequences
                "PF00400_beta":None,#TEST DATA
                "aminopeptidase":  None, #another real sequences example
                "PF00096": None} #another real sequences example
    if print_dict:
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
                "simulations_jj_1": "jj1",
                "simulations_jj_2": "jj2",
                "Coral_Faviina":"Coral fluorescent proteins (CFP) Faviina clade",  # Faviina clade from coral sequences
                "Coral_all":"Coral fluorescent proteins (CFP) clade",  # All Coral sequences (includes Faviina clade and additional sequences)
                "PF00400":"WD40 125 sequences",
                "PF00400_beta": "WD40 125 sequences", #TODO:Remove
                "aminopeptidase":"Amino Peptidase",
                "PF00096":"PF00096 protein kinases"}
    return datasets,datasets_full_names
def create_draupnir_dataset(name,use_custom,script_dir,args,build=False,fasta_file=None,tree_file=None,alignment_file=None):
    """In:
    :param str name: Dataset name
    :param bool use_custom: True (uses a custom dataset, located in datasets/custom/"folder_name" ) or False (uses a Draupnir default dataset (used in the publication))
    :param str script_dir: Working directory of Draupnir #TODO: remove
    :param bool build: Activates the construction of the dataset, might take a while if it requires to build tree, so it's recommended to use the pre-saved files
    :param str or None fasta_file: Path to NOT aligned sequences
    :param str or None tree_file: Path to Newick tree, format 1 in ete3
    :param str or None alignment_file: Path to pre-aligned sequences
    :returns namedtuple build_config:
        :str alignment-file:
        :bool use_ancestral: True (patristic_matrix_train = patristic_matrix_full (leaves + ancestors)), False (patristic_matrix_train = patristic_matrix) otherwise we remove the ancestral nodes from patristic_matrix_train. Necessary for some datasets
        :int n_test: percentage of train/leaves sequences to be used as test, i-e n_test = 20 ---> 20% leaves will be th etest datasets
        build_graph: make a graph for CNN #TODO: remove?
        :int aa_prob: Number of amino acid probabilities (21 or 24), depends on the different types of amino acids in the sequence alignment
        :bool triTSNE: Whether to plot TSNE in 3D (True) or not #TODO: Remove
        :bool leaves_testing: True (uses all the leaf's evolutionary distances for training, it only observes (n-n_test) leafsequences. USE WITH n_test), False (uses all the leaf's evolutionary distances for training
                            and observes all the leaf sequences. Use with datasets without ancestors for testing, only generate sequences).
        """
    BuildConfig = namedtuple('BuildConfig',['alignment_file','use_ancestral','n_test','build_graph',"aa_prob","triTSNE","leaves_testing","script_dir","no_testing"],module="build_config") #__name__ + ".namespace"
    SettingsConfig = namedtuple("SettingsConfig", ["one_hot_encoding", "model_design", "aligned_seq","data_folder","full_name","tree_file"],module="settings_config")
    if args.one_hot_encoded:
        warnings.warn("Draupnir was constructed to be used with integers for Categorical likelihood,not OneHotCategorical. And blosum-encoding for the guide. You can build"
                      "the one-hot-encoded dataset for other purposes")

    #script_dir = os.path.dirname(os.path.abspath(__file__))
    if not use_custom:
        warnings.warn("You have selected a pre-defined dataset, if not present, it will be downloaded. Otherwise set use_custom to True")
        root_sequence_name = available_datasets()[0][name]
        full_name = available_datasets()[1][name]
        storage_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "data")) #changed from "datasets/default"
        dir_name = '{}/{}'.format(storage_folder,name)
        dict_urls = {
                "aminopeptidase":"https://drive.google.com/drive/folders/1fLsOJbD1hczX15NW0clCgL6Yf4mnx_yl?usp=sharing",
                "benchmark_randall_original_naming":"https://drive.google.com/drive/folders/1oE5-22lqcobZMIguatOU_Ki3N2Fl9b4e?usp=sharing",
                "Coral_all":"https://drive.google.com/drive/folders/1IbfiM2ww5PDcDSpTjrWklRnugP8RdUTu?usp=sharing",
                "Coral_Faviina":"https://drive.google.com/drive/folders/1Ehn5xNNYHRu1iaf7vS66sbAESB-dPJRx?usp=sharing",
                "PDB_files_Draupnir_PF00018_116":"https://drive.google.com/drive/folders/1YJDS_oHHq-5qh2qszwk-CucaYWa9YDOD?usp=sharing",
                "PDB_files_Draupnir_PF00400_185": "https://drive.google.com/drive/folders/1LTOt-dhksW1ZsBjb2uzi2NB_333hLeu2?usp=sharing",
                "PF00096":"https://drive.google.com/drive/folders/103itCfxiH8jIjKYY9Cvy7pRGyDl9cnej?usp=sharing",
                "PF00400":"https://drive.google.com/drive/folders/1Ql10yTItcdX93Xpz3Oh-sl9Md6pyJSZ3?usp=sharing",
                "SH3_pf00018_larger_than_30aa":"https://drive.google.com/drive/folders/1Mww3uvF_WonpMXhESBl9Jjes6vAKPj5f?usp=sharing",
                "simulations_blactamase_1":"https://drive.google.com/drive/folders/1ecHyqnimdnsbeoIh54g2Wi6NdGE8tjP4?usp=sharing",
                "simulations_calcitonin_1":"https://drive.google.com/drive/folders/1jJ5RCfLnJyAq0ApGIPrXROErcJK3COvK?usp=sharing",
                "simulations_insulin_2":"https://drive.google.com/drive/folders/1xB03AF_DYv0EBTwzUD3pj03zBcQDDC67?usp=sharing",
                "simulations_PIGBOS_1":"https://drive.google.com/drive/folders/1KTzfINBVo0MqztlHaiJFoNDt5gGsc0dK?usp=sharing",
                "simulations_sirtuins_1":"https://drive.google.com/drive/folders/1llT_HvcuJQps0e0RhlfsI1OLq251_s5S?usp=sharing",
                "simulations_src_sh3_1":"https://drive.google.com/drive/folders/1tZOn7PrCjprPYmyjqREbW9PFTsPb29YZ?usp=sharing",
                "simulations_src_sh3_2":"https://drive.google.com/drive/folders/1ji4wyUU4aZQTaha-Uha1GBaYruVJWgdh?usp=sharing",
                "simulations_src_sh3_3":"https://drive.google.com/drive/folders/13xLOqW2ldRNm8OeU-bnp9DPEqU1d31Wy?usp=sharing"

            }
        #download=False #TODO: Remove
        if os.path.isdir(dir_name):
            if not os.listdir(dir_name):
                print("Directory is empty")
                #if download:
                os.remove(dir_name)
                print("Data directory is missing. Downloading, this might take a while. If you see an error like \n"
                      " 'Cannot retrieve the public link of the file. You may need to change the permission to <Anyone with the link>, or have had many accesses', \n"
                      "just wait, too many requests have been made to the google drive folder \n"
                      "Otherwise just download the data sets manually from the google drive urls : \n {}".format(
                    dict_urls[name]))
                gdown.download_folder(dict_urls[name], output='{}/{}'.format(storage_folder, name), quiet=True,
                                      use_cookies=False, remaining_ok=True)
                # else:
                #     pass


            else:
                print("Dataset is ready in the folder!")
        else:
            print("Data directory is missing. Downloading, this might take a while. If you see an error like \n"
                  " 'Cannot retrieve the public link of the file. You may need to change the permission to <Anyone with the link>, or have had many accesses', \n"
                  "just wait, too many requests have been made to the google drive folder \n"
                  "Otherwise just download the data sets MANUALLY from the google drive urls : \n {}".format(dict_urls[name]))
            gdown.download_folder(dict_urls[name], output='{}/{}'.format(storage_folder,name),quiet=True, use_cookies=False,remaining_ok=True)

        # Highlight: Simulation datasets, Simulations might produce stop codons---Use probabilities == 21
        if name.startswith("simulations"):
            # leaves_prot = "{}/{}/jj1_True_Pep_alignment.FASTA".format(storage_folder,name)
            # internal_prot = "{}/{}/jj1_pep_Internal_Nodes_True_alignment.FASTA".format(storage_folder,name)
            # DraupnirUtils.remove_stop_codons(leaves_prot,is_prot=True)
            # DraupnirUtils.remove_stop_codons(internal_prot,is_prot=True)
            # exit()
            alignment_file = "{}/{}/{}_True_Pep_alignment.FASTA".format(storage_folder,name,root_sequence_name)
            tree_file = "{}/{}/{}_True_Rooted_tree_node_labels.tre".format(storage_folder,name,root_sequence_name)
            build_config = BuildConfig(alignment_file=alignment_file,
                                       use_ancestral=True,
                                       n_test=0,
                                       build_graph=True,
                                       aa_prob=21,
                                       triTSNE=False,
                                       leaves_testing=False,
                                       script_dir=script_dir,
                                       no_testing=False)
            if build:
                DraupnirUtils.create_dataset(name,
                               one_hot_encoding=args.one_hot_encoded,
                               tree_file= tree_file,
                               alignment_file="{}/{}/{}_True_Pep_alignment.FASTA".format(storage_folder,name,root_sequence_name),
                               aa_probs=21,
                               rename_internal_nodes=True,
                               storage_folder=storage_folder)

        elif name == "benchmark_randall_original_naming":
            alignment_file = "{}/{}/benchmark_randall_original_naming.mafft".format(storage_folder,name)
            tree_file = "{}/{}/RandallBenchmarkTree_OriginalNaming.tree".format(storage_folder,name)
            build_config = BuildConfig(alignment_file=alignment_file,
                                       use_ancestral=True, n_test=0,
                                       build_graph=True,aa_prob=21,
                                       triTSNE=False,leaves_testing=False,script_dir=script_dir,no_testing=False)
            if build:
                benchmark_randalls_dataset_train(name, storage_folder,args.one_hot_encoded,aa_prob=21)

        elif name == "SH3_pf00018_larger_than_30aa":# Highlight: SRC Kinases, SH3 domain with PDB structures
            alignment_file = "{}/SH3_pf00018_larger_than_30aa/SH3_pf00018_larger_than_30aa.mafft".format(storage_folder) #I hope it's the correct one
            tree_file = "{}/SH3_pf00018_larger_than_30aa/SH3_pf00018_larger_than_30aa.mafft.treefile".format(storage_folder)
            build_config = BuildConfig(alignment_file=alignment_file,
                                       use_ancestral=False,
                                       n_test=0, #i.e n_test = 20 --> 20% sequences for testing---> use with leaves testing True
                                       build_graph=False,
                                       aa_prob=21,
                                       triTSNE=False,
                                       leaves_testing=False,#turn to true to split the leaves into a train and test dataset
                                       script_dir=script_dir,
                                       no_testing=True)
            warnings.warn("Remember with this dataset we use part of the leaves (default 20%) as test and part as training unless leaves_testing == False")
            if build_config.leaves_testing: assert build_config.n_test > 0, "Please select a % of leaves for testing with n_test"
            else: assert build_config.n_test == 0; "Please do not test leaves, set n_test = 0"
            if build:
                family_name = "PF00018;" #protein family name, src-sh3 domain
                pfam_dict, pdb_list = DraupnirUtils.Pfam_parser(family_name,storage_folder,first_match=True,update_pfam=False)
                DraupnirUtils.download_PDB(pdb_list, "{}/PDB_files_Draupnir_{}_{}".format(storage_folder,family_name.strip(";"),len(pdb_list)))
                DraupnirUtils.create_dataset(name_file=name,
                                             one_hot_encoding=args.one_hot_encoded,
                                             method="iqtree",
                                             PDB_folder="{}/PDB_files_Draupnir_{}_{}".format(storage_folder,family_name.strip(";"),len(pdb_list)),
                                             alignment_file=alignment_file,
                                             tree_file=tree_file,
                                             pfam_dict=pfam_dict,
                                             rename_internal_nodes=True,
                                             storage_folder=storage_folder)
        elif name == "PF00096":#PKKinases
            alignment_file = "{}/PF00096/PF00096.fasta".format(storage_folder)
            tree_file = "{}/PF00096/PF00096.fasta.treefile".format(storage_folder)
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
                                             one_hot_encoding=args.one_hot_encoded,
                                             method="iqtree",
                                             alignment_file=alignment_file,
                                             tree_file=tree_file,
                                             rename_internal_nodes=True,
                                             min_len=0,
                                             storage_folder=storage_folder)
        elif name == "aminopeptidase":#
            alignment_file = "{}/aminopeptidase/2MAT_BLAST90.fasta".format(storage_folder)
            tree_file = "{}/aminopeptidase/2MAT_BLAST90.fasta.treefile".format(storage_folder)
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
                                             one_hot_encoding=args.one_hot_encoded,
                                             method="iqtree",
                                             alignment_file=alignment_file,
                                             tree_file=tree_file,
                                             rename_internal_nodes=True,
                                             storage_folder=storage_folder)
        elif name == "PF00400":
            alignment_file = "{}/PF00400/PF00400.mafft".format(storage_folder)
            tree_file = "{}/PF00400/PF00400.mafft.treefile".format(storage_folder)
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
                pfam_dict, pdb_list = DraupnirUtils.Pfam_parser(family_name,storage_folder,first_match=True, update_pfam=False)
                #DraupnirUtils.download_PDB(pdb_list, "/home/lys/Dropbox/PhD/DRAUPNIR/PDB_files_Draupnir_{}_{}".format(family_name.strip(";"), len(pdb_list)))
                DraupnirUtils.create_dataset(name_file=name,
                                             one_hot_encoding=args.one_hot_encoded,
                                             method="iqtree",
                                             PDB_folder="{}/PDB_files_Draupnir_{}_{}".format(storage_folder,
                                                                                             family_name.strip(";"),
                                                                                             len(pdb_list)),
                                             alignment_file=alignment_file,
                                             tree_file=tree_file,
                                             pfam_dict=pfam_dict,
                                             rename_internal_nodes=True,
                                             min_len=30,
                                             storage_folder=storage_folder)
        elif name == "PF00400_beta": #TODO: remove, only for fixing one hot
            alignment_file = "{}/PF00400_beta/PF00400_beta.mafft".format(storage_folder)
            tree_file = "{}/PF00400_beta/PF00400_beta.mafft.treefile".format(storage_folder)
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
                pfam_dict, pdb_list = DraupnirUtils.Pfam_parser(family_name, storage_folder, first_match=True,update_pfam=False)
                # DraupnirUtils.download_PDB(pdb_list, "/home/lys/Dropbox/PhD/DRAUPNIR/PDB_files_Draupnir_{}_{}".format(family_name.strip(";"), len(pdb_list)))
                DraupnirUtils.create_dataset(name_file=name,
                                             one_hot_encoding=args.one_hot_encoded,
                                             method="iqtree",
                                             PDB_folder="/home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src/draupnir/data/PDB_files_test",
                                             alignment_file=alignment_file,
                                             tree_file=tree_file,
                                             pfam_dict=pfam_dict,
                                             rename_internal_nodes=True,
                                             min_len=30,
                                             storage_folder=storage_folder)
        elif name == "Coral_Faviina":
            alignment_file = "{}/{}/Coral_Faviina_Aligned_Protein.fasta".format(storage_folder,name)
            tree_file = "{}/{}/c90.PP.tree".format(storage_folder,name)
            build_config = BuildConfig(alignment_file=alignment_file,use_ancestral=False,n_test=0,build_graph=False,aa_prob=21,triTSNE=False,leaves_testing=False,script_dir=script_dir,no_testing=False)
            if build:
                DraupnirUtils.create_dataset(name_file=name,
                                             one_hot_encoding=args.one_hot_encoded,
                                             tree_file=tree_file,
                                             alignment_file=alignment_file,
                                             aa_probs=21,
                                             rename_internal_nodes=True,
                                             storage_folder=storage_folder)
        elif name == "Coral_all":
            alignment_file = "{}/{}/Coral_all_Aligned_Protein.fasta".format(storage_folder,name)
            tree_file = "{}/{}/c90.PP.tree".format(storage_folder,name)
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
                                             one_hot_encoding=args.one_hot_encoded,
                                             PDB_folder=None,
                                             tree_file="{}/{}/c90.PP.tree".format(storage_folder,name),
                                             alignment_file=alignment_file,
                                             aa_probs=23,
                                             rename_internal_nodes=True,
                                             storage_folder=storage_folder)
        else:
             print("..................")
             raise NameError("Name {} not in the available default datasets".format(name))

    else:
        warnings.warn("You have selected to use a custom dataset, make sure to have at least a fasta file ready")
        root_sequence_name = None
        full_name = name
        assert any(v is not None for v in [fasta_file,alignment_file,tree_file]) != False,"Provide at least 1 file path: fasta_file or alignment_file or alignment_file + tree_file"
        build_config = BuildConfig(alignment_file=alignment_file,
                                   use_ancestral=False,
                                   n_test=0,
                                   build_graph=False,
                                   aa_prob=21,
                                   triTSNE=False,
                                   leaves_testing=False,
                                   script_dir=script_dir,
                                   no_testing=True)
        if build:
            if fasta_file is not None and alignment_file is None:
                alignment_file = tree_file = None
                storage_folder = os.path.dirname(os.path.dirname(fasta_file))
            else:
                assert alignment_file is not None, "Provide at least 1 file path: alignment_file or alignment_file + tree_file"
                storage_folder = os.path.dirname(os.path.dirname(alignment_file))

            tree_file = DraupnirUtils.create_dataset(name_file=name,
                                             one_hot_encoding=args.one_hot_encoded,
                                             PDB_folder=args.pdb_folder,
                                             tree_file=tree_file,
                                             alignment_file=alignment_file,
                                             fasta_file=fasta_file,
                                             aa_probs=args.aa_probs,
                                             rename_internal_nodes=True,
                                             storage_folder=storage_folder)
        else:
            assert tree_file is not None, "Please provide a tree file in the same folder as the alignment or fasta sequences file, or first build the tree"
            assert alignment_file is not None, "Please provide a alignment file inside the same folder as the alignment or fasta sequences file or first build the alignment"
            storage_folder = os.path.dirname(os.path.dirname(alignment_file))

    settings_config = SettingsConfig(one_hot_encoding=args.one_hot_encoded,
                             model_design="GP_VAE",
                             aligned_seq=True,
                             data_folder="{}/{}".format(storage_folder,name), #["{}".format(storage_folder) if use_custom else "{}/{}".format(storage_folder,name)][0]
                             full_name=full_name,
                             tree_file=tree_file)
    return build_config,settings_config, root_sequence_name

def benchmark_randalls_dataset_train(name,storage_folder,one_hot_encoded,aa_prob):
    """Processing of the leaves dataset from "An experimental phylogeny to benchmark ancestral sequence reconstruction"
    :param str name: project dataset name
    :param int aa_prob: amino acid probabilities"""
    observed_nodes = [19,18,17,16,15,14,13,12,11,10,9,8,7,6,4,5,3,2,1] #I have this in a list for a series of past reasons
    sequences_file = "benchmark_randall_original_naming/original_data/RandallExperimentalPhylogenyAASeqs.fasta"
    #Select the sequences of only the observed nodes
    full_fasta = SeqIO.parse(sequences_file, "fasta")
    with open("{}/original_data/Randall_Benchmark_Observed.fasta".format(storage_folder), "w") as output_handle:
        observed_fasta = []
        for seq in full_fasta:
            if int(seq.id) in observed_nodes:
                observed_fasta.append(seq)
        SeqIO.write(observed_fasta, output_handle, "fasta")
    DraupnirUtils.create_dataset(name,
                   one_hot_encoding=one_hot_encoded,
                   fasta_file="{}/original_data/Randall_Benchmark_Observed.fasta",
                   alignment_file="{}/benchmark_randall_original.mafft".format(storage_folder),
                   tree_file="{}/RandallBenchmarkTree_OriginalNaming.tree".format(storage_folder),
                   aa_probs=aa_prob,
                   rename_internal_nodes=False)
def benchmark_randalls_dataset_test(settings_config,aa_probs=21):
    """Processing of the internal nodes dataset from "An experimental phylogeny to benchmark ancestral sequence reconstruction
    :param str scriptdir
    :param int aa_probs"""
    internal_nodes = [21,30,37,32,31,34,35,36,33,28,29,22,23,27,24,26,25]
    sequences_file = "{}/original_data/RandallExperimentalPhylogenyAASeqs.fasta".format(settings_config.data_folder)
    # Select the sequences of only the observed nodes
    full_fasta = SeqIO.parse(sequences_file, "fasta")
    aminoacid_names= DraupnirUtils.aminoacid_names_dict(aa_probs)
    internal_fasta_dict = {}
    for seq in full_fasta:
        if int(seq.id) in internal_nodes:
            seq_numbers =[]
            for aa_name in seq.seq:
                #aa_number = int(np.where(np.array(aminoacid_names) == aa_name)[0][0]) + add_on
                aa_number = aminoacid_names[aa_name]
                seq_numbers.append(aa_number)
            internal_fasta_dict[int(seq.id)] = [seq.seq,seq_numbers]
    max_length = max([int(len(sequence[0])) for idx,sequence in internal_fasta_dict.items()]) #225

    dataset = np.zeros((len(internal_fasta_dict), max_length + 1 + 1, 30),dtype=object)  # 30 dim to accomodate git vectors. Careful with the +2 (to include git, seqlen)
    for i, (key,val) in enumerate(internal_fasta_dict.items()):
        # aligned_seq = list(alignment[i].seq.strip(",")) # I don't think this made sense, cause files could be in wrong order?
        aligned_seq = list(internal_fasta_dict[key][0].strip(","))
        no_gap_indexes = np.where(np.array(aligned_seq) != "-")[0] + 2  # plus 2 in order to make the indexes fit in the final dataframe
        dataset[i, 0,0] = len(internal_fasta_dict[key][1]) #Insert seq len and git vector
        dataset[i,0,1] = key #position in the tree
        dataset[i, 0, 2] =  0 #fake distance to the root
        dataset[i, no_gap_indexes,0] = internal_fasta_dict[key][1] # Assign the aa info (including angles) to those positions where there is not a gap

    return dataset, internal_nodes
def load_randalls_benchmark_ancestral_sequences(settings_config):
    dataset_test,internal_names_test = benchmark_randalls_dataset_test(settings_config)
    dataset_test = np.array(dataset_test, dtype="float64")
    dataset_test = torch.from_numpy(dataset_test)
    return dataset_test,internal_names_test
def load_simulations_ancestral_sequences(name,settings_config,align_seq_len,tree_levelorder_names,root_sequence_name,aa_probs,script_dir):
    """Load and format the ancestral sequences from the EvolveAGene4 simulations
    :param name
    :param settings_config
    :param align_seq_len
    :param tree_levelorder_names: names of the nodes in tree level order
    :param root_sequence_name
    :param aa_probs
    :param script_dir
    """
    ancestral_file = "{}/{}_pep_Internal_Nodes_True_alignment.FASTA".format(settings_config.data_folder,root_sequence_name)

    # Select the sequences of only the observed nodes
    ancestral_fasta = SeqIO.parse(ancestral_file, "fasta")
    aminoacid_names = DraupnirUtils.aminoacid_names_dict(aa_probs)
    internal_fasta_dict = {}
    tree_level_order_names = np.char.strip(tree_levelorder_names, 'I') #removing the letter added while processing the full tree

    for seq in ancestral_fasta:
            seq_numbers =[]
            #Highlight: replace all stop codons with a gap and also the sequence coming after it
            sequence_no_stop_codons = str(seq.seq).split("*", 1)[0]
            len_diff = len(str(seq.seq)) - len(sequence_no_stop_codons)
            sequence_no_stop_codons = sequence_no_stop_codons + "-"*len_diff
            #for aa_name in seq.seq :
            for aa_name in sequence_no_stop_codons:
                #aa_number = int(np.where(np.array(aminoacid_names) == aa_name)[0][0])
                aa_number = aminoacid_names[aa_name]
                seq_numbers.append(aa_number)
            seq_id = np.where(np.array(tree_level_order_names) == seq.id.strip("Node"))[0][0]
            #internal_fasta_dict[int(seq_id)] = [seq.seq,seq_numbers]
            internal_fasta_dict[int(seq_id)] = [sequence_no_stop_codons, seq_numbers]

    max_lenght_internal_aligned = max([int(len(sequence[0])) for idx, sequence in internal_fasta_dict.items()])  # Find the largest sequence without being aligned
    print("Creating aligned TEST simulation dataset...")
    dataset_test = np.zeros((len(internal_fasta_dict), max_lenght_internal_aligned + 2 , 30),dtype=object)
    for i, (key, val) in enumerate(internal_fasta_dict.items()):
        aligned_seq = list(internal_fasta_dict[key][0])
        dataset_test[i, 0, 1] = key  # name in the tree
        dataset_test[i, 0, 0] =  len(str(internal_fasta_dict[key][0]).replace("-","")) # Fill in the sequence lenght
        dataset_test[i, 2:,0] = internal_fasta_dict[key][1]

    leaves_names_test = internal_fasta_dict.keys()
    dataset_test = np.array(dataset_test, dtype="float64")
    dataset_test = torch.from_numpy(dataset_test)
    return dataset_test,leaves_names_test,max_lenght_internal_aligned


def load_coral_fluorescent_proteins_ancestral_sequences(name,ancestral_file,tree_levelorder_names,aa_probs):
    """Loads the 5 ancestral root nodes from the coral fluorescent proteins as the data set test.
    :param str name
    :param str ancestral_file
    :param list tree_levelorder_names
    :param int aa_probs
    """
    ancestral_fasta = SeqIO.parse(ancestral_file, "fasta")
    if name == "Coral_Faviina":
        root = "A35" #TODO: root detection system?
        nodes_dict = {"all-fav0":root,"all-fav1":root,"all-fav2":root,"all-fav3":root,"all-fav4":root}
    elif name == "Coral_all":
        root = "A71"
        nodes_dict = {"allcor0": root, "allcor1": root, "allcor2": root, "allcor3": root, "allcor4": root}
    elif name == "Cnidaria":
        print("Fix CFPTest for Cnidaria. The tree does not match the original, cannot be used")
        exit()
        ancestor_coral = ""
        ancestor_faviina = ""
        nodes_dict = {"allcor0": ancestor_coral, "allcor1": ancestor_coral, "allcor2": ancestor_coral, "allcor3": ancestor_coral, "allcor4": ancestor_coral,
                      "all-fav0":ancestor_faviina,"all-fav1":ancestor_faviina,"all-fav2":ancestor_faviina,"all-fav3":ancestor_faviina,"all-fav4":ancestor_faviina}

    aminoacid_names = DraupnirUtils.aminoacid_names_dict(aa_probs)
    test_nodes_names = []
    internal_fasta_dict ={}
    for seq in ancestral_fasta:
        if seq.id in nodes_dict.keys():
            seq_numbers = []
            for aa_name in seq.seq:
                #aa_number = int(np.where(np.array(aminoacid_names) == aa_name)[0][0]) + add_on
                aa_number = aminoacid_names[aa_name]
                seq_numbers.append(aa_number)
            id = nodes_dict[seq.id]
            seq_id = np.where(np.array(tree_levelorder_names) == id)[0][0]
            test_nodes_names.append(seq_id)
            internal_fasta_dict[seq.id] = [seq_id,seq.seq, seq_numbers]

    max_lenght_internal_aligned = max([int(len(sequence[1])) for idx, sequence in internal_fasta_dict.items()])  # Find the largest sequence without being aligned
    print("Creating aligned Coral Faviina dataset...")
    dataset_test = np.zeros((len(internal_fasta_dict), max_lenght_internal_aligned + 2, 30), dtype=object)
    for i, (key, val) in enumerate(internal_fasta_dict.items()):
        dataset_test[i, 0, 1] = int(val[0])  # name in the tree
        dataset_test[i, 0, 0] = len(str(internal_fasta_dict[key][1]).replace("-", ""))  # Fill in the sequence lenght
        dataset_test[i, 2:, 0] = internal_fasta_dict[key][2]

    return dataset_test.astype(float), test_nodes_names, max_lenght_internal_aligned,nodes_dict

