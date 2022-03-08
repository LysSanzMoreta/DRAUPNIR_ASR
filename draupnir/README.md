DRAUPNIR: "Beta library version for performing ASR using a tree-structured Variational Autoencoder"

##Extra requirements for tree inference:

IQ-Tree: http://www.iqtree.org/doc/Quickstart
```
conda install -c bioconda iqtree
```
RapidNJ: https://birc.au.dk/software/rapidnj
```
conda config --add channels bioconda
conda install rapidnj
```
#Extra requirements for fast patristic matrix construction

Install R (R version 4.1.2 (2021-11-01) -- "Bird Hippie"
) 
```
sudo apt update & sudo apt upgrade
sudo apt -y install r-base
```

together with ape 5.5 and TreeDist 2.3 libraries

```
install.packages(c("ape","TreeDist"))
```


#Draupnir Install

```
pip install draupnir
```

#Example
```
    import pyro
    import torch
    import draupnir
    import argparse
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pyro.enable_validation(False)
    use_cuda=True
    if use_cuda:
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch.set_default_tensor_type(torch.DoubleTensor)
        device = "cpu"
    draupnir.available_datasets(print_dict=True)
    build_config,settings_config, root_sequence_name = draupnir.create_draupnir_dataset("simulations_blactamase_1", #default dataset
                                                           use_custom=False, #default dataset
                                                           script_dir=script_dir,
                                                           build=False, # True: construct the dataset, False: use the stored dataset
                                                           fasta_file=None, # in this case, it will be read from /draupnir/src/data
                                                           tree_file=None, #in this case, it will be read from /draupnir/src/data
                                                           alignment_file=None) #in this case, #it will be read from /draupnir/src/data
    #draupnir.draw_tree_simple(args.dataset_name,settings_config) # to draw a tree, only after the dataset has been built
    draupnir.run(args.dataset_name,root_sequence_name,args,device,settings_config,build_config,script_dir)
```

#How long should I run my model?

1) While it is training:
   - Check for the Percent_ID.png plot, if the training accuracy has peaked to almost 100%, run for at least ~1000 epochs more to guarantee full learning
   - Check for stabilization of the error loss: ELBO_error.png
   - Check for stabilization of the entropy: Entropy_convergence.png
2) After training:
   - Observe the latent space: 
      1) t_SNE, UMAP and PCA plots: Is it organized by clades? Although, not every data set will present tight clustering of the tree clades though but there should be some organization
      2) Distances_GP_VAE_z_vs_branch_lengths_Pairwise_distance_INTERNAL_and_LEAVES plot: Is there a positive correlation? If there is not a good correlation but the train percent identity is high, it will still be a valid run 
   - Observe the sampled training (leaves) sequences and test (internal) sequences: Navigate to the Train_argmax and Test_argmax folders and look for the .fasta files
   - Calculate mutual information: 
     - First: Run Draupnir with the MAP & Marginal version and Variational version, or just the Variational 
     - Second: Use the draupnir.calculate_mutual_information() with the paths to the folders with the trained runs. 





