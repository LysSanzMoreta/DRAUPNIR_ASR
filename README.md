
DRAUPNIR: "Beta library version for performing ASR using a tree-structured Variational Autoencoder"

<p align="center">
<img src="https://github.com/LysSanzMoreta/DRAUPNIR_ASR/blob/main/draupnir/src/draupnir/images/draupnir_logo.png" height="auto" width="790" style="border-radius:50%">
</p>

**Extra requirements for tree inference:**

#These are not necessary if you have your own tree file

IQ-Tree: http://www.iqtree.org/doc/Quickstart
```
conda install -c bioconda iqtree
```
RapidNJ: https://birc.au.dk/software/rapidnj
```
conda config --add channels bioconda
conda install rapidnj
```
**Extra requirements for fast patristic matrix construction**

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

**Draupnir pip install**

```
pip install draupnir
```

**Example**
```
See Draupnir_example.py
```
**Which guide to use?**

By experience, use delta_map, the marginal results (Test folder) are the most stable.
It is recommended to run the model both with the variational and the delta_map guides and compare outputs using the mutual information.
If necessary run the variational guide longer than the delta_map, since it has more parameters to infere and takes longer.

**How long should I run my model?**

0) Before training: 
   - It is recommended to train for at least 10000 epochs in datasets <800 leaves. See article for inspiration, the runtimes where extended to achieve maximum benchmarking accuracy, but it should not be necessary.
1) While it is training:
   - Check for the Percent_ID.png plot, if the training accuracy has peaked to almost 100%, run for at least ~1000 epochs more to guarantee full learning
   - Check for stabilization of the error loss: ELBO_error.png
   - Check for stabilization of the entropy: Entropy_convergence.png
2) After training:
   - Observe the latent space: 
      1) t_SNE, UMAP and PCA plots: Is it organized by clades? Although, not every data set will present tight clustering of the tree clades though but there should be some organization
      <p align="center">
      <img src="https://github.com/LysSanzMoreta/DRAUPNIR_ASR/blob/main/draupnir/src/draupnir/images/LatentBlactamase.png" alt="Latent space" width="300" />
      </p>
      
      2) Distances_GP_VAE_z_vs_branch_lengths_Pairwise_distance_INTERNAL_and_LEAVES plot: Is there a positive correlation? If there is not a good correlation but the train percent identity is high, it will still be a valid run 
   - Observe the sampled training (leaves) sequences and test (internal) sequences: Navigate to the Train_argmax and Test_argmax folders and look for the .fasta files
   - Calculate mutual information: 
     - First: Run Draupnir with the MAP & Marginal version and Variational version, or just the Variational 
     - Second: Use the draupnir.calculate_mutual_information() with the paths to the folders with the trained runs. 
     
     ![alt text](https://github.com/LysSanzMoreta/DRAUPNIR_ASR/blob/main/draupnir/src/draupnir/images/MI.png)


**If this library is useful for your research please cite:**

```
@inproceedings{moreta2021ancestral,
  title={Ancestral protein sequence reconstruction using a tree-structured Ornstein-Uhlenbeck variational autoencoder},
  author={Moreta, Lys Sanz and R{\o}nning, Ola and Al-Sibahi, Ahmad Salim and Hein, Jotun and Theobald, Douglas and Hamelryck, Thomas},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```









