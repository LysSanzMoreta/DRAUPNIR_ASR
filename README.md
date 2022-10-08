
DRAUPNIR: "Beta library version for performing ASR using a tree-structured Variational Autoencoder"

<p align="center">
<img src="https://github.com/LysSanzMoreta/DRAUPNIR_ASR/blob/main/draupnir/src/draupnir/images/draupnir_logo.png" height="auto" width="790" style="border-radius:50%">
</p>

**Extra requirements for tree inference:**

#These are NOT necessary if you have your own tree file or for using the default datasets

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

#Recommended if you have more than 200 sequences. The patristic matrix is constructed only once

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


**Datasets**
#They are recommended to use with the pipeline, look into datasets.py for more details
```
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
```
|                      Dataset                      | Number of leaves | Alignment lenght | Name                              |
|:-------------------------------------------------:|:----------------:|:----------------:|-----------------------------------|
| Randall's Coral fluorescent proteins (CFP)        | 19               | 225              | benchmark_randall_original_naming |
| Coral fluorescent proteins (CFP) Faviina subclade | 35               | 361              | Coral_Faviina                     |
| Coral fluorescent proteins (CFP) subclade         | 71               | 272              | Coral_all                         |
| Simulation $\beta$-Lactamase                      | 32               | 314              | simulations_blactamase_1          |
| Simulation Calcitonin                             | 50               | 71               | simulations_calcitonin_1          |
| Simulation SRC-Kinase SH3 domain                  | 100              | 63               | simulations_src_sh3_1             |
| Simulation Sirtuin                                | 150              | 477              | simulations_sirtuins_1            |
| Simulation SRC-kinase SH3 domain                  | 200              | 128              | simulations_src_sh3_3             |
| Simulation PIGBOS                                 | 300              | 77               | simulations_PIGBOS_1              |
| Simulation Insulin                                | 400              | 558              | simulations_insulin_2             |
| Simulation SRC-kinase SH3 domain                  | 800              | 99               | simulations_src_sh3_2             |


**What do the folders mean?**

1) If you selected **delta_map** guide:
   1) Train_Plots: Contains information related to the inference of the train sequences (the leaves). They are samples obtained by using the MAP estimates of the logits.
   2) Train_argmax_Plots: Single sequence per leaf obtained by the using the most likely amino acids indicated by the logits ("argmax the logits")
   3) Test_Plots: Samples for the test sequences (ancestors). In this case they contain the sequences sampled using the marginal probability approach (equation 5 in the paper)
   4) Test_argmax_Plots: Contains the most voted sequence from the samples in Test_Plots.
   5) Test2_Plots: Samples for the test sequences (ancestors). In this case they contain the sequences sampled using the MAP estimated of the logits. 
   6) Test2_argmax_Plots:  Samples for the test sequences (ancestors). In this case they contain the most likely amino acids indicated by the logits ("argmax the logits") (equation 4 in the paper)
2) If you selected **variational** guide:
   1) Train_Plots: Contains information related to the inference of the train sequences (the leaves). They are samples obtained by using the MAP estimates of the logits.
   2) Train_argmax_Plots: Single sequence per leaf obtained by the using the most likely amino acids indicated by the logits ("argmax the logits")
   3) Test_Plots: Samples for the test sequences (ancestors). In this case they contain the sequences sampled using the full variational probability approach (equation 6 in the paper)
   4) Test_argmax_Plots: Contains the most voted sequence from the samples in Test_Plots.
   5) Test2_Plots == Test_Plots 
   6) Test2_argmax_Plots == Test_argmax_Plots

**Where are my ancestral sequences?**

- In each of the folders there should be a fasta file <dataset-name>_sampled_nodes_seq.fasta

- Each of the sequences in the file should be identified as <node-name-input-tree>//<tree-level-order>\_sample\_<sample-number>

    >Node_A1//1.0_sample_0

    
**If this library is useful for your research please cite:**

```
@inproceedings{moreta2021ancestral,
  title={Ancestral protein sequence reconstruction using a tree-structured Ornstein-Uhlenbeck variational autoencoder},
  author={Moreta, Lys Sanz and R{\o}nning, Ola and Al-Sibahi, Ahmad Salim and Hein, Jotun and Theobald, Douglas and Hamelryck, Thomas},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```









