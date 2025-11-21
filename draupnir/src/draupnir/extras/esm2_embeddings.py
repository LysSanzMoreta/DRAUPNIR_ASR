import os
import ntpath
import torch
from Bio import SeqIO
import sys
import numpy as np
import dill
import umap
local_repository=True
if local_repository:
    sys.path.insert(1,"/home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src/draupnir/esm-main")
    import esm
else:#pip installed module
    import esm

from esm import pretrained
import matplotlib.pyplot as plt

local_repository=True
if local_repository:
    sys.path.insert(1,"/home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src")
    import draupnir
else:#pip installed module
    import draupnir

import draupnir.load_utils as DraupnirLoadUtils


def read_fasta(args):
    """Read sequences from a FASTA file."""
    sequences = []
    for record in SeqIO.parse(args.fasta, "fasta"):
        sequences.append((str(record.id), str(record.seq)))
    return sequences

def generate_esm2_embeddings(args):
    """Generate ESM-2 embeddings for a list of sequences.

    ESM2(
  (embed_tokens): Embedding(33, 1280, padding_idx=1)
  (layers): ModuleList(
    (0-32): 33 x TransformerLayer(
      (self_attn): MultiheadAttention(
        (k_proj): Linear(in_features=1280, out_features=1280, bias=True)
        (v_proj): Linear(in_features=1280, out_features=1280, bias=True)
        (q_proj): Linear(in_features=1280, out_features=1280, bias=True)
        (out_proj): Linear(in_features=1280, out_features=1280, bias=True)
        (rot_emb): RotaryEmbedding()
      )
      (self_attn_layer_norm): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
      (fc1): Linear(in_features=1280, out_features=5120, bias=True)
      (fc2): Linear(in_features=5120, out_features=1280, bias=True)
      (final_layer_norm): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
    )
  )
  (contact_head): ContactPredictionHead(
    (regression): Linear(in_features=660, out_features=1, bias=True)
    (activation): Sigmoid()
  )
  (emb_layer_norm_after): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
  (lm_head): RobertaLMHead(
    (dense): Linear(in_features=1280, out_features=1280, bias=True)
    (layer_norm): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
      )
    )

    """

    sequences_name = ntpath.dirname(args.fasta).split("/")[-1]

    sequences = read_fasta(args)


    # Load the model
    model_name = "esm2_t33_650M_UR50D"
    #model_name = "esm2_t36_3B_UR50D"


    output_name = f"{sequences_name}_{model_name}_embeddings.npy"

    if not os.path.exists(output_name) or args.overwrite:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_name == "esm2_t33_650M_UR50D":
            model, alphabet = pretrained.esm2_t33_650M_UR50D() #esm1b_t33_650M_UR50S #esm2_t36_3B_UR50D()
        elif model_name == "esm2_t36_3B_UR50D":
            model, alphabet = pretrained.esm2_t36_3B_UR50D()
            device = "cpu"
        batch_converter = alphabet.get_batch_converter()
        model.eval()
        node_names = np.array(list(zip(*sequences))[0])


        batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)


        # Run the model
        # Extract per-residue representations
        model = model.to(device)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad(): #todo: return contacts is very expensive, do not return unless necessary
            results = model(batch_tokens, repr_layers=[33], return_contacts=False,need_head_weights=False)

        token_representations = results["representations"][33].detach().cpu().numpy().astype(object) #[N,L,1280]

        embeddings = np.pad(token_representations, pad_width=((0, 0), (1, 0), (0, 0)), mode='constant') #adds padding of zeroes to get [N, 2 + L, feat_dim]
        embeddings[:,0,0] = node_names


        # Generate per-sequence embeddings (mean pooling): the sequence representations average across the length dimension
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            average = token_representations[i, 1: tokens_len - 1].mean(0)
            average = np.hstack([node_names[i],average])
            sequence_representations.append(average)

        # Look at the unsupervised self-attention map contact predictions
        if "contacts" in results.keys():
            for (_, seq), tokens_len, attention_contacts in zip(sequences, batch_lens, results["contacts"]):
                plt.matshow(attention_contacts[: tokens_len, : tokens_len])
                plt.title(seq)
                plt.show()
        else:
            print("Contacts not computed")

        np.save(output_name,embeddings)
        np.save(f"{sequences_name}_{model_name}_sequence_representations.npy",sequence_representations)

        return embeddings, sequence_representations

    else:
        embeddings = np.load(output_name,allow_pickle=True)

        sequence_representations = np.load(f"{sequences_name}_{model_name}_sequence_representations.npy",allow_pickle=True)

        return embeddings,sequence_representations




def plot_latent_space_umap_by_clade_leaves(latent_space, clades_dict_all, results_dir):
    """UMAP projection of a z-dimensional latent space onto a 2D space. The latent space represents the sequences in the tree. The latent space is coloured according to
    the clade membership
    :param tensor latent_space: [n_leaves,1+ z_dim], first column contains the nodes indexes
    :param namedtuple additional_load
    :param int epoch
    :param str results_dir
    :param bool triTSNE: boolean paramter to indicate whether to perform a 3D latent space projection, Not used right now
    """
    # Create a two dimensional t-SNE projection of the z dim latent space
    print("Building UMAP plot COLOURED by clades")
    #annotate = [True if latent_space.shape[0] < 100 else False][0]
    annotate = False
    stripped = True
    #n_cols = DraupnirUtils.Define_batch_size(latent_space.shape[0], batch_size=False,benchmarking=True)
    reducer = umap.UMAP()
    umap_proj =  reducer.fit_transform(latent_space[:,1:])
    #tsne_proj = TSNE(n_components=2).fit_transform(latent_space[:, 1:])


    color_map_name = "nipy_spectral" if len(clades_dict_all) > 148 else "148colormap" if len(clades_dict_all) > 21 else "21colormap"
    clrs = plt.get_cmap(color_map_name, len(clades_dict_all))
    #n_cols = int(len(clades_dict_all.keys()) / 30) + 1
    if stripped:
        fig, ax = plt.subplots(figsize=(22, 20))
        for idx, (clade, nodes) in enumerate(clades_dict_all.items()): #the clades_dict_all is transformed to contain the indexes in the original, here is the letters directly
            sequences = list(nodes["leaves"])
            indexes = (latent_space[:, 0][..., None] == sequences).any(-1)
            ax.scatter(umap_proj[indexes, 0], umap_proj[indexes, 1], color=clrs(idx), label=clade, alpha=1, s=700)
            if annotate:
                for name, point in zip(sequences, umap_proj[indexes]):
                    ax.annotate(name, xy=(point[0], point[1]), size=7)  # xytext=(1,1)
        #plt.legend(loc='upper left', prop={'size': 25},ncol=1, shadow=True)
        plt.tight_layout()
        plt.axis("off")
    else:
        fig, ax = plt.subplots(figsize=(22, 20))

        for idx,(clade, nodes) in enumerate(clades_dict_all.items()):
                sequences = nodes["leaves"]
                indexes = (latent_space[:, 0][..., None] == sequences).any(-1)
                ax.scatter(umap_proj[indexes, 0], umap_proj[indexes, 1], color=clrs(idx), label=clade, alpha=1)
                if annotate:
                    for name, point in zip(sequences, umap_proj[indexes]):
                        ax.annotate(name, xy=(point[0], point[1]), size=7)  # xytext=(1,1)
        plt.legend(title='Clades', bbox_to_anchor=(1.01, 1), loc='upper left', prop={'size': 10},ncol=1, shadow=True,fontsize=10)

    plt.title("UMAP projection of ESM's sequence representations coloured by clade",fontsize=20)
    plt.tight_layout(pad=3.0)
    plt.savefig("{}/UMAP_z_space_by_clade_only_leaves.png".format(results_dir))



def main(args):

    embeddings, sequence_representations = generate_esm2_embeddings(args)

    clades_dict_all = DraupnirLoadUtils.load_serialized(open(args.clades_dict,"rb"))


    plot_latent_space_umap_by_clade_leaves(sequence_representations, clades_dict_all, args.output_dir)






if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", default="/home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src/draupnir/data/simulations_src_sh3_3/SRC_SH3_pep_Unaligned.FASTA", type=str, help="Input FASTA file")
    parser.add_argument("--clades-dict", default="/home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src/draupnir/data/simulations_src_sh3_3/simulations_src_sh3_3_Clades_dict_all.p", type=str, help="")
    parser.add_argument("--output-dir", default="/home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src/draupnir/extras", type=str, help="Output torch file")
    parser.add_argument("--overwrite", default="", type=str, help="overwrite embeddings")

    args = parser.parse_args()
    main(args)