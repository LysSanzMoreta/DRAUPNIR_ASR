"""
2021: aleatoryscience
Lys Sanz Moreta
Draupnir : GP prior VAE for Ancestral Sequence Resurrection
=======================
"""
import argparse
import os.path
from Bio.PDB import MMCIFParser, PDBIO
import pnerf as pnerf
from collections import namedtuple
import pymol2
from superposition import *
import calculate_coords as nerf
sys.path.append("./draupnir/draupnir")
import Draupnir_utils as DraupnirUtils
import datetime
SamplingOutput = namedtuple("SamplingOutput",["aa_sequences","latent_space","logits","phis","psis","mean_phi","mean_psi","kappa_phi","kappa_psi"])
SuperpositionInput = namedtuple("SuperpositionInput",["observed_pdb","predicted_pdb"])


def write_transformed_PDB(initialPDB, Rotation, Translation, file_name, folder_location):
    ''' Transform by rotating and translating the atom coordinates from the original PDB file and rewrite it '''
    file_path = os.path.join(folder_location,file_name)

    try:
        parser = MMCIFParser()
        structure = parser.get_structure('%s' % (file_name), initialPDB)
    except:
        parser = PDB.PDBParser()
        structure = parser.get_structure('%s' % (file_name), initialPDB)

    for atom in structure.get_atoms():
        atom.transform(Rotation, Translation)
    io = PDBIO()
    io.set_structure(structure)
    io.save("{}".format(file_path))

def write_pdb_file(structure, file_name, file_location, only_calphas=True):
    """Write a completely new PDB file with the C_alpha trace coordinates extracted from the model. It adds an intermediate coordinate between the C_alpha atoms in order to be able
    to visualize a conected line in Pymol"""
    file_name = os.path.join(file_location,file_name)
    if os.path.isfile(file_name):
        os.remove(file_name)
    else:
        pass
    if only_calphas:
        # Create an expanded array: Contains an extra row between each C_alpha atom, for the intermediate coordinate
        expanded_structure = np.ones(shape=(2 * len(structure) - 1, 3))
        # Calculate the average coordinate between each C alpha atom
        averagearray = np.zeros(shape=(len(structure) - 1, 3))
        for index, row in enumerate(structure):
            if index != len(structure) and index != len(structure) - 1:
                averagearray[int(index)] = (structure[int(index)] + structure[int(index) + 1]) / 2
            else:
                pass
        # The even rows of the 'expanded structure' are simply the rows of the original structure
        # The odd rows are the intermediate coordinate
        expanded_structure[0::2] = structure
        expanded_structure[1::2] = averagearray
        structure = expanded_structure
        res_name = list("ALA")
        res_type = [" "] * (4 - len(list("CA"))) + list("CA")  # atom full name
        element = "C"
        for i in range(len(structure)):
            with open(file_name, 'a') as f:
                line = [" "] * 80
                line[0:6] = ["A", "T", "O", "M", " ", " "]
                line[6:11] = [" "] * (5 - len(list(str(i+1)))) + list(str(i+1))  # atom serial number,start counting at 1
                line[12:16] = res_type #residue type
                line[17:20] = res_name  # residue name
                line[21] = "A"  # chain
                line[22:26] = [" "] * (4 - len(list(str(i+1)))) + list(str(i+1))  # residue sequence number, here is a fake one
                line[26] = " "  # insertion code
                line[30:38] = [" "] * (8 - len(list("{0:.3f}".format(structure[i, 0])))) + list("{0:.3f}".format(structure[i, 0]))  # x
                line[38:46] = [" "] * (8 - len(list("{0:.3f}".format(structure[i, 1])))) + list("{0:.3f}".format(structure[i, 1]))  # y
                line[46:54] = [" "] * (8 - len(list("{0:.3f}".format(structure[i, 2])))) + list("{0:.3f}".format(structure[i, 2]))  # z
                line[54:60] = [" "] * (5 - len(list(str(1.02)))) + list(str(1.02))  # occupancy
                line[60:66] = [" "] * (5 - len(list(str(11.92)))) + list(str(11.92))  # bfactor?
                # line[72:76] = list(str("A1")) #segid
                line[77:78] = list(element)
                f.write("{}\n".format("".join(line)))
    else:
        res_name = list("ALA")
        res_type = ["N","CA","C"]*structure.shape[0]
        element = ["N","C","C"]*structure.shape[0]
        res_number = 0
        for i in range(len(structure)):
            if i % 3 == 0:
                res_number += 1
            with open(file_name, 'a') as f:
                line = [" "] * 80
                line[0:6] = ["A", "T", "O", "M", " ", " "]
                line[6:11] = [" "] * (5 - len(list(str(i+1)))) + list(str(i+1))  # atom serial number, start counting at 1
                line[12:16] = [" "] * (4 - len(list(res_type[i]))) + list(res_type[i])  # atom full name #residue type
                line[17:20] = res_name  # residue name
                line[21] = "A"  # chain
                line[22:26] = [" "] * (4 - len(list(str(res_number)))) + list(str(res_number))  # residue sequence number
                line[26] = " "  # insertion code
                line[30:38] = [" "] * (8 - len(list("{0:.3f}".format(structure[i, 0])))) + list("{0:.3f}".format(structure[i, 0]))  # x
                line[38:46] = [" "] * (8 - len(list("{0:.3f}".format(structure[i, 1])))) + list("{0:.3f}".format(structure[i, 1]))  # y
                line[46:54] = [" "] * (8 - len(list("{0:.3f}".format(structure[i, 2])))) + list("{0:.3f}".format(structure[i, 2]))  # z
                line[54:60] = [" "] * (5 - len(list(str(1.02)))) + list(str(1.02))  # occupancy
                line[60:66] = [" "] * (5 - len(list(str(11.92)))) + list(str(11.92))  # bfactor?
                # line[72:76] = list(str("A1")) #segid
                line[77:78] = list(element[i])
                f.write("{}\n".format("".join(line)))

def convert_to_coordinates(dihedrals,protein_len,folder_structures,protein_name):
    # Highlight: P-NERF
    points = pnerf.dihedral_to_point(dihedrals.unsqueeze(1))
    coordinates = pnerf.point_to_coordinate(points.float(),
                                                      num_fragments=6)  # [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS] = [len*3,n_prot,3]
    protein_coordinates = coordinates.reshape(protein_len, 3,3)
    carbon_alphas = protein_coordinates[:,1,:]  #if we consider that the second dimension is the backbone atoms (N,C-alpha,C), the middle one is C-alpha
    # Highlight: NERF
    # calculated_coords = nerf.place_residues(dihedrals) #list of tuples of len 3 that contain 3 arrays
    #TODO: Change to the other write_pdb_file
    protein_coordinates = torch.tensor_split(protein_coordinates,protein_coordinates.shape[0], dim=0)
    protein_coordinates = [torch.tensor_split(atom_array, 3, dim=1) for atom_array in protein_coordinates]
    protein_coordinates = [atom_array.view(-1).numpy() for position in protein_coordinates for atom_array in position]
    predicted_protein_coordinates = [(n, ca, cb) for n, ca, cb in three_tuple_split(protein_coordinates)]
    nerf.write_to_pdb(predicted_protein_coordinates,
                      filename="{}/{}.pdb".format(folder_structures, protein_name))
    return carbon_alphas, coordinates


def three_tuple_split(l):
    for i in range(0, len(l), 3):
        yield l[i:i + 3]
def load_angles(args):

    load_dict = torch.load("{}".format(args.load_file))

    type = ["test" if os.path.basename(args.load_file).startswith("test") else "train"][0]
    dataset = load_dict["dataset"]
    correspondence_dict = load_dict["correspondence_dict"]
    aa_predictions = load_dict["aa_predictions"].cpu()
    phis = load_dict["phis"].detach().cpu() #[n_samples,n_nodes,max_len ] // just 2 samples for now
    psis = load_dict["psis"].detach().cpu()
    omegas = torch.full(phis.shape,180)
    folder_structures = "/home/lys/Dropbox/PhD/DRAUPNIR/Draupnir_Structures"
    only_calphas = False

    for n in range(args.n_samples):
        # Highlight: convert to [n_steps,batch_size,n_dihedrals] = [len,n_proteins,3 angles]
        predicted_aa = aa_predictions[n].transpose(1,0).unsqueeze(2)
        observed_aa = dataset[:, 2:, 0]
        phi = phis[n].transpose(1,0).unsqueeze(2)
        psi = psis[n].transpose(1,0).unsqueeze(2)
        omega = omegas[n].transpose(1,0).unsqueeze(2)
        for p in range(args.n_proteins):
            protein_name = correspondence_dict[dataset[p, 0, 1].item()]
            print("################ Protein {} Sample {} #####################".format(protein_name,n))
            folder_name = "{}_{}_sample_{}_{}".format(protein_name,type,n,now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"))
            protein_folder = "{}/{}".format(folder_structures,folder_name)
            DraupnirUtils.folders(folder_name, folder_structures)
            #Highlight: Predicted protein
            predicted_dihedrals = torch.cat([phi[:,p].unsqueeze(1), psi[:,p].unsqueeze(1), omega[:,p].unsqueeze(1)], dim=2)  # [len,n_proteins,3]
            predicted_dihedrals = predicted_dihedrals[predicted_aa[:,p] !=0] #[len,3]remove angles assigned to gaps
            predicted_len = predicted_dihedrals.shape[0]
            predicted_carbon_alphas, predicted_atoms = convert_to_coordinates(predicted_dihedrals,predicted_len,protein_folder,"{}_predicted".format(protein_name))
            if only_calphas:
                predicted_atoms = predicted_carbon_alphas
            else:
                predicted_atoms = predicted_atoms.squeeze(1)
            #Highlight: Observed/Real protein
            observed_dihedrals = dataset[p,2:,1:3]
            observed_dihedrals = observed_dihedrals[observed_aa[p] != 0]
            observed_len = int(dataset[p, 0, 0].item())
            observed_dihedrals = torch.cat([observed_dihedrals.cpu(),torch.full((observed_dihedrals.shape[0],1),180)],dim=1)
            observed_carbon_alphas,observed_atoms = convert_to_coordinates(observed_dihedrals, observed_len, protein_folder,"{}_observed".format(protein_name))
            if only_calphas:
                observed_atoms = observed_carbon_alphas
            else:
                observed_atoms = observed_atoms.squeeze(1)
            print("Predicted len {}, Observed len {}".format(predicted_len,observed_len))
            if predicted_len == observed_len:
                #Highlight: Theseus superposition
                max_var = data_management.Max_variance(observed_atoms)  # calculate the max pairwise distance to origin of the structure to set as value for max variance in the prior for mean struct
                average = data_management.Average_Structure((observed_atoms,predicted_atoms))
                # print(predicted_atoms.shape)
                # print(observed_atoms.shape)
                data_obs = max_var, observed_atoms,predicted_atoms
                #Highlight: PDB parser https://github.com/biopython/biopython/blob/master/Bio/PDB/PDBParser.py
                T2, R, M, X1, X2, distances, info, duration = superposition_model.Run(data_obs, average, "{}".format(protein_name),protein_folder)
                write_pdb_file(M, '{}_M.pdb'.format(protein_name),protein_folder,only_calphas=only_calphas)
                write_pdb_file(X1, '{}_result_X1.pdb'.format("{}_observed".format(protein_name)),protein_folder, only_calphas=only_calphas)
                write_pdb_file(X2, '{}_result_X2.pdb'.format("{}_predicted".format(protein_name)),protein_folder, only_calphas=only_calphas)
                write_transformed_PDB("{}/{}_predicted.pdb".format(protein_folder,protein_name), np.transpose(R), -T2,"{}.pdb".format("{}_predicted_transformed".format(protein_name)),protein_folder)
                # create a new PyMOL instance
                p = pymol2.PyMOL()
                new_cmd = p.cmd  # get its cmd
                p.start()
                data_management.Pymol_newinstance(new_cmd,protein_folder,["{}/{}_result_X1.pdb".format(protein_folder,"{}_observed".format(protein_name)), "{}/{}_result_X2.pdb".format(protein_folder,"{}_predicted".format(protein_name))])
                p.stop()
            else:
                print("Predicted and observed sequences have different lengths")
                # create a new PyMOL instance, Highlight: Does not work in Pycharm, run from terminal
                p = pymol2.PyMOL()
                new_cmd = p.cmd  # get its cmd
                p.start()
                data_management.Pymol_newinstance(new_cmd,protein_folder,["{}/{}_observed.pdb".format(protein_folder,protein_name),"{}/{}_predicted.pdb".format(protein_folder,protein_name)])
                p.stop()

if __name__ == "__main__":
    now = datetime.datetime.now()
    #Highlight: Single GRU with 1 layer
    load_folder = "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_SH3_pf00018_larger_than_30aa_2021_08_12_13h00min42s587956ms_5epochs_NO_REMOVE/Train_Plots/train_info_dict.torch"
    #load_folder = "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_SH3_pf00018_larger_than_30aa_2021_08_12_13h00min42s587956ms_5epochs_NO_REMOVE/Test_Plots/test_info_dict.torch"
    #load_folder = "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_SH3_pf00018_larger_than_30aa_2021_08_10_20h15min38s132484ms_30000epochs_angles/Train_Plots/train_info_dict.torch"
    #Highlight: Double SRU with 1 layer each, open structures
    #load_folder = "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_SH3_pf00018_larger_than_30aa_2021_08_26_15h51min34s041768ms_15000epochs/Train_Plots/train_info_dict.torch"
    #Highlight: Double SRU with 3 layers each, okeish
    #load_folder = "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_SH3_pf00018_larger_than_30aa_2021_09_01_12h55min42s876184ms_5089_out_of_20000epochs/Train_Plots/train_info_dict.torch"
    #Highlight: Single SRU with 6 layers : okeish
    #load_folder = "/home/lys/Dropbox/PhD/DRAUPNIR/PLOTS_GP_VAE_SH3_pf00018_larger_than_30aa_2021_09_01_11h32min20s013854ms_11505_out_of_20000epochs/Train_Plots/train_info_dict.torch"

    parser = argparse.ArgumentParser(description="Draupnir 3D args")
    parser.add_argument('-f', '--load_file', default=load_folder, type=str, help='.torch file where the angles and other information asre stored')
    parser.add_argument('-n', '--n-samples', default=3, type=str,help='number of samples to superimpose/plot')
    parser.add_argument('-np', '--n-proteins', default=11, type=str, help='number of proteins to superimpose/plot')
    args = parser.parse_args()
    load_angles(args)
    #calling_superposition()