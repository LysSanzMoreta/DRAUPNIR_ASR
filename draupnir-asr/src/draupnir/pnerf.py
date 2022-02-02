"""
pNeRF algorithm for parallelized conversion from torsion (dihedral) angles to
Cartesian coordinates implemented with PyTorch.
Reference implementation in tensorflow by Mohammed AlQuraishi:
    https://github.com/aqlaboratory/pnerf/blob/master/pnerf.py
Paper (preprint) by Mohammed AlQuraishi:
    https://www.biorxiv.org/content/early/2018/08/06/385450
PyTorch implementation by Felix Opolka
"""
import math
import collections
import numpy as np
import torch
import torch.nn.functional as F

# Constants
NUM_DIMENSIONS = 3 #number of atoms? N, C_alpha, C
NUM_DIHEDRALS = 3
BOND_LENGTHS = np.array([145.801, 152.326, 132.868], dtype=np.float32)
BOND_ANGLES = np.array([2.124, 1.941, 2.028], dtype=np.float32)


def dihedral_to_point(dihedral, bond_lengths=BOND_LENGTHS,
                      bond_angles=BOND_ANGLES):
    """
    Takes triplets of dihedral angles (phi, psi, omega) and returns 3D points
    ready for use in reconstruction of coordinates.SRF? Bond lengths and angles
    are based on idealized averages.
    :param dihedral: [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]
    :return: Tensor containing points of the protein's backbone atoms.
    Shape [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS], where NUM_STEPS = Protein lenght, batch_size = number of proteins, num_dim = num_atoms
    """
    num_steps = dihedral.shape[0]
    batch_size = dihedral.shape[1]
    # c_tilde = [ [r.cos(theta)],[r.cos(torsion).sin(theta)],[r.sin(torsion).sin(theta)]], Eq.1 , torsion angles = dihedral angles

    r_cos_theta = torch.tensor(bond_lengths * np.cos(np.pi - bond_angles)) # r.cos(theta)
    r_sin_theta = torch.tensor(bond_lengths * np.sin(np.pi - bond_angles)) #r.sin(theta)

    point_x = r_cos_theta.view(1, 1, -1).repeat(num_steps, batch_size, 1)  # [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]---> same r.cos(theta) (repeat shape [1,1,3])across the length of the protein
    point_y = torch.cos(dihedral) * r_sin_theta  # [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS] #multiply all the dihedral angles by r.sin(theta)
    point_z = torch.sin(dihedral) * r_sin_theta  # [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]
    point = torch.stack([point_x, point_y, point_z]) # [NUM_DIMS, NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]  torch.stack: Concatenates a sequence of tensors along a NEW dimension,the new dim = atom?

    point_perm = point.permute(1, 3, 2, 0)
    point_final = point_perm.contiguous().view(num_steps*NUM_DIHEDRALS,
                                               batch_size,
                                               NUM_DIMENSIONS)

    return point_final#c_tilde?


def point_to_coordinate(points, num_fragments=6):
    """
    Takes points from dihedral_to_point and sequentially converts them into
    coordinates of a 3D structure.

    Reconstruction is done in parallel by independently reconstructing
    num_fragments and the reconstituting the chain at the end in reverse order.
    The core reconstruction algorithm is NeRF, based on
    DOI: 10.1002/jcc.20237 by Parsons et al. 2005.
    The parallelized version is described in
    https://www.biorxiv.org/content/early/2018/08/06/385450.
    :param points: Tensor containing points as returned by `dihedral_to_point`.
    Shape [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]
    :param num_fragments: Number of fragments in which the sequence is split
    to perform parallel computation.
    :return: Tensor containing correctly transformed atom coordinates.
    Shape [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]
    """

    # Compute optimal number of fragments if needed
    total_num_angles = points.shape[0] # NUM_STEPS x NUM_DIHEDRALS
    if num_fragments is None:
        num_fragments = int(math.sqrt(total_num_angles))


    # Initial three coordinates (specifically chosen to eliminate need for
    # extraneous matmul)
    Triplet = collections.namedtuple('Triplet', 'a, b, c') #collections.namedtuple('Triplet', 'n, ca, c')---better naming even if they are fake
    batch_size = points.shape[1] #number of proteins
    # init_matrix = np.array([[-np.sqrt(1.0 / 2.0), np.sqrt(3.0 / 2.0), 0],
    #                      [-np.sqrt(2.0), 0, 0], [0, 0, 0]],
    #                     dtype=np.float32) #equation 4 transposed?
    # init_matrix = torch.from_numpy(init_matrix)
    #Highlight: init matrix are the initial coordinates for the first 3 atoms
    init_matrix = torch.tensor([[-np.sqrt(1.0 / 2.0), np.sqrt(3.0 / 2.0), 0],
                                [-np.sqrt(2.0), 0, 0],
                                [0, 0, 0]],dtype=torch.float32) #equation 4 transposed?

    #Highlight: repeat the initial triplet of coordinates for as many fragments and proteins as there are
    init_coords = [row.repeat([num_fragments * batch_size, 1]).view(num_fragments, batch_size, NUM_DIMENSIONS) for row in init_matrix]
    #Highlight: assigning the the initial coordinates to the named tuple
    init_coords = Triplet(*init_coords)                                     # NUM_DIHEDRALS x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS]

    # Highlight: Pad points to yield equal-sized fragments, makes the length of the sequence dividible by n_fragments
    padding = ((num_fragments - (total_num_angles % num_fragments))% num_fragments)   # NUM_FRAGMENTS -((NUM_STEPS x NUM_DIHEDRALS)%NUM-FRAGMENTS)%NUM_FRAGMENTS
    points = F.pad(points, (0, 0, 0, 0, 0, padding))                                  # [NUM_STEPS*N_DIHEDRALS + PADDING, BATCH_SIZE, NUM_DIMENSIONS]
    points = points.view(num_fragments, -1, batch_size, NUM_DIMENSIONS)               # [NUM_FRAGS, FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS], FRAG_SIZE = (NUM_STEPS*N_DIHEDRALS + PADDING)%NUM_FRAGS
    points = points.permute(1, 0, 2, 3)                                               # [FRAG_SIZE, NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS]

    # Extension function used for single atom reconstruction and whole fragment
    # alignment
    def extend(prev_three_coords, point, multi_m):
        """
        Aligns an atom or an entire fragment depending on value of `multi_m`
        with the preceding three atoms.
        :param prev_three_coords: Named tuple storing the last three atom
        coordinates ("a", "b", "c") where "c" is the current end of the
        structure (i.e. closest to the atom/ fragment that will be added now).
        Shape NUM_DIHEDRALS x [NUM_FRAGS/0, BATCH_SIZE, NUM_DIMENSIONS].
        First rank depends on value of `multi_m`.
        :param point: Point describing the atom that is added to the structure.
        Shape [NUM_FRAGS/FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS]
        First rank depends on value of `multi_m`.
        :param multi_m: If True, a single atom is added to the chain for
        multiple fragments in parallel. If False, an single fragment is added.
        Note the different parameter dimensions.
        :return: Coordinates of the atom/ fragment.
        """
        #HIGHLIGHT: COMPONENTS FOR M at EQ.4 at https://onlinelibrary.wiley.com/doi/10.1002/jcc.20237
        # M = [bc,nxbc,n]

        bc = F.normalize(prev_three_coords.c - prev_three_coords.b, dim=-1) # vector the last 2 atoms ck-1 and ck-2, [num_fragments, batch_size, num_dims]
        n = F.normalize(torch.cross(prev_three_coords.b - prev_three_coords.a,bc), dim=-1)

        if multi_m:     # multiple fragments, one atom at a time
            m = torch.stack([bc, torch.cross(n, bc), n]).permute(1, 2, 3, 0)
        else:           # single fragment, reconstructed entirely at once.
            s = point.shape + (3,) #[frag size, 1, 3, 3]
            m = torch.stack([bc, torch.cross(n, bc), n]).permute(1, 2, 0)
            m = m.repeat(s[0], 1, 1).view(s)
        #highlight: d = m.d_2 + c
        coord = torch.squeeze(torch.matmul(m, point.unsqueeze(3)),dim=3) + prev_three_coords.c #[n_fragments,batch_size,n_coord], 3 start
        return coord

    # Loop over FRAG_SIZE in NUM_FRAGS parallel fragments, sequentially
    # generating the coordinates for each fragment across all batches
    coords_list = [None] * points.shape[0]                                  # FRAG_SIZE = (NUM_STEPS*N_DIHEDRALS + PADDING)%NUM_FRAGS
    prev_three_coords = init_coords

    for i in range(points.shape[0]):    # Iterate over FRAG_SIZE
        coord = extend(prev_three_coords, points[i], True) # points[i] = [n_fragments,1,n_dih]
        coords_list[i] = coord
        prev_three_coords = Triplet(prev_three_coords.b, # a becames b
                                    prev_three_coords.c, # b becomes c
                                    coord) # c becomes d, the new atom

    coords_pretrans = torch.stack(coords_list).permute(1, 0, 2, 3)

    # Loop backwards over NUM_FRAGS to align the individual fragments. For each
    # next fragment, we transform the fragments we have already iterated over
    # (coords_trans) to be aligned with the next fragment
    coords_trans = coords_pretrans[-1]
    for i in reversed(range(coords_pretrans.shape[0]-1)):
        # Transform the fragments that we have already iterated over to be
        # aligned with the next fragment `coords_trans`
        transformed_coords = extend(Triplet(*[di[i]
                                              for di in prev_three_coords]),
                                    coords_trans, False)
        coords_trans = torch.cat([coords_pretrans[i], transformed_coords], 0)

    coords = F.pad(coords_trans[:total_num_angles-1], (0, 0, 0, 0, 1, 0))

    return coords