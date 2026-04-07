import itertools
import time
import datetime
from ete3 import Tree as TreeEte3
from ete3 import TreeStyle
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import functools
from collections import defaultdict



dataset = np.load("/home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src/draupnir/data/simulations_src_sh3_2/simulations_src_sh3_2_dataset_numpy_NOT_aligned_integers.npy",allow_pickle=True)


def calculate_patristic_cladistic(tree):
    nodes_and_leafs_names = [node.name for node in tree.traverse()]
    n_elements = len(nodes_and_leafs_names)

    I = pd.Index(nodes_and_leafs_names, name="rows")
    C = pd.Index(nodes_and_leafs_names, name="columns")
    patristic_matrix = pd.DataFrame(data=np.zeros((n_elements, n_elements)), index=I, columns=C)
    cladistic_matrix = pd.DataFrame(data=np.zeros((n_elements, n_elements)), index=I, columns=C)

    def calc_distance(t1,t2):
        cladistic_dist =  tree.get_distance(t1, t2, topology_only=True)
        patristic_dist = tree.get_distance(t1, t2, topology_only=False)
        return ((t1,t2,cladistic_dist,patristic_dist))

    nodes_and_leafs_names_i = []
    nodes_and_leafs_names_j = []
    for i, t1 in enumerate(nodes_and_leafs_names):
        for j, t2 in enumerate(list(nodes_and_leafs_names)[i + 1:]):
            nodes_and_leafs_names_i.append(t1)
            nodes_and_leafs_names_j.append(t2)

    start = time.time()
    results = Parallel(n_jobs=-2)(delayed(functools.partial(calc_distance))(t1,t2) for t1, t2 in itertools.zip_longest(nodes_and_leafs_names_i,nodes_and_leafs_names_j))

    # t1,t2,cladistic_distances,patristic_distances = list(zip(*results))
    # cladistic_matrix.loc[t1,t2]= cladistic_distances #todo: fix at some point
    # patristic_matrix.loc[t1,t2]= patristic_distances

    results_dict = defaultdict(list)
    for result in results: #group results by leaf
        results_dict[result[0]].append(result[1:])

    for key,vals in results_dict.items(): #we only iterate the leaves
        t2,cladistic_dist,patristic_dist = list(zip(*vals))
        patristic_matrix.loc[key,t2] = patristic_dist
        cladistic_matrix.loc[key,t2] = cladistic_dist

    stop = time.time()
    print("Current total time NEW: {}".format(str(datetime.timedelta(seconds=stop - start))))

    return patristic_matrix,cladistic_matrix


def calculate_pairwise_distance(dataset,batched=False):

    """"""

    node_names = dataset[:,0,0]
    #col_names = np.concatenate([np.array([float("-inf")]),node_names])
    dataset_sequences = dataset[:,3:,0]
    max_len = dataset_sequences.shape[1]
    dataset_sequences = dataset_sequences[:,:,None]

    pairwise_sim_fx = lambda d : (d[None,:] == d[:,None]).all((-1)).astype(float).sum(-1)/max_len

    if batched:#todo this would be dromi basically
        pass

    else:
        pairwise_sim = pairwise_sim_fx(dataset_sequences)
    pairwise_sim = pd.DataFrame(pairwise_sim,columns=node_names,index=node_names)
    #pairwise_sim = np.hstack([node_names[:,None],pairwise_sim])
    return pairwise_sim


print(calculate_pairwise_distance(dataset))


# toy_data = np.array([[1,4,5,6],[1,0,5,6],[0,6,5,6],[9,4,5,6]])
# calculate_pairwise_distance(toy_data)

