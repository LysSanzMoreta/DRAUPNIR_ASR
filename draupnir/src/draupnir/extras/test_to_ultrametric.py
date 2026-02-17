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

tree_file = "/home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src/draupnir/data/simulations_src_sh3_3/SRC_SH3_True_Rooted_tree_node_labels.tre"

tree = TreeEte3(tree_file,format=1,quoted_node_names=True)


# ts = TreeStyle()
# ts.show_leaf_name = True
# ts.show_branch_length = True
# ts.show_branch_support = True
# tree.show(tree_style=ts)




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


patristic_matrix,cladistic_matrix = calculate_patristic_cladistic(tree)


exit()
print(f"{tree.get_farthest_leaf()}")
diameter = patristic_matrix.max().max()
print(f"diameter: {diameter}")

print("-------------------------------")
#directly convert the patristic matrix to ultrametric #https://www.r-bloggers.com/2021/07/three-ways-to-check-and-fix-ultrametric-phylogenies/

#print(patristic_matrix)

root_row = patristic_matrix.loc["1",:]



#leaves_matrix = patristic_matrix.loc[patristic_matrix.columns.str.startswith("A"),patristic_matrix.columns.str.startswith("A")]
diameter = patristic_matrix.max().max()
print(f"diameter: {diameter}")
tree_height = root_row.max()
print(f"tree height: {tree_height}")


print("-------------------------------")

tree2 = tree.copy()

tree2.convert_to_ultrametric(tree_length=1)
patristic_matrix,cladistic_matrix = calculate_patristic_cladistic(tree2)

diameter = patristic_matrix.max().max()
print(f"diameter: {diameter}")
print(f"{tree2.get_farthest_leaf()}")






#
#
# np.testing.assert_array_equal(patristic_matrix1, patristic_matrix2)
# np.testing.assert_array_equal(patristic_matrix1, patristic_matrix2)


