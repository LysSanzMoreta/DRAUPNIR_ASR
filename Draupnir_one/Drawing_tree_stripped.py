from ete3 import Tree, faces, AttrFace, TreeStyle, NodeStyle, Tree, TextFace
# from PyQt4 import QtCore
# from PyQt4.QtGui import QGraphicsRectItem, QGraphicsSimpleTextItem, \
#     QGraphicsEllipseItem, QColor, QPen, QBrush
import matplotlib.pyplot as plt
import matplotlib
import pickle, os

def Renaming():
    "Rename the internal nodes, unless the given newick file already has the names on it"
    # Rename the internal nodes
    leafs_names = tree.get_leaf_names()
    edge = len(leafs_names)
    internal_nodes_names = []
    for node in tree.traverse():  # levelorder (nodes are visited in zig zag order from root to leaves)
        if not node.is_leaf():
            node.name = "A%d" % edge
            internal_nodes_names.append(node.name)
            edge += 1


def Renaming_simulations(with_indexes=False):
    "Rename the internal nodes to I + number, in simulations the leaves have the prefix A. With indexes shows the names used when transferring to an array"
    print("Renaming tree from simulations")
    leafs_names = tree.get_leaf_names()
    edge = len(leafs_names)
    internal_nodes_names = []
    if with_indexes:
        for idx, node in enumerate(
                tree.traverse()):  # levelorder (nodes are visited in zig zag order from root to leaves)
            if not node.is_leaf():
                node.name = "I" + node.name + "/{}".format(idx)
                internal_nodes_names.append(node.name)
                edge += 1
            else:
                node.name = node.name + "/{}".format(idx)
                # edge += 1

    else:
        for node in tree.traverse():  # levelorder (nodes are visited in zig zag order from root to leaves)
            if not node.is_leaf():
                node.name = "I" + node.name
                internal_nodes_names.append(node.name)
                edge += 1

def my_layout(node):
    "Adds the internal nodes names"
    if node.is_leaf():
        # If terminal node, draws its name
        name_face = AttrFace("name",fsize=8,fgcolor="black")
    else:
        # If internal node, draws label with smaller font size
        name_face = AttrFace("name", fsize=8,fgcolor="black")
    # Adds the name face to the image at the preferred position
    faces.add_face_to_node(name_face, node, column=0, position="branch-right")
def colour_tree_by_clades(clades_dict, tree, rename_internal_nodes=True):
    color_map21 = matplotlib.colors.ListedColormap(
        ["plum", "navy", "turquoise", "peachpuff", "palevioletred", "red", "darkorange", "yellow", "lime", "green",
         "dodgerblue", "blue", "purple", "magenta", "grey", "maroon", "lightcoral", "olive", "teal", "goldenrod",
         "black"])

    color_map_name = [color_map21 if len(clades_dict_all) <= 21 else 'gist_rainbow'][0]
    clrs = plt.get_cmap(color_map_name, len(clades_dict))
    clrs = [matplotlib.colors.rgb2hex(clrs(i)) for i in range(clrs.N)]
    node_styles = []
    for color in clrs:
        ns = NodeStyle()
        ns["fgcolor"] = color #tips color
        ns["size"] = 6
        ns["vt_line_width"] = 1
        ns["hz_line_width"] = 1
        node_styles.append(ns)
    for node_style,(clade, nodes) in zip(node_styles,clades_dict.items()):
        clade_ancestor = tree.get_common_ancestor(nodes["leaves"])
        clade_ancestor.set_style(node_style)
        for node in clade_ancestor.get_descendants():
            node.set_style(node_style)

    if rename_internal_nodes:
        if name.startswith("simulations"): Renaming_simulations(with_indexes=True)
        else:Renaming()
    ts = TreeStyle()
    # Do not add leaf names automatically
    ts.show_leaf_name = False
    # Use my custom layout
    #ts.layout_fn = my_layout
    ts.show_branch_length = False
    ts.show_scale = False
    #new stuff
    # ts.legend= False
    # ts.force_topology = True
    # ts.show_border = True
    # ts.scale_length = False
    try:
        tree.render("Tree_Pictures/return_{}_colored_by_clades_stripped.png".format(name), w=1000, units="mm", tree_style=ts)
    except:
        tree.render("Tree_Pictures/return_{}_colored_by_clades_stripped.png".format(name), w=1000, units="mm")




if __name__ == "__main__":
    #t, ts = get_example_tree()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    datasets = {0:["benchmark_randall", None, None, None],  #the tree is inferred
                1:["benchmark_randall_original",None, None,None],  #uses the original tree but changes the naming of the nodes (because the original tree was not rooted)
                2:["benchmark_randall_original_naming",None,None,None],  #uses the original tree and it's original node naming
                3:["SH3_pf00018_larger_than_30aa",None,None,None],  #SRC kinases domain SH3 ---> Leaves and angles testing
                4:["simulations_blactamase_1",1,"BLactamase","BetaLactamase_seq"],  #EvolveAGene4 Betalactamase simulation # 32 leaves
                5:["simulations_src_sh3_1",1, "SRC_simulations","SRC_SH3"],  #EvolveAGene4 SRC SH3 domain simulation 1 #100 leaves
                6:["simulations_src_sh3_2",2, "SRC_simulations","SRC_SH3"],  #EvolveAGene4 SRC SH3 domain simulation 2 #800 leaves
                7: ["simulations_src_sh3_3", 3, "SRC_simulations", "SRC_SH3"],# EvolveAGene4 SRC SH3 domain simulation 2 #200 leaves
                8: ["simulations_sirtuins_1",1, "Sirtuin_simulations", "Sirtuin_seq"],  # EvolveAGene4 Sirtuin simulation #150 leaves
                9: ["simulations_calcitonin_1", 1, "Calcitonin_simulations", "Calcitonin_seq"],# EvolveAGene4 Calcitonin simulation #50 leaves
                10: ["simulations_mciz_1",1, "Mciz_simulations","Mciz_seq"],  # EvolveAGene4 MciZ simulation # 1600 leaves
                11:["Douglas_SRC",None,None,None],  #Douglas's Full SRC Kinases #Highlight: The tree is not similar to the one in the paper, therefore the sequences where splitted in subtrees according to the ancetral sequences in the paper
                12:["ANC_A1_subtree",None,None,None],  #highlight: 3D structure not available #TODO: only working at dragon server
                13:["ANC_A2_subtree",None,None,None],  #highlight: 3D structure not available
                14:["ANC_AS_subtree",None,None,None],
                15:["ANC_S1_subtree",None,None,None],  #highlight: 3D structure not available
                16:["Coral_Faviina",None,None,None],  #Faviina clade from coral sequences
                17:["Coral_all",None,None,None],  # All Coral sequences (includes Faviina clade and additional sequences)
                18:["Cnidarian",None,None,None],  # All Coral sequences plus other fluorescent cnidarians #Highlight: The tree is too different to certainly locate the all-coral / all-fav ancestors
                19:["PKinase_PF07714",None,None,None],
                20:["simulations_CNP_1", 1, "CNP_simulations", "CNP_seq"],  # EvolveAGene4 CNP simulation # 1000 leaves
                21:["simulations_insulin_1", 1, "Insulin_simulations", "Insulin_seq"],  # EvolveAGene4 Insulin simulation #50 leaves
                22:["PF01038_msa",None,None,None],
                }
    name,dataset_number,simulation_folder,root_sequence_name = datasets[9]
    clades_dict_all = pickle.load(open('{}/Mixed_info_Folder/{}_Clades_dict_all.p'.format(script_dir, name), "rb"))
    if name.startswith("simulations"):
        tree_file = "Datasets_Simulations/{}/Dataset{}/{}_True_Rooted_tree_node_labels.tre".format(simulation_folder,dataset_number,root_sequence_name)
    elif name == "benchmark_randall_original_naming":
        tree_file = "AncestralResurrectionStandardDataset/RandallBenchmarkTree_OriginalNaming.tree"
    elif name.endswith("_subtree"):
        tree_file = "/home/lys/Dropbox/PhD/DRAUPNIR/Douglas_SRC_Dataset/{}/{}_ALIGNED.mafft.treefile".format(name,name.replace("_subtree",""))
    else:
        tree_file = "Mixed_info_Folder/{}.mafft.treefile".format(name)
    tree = Tree(tree_file, format=1, quoted_node_names=True)
    colour_tree_by_clades(clades_dict_all,tree)
