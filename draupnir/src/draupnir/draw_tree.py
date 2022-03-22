"""
=======================
2022: Lys Sanz Moreta
Draupnir : Ancestral protein sequence reconstruction using a tree-structured Ornstein-Uhlenbeck variational autoencoder
=======================
"""
#from ete3 import Tree, faces, AttrFace, TreeStyle, NodeStyle, Tree, TextFace
try:
    from ete3 import Tree, faces, AttrFace, TreeStyle,NodeStyle,TextFace
except:
    pass

import matplotlib.pyplot as plt
import matplotlib
import pickle, os

def renaming(tree):
    """Rename the internal nodes, unless the given newick file already has the names on it
    :param ete3-tree tree: ete3 tree in format 1"""
    # Rename the internal nodes
    leafs_names = tree.get_leaf_names()
    edge = len(leafs_names)
    internal_nodes_names = []
    for node in tree.traverse():  # levelorder (nodes are visited in zig zag order from root to leaves)
        if not node.is_leaf():
            node.name = "A%d" % edge
            internal_nodes_names.append(node.name)
            edge += 1


def renaming_simulations(tree,with_indexes=False):
    """Rename the internal nodes to I + number, in simulations the leaves have the prefix A. With indexes shows the names used when transferring to an array
    :param ete3-tree tree: ete3 format 1"""
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

def my_layout_complex(node):
    """Ete3 layout that adds the internal nodes names. It is a plug-in for rendering tree images
    :param ete3-node node: node from an ete3 tree"""
    if node.is_leaf():
        # If terminal node, draws its name
        name_face = AttrFace("name",fsize=8,fgcolor="StateGrey")
    else:
        # If internal node, draws label with smaller font size
        name_face = AttrFace("name", fsize=8,fgcolor="Black") #color "Peru" for the gist rainbow palette
    # Adds the name face to the image at the preferred position
    faces.add_face_to_node(name_face, node, column=0, position="branch-right")

def my_layout_simple(node):
    """Ete3 layout that adds the internal nodes names. It is a plug-in for rendering tree images
     :param ete3-node node: node from an ete3 tree"""
    if node.is_leaf():
        # If terminal node, draws its name
        name_face = AttrFace("name",fsize=8,fgcolor="black")
    else:
        # If internal node, draws label with smaller font size
        name_face = AttrFace("name", fsize=8,fgcolor="black")
    # Adds the name face to the image at the preferred position
    faces.add_face_to_node(name_face, node, column=0, position="branch-right")
def colour_tree_by_clades_simple(name,clades_dict_all, tree, data_folder,rename_internal_nodes=True):
    """Returns the image of the tree with the nodes coloured by clades, as shown in the article ICLR 2022
    'Ancestral sequence reconstruction using a tree-structured Ornstein-Ulenbeck process'
    :param str name: name of the dataset
    :param defaultdict clades_dict_all: nested dictionary containing all the nodes of the tree divided by clades , i.e {"clade_0":{"internal":[],"leaves":[]}}
    :param ete3-tree: Ete3 formatted tree
    :param str data_folder: folder where the files will be stores, such as the sequences fasta file or the alignment file
    :param bool rename_internal_nodes: If true, the internal nodes are assigned a letter in front of the number (the number is given by the tree file)"""
    color_map21 = matplotlib.colors.ListedColormap(
        ["plum", "navy", "turquoise", "peachpuff", "palevioletred", "red", "darkorange", "yellow", "lime", "green",
         "dodgerblue", "blue", "purple", "magenta", "grey", "maroon", "lightcoral", "olive", "teal", "goldenrod",
         "black"])

    color_map_name = [color_map21 if len(clades_dict_all) <= 21 else 'gist_rainbow'][0]
    clrs = plt.get_cmap(color_map_name, len(clades_dict_all))
    clrs = [matplotlib.colors.rgb2hex(clrs(i)) for i in range(clrs.N)]
    node_styles = []
    for color in clrs:
        ns = NodeStyle()
        ns["fgcolor"] = color #tips color
        ns["size"] = 6
        ns["vt_line_width"] = 1
        ns["hz_line_width"] = 1
        node_styles.append(ns)
    for node_style,(clade, nodes) in zip(node_styles,clades_dict_all.items()):
        clade_ancestor = tree.get_common_ancestor(nodes["leaves"])
        clade_ancestor.set_style(node_style)
        for node in clade_ancestor.get_descendants():
            node.set_style(node_style)

    if rename_internal_nodes:
        if name.startswith("simulations"): renaming_simulations(tree,with_indexes=True)
        else:renaming(tree)
    ts = TreeStyle()
    # Do not add leaf names automatically
    ts.show_leaf_name = False
    # Use my custom layout
    #ts.layout_fn = my_layout_simple
    ts.show_branch_length = False
    ts.show_scale = False
    #new stuff
    # ts.legend= False
    # ts.force_topology = True
    # ts.show_border = True
    # ts.scale_length = False
    try:
        tree.render("{}/return_{}_colored_by_clades_simple.pdf".format(data_folder,name), w=1000, units="mm", tree_style=ts)
    except:
        tree.render("{}/return_{}_colored_by_clades_stripped.pdf".format(data_folder,name), w=1000, units="mm")

def colour_tree_by_clades_complex(name,clades_dict_all, tree, data_folder,rename_internal_nodes=True):
    """Returns the image of the tree coloured by clades, coloured rectangles and with nodes names
    :param str name: name of the dataset
    :param defaultdict clades_dict_all: nested dictionary containing all the nodes of the tree divided by clades , i.e {"clade_0":{"internal":[],"leaves":[]}}
    :param ete3-tree: Ete3 formatted tree
    :param str data_folder: folder where the files will be stores, such as the sequences fasta file or the alignment file
    :param bool rename_internal_nodes: If true, the internal nodes are assigned a letter in front of the number (the number is given by the tree file)"""
    color_map21 = matplotlib.colors.ListedColormap(
        ["plum", "navy", "turquoise", "peachpuff", "palevioletred", "red", "darkorange", "yellow", "lime", "green",
         "dodgerblue", "blue", "purple", "magenta", "grey", "maroon", "lightcoral", "olive", "teal", "goldenrod",
         "black"])

    color_map_name = [color_map21 if len(clades_dict_all) <= 21 else 'nipy_spectral'][0]
    clrs = plt.get_cmap(color_map_name, len(clades_dict_all))
    clrs = [matplotlib.colors.rgb2hex(clrs(i)) for i in range(clrs.N)]
    node_styles = []
    for color in clrs:
        ns = NodeStyle()
        ns["bgcolor"] = color
        ns["vt_line_width"] = 1
        ns["hz_line_width"] = 1
        node_styles.append(ns)
    for node_style,(clade, nodes) in zip(node_styles,clades_dict_all.items()):
        clade_members = tree.get_common_ancestor(nodes["leaves"])
        clade_members.set_style(node_style)
    if rename_internal_nodes:
        if name.startswith("simulations"): renaming_simulations(tree,with_indexes=True)
        else:renaming(tree)
    ts = TreeStyle()
    # Do not add leaf names automatically
    ts.show_leaf_name = False
    # Use my custom layout
    ts.layout_fn = my_layout_complex
    ts.show_branch_length = True

    try:
        tree.render("{}/return_{}_colored_by_clades_facetted.pdf".format(data_folder,name), dpi=600, units="mm", tree_style=ts)
    except:
        tree.render("{}/return_{}_colored_by_clades_facetted.pdf".format(data_folder,name), dpi=600, units="mm")



def draw_tree_simple(name,settings_config):
    """Draws a tree where each of the nodes is coloured by clade membership. The names of the nodes are not shown. Trees as shown in the article.
    :param str name
    :param namedtuple settings_config """
    print("Drawing tree at {}".format(settings_config.data_folder))
    clades_dict_all = pickle.load(open('{}/{}_Clades_dict_all.p'.format(settings_config.data_folder, name), "rb"))
    tree_file = settings_config.tree_file
    tree = Tree(tree_file, format=1, quoted_node_names=True)
    colour_tree_by_clades_simple(name,clades_dict_all,tree,settings_config.data_folder)

def draw_tree_facets(name,settings_config):
    """Draws a tree where each of the clades is coloured by a rectangle. The names of the nodes are shown
    :param str name
    :param namedtuple settings_config"""
    print("Drawing tree at {}".format(settings_config.data_folder))
    clades_dict_all = pickle.load(open('{}/{}_Clades_dict_all.p'.format(settings_config.data_folder, name), "rb"))
    tree_file = settings_config.tree_file
    tree = Tree(tree_file, format=1, quoted_node_names=True)
    colour_tree_by_clades_complex(name,clades_dict_all,tree,settings_config.data_folder)


