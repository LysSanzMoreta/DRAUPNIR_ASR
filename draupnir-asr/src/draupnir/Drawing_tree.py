from ete3 import Tree, faces, AttrFace, TreeStyle, NodeStyle, Tree
# from PyQt4 import QtCore
# from PyQt4.QtGui import QGraphicsRectItem, QGraphicsSimpleTextItem, \
#     QGraphicsEllipseItem, QColor, QPen, QBrush
import matplotlib.pyplot as plt
import matplotlib
import pickle, os

# class InteractiveItem(QGraphicsRectItem):
#     def __init__(self, *arg, **karg):
#         QGraphicsRectItem.__init__(self, *arg, **karg)
#         self.node = None
#         self.label = None
#         self.setCursor(QtCore.Qt.PointingHandCursor)
#         self.setAcceptsHoverEvents(True)
#
#     def hoverEnterEvent (self, e):
#         # There are many ways of adding interactive elements. With the
#         # following code, I show/hide a text item over my custom
#         # DynamicItemFace
#         if not self.label:
#             self.label = QGraphicsRectItem()
#             self.label.setParentItem(self)
#             # This is to ensure that the label is rendered over the
#             # rest of item children (default ZValue for items is 0)
#             self.label.setZValue(1)
#             self.label.setBrush(QBrush(QColor("white")))
#             self.label.text = QGraphicsSimpleTextItem()
#             self.label.text.setParentItem(self.label)
#
#         self.label.text.setText(self.node.name)
#         self.label.setRect(self.label.text.boundingRect())
#         self.label.setVisible(True)
#
#     def hoverLeaveEvent(self, e):
#         if self.label:
#             self.label.setVisible(False)
#
# def random_color(h=None):
#     """Generates a random color in RGB format."""
#     if not h:
#         h = random.random()
#     s = 0.5
#     l = 0.5
#     return _hls2hex(h, l, s)
#
# def _hls2hex(h, l, s):
#     return '#%02x%02x%02x' %tuple(map(lambda x: int(x*255),
#                                       colorsys.hls_to_rgb(h, l, s)))
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
# def ellipse_face(node, *args, **kargs):
#     """ This is my item generator. It must receive a node object, and
#     returns a Qt4 graphics item that can be used as a node face.
#     """
#
#     # receive an arbitrary number of arguments, in this case width and
#     # height of the faces
#     width = args[0][0]
#     height = args[0][1]
#
#     ## Creates a main master Item that will contain all other elements
#     ## Items can be standard QGraphicsItem
#     # masterItem = QGraphicsRectItem(0, 0, width, height)
#
#     # Or your custom Items, in which you can re-implement interactive
#     # functions, etc. Check QGraphicsItem doc for details.
#     masterItem = InteractiveItem(0, 0, width, height)
#
#     # Keep a link within the item to access node info
#     masterItem.node = node
#
#     # I dont want a border around the masterItem
#     masterItem.setPen(QPen(QtCore.Qt.NoPen))
#
#     # Add ellipse around text
#     ellipse = QGraphicsEllipseItem(masterItem.rect())
#     ellipse.setParentItem(masterItem)
#     # Change ellipse color
#     ellipse.setBrush(QBrush(QColor( random_color())))
#
#     # Add node name within the ellipse
#     text = QGraphicsSimpleTextItem(node.name)
#     text.setParentItem(ellipse)
#     text.setPen(QPen(QPen(QColor("white"))))
#
#     # Center text according to masterItem size
#     tw = text.boundingRect().width()
#     th = text.boundingRect().height()
#     center = masterItem.boundingRect().center()
#     text.setPos(center.x()-tw/2, center.y()-th/2)
#
#     return masterItem
def my_layout(node):
    "Adds the internal nodes names"
    if node.is_leaf():
        # If terminal node, draws its name
        name_face = AttrFace("name",fsize=8,fgcolor="StateGrey")
    else:
        # If internal node, draws label with smaller font size
        name_face = AttrFace("name", fsize=8,fgcolor="Black") #color "Peru" for the gist rainbow palette
    # Adds the name face to the image at the preferred position
    faces.add_face_to_node(name_face, node, column=0, position="branch-right")
def colour_tree_by_clades(clades_dict, tree, rename_internal_nodes=True):
    color_map21 = matplotlib.colors.ListedColormap(
        ["plum", "navy", "turquoise", "peachpuff", "palevioletred", "red", "darkorange", "yellow", "lime", "green",
         "dodgerblue", "blue", "purple", "magenta", "grey", "maroon", "lightcoral", "olive", "teal", "goldenrod",
         "black"])

    color_map_name = [color_map21 if len(clades_dict_all) <= 21 else 'nipy_spectral'][0]
    clrs = plt.get_cmap(color_map_name, len(clades_dict))
    clrs = [matplotlib.colors.rgb2hex(clrs(i)) for i in range(clrs.N)]
    node_styles = []
    for color in clrs:
        ns = NodeStyle()
        ns["bgcolor"] = color
        ns["vt_line_width"] = 1
        ns["hz_line_width"] = 1
        node_styles.append(ns)
    for node_style,(clade, nodes) in zip(node_styles,clades_dict.items()):
        clade_members = tree.get_common_ancestor(nodes["leaves"])
        clade_members.set_style(node_style)
    if rename_internal_nodes:
        if name.startswith("simulations"): Renaming_simulations(with_indexes=True)
        else:Renaming()
    ts = TreeStyle()
    # Do not add leaf names automatically
    ts.show_leaf_name = False
    # Use my custom layout
    ts.layout_fn = my_layout
    ts.show_branch_length = True

    try:
        tree.render("Tree_Pictures/return_{}_colored_by_clades.pdf".format(name), dpi=600, units="mm", tree_style=ts)
    except:
        tree.render("Tree_Pictures/return_{}_colored_by_clades.pdf".format(name), dpi=600, units="mm")




if __name__ == "__main__":
    #t, ts = get_example_tree()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_number,simulation_folder,root_sequence_name = 1,"BLactamase","BetaLactamase_seq"
    #dataset_number,simulation_folder,root_sequence_name = 1, "SRC_simulations","SRC_SH3" #100 leaves
    # dataset_number,simulation_folder,root_sequence_name = 2, "SRC_simulations","SRC_SH3" #800 leaves
    #dataset_number, simulation_folder, root_sequence_name = 3, "SRC_simulations", "SRC_SH3"  # 200 leaves
    datasets = {0: "benchmark_randall",  # the tree is inferred
                1: "benchmark_randall_original",# uses the original tree but changes the naming of the nodes (because the original tree was not rooted)
                2: "benchmark_randall_original_naming",  # uses the original tree and it's original node naming
                3: "SH3_pf00018_larger_than_30aa",  # SRC kinases domain SH3 ---> Leaves and angles testing
                4: "simulations_blactamase_{}".format(dataset_number),  # EvolveAGene4 Betalactamase simulation
                5: "simulations_src_sh3_{}".format(dataset_number),  # EvolveAGene4 SRC SH3 domain simulation
                6: "Douglas_SRC",  # Douglas's SRC Kinases #TODO: Realign, tree and ancestral sequences?
                7: "ANC_AS_subtree",
                8: "Coral_Faviina",  # Faviina clade from coral sequences
                9: "Coral_all",  # All Coral sequences (includes Faviina and more)
                10: "Cnidarian",# coral sequences plus other fluorescent cnidarians #Highlight: The tree is too different to certainly locate the all-coral / all-fav ancestors
                11: "Sirtuin_PF02146",  # SIR2 protein
                12: "PKinase_PF07714",
                13:"PF01038_msa",
                14:"5azu_dataset",
                15: "PF01038_lipcti_msa_fungi"}  # Protein Kinases, relatively big dataset, 324 seq, might need some cleaning
    name = datasets[15]
    clades_dict_all = pickle.load(open('{}/Mixed_info_Folder/{}_Clades_dict_all.p'.format(script_dir, name), "rb"))
    if name.startswith("simulations"):
        tree_file = "Datasets_Simulations/{}/Dataset{}/{}_True_Rooted_tree_node_labels.tre".format(simulation_folder,dataset_number,root_sequence_name)
    elif name == "benchmark_randall_original_naming":
        tree_file = "AncestralResurrectionStandardDataset/RandallBenchmarkTree_OriginalNaming.tree"
    elif name.endswith("_subtree"):
        tree_file = "/home/lys/Dropbox/PhD/DRAUPNIR/Douglas_SRC_Dataset/{}/{}_ALIGNED.mafft.treefile".format(name,name.replace("_subtree",""))
    elif name in ["PF01038_msa","PF01038_lipcti_msa_fungi"]:
        tree_file = "/home/lys/Dropbox/PhD/DRAUPNIR/NovozDataset/{}.fasta.treefile".format(name)
    elif name == "5azu_dataset":
        tree_file = ""
    else:
        tree_file = "Mixed_info_Folder/{}.mafft.treefile".format(name)

    tree = Tree(tree_file, format=1, quoted_node_names=True)
    colour_tree_by_clades(clades_dict_all,tree)
