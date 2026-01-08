import json

import pandas as pd
import re
from pandas._config import get_option
import imgkit
#awk '{ if (length($1) > max) max = length($1) } END { print max }' /home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src/draupnir/data/hmmer/CBM_HMMSEARCH.txt
folder = "/home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src/draupnir/data/hmmer"
def reformat():
    row1 = "sp|A0A0B4GLB5|MLAC1_METGA -          AA1_1.hmm            -            7.7e-33  118.8   0.6   1.1e-23   88.8   0.3   2.5   2   1   0   2   2   2   2"
    starts = [m.start() for m in re.finditer(r'\S+', row1)] #ESTIMATES INITIAL VALUES, BUT NEED TO BE CORRECTED AFTERWARDS
    widths = [e - s for s, e in zip(starts, starts[1:])] + [200]#last column
    widths = [25, 8, 18, 13, 14, 8, 6, 10, 7, 6, 6, 4, 4, 4, 4, 4, 4,3,200]
    columnsnames = ["target name", "accession", "query name", "accession","E-value",  "score",  "bias", "E-value2",  "score",  "bias", "exp", "reg", "clu", "ov", "env", "dom", "rep", "inc", "description of target"]
    df = pd.read_fwf(f"{folder}/CBM_HMMSEARCH.txt", widths=widths,header=None, skiprows=1)
    df.columns = columnsnames
    df.to_csv(f"{folder}/CBM_HMMSEARCH_reformatted.txt",sep="\t",index=False)
    print(df[["query name"]].values.tolist())

#reformat()


df = pd.read_csv(f"{folder}/CBM_HMMSEARCH_reformatted.txt",sep="\t")
df = df[df["query name"].str.startswith("CBM")]
values_counts_df = df[["query name"]].value_counts().rename_axis('unique_values').reset_index(name='counts')

def create_counts_image(values_counts_df,name):
    metric_df_styled = values_counts_df.style.format(na_rep="-", escape="latex", precision=2).background_gradient(axis=None,cmap="YlOrBr",)
    sparse_index = get_option("styler.sparse.index")
    sparse_columns = get_option("styler.sparse.columns")
    html = metric_df_styled._render_html(sparse_index, sparse_columns, None, None)

    with open(f"{folder}/temp.html", "w") as f:
        f.write(html)

    # Convert HTML → PNG
    options = {"format": "png", "encoding": "UTF-8"}
    imgkit.from_file(f"{folder}/temp.html",
                     f'{folder}/{name}', options=options)

create_counts_image(values_counts_df,"cmb_hmmsearch_swissprot_family_value_counts.png")

#classification according to sequence similarity and phylogenetics clustering
#https://www.cazy.org/CBM1.html
folds_dict = {
    "CBM1": ["PDOC00486",r"three-stranded antiparallel $\beta$-sheet",r"$\beta$-sheet sandwich","cellulose binding"],
    "CBM2":["PDOC00485",r"$\beta$-sheet domain containing a planar face",r"$\beta$-sheet sandwich","cellulose,chitin,xylan binding"],
    "CBM3":["PDOC51172",r"nine $\beta$-strands, two antiparallel $\beta$-sheets that stack face-to-face to form a $\beta$ sandwich with jelly roll topology", r"$\beta$-sheet sandwich", "crystaline cellulose binding"],
    "CBM4": ["","",r"$\beta$-sheet sandwich",r"xylan,$\beta$-1,3-glucan binding"],
    "CBM5":["",r"Type A CBM. Composed by five $\beta$-strands, only 2 of the, are forming an antiparallel $\beta$ sheet","L-shaped or ski-boot fold", "chitin, cellulose binding"], #Modules of approx. 60 residues found in bacterial enzymes. Chitin-binding described in several cases. Distantly related to the CBM12 family
    "CBM6":["",r"five antiparallel $\beta$-strands on one face and four anti-parallel $\beta$-strands on the other face, connected by loops with variable lengths",r"$\beta$-sheet sandwich", r"xylan, $\beta$-1,3-glucan binding"],
    "CBM7": ["","","", ""], #deleted entry in cazy
    "CBM8": ["","",r"$\beta$-sheet sandwich", "cellooligosaccharides binding"],
    "CBM9": ["","Type A CBM. 3 Calcium binding sites",r"$\beta$-sheet sandwich","cellooligosaccharides binding"], #Modules of approx. 170 residues found so far only in xylanases. The cellulose-binding function has been demonstrated in one case.
    "CBM10": ["",r"5 $\beta$-strands, organized as 2 antiparallel sheets, one of three strands ($\beta$-sheet 1) and one of two ($\beta$-sheet 2) that are approximately perpendicular to each other. The structure also contains a short stretch of α-helix. The protein is stabilized by two disulfide bridges","OB-fold","crystalline-cellulose binding"], #only cbm family with this fold?
    "CBM11": ["","","", r"cellulose,$\beta$-glucan binding"],
    "CBM12": ["","","", "chitin binding"],
    "CBM13": ["","",r"$\beta$-trefoil", r"galactose, lactose, xylan, $\beta$-1,3-glucan binding"],
    "CBM14": ["",r"central $\beta$-sheet (three anti-parallel $\beta$-strands) linked to a small $\beta$-sheet (two anti-parallel $\beta$-strands) by two aromatic residues","hevein-like fold", "chiting binding"],
    "CBM15": ["",r"classic $\beta$-jelly roll, predominantly consisting of five major anti-parallel $\beta$-strands on the two face",r"$\beta$-sheet sandwich", "xylan,xylooligosaccharides binding"],
    "CBM16": ["","",r"$\beta$-sheet sandwich", r"planar $\beta$-1,4-glucans binding"],
    "CBM17": ["","","", ""], #Modules of approx. 200 residues. Binding to amorphous cellulose, cellooligosaccharides and derivatized cellulose has been demonstrated.
    "CBM18": ["","","", "chiting binding"], #Modules of approx. 40 residues
    "CBM19": ["","","", "chiting binding"], #Modules of 60-70 residues
    "CBM20": ["PDOC51166",r"seven $\beta$-stands forming an open-sided distorted $\beta$-barrel",r"$\beta$-sheet sandwich", "starch binding"],
    "CBM21": ["PDOC51159","starch-binding domain", r"$\beta$-sheet sandwich", "starch binding"], #Modules of approx. 100 residues
    "CBM22": ["",r"canonical $\beta$-sandwich fold in which the two $\beta$-sheets both contain five antiparallel $\beta$-strands", r"$\beta$-sheet sandwich", "xylan, glucans binding"],
    "CBM23": ["","","", "manan binding"],
    "CBM24": ["","","", r"$\alpha-1,3-glucan (mutan)-binding"],
    "CBM25": ["","",r"$\beta$-sheet sandwich", "starch binding"],
    "CBM26": ["","",r"$\beta$-sheet sandwich", "starch binding"],
    "CBM27": ["","","", "mannan binding"],
    "CBM28": ["","",r"$\beta$-sheet sandwich", "non-crystalline, cellulose, glucan binding"],
    "CBM29": ["","",r"$\beta$-sheet sandwich", "mannan, glucomannan binding"],
    "CBM30": ["","",r"$\beta$-sheet sandwich", "cellulose binding"], #not sure abou the fold: https://www.rcsb.org/structure/1WMX
    "CBM31": ["",r"8 $\beta$-strands, 2 intra-molecular disulfide bonds, jelly-roll fold, immunoglobulin fold",r"$\beta$-sheet sandwich", r"$\beta$-1,3-xylan binding"],
    "CBM32": ["","",r"$\beta$-sheet sandwich", r"galactose,glucose,polygalacturonic acid, LacNAc, $\beta$-1,3-glucan binding"],
    "CBM33": ["","Renamed to AA10 module family: Copper-dependent lytic polysaccharide monooxygenases","", "chintin, cellulose binding"],
    "CBM34": ["","","", "starch binding"], #structure of cbm not clear
    "CBM35": ["",r"$\beta$-sandwich fold in which the two $\beta$-sheets containing four and five antiparallel $\beta$-strands, respectively, are connected entirely by loops. $\beta$-sheets are twisted like a barrel",r"$\beta$-sheet sandwich", r"xylan,mannan,$\beta$-galactan binding"],
    "CBM36": ["","",r"$\beta$-sheet sandwich", "xylan and xylooligosaccharides binding"],
    "CBM37": ["","","", "broad binding"],
    "CBM38": ["","","", "inulin binding"],
    "CBM39": ["",r"8 $\beta$-strands and has a typical Immunoglobulin fold",r"$\beta$-sheet sandwich", r"$\beta$-1,3-glucan binding"],
    "CBM40": ["",r"Lectin-like structure. 2 $\beta$-sheets of five and six antiparallel $\beta$-strands and two $\alpha$-helices, one within the sheets and another at the C-terminus, packed against $\beta$-strand 7",r"$\beta$-sheet sandwich", "sialic acid binding"],#r"$\beta$-sandwich fold consisting of two antiparallel $\beta$-sheets"
    "CBM41": ["","Type B CBM. Forms a concave-shaped binding groove",r"$\beta$-sheet sandwich", "glucans, amylose, amylopectin, pullulan saccharides binding"],
    "CBM42": ["","Type C CBMs that bind termini of glycans with pocket-type binding sites for short oligosaccharides",r"$\beta$-trefoil", "arabinofuranose binding"],
    "CBM43": ["","Domain containing two parallel alpha-helices forming an angle of approximately 55 degrees , a small antiparallel beta-sheet with two short strands, and a 3-10 helix turn, all connected by long coil segments, resembling a novel type of folding among allergens","unclear", r"$\beta$-1,3-glucan binding"],
    "CBM44": ["",r"2 antiparallel $\beta$-sheets form a concave and a convex surface",r"$\beta$-sheet sandwich", "cellulose, xyloglucan binding"],
    "CBM45": ["","","", "starch binding"], #very little is know, not structures
    "CBM46": ["",r"The CBM displays a classic $\beta$-sandwich jelly roll fold. The two $\beta$-sheets contain four anti-parallel $\beta$-strands",r"$\beta$-sheet sandwich", "cellulose binding"],
    "CBM47": ["",r"8-stranded $\beta$-sandwich fold, which is comprised of a five-stranded anti-parallel $\beta$-sheet on one side and a three-stranded anti-parallel $\beta$-sheet on the other",r"$\beta$-sheet sandwich", "fucose binding"],
    "CBM48": ["","",r"$\beta$-sheet sandwich", r"various linear and cyclic $\alpha$ glycans binding"],
    "CBM49": ["","",r"$\beta$-barrel fold", r"crystalline cellulose, $\beta$-1,3-glucan binding"],
    "CBM50": ["",r"Also known as LysM domains, consist of two helices packing against one side of the two-stranded antiparallel $\beta$-sheet",r"$\beta\alpha\alpha\beta$ fold", " N-acetylglucosamine residues (found in chitin or peptidoglycans) binding"],
    "CBM51": ["","Also know as NPCBM domains",r"$\beta$-sheet sandwich", "galactose, blood group A/B-antigens binding"], #domain found in exoglycosidases (found in mucing-degrading gut bacteria) responsible for generating ABo universal blood?
    "CBM52": ["","","", r"$\beta$-1,3-glucan binding"],
    "CBM53": ["","","", "starch binding"],
    "CBM54": ["",r"12 coiled and right-handed $\beta$-helix structure, 34 $\beta$-strands that form three parallel $\beta$-sheets",r"right-handed $\beta$-helix domain", r"xylan, chitin, $\beta$-1,3-glucan binding"],
    "CBM55": ["","Type A CBM. No strucure available, contains 9 conserved cysteines that suggest a disulfide knot","", "chitin binding"],
    "CBM56": ["",r"$\beta$-sandwich fold comprising two opposing 4-stranded $\beta$-sheets with the very last $\beta$-strand in the fold being broken up by a bulge in its middle",r"$\beta$-sheet sandwich", r"insoluble $\beta$-1,3-glucan binding"],
    "CBM57": ["",r"Malectin (family representation) differs from the currently known lectins, however, in having α-helices and extensions of the $\beta$-sandwich arrangement, which would classify it as a new type of carbohydrate recognition domain.",r"$\beta$-sheet sandwich", "Glc2-N-glycan binding"], #malectin protein (n-glycosilation)
    "CBM58": ["",r"Type B CBM, 8 $\beta$ strands, only found in GH13 family",r"$\beta$-sheet sandwich", "maltoheptaose, acarbose binding"],
    "CBM59": ["","",r"$\beta$-sheet sandwich", "mannan, xylan and cellulose binding"],
    "CBM60": ["",r"Calcium dependent CBM, eight $\beta$-strands in two antiparallel $\beta$-sheets, each of four strands",r"distorted $\beta$-jelly-roll fold", "xylan binding"],
    "CBM61": ["","",r"$\beta$-jellyroll fold", r"$\beta$-1,4-galactan binding"],
    "CBM62": ["",r"The data showed that an axial O4 is a key determinant for the specificity of CtCBM62, explaining why the ligand binding pocket targets galactose and arabinopyranose, as opposed to mannose, glucose and xylose. Five antiparallel $\beta$-strands on one face ($\beta$1, 2, 4, 5 and 7) and three antiparallel $\beta$-strands on the other face ($\beta$3, 6 and 8). Two α-helixes and five loops on top of the $\beta$-jelly-roll complete the structure.",r"$\beta$-jelly-roll fold,", "xyloglucan, arabinogalactan and galactomannan binding"],
    "CBM63": ["",r"Type A CBM almost exclusively found in expansins. Formed by two sets of four anti-parallel $\beta$-strands that form a $\beta$-sandwich",r"$\beta$-sheet sandwich", "cellulose binding"],
    "CBM64": ["",r"Type A CBM. 2 antiparallel $\beta$-sheets with nine $\beta$-strands",r"$\beta$-jelly-roll-like fold", "cellulose binding"],
    "CBM65": ["","Type B CBM",r"$\beta$-sheet sandwich", "xyloglucan binding"],
    "CBM66": ["",r"Type C CBM. $\beta$-sandwich fold in which the two $\beta$-sheets contain seven and six antiparallel $\beta$-strands",r"$\beta$-sheet sandwich", "fructans residues binding"],
    "CBM67": ["","Type C CBM",r"$\beta$-jelly-roll fold", "L-rhamnose binding"],
    "CBM68": ["","","", "maltotriose, maltotetraose binding"],
    "CBM69": ["","","novel fold", "starch binding"],
    "CBM70": ["",r"2 opposing 5-stranded antiparallel $\beta$-sheets",r"$\beta$-jelly-roll fold", "hyaluran binding"],
    "CBM71": ["",r"Type C CBM. Opposing sheets of 4- and 5-antiparallel $\beta$-strands and a bound structural metal ion modelled as a calcium",r"$\beta$-sheet sandwich", "lactose, LacNac binding"],
    "CBM72": ["","","", r"$\beta$-1,3-glucan binding, xylan, $\beta$-mannan, insoluble cellulose binding"],
    "CBM73": ["","structure derived from homology",r"$\beta$-sheet sandwich", "chitin binding"],
    "CBM74": ["",r"core $\beta$-sandwich fold of two sheets with five antiparallel $\beta$-strands",r"$\beta$-sheet sandwich", "starch binding"],
    "CBM75": ["","Found exclusively in Ruminococci","unknown", "xyloglucan binding"],
    "CBM76": ["","Found exclusively in Ruminococci", "unknown", r"$\beta$-glucans binding"],
    "CBM77": ["",r"Type B CBM. 13 antiparallel $\beta$-strands are organized in two $\beta$-sheets",r"$\beta$-sheet sandwich", "pectin binding"], #very selective pectin binding
    "CBM78": ["",r"Type B CBM. 2 $\beta$ sheets",r"$\beta$-sheet sandwich", r"$\beta$-1,4-glucans (xyloglucan) binding"],
    "CBM79": ["",r"Type B CBM. 12 antiparallel $\beta$-strands are organized in two $\beta$-sheets 1 and 2",r"$\beta$-sheet sandwich", r"$\beta$-glucans binding"],
    "CBM80": ["",r"2 $\beta$-sheets comprising of 4 anti-parallel $\beta$-strands each",r"$\beta$-sheet sandwicrh", r"$\beta$-glycans"], #broad specificity
    "CBM81": ["","Type A and B CBM (rare). 4 and 5 beta-strands connected by loops",r"$\beta$-sheet sandwich", r"$\beta$-1,4-, $\beta$-1,3,-glucans, xyloglucan, avicel and cellooligosaccharides binding"],
    "CBM82": ["","Fold similar to CBM41 family",r"$\beta$-sheet sandwich", "starch binding"],
    "CBM83": ["","Fold possibly similar to CBM41 family","unknown", "starch binding"],
    "CBM84": ["",r"3 unconnected $\beta$-strands with highly disordered residues",r"$\beta$-sheet sandwich", "xanthan binding"],
    "CBM85": ["","","", "broad binding of glycans"], #no info
    "CBM86": ["",r"Type B CMB. 2 sheets formed by 11 antiparallel $\beta$-stands and two helical turns connected by loops.",r"$\beta$-sheet sandwich", "xylan binding"],
    "CBM87": ["",r"α / $\beta$ /α-domain. The central $\beta$-sheet consists of seven strands, with the sixth stand running anti-parallel to the other stands",r"Rossman-like fold ", r"α-1,4-N-acetylgalactosamine-rich regions of galactosaminogalactan binding"], #reductase-like fold?
    "CBM88": ["","","", "terminal galactose in galactoxyloglucan and galactomannan"],
    "CBM89": ["",r"14 helical turns establishing a parallel $\beta$-helix fold",r"$\beta$-helix", "beechwood xylan and rye arabinoxylan binding"],
    "CBM90": ["","Found in PL28 ulvan lyase from Nonlabens ulvanivorans","", "ulvan binding"],
    "CBM91": ["",r"CBM91 are often connected to $\beta$-xylosidases belonging to glycoside hydrolase family 43 ","", "xylans (birchwood and oat spelt)"], #most likely beta sheet sandwich
    "CBM92": ["",r"12 $\beta$-strands arranged into 3 subdomains",r"$\beta$-trefoil ", r"$\beta$-1,3- and $\beta$-1,6-glucan binding"],
    "CBM93": ["",r"Located at the N-terminus of GH33 sialidases $\beta$-sandwich style fold with a central cleft; however, the structure is more open.",r"$\beta$-sheet sandwich", "broad-specificity glycan binding"],
    "CBM94": ["",r"Present in N-acetylglucosaminyltransferase. 9 $\beta$-strands and three short α-helices,",r"$\beta$-sheet sandwich", "N-Acetylglucosamine binding"],
    "CBM95": ["","",r"$\beta$-jellyroll fold", "pectic rhamnogalacturonan-I binding"],
    "CBM96": ["",r"2 antiparallel $\beta$-sheets consisting of 11 $\beta$-strands",r"$\beta$-sheet sandwich", "alginate binding"],
    "CBM97": ["","",r"$\beta$-sheet sandwich", "polygalacturonic acid binding"],
    "CBM98": ["",r"7 $\beta$ strands",r"$\beta$-sheet sandwich", "amylopectin binding"],
    "CBM99": ["",r"Predicted as $\beta$-sheet sandwich fold","", "porphyran binding"],
    "CBM100": ["",r"10 $\beta$-strands and 2 helices",r"$\beta$-sheet sandwich", "chondroitin sulfate binding "],
    "CBM101": ["",r"Predicted as $\beta$-sheet sandwich fold","", "agarose, weak porphyran binding"],
    "CBM102": ["",r"Predicted as $\beta$-sheet sandwich fold","", "agarose, laminarin, weak porphyran binding"],
    "CBM103": ["",r"Type C CBM. $\beta$-barrel fold with an additional $\beta$-strand pair and two short α-helices",r"$\beta$-barrel fold", "laminarin, lichenan, laminaritriose binding"],
    "CBM104": ["","","", "crystalline cellulose binding"],
    "CBM105": ["",r"Predicted as $\beta$-sheet sandwich fold","", "chondroitin sulfate binding"],
    "CBM106": ["",r"2 antiparallel $\beta$-sheets formed by 11 $\beta$-strands and 4 helixes",r"$\beta$-sheet sandwich", "alginate binding"],
    "CBM107": ["","","", "homogalacturonan, rhamnogalacturonan-I binding"],
}


folds_dict = json.load(open(f"{folder}/cazy_dict.json","r"))

df["prosite_id"] = df["query name"].map(lambda x: folds_dict[x.replace(".hmm","")][0] if x.replace(".hmm","") in folds_dict else x)
df["description"] = df["query name"].map(lambda x: folds_dict[x.replace(".hmm","")][1] if x.replace(".hmm","") in folds_dict else x)
df["structural_fold"] = df["query name"].map(lambda x: folds_dict[x.replace(".hmm","")][2] if x.replace(".hmm","") in folds_dict else x)
df["binding_substrate"] = df["query name"].map(lambda x: folds_dict[x.replace(".hmm","")][3] if x.replace(".hmm","") in folds_dict else x)


df.to_csv(f"{folder}/CBM_HMMSEARCH_Cazy_CBMs.txt",sep="\t",index=False)


values_counts_df = df[["query name"]].value_counts().rename_axis('unique_values').reset_index(name='counts')

values_counts_df["structural_fold"] = values_counts_df["unique_values"].map(lambda x: folds_dict[x.replace(".hmm","")][2] if x.replace(".hmm","") in folds_dict else x)
values_counts_df["binding_substrate"] = values_counts_df["unique_values"].map(lambda x: folds_dict[x.replace(".hmm","")][3] if x.replace(".hmm","") in folds_dict else x)


create_counts_image(values_counts_df,"cmb_hmmsearch_swissprot_family_value_counts_CAZY.png")




