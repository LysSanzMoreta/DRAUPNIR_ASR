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



metric_df_styled = values_counts_df.style.format(na_rep="-", escape="latex", precision=2).background_gradient(axis=None,cmap="YlOrBr",)
sparse_index = get_option("styler.sparse.index")
sparse_columns = get_option("styler.sparse.columns")
html = metric_df_styled._render_html(sparse_index, sparse_columns, None, None)

with open(f"{folder}/temp.html", "w") as f:
    f.write(html)

# Convert HTML → PNG
options = {"format": "png", "encoding": "UTF-8"}
imgkit.from_file(f"{folder}/temp.html",
                 f'{folder}/cmb_hmmsearch_swissprot_family_value_counts.png', options=options)


#https://www.cazy.org/CBM1.html
folds_dict = {
    "CBM1": ["PDOC00486",r"three-stranded antiparallel $\beta-sheet",r"$\beta-sheet sandwich","cellulose binding"],
    "CBM2":["PDOC00485",r"a \beta-sheet domain containing a planar face",r"$\beta-sheet sandwich","cellulose,chitin,xylan binding"],
    "CBM3":["PDOC51172",r"nine β-strands, two antiparallel β-sheets that stack face-to-face to form a β sandwich with jelly roll topology", r"$\beta-sheet sandwich", "crystaline cellulose binding"],
    "CBM4": ["","",r"$\beta-sheet sandwich",r"xylan,$\beta-glucan binding"],
    "CBM5":["","",""], #Modules of approx. 60 residues found in bacterial enzymes. Chitin-binding described in several cases. Distantly related to the CBM12 family
    "CBM6":[""," five antiparallel β-strands on one face and four anti-parallel β-strands on the other face, connected by loops with variable lengths","r$\beta-sheet sandwich", "xylan binding"],
    "CBM7": ["","","", ""], #deleted entry in cazy
    "CBM8": ["","",r"$\beta-sheet sandwich", "cellulose and others binding"],
    "CBM9": ["","",r"$\beta-sheet sandwich",""], #Modules of approx. 170 residues found so far only in xylanases. The cellulose-binding function has been demonstrated in one case.
    "CBM10": ["",r"five $\beta-strands, organized as two antiparallel sheets, one of three strands ($\beta-sheet 1) and one of two (β-sheet 2) that are approximately perpendicular to each other. The structure also contains a short stretch of α-helix. The protein is stabilized by two disulfide bridges","OB-fold","crystalline-cellulose binding"], #only cbm family with this fold?
    "CBM11": ["","","", r"cellulose,$\beta-glucan binding"],
    "CBM12": ["","","", "chitin binding"],
    "CBM13": ["","","β-trefoil", "galactose, lactose, xylan binding"],
    "CBM14": ["",r"central $\beta-sheet (three anti-parallel $\beta-strands) linked to a small $\beta-sheet (two anti-parallel β-strands) by two aromatic residues","hevein-like fold", "chiting binding"],
    "CBM15": ["","classic β-jelly roll, predominantly consisting of five major anti-parallel β-strands on the two face",r"$\beta-sheet sandwich", "xylan,xylooligosaccharides binding"],
    "CBM16": ["","",r"$\beta-sheet sandwich", r"planar $\beta-1,4-glucans binding"],
    "CBM17": ["","","", ""], #Modules of approx. 200 residues. Binding to amorphous cellulose, cellooligosaccharides and derivatized cellulose has been demonstrated.
    "CBM18": ["","","", "chiting binding"], #Modules of approx. 40 residues
    "CBM19": ["","","", "chiting binding"], #Modules of 60-70 residues
    "CBM20": ["PDOC51166","seven β-stands forming an open-sided distorted β-barrel",r"$\beta-sheet sandwich", "starch binding"],
    "CBM21": ["PDOC51159","","starch-binding domain, ", r"$\beta-sheet sandwich", "starch binding"], #Modules of approx. 100 residues
    "CBM22": ["","canonical β-sandwich fold in which the two β-sheets both contain five antiparallel β-strands", r"$\beta-sheet sandwich", "xylan, glucans binding"],
    "CBM23": ["","","", "manan binding"],
    "CBM24": ["","","", r"$\alpha-1,3-glucan (mutan)-binding"],
    "CBM25": ["","",r"$\beta-sheet sandwich", "starch binding"],
    "CBM26": ["","",r"$\beta-sheet sandwich", "starch binding"],
    "CBM27": ["","","", "mannan binding"],
    "CBM28": ["","",r"$\beta-sheet sandwich", "non-crystalline, cellulose, glucan binding"],
    "CBM29": ["","",r"$\beta-sheet sandwich", "mannan, glucomannan binding"],
    "CBM30": ["","","", ""],
    "CBM30": ["","","", ""],
    "CBM30": ["","","", ""],
    "CBM30": ["","","", ""],
    "CBM30": ["","","", ""],
}
