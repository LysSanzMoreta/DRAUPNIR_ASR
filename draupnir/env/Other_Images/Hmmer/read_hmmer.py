import pandas as pd
import re
import dataframe_image as dfi
#awk '{ if (length($1) > max) max = length($1) } END { print max }' /home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src/draupnir/data/hmmer/CBM_HMMSEARCH.txt

def reformat():
    row1 = "sp|A0A0B4GLB5|MLAC1_METGA -          AA1_1.hmm            -            7.7e-33  118.8   0.6   1.1e-23   88.8   0.3   2.5   2   1   0   2   2   2   2"
    starts = [m.start() for m in re.finditer(r'\S+', row1)] #ESTIMATES INITIAL VALUES, BUT NEED TO BE CORRECTED AFTERWARDS
    widths = [e - s for s, e in zip(starts, starts[1:])] + [200]#last column
    widths = [25, 8, 18, 13, 14, 8, 6, 10, 7, 6, 6, 4, 4, 4, 4, 4, 4,3,200]
    columnsnames = ["target name", "accession", "query name", "accession","E-value",  "score",  "bias", "E-value2",  "score",  "bias", "exp", "reg", "clu", "ov", "env", "dom", "rep", "inc", "description of target"]
    df = pd.read_fwf("/home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src/draupnir/data/hmmer/CBM_HMMSEARCH.txt", widths=widths,header=None, skiprows=1)
    df.columns = columnsnames
    df.to_csv("/home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src/draupnir/data/hmmer/CBM_HMMSEARCH_reformatted.txt",sep="\t",index=False)
    print(df[["query name"]].values.tolist())

#reformat()

folder = "/home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src/draupnir/data/hmmer"
df = pd.read_csv("/home/lys/Dropbox/PhD/DRAUPNIR_ASR/draupnir/src/draupnir/data/hmmer/CBM_HMMSEARCH_reformatted.txt",sep="\t")



values_counts_df = df[["query name"]].value_counts().rename_axis('unique_values').reset_index(name='counts')

metric_df_styled = values_counts_df.style.format(na_rep="-", escape="latex", precision=2).background_gradient(axis=None,cmap="YlOrBr",)
dfi.export(metric_df_styled, f'{folder}/cmb_hmmsearch_swissprot_family_value_counts.png', max_cols=-1,max_rows=-1, table_conversion="chrome")

#metric_df_styled.export_png(f'{folder}/cmb_hmmsearch_swissprot_family_value_counts.png',max_cols=-1,max_rows=-1)



