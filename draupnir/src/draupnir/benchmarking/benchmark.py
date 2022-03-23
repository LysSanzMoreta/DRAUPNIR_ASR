import datetime
from Bio.Phylo.PAML import codeml,baseml
from Bio import SeqIO, AlignIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import subprocess,sys, time,os,pathlib,ntpath,glob,re
import shutil,click, time
import  argparse
def remove_stop_codons(alignment_file_fasta):
    codon_stop_array = ["TAG", "TGA", "TAA", "UGA", "UAA", "UAG"]

    records_list=[]
    for record in SeqIO.parse(alignment_file_fasta, "fasta"):
        tempRecordSeq = list(record.seq)
        for index in range(0, len(record.seq), 3):
            codon = record.seq[index:index + 3]
            if codon in codon_stop_array:
                print("There are stop codons in position {}".format(index))
                tempRecordSeq[index:index + 3] = "---"
                #del tempRecordSeq[index:index + 3]
        record = SeqRecord(Seq("".join(tempRecordSeq)), id=record.id, description="",annotations={"molecule_type": "dna"})
        records_list.append(record)
    SeqIO.write(records_list, alignment_file_fasta, "fasta")
    return alignment_file_fasta
def check_exists(file_path, working_dir):
    "Checks if the input files exist in the working directory, otherwise it copies them there"
    file_name = ntpath.basename(file_path)

    join_path = os.path.join(working_dir,file_name)

    if not os.path.exists(join_path):
        shutil.copyfile(file_path,join_path)
    return join_path #new path
def Run_PAML(alignment_file_fasta,tree_file_newick6,palm_out_dir,working_dir,use_codeml_aa=True,use_codeml_codons=False):
    """Implemented as in https://evosite3d.blogspot.com/2014/09/tutorial-on-ancestral-sequence.html
    All the files need to be in the same folder
    As for the article: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5027606/#S1
    The 19 ‘modern' sequences at the tips (leaves) of the FP phylogeny were used to computationally reconstruct infer ancestral sequences at all
     internal nodes of the tree using the evolved (known) topology. Marginally reconstructed ancestral sequences were inferred using Bayesian
      approaches that incorporated the WAG amino acid replacement matrix (PAML, FastML and PhyloBayes [CAT]), with or without rate variation
      as modelled by a discrete gamma distribution (four rate categories), and ancestral sequences were also inferred with the MP criterion
      (as implemented in PAML). DNA and codon-based analyses were performed only in PAML using HKY85+GC and M0(F3x4), respectively. ProtTest v3.2
       was used to analyse the various models according to the AIC criterion (AIC weight was 100% for WAG_Г)33."""
    # def check_exists(file_path):
    #     "Checks if the input files exist in the working directory, otherwise it copies them there"
    #     file_name = ntpath.basename(file_path)
    #     join_path = os.path.join(working_dir,file_name)
    #     if not os.path.exists(join_path):
    #         shutil.copyfile(file_path,join_path)
    #     return join_path #new path

    alignment_file_fasta= check_exists(alignment_file_fasta,working_dir)
    if sequence_input_type == "DNA":
        alignment_file_fasta= remove_stop_codons(alignment_file_fasta)
    tree_file_newick6=check_exists(tree_file_newick6,working_dir)
    if use_codeml_aa:
        print("Using codeml, codon substitution models with aa in this case")
        print("REMEMBER: THE TREE SHALL NOT CONTAIN REPEATED NAMES AND NO INTERNAL NODES NAMES!!!")
        #palm_out = open(palm_out_dir,"w+").close()

        cml = codeml.Codeml(
            alignment=alignment_file_fasta,
            tree=tree_file_newick6,
            out_file=palm_out_dir,
            working_dir=working_dir)

        cml.set_options(noisy=9)
        cml.set_options(verbose=2) #2: Include detail info on post prob per node, otherwise it does not
        cml.set_options(runmode=0) #* 0: user tree; 1: semi-automatic; 2: automatic
                                   #* 3: StepwiseAddition; (4,5):PerturbationNNI
        cml.set_options(seqtype=2) #* 1:codons; 2:AAs; 3:codons(3nt-->AAs)
        #cml.set_options(NSsites=0)  #* 0:one w;1:neutral;2:selection; 3:discrete;4:freqs;
                                    #* 5:gamma;6:2gamma;7:beta;8:beta&w;9:beta&gamma;
                                    #* 10:beta&gamma+1; 11:beta&normal>1; 12:0&2normal>1;
                                    #* 13:3normal>0
        cml.set_options(clock=0) #*0 = no molecular rate, genes are evolving a different rate ---> for unrooted trees
                                 #*1 = global clock, and all branches have the same rate.---> rooted tree
                                 #*2 =local clock models, the user specifies at which rates the branches evolve
                                 #*3 = multiple genes
        cml.set_options(aaDist=0) #* 0:equal, +:geometric; -:linear, 1-6:G1974,Miyata,c,p,v,a * 7:AAClasses
        cml.set_options(getSE=0) #0, 1, or 2 tells whether we want estimates of the standard errors of estimated parameters.
        cml.set_options(RateAncestor=1) #force the empirical bayesian estimation (parsimony) of ancestral states
        cml.set_options(aaRatefile="/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/PAML/Matrix_aa/wag.dat") #wag.dat * only used for aa seqs with model=empirical(_F)
                                                                                                                #* dayhoff.dat, jones.dat, wag.dat, mtmam.dat, or your own
        cml.set_options(method=1) #method = 0 implements the old algorithm in PAML, which updates all parameters including branch lengths simultaneously
                                  #method = 1 specifies an algorithm newly implemented in PAML, which updates branch lengths one by one. --> only works with clock=0
        cml.set_options(model=2) #* models for codons:
                                    # * 0:one ω ratio for all branches,
                                    # * 1: means one ratio for each branch (the free-ratio model) --->BEFORE
                                    # * 2:2 or more dN/dS ratios for branches (needs to be set by user) --> generates nan values with wag.dat
                                 #* models for AAs or codon-translated AAs:
                                    # * 0:poisson, 1:proportional,2:Empirical,3:Empirical+F
                                    # * 5:FromCodon0, 6:FromCodon1, 8:REVaa_0, 9:REVaa(nr=189)
        cml.set_options(fix_alpha=0) #* 0: estimate gamma shape parameter;
                                     #* 1: fix it at alpha
        cml.set_options(alpha=0.5) #initial or fixed alpha, 0:infinity (constant rate) for the gamma distribution
        cml.set_options(cleandata=0) #keep all the ambigous data ("-","X")  --> remove sites with ambiguity data (1:yes, 0:no)?
        cml.set_options(fix_blength=0) # 0= (ignore) means that branch lengths in the tree file (if they exist) will be ignored and initial branch lengths are generated using pairwise distances and random numbers.
                                      # 1 (initial) means that branch lengths in the tree file will be used as initial values for ML iteration.
                                       # 2 (fixed) means that branch lengths will be fixed at those values given in the tree file and not estimated by ML
                                       # 3 (proportional) means that branch lengths will be proporational to those given in the tree file, and the proportionality factor is estimated by ML
                                       # –1 (random) means that random initial branch lengths will be used. T
        path = str(pathlib.Path(palm_out_dir).parent)
        cml.ctl_file = "{}/control_file_{}.ctl".format(path,name)
        cml.write_ctl_file()
        #Highlight: Output files of interest are : .mlc (contains information on evolutionary rates); .rst (contains ancestral states for sites and for nodes)
        #Highlight: PAML and FASTML do not predict gaps, they either ignore those columns or simply put the most likely aa (which is the most frequent aa found in the column)
        print("Go to {} and run codeml ctl_filename".format(path))
        exit()
        cml.run("{}/control_file_{}.ctl".format(path,name),verbose=True) #Does not work when running it from python script, therefore just run directly on terminal
    elif use_codeml_codons:
        cml = codeml.Codeml(
            alignment=alignment_file_fasta,
            tree=tree_file_newick6,
            out_file=palm_out_dir,
            working_dir=working_dir)

        cml.set_options(noisy=9)
        cml.set_options(verbose=2)  # 2: Include detail info on post prob per node, otherwise it does not
        cml.set_options(runmode=0)  # * 0: user tree; 1: semi-automatic; 2: automatic
        # * 3: StepwiseAddition; (4,5):PerturbationNNI
        cml.set_options(seqtype=1)  # * 1:codons; 2:AAs; 3:codons(3nt-->AAs)
        # cml.set_options(NSsites=0)  #* 0:one w;1:neutral;2:selection; 3:discrete;4:freqs;
        # * 5:gamma;6:2gamma;7:beta;8:beta&w;9:beta&gamma;
        # * 10:beta&gamma+1; 11:beta&normal>1; 12:0&2normal>1;
        # * 13:3normal>0
        cml.set_options(clock=0)  # *0 = no molecular rate, genes are evolving a different rate ---> for unrooted trees
        # *1 = global clock, and all branches have the same rate.---> rooted tree
        # *2 =local clock models, the user specifies at which rates the branches evolve
        # *3 = multiple genes
        cml.set_options(aaDist=0)  # * 0:equal, +:geometric; -:linear, 1-6:G1974,Miyata,c,p,v,a * 7:AAClasses
        cml.set_options(getSE=0)  # 0, 1, or 2 tells whether we want estimates of the standard errors of estimated parameters.
        cml.set_options(RateAncestor=1)  # force the empirical bayesian estimation (parsimony) of ancestral states
        cml.set_options(aaRatefile="/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/PAML/Matrix_aa/wag.dat")  # wag.dat * only used for aa seqs with model=empirical(_F)
        # * dayhoff.dat, jones.dat, wag.dat, mtmam.dat, or your own
        cml.set_options(method=1)  # method = 0 implements the old algorithm in PAML, which updates all parameters including branch lengths simultaneously
        # method = 1 specifies an algorithm newly implemented in PAML, which updates branch lengths one by one. --> only works with clock=0
        cml.set_options(model=0)  # * models for codons:
        # * 0:one ω ratio for all branches,
        # * 1: means one ratio for each branch (the free-ratio model)
        # * 2:2 or more dN/dS ratios for branches (needs to be set by user)
        # * models for AAs or codon-translated AAs:
        # * 0:poisson, 1:proportional,2:Empirical,3:Empirical+F
        # * 5:FromCodon0, 6:FromCodon1, 8:REVaa_0, 9:REVaa(nr=189)
        cml.set_options(fix_alpha=0)  # * 0: estimate gamma shape parameter;
                                     # * 1: fix it at alpha
        cml.set_options(alpha=0.5)  # initial or fixed alpha, 0:infinity (constant rate) for the gamma distribution
        cml.set_options(cleandata=0)  # keep all the ambigous data ("-","X")  --> remove sites with ambiguity data (1:yes, 0:no)?
        cml.set_options(fix_blength=0)  # 0= (ignore) means that branch lengths in the tree file (if they exist) will be ignored and initial branch lengths are generated using pairwise distances and random numbers.
        # 1 (initial) means that branch lengths in the tree file will be used as initial values for ML iteration.
        # 2 (fixed) means that branch lengths will be fixed at those values given in the tree file and not estimated by ML
        # 3 (proportional) means that branch lengths will be proporational to those given in the tree file, and the proportionality factor is estimated by ML
        # –1 (random) means that random initial branch lengths will be used. T
        path = str(pathlib.Path(palm_out_dir).parent)
        cml.ctl_file = "{}/control_file_{}.ctl".format(path, name)
        cml.write_ctl_file()
        # Highlight: Output files of interest are : .mlc (contains information on evolutionary rates); .rst (contains ancestral states for sites and for nodes)
        # Highlight: PAML and FASTML do not predict gaps, they either ignore those columns or simply put the most likely aa (which is the most frequent aa found in the column)
        print("Go to {} and run codeml ctl_filename".format(path))
        print("{}/control_file_{}.ctl".format(path, name))
        exit()
        cml.run("{}/control_file_{}.ctl".format(path, name), verbose=True)

def Convert_to_Format(alignment_file_fasta,mol_type,out_format):
    """File conversion needed from fasta to nexus to infer ancestral sites for MrBayes
    Alternative: seqmagick convert --output-format nexus --alphabet dna BetaLactamase_seq_True_alignment.FASTA BetaLactamase_seq_True_alignment.nex
    mol_type = DNA or protein
    """
    path = pathlib.Path(alignment_file_fasta).parent
    new_name= str(path) + "/" +ntpath.basename(alignment_file_fasta).split(".")[0] + "." + out_format
    if out_format == "phylip-relaxed":
        new_name = str(path) + "/" + ntpath.basename(alignment_file_fasta).split(".")[0] + "." + "phylip"

    records = AlignIO.read(alignment_file_fasta, "fasta")
    records = [SeqRecord(seq.seq,id=seq.id,annotations={"molecule_type": mol_type}) for seq in records]
    SeqIO.write(records,new_name , out_format)

def Run_MrBayes():

    """
    Command line interface
    Requires nexus file!


    mrbayes
    execute filename.nex
    MrBayes parse
    /parsemb.pl --cutoff 0.99 --unknown N BLactamase/Dataset1/BLactamase_True_alignment.nex.pstat 2> /dev/null | fold > BLactamase/Dataset1/AncestralSeq.fasta
    """

def Run_MEGA():
    """
    GUI interface, call with:
    megax
    https://www.youtube.com/watch?v=djju9WFMvn0
    ExtAncSeqMEGA anc.txt
    """

def Run_PhyloBayes(alignment_file_phylip,tree_file_newick6,working_dir):
    """In:
    alignment_file : Phylip format (DNA sequences). Nodes labels alphanumeric, only leaves nodes
    tree-file: Newick format 6. Uses a tree file with only tree leaves labels as fixed topology and patristic distances---> No internal labels and labels must be alphanumeric (not just numeric)!
    working_dir : Directory where to run the program (inpt files have to be there)
    Understanding PhyloBayes: https://github.com/wrf/graphphylo"""
    #Highlight: Warning if pb is not in the path, start PyCharm from the terminal
    #Highlight: Warning chainname ALWAYS at the end of the command
    print("Running PhyloBayes!!!...")
    alignment_file_phylip= check_exists(alignment_file_phylip,working_dir)
    tree_file_newick6=check_exists(tree_file_newick6,working_dir)

    chainname_a = "A"
    chainname_b = "B"
    all_files = [file for file in glob.glob('{}/*'.format(working_dir)) if ntpath.basename(file).startswith(chainname_a+".") or ntpath.basename(file).startswith(chainname_b+".")]
    if click.confirm('Do you want to delete previous files?', default=False):
        print("Deleting files from previous run ...")
        for file in all_files:
            os.remove(file)
        try:
            os.remove("{}/tracecomp.contdiff".format(working_dir))
        except:
            pass
    #"-s saves all the files (otherwise .chain file is not saved!)"
    print("Running chains  {} {}".format(chainname_a,chainname_b))

    proc1=subprocess.Popen(args=["pb","-s","-f","-T",tree_file_newick6,"-cat", "-gtr","-d", alignment_file_phylip,chainname_a],cwd=working_dir,stdout=open(os.devnull, 'wb'),stderr=open(os.devnull, 'wb')) #stdin=PIPE, stderr=PIPE, stdout=PIPE
    proc2 = subprocess.Popen(args=["pb", "-s", "-f","-T",tree_file_newick6,"-cat", "-gtr","-d", alignment_file_phylip,chainname_b],cwd=working_dir,stdout=open(os.devnull, 'wb'),stderr=open(os.devnull, 'wb'))
    #Start running chains in the background and check that they are working

    #Wait a little to get results
    print("Waiting for chain mixing")
    proc3 = subprocess.Popen(args=["sleep","40"]) #switched from 40 to 2
    proc3.communicate()
    #maxdiff < 0.1 and minimum effective size > 300: good run;
    maxdiff = 1 #starting point
    minimum_effective_size = 1
    while not minimum_effective_size > 300 or not maxdiff < 0.1 :
        result = False
        while not result:
            subprocess.call(args=["tracecomp", "-x", "100", chainname_a, chainname_b],
                            cwd=working_dir)  # ,stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb'))
            subprocess.Popen(args=["bpcomp", "-x", "100", "10", chainname_a, chainname_b],
                             cwd=working_dir)  # ,stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb'))

            try:
                tracecomp_file = open("{}/tracecomp.contdiff".format(working_dir),"r+")
                bpcomp_file = open("{}/bpcomp.bpdiff".format(working_dir), "r+")
                result = True
            except:
                print("waiting for more chain mixing....")
                proc5 = subprocess.Popen(args=["sleep", "20"])
                proc5.communicate()
                #tracecomp_file = open("{}/tracecomp.contdiff".format(working_dir), "r+")
                #bpcomp_file = open("{}/bpcomp.bpdiff".format(working_dir), "r+")
        trf = tracecomp_file.read().split('\n')
        ll = trf[trf.index("name                effsize\trel_diff") + 2]
        reldiff = float(ll.split()[2])
        print("rel diff {}".format(reldiff))
        minimum_effective_size = float(ll.split()[1])
        print("minimum effective size {}".format(minimum_effective_size))
        bpc = bpcomp_file.read().split('\n')
        maxdiff = float(bpc[1].split()[2])
        print("max diff {}".format(maxdiff))
        print("Running chain longer")
        proc7 = subprocess.Popen(args=["sleep", "20"])
        proc7.communicate()

    print("Final maxdiff {}".format(maxdiff))
    print("Final minimum effective size {}".format(minimum_effective_size))
    print("Exiting loop. Waiting...")
    proc8 = subprocess.Popen(args=["sleep", "20"])
    proc8.communicate()
    print("sampling ancestral sequences...")
    subprocess.call(args=["ancestral","-x","100","100", chainname_a],cwd=working_dir,shell=False)#,stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb'))  # compute the ancestral sequences
    proc10 = subprocess.Popen(args=["sleep", "20"])
    proc10.communicate()
    print("Stopping chains")
    #Highlight: Overwrites the .run files with 0
    subprocess.call('echo "0" > {}.run'.format(chainname_a), shell=True,cwd=working_dir)
    subprocess.call('echo "0" > {}.run'.format(chainname_b), shell=True, cwd=working_dir)
    #proc9 = subprocess.Popen(args=["/bin/echo", "0", ">", "{}.run".format(chainname_b)],cwd=working_dir)  # Stop the sampling for chain b

def Run_BaliPhy(alignment_file):
    """
    WARNING: BaliPhy does not allow for fixed input tree topology
    From Douglas's paper:
    Both phylogeny and
    alignment were co-estimated using the Bayesian BAli-Phy software package (Fig. S1)
    (34). The analysis was performed using the RS07 insertion/deletion model, LG amino
    acid substitution matrix, estimating equilibrium amino acid frequencies, with gamma
    distributed rates across sites (four categories). Two independent chains were run until the
    ASDSF and PSRF-80%CI criteria fell below 0.01 and 1.01, respectively.

    >>Sampled alignments , including ancestral seqs, for the nth partition are in the file. C1.Pn.fastas.
    >>Ancestral states in these alignments are randomly sampled from their joint posterior and do not represent the most probable ancestral state.
    The alignment of ancestral sequences is also inferred, so these sequences may contain gaps.
    The length of ancestral sequences may vary between samples when the length of the ancestral sequence is uncertain.

    NOTE to self: It it gives statreport: Error! The 0th column name is blank!, just keep running the chain for a long time, eventually it will work
    BaliPhy flags:

    First arg---ALigned sequences
    I---None-Run a traditional fixed alignment tree
    A----DNA
    S---substitution model"""

    #alignment_file = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/BaliPhy/{}/Dataset{}/{}_True_alignment.FASTA".format(simulation_folder,dataset_number,root_sequence_name)
    #alignment_file = "/home/lys/Dropbox/PhD/DRAUPNIR/GPFCoralDataset/Coral_Faviina/Faviina_Aligned_DNA.fasta"
    #alignment_file = "/home/lys/Dropbox/PhD/DRAUPNIR/GPFCoralDataset/Coral_all/Coral_all_Aligned_DNA.fasta"
    #alignment_file = "/home/lys/Dropbox/PhD/DRAUPNIR/GPFCoralDataset/Cnidarian_FP/Cnidarian_Aligned_DNA.fasta"
    #alignment_file = "/home/lys/Dropbox/PhD/DRAUPNIR/Douglas_SRC_Dataset/NewNaming/SrcAblAlign.fasta"
    alignment_file = "/home/lys/Dropbox/PhD/DRAUPNIR/Douglas_SRC_Dataset/SrcAblALIGNED.mafft"
    print(alignment_file)

    working_dir = pathlib.Path(alignment_file).parent
    file_name = ntpath.basename(alignment_file).split(".")[0]
    #Highlight: delete folders from previous runs
    all_folders = [folder for folder in glob.glob('{}/*'.format(working_dir)) if file_name + "-" in folder]
    print("Running BaliPhy...")
    if click.confirm('Do you want to delete previous files?', default=True):
        print("Deleting previous run ...")
        for folder in all_folders:
            shutil.rmtree(folder)
        try:
            os.remove("{}/c50.PP.tree".format(working_dir))
            os.remove("{}/c70.PP.tree".format(working_dir))
            os.remove("{}/Statreport.txt".format(working_dir))
            os.remove("{}/Bootstrap".format(working_dir))
        except:pass
    #Highlight: Run several chains
    n_chains = 4
    for i in range(1,n_chains+1):
        print("Started Chain {}".format(i))
        proc = subprocess.Popen(args=["bali-phy",alignment_file,"-S","gtr+Rates.gamma[4]+inv","-I","none","--set", "write-fixed-alignments=true"],
                                cwd=working_dir,
                                stdout=open(os.devnull, 'wb'),
                                stderr=open(os.devnull, 'wb'))
        print("Waiting....")
        procsleep = subprocess.Popen(args=["sleep", "10"])
        procsleep.communicate()
    proc6 = subprocess.Popen(args=["sleep", "20"])
    proc6.communicate()
    #Store the new folders
    all_folders = [folder for folder in glob.glob('{}/*'.format(working_dir)) if file_name + "-" in folder]
    max_psrf = 1.01  # should be lower than this
    max_asdsf = 0.01 #should be lower than this
    psrf=2
    asdsf = 1
    # Highlight: Checking PSRF (Potential Scale Reduction factors)--> Differences among the posterior distributions of the different chains
    all_logs = list(map(lambda folder: "{}/C1.log".format(ntpath.basename(folder)), all_folders))
    all_trees = list(map(lambda folder: "{}/C1.trees".format(ntpath.basename(folder)), all_folders))
    while not psrf <= max_psrf or not asdsf <= max_asdsf:
        subprocess.run(["statreport"] + all_logs, shell=False, cwd=working_dir,stdout=open("{}/Statreport.txt".format(working_dir), 'w+'))
        try:
            psrf_file =  open("{}/Statreport.txt".format(working_dir), "r+").read().split('\n')
            psrf = float(psrf_file[psrf_file.index("Increasing: iter")-3].split()[2])
            ess = psrf_file[psrf_file.index("Increasing: iter") - 5].split()[2] #ess = effective sample size, it's named as Ne---> split ESS values
            # Highlight: Checking topology convergence ASDSF (Average Standard Deviation of Split Frequencies )
            subprocess.call(["trees-bootstrap"] + all_trees, shell=False, cwd=working_dir,stdout=open("{}/Bootstrap.txt".format(working_dir), 'wb'))
            asdsf_file = open("{}/Bootstrap.txt".format(working_dir), 'r+').read().split('\n')
            asdsf = float(asdsf_file[-3].split()[2])
            msdsf = asdsf_file[-3].split()[-1] #maximum of sdsf values--> Range of variation in posterior probabilities across the runs for the split for the most variation
            split_ess = asdsf_file[-2].split()[3]
            print("ASDSF is {}".format(asdsf))
            print("PSRF is {}".format(psrf))
        except:
            print("Running chain longer")
            proc5 = subprocess.Popen(args=["sleep", "10"])
            proc5.communicate()
    #pattern = re.compile("\bBetaLactamase_seq_True_alignment-\b")


    #Highlight: Finding the majority consensus tree
    all_trees  = list(map(lambda folder: "{}/C1.trees".format(ntpath.basename(folder)), all_folders))
    subprocess.call(["trees-consensus"]+all_trees,shell=False,cwd=working_dir,stdout=open("{}/c50.PP.tree".format(working_dir), 'wb'))
    subprocess.call(["trees-consensus","--consensus=0.9"] + all_trees, shell=False, cwd=working_dir,stdout=open("{}/c90.PP.tree".format(working_dir), 'wb'))

    if click.confirm('Do you want to kill all bali-phy processes?', default=True):
        print("Done,killing all Baliphy processes")
        subprocess.call('killall bali-phy', shell=True, cwd=working_dir)

def Run_PAUP():
    """
    https://www.cs.rice.edu/~ogilvie/phylogenetics-workshop/2019/02/20/paup-tutorial.html
    https://alohachou0.medium.com/paup-beast2-philip-fastcd-dd743b9698b3
    Runs with python 2.7
    Requires Nexus format"""
    alignment_file = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/BaliPhy/{}/Dataset{}/{}_True_alignment.nex".format(simulation_folder, dataset_number, simulation_folder)
    working_dir = pathlib.Path(alignment_file).parent
    subprocess.call(["paup",alignment_file],shell=False,cwd=working_dir)

def Run_IQTree(alignment_fasta, tree_necwickformat6,working_dir):
    "Find the best substitution model and reconstructs the ancestral sequences, not insertions/deletions reconstructions"
    alignment_fasta_path = check_exists(alignment_fasta,working_dir)
    subprocess.run(args=["iqtree","-s",alignment_fasta_path,"-m","TEST","-asr", "-te",tree_necwickformat6,"-nt","AUTO"],stderr=sys.stderr, stdout=sys.stdout)

def Run_FastML(alignment_file_fasta,tree_file_newick6,working_dir,fastml_out_dir):
    """Upload files to online server http://fastml.tau.ac.il/ ---> does not finish
    or download binaries from http://fastml.tau.ac.il/source.php
    store perl FastML_Wrapper.pl as alias fastml under bash_aliases
    Input:
    alignment_file_fasta: File in PHYLIP format containing aligned sequences (leaves) in fasta format. Labels must be alphanumeric (not just numeric).
     In principle they can be nucleotides or aa, but aa is not working
    tree_file_newick6 : Tree file in newick format 6 (see Ete3 formats) -->  No internal labels and labels must be alphanumeric (not just numeric)

    Output:
    Character reconstruction - two methods are implemented: the joint and the marginal.
    In the joint reconstruction, one finds the set of all the internal nodes sequences.
    In the marginal reconstruction, one infers the most likely sequence in a specific internal node.
    The results of these two estimation methods are not necessarily the same [1, 2]. Both methods are based
    on maximum likelihood (ML) algorithms and on an empirical Bayesian approach taking into account the rate variation among sites of the MSA."""

    #"perl","FastML_Wrapper.pl"
    program={"fastml":"/home/lys/FastML.v3.11/www/fastml/FastML_Wrapper.pl"}
    seqtype = ["nuc" if sequence_input_type == "DNA" else "aa"][0]
    submatrix = ["GTR" if sequence_input_type == "DNA" else "WAG"][0] #changed from WAG


    subprocess.call(["perl",program["fastml"],"--MSA_File",alignment_file_fasta,"--seqType",seqtype,"--SubMatrix",submatrix,"--indelReconstruction", "ML","--Tree",tree_file_newick6,"--outDir",fastml_out_dir],
                            shell=False,
                            env=dict(ENV='/home/lys/FastML.v3.11/www/fastml'),
                            cwd=working_dir)

def main():
    #TODO: Generalize with repository location = "/home/lys/Dropbox/PhD"
    if name.startswith("simulations"):
        file_origin = "/home/lys/Dropbox/PhD/DRAUPNIR/Datasets_Simulations/{}/Dataset{}".format(simulation_folder,dataset_number)
        alignment_file_fasta_DNA = "{}/{}_True_alignment.FASTA".format(file_origin, root_sequence_name)
        unaligned_file_fasta_DNA = "{}/{}_Unaligned.FASTA".format(file_origin,root_sequence_name)
        alignment_file_phylip_DNA = "{}/{}_True_alignment.phylip".format(file_origin, root_sequence_name)
        if not os.path.exists(alignment_file_phylip_DNA):
            Convert_to_Format(alignment_file_fasta_DNA, "DNA", "phylip")
        alignment_file_fasta_PEP = "{}/{}_True_Pep_alignment.FASTA".format(file_origin, root_sequence_name)
        alignment_file_phylip_PEP = "{}/{}_True_Pep_alignment.phylip".format(file_origin, root_sequence_name)
        if not os.path.exists(alignment_file_phylip_PEP):
            Convert_to_Format(alignment_file_fasta_PEP, "PROT", "phylip")

        codeml_working_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/PAML/CodeML/{}/Dataset{}/{}".format(simulation_folder, dataset_number,sequence_input_type,)
        codeml_out_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/PAML/CodeML/{}/Dataset{}/{}/PAML_ARS_codeml_{}.out".format(simulation_folder, dataset_number, sequence_input_type,simulation_folder + "Dataset_{}".format(dataset_number))

        tree_file_newick6 = "{}/{}_True_Rooted_tree_node_labels.format6newick".format(file_origin,root_sequence_name)
        tree_file_newick7 = "{}/{}_True_Rooted_tree_node_labels.format7newick".format(file_origin, root_sequence_name)
        tree_file_newick = "{}/{}_True_Rooted_tree_node_labels.newick".format(file_origin,root_sequence_name)
        phylobayes_working_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/PhyloBayes/{}/Dataset{}/{}".format(simulation_folder,dataset_number,sequence_input_type)
        iqtree_out_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/IQTree/{}/Dataset{}/{}".format(simulation_folder, dataset_number, sequence_input_type)
        fastml_working_dir = file_origin
        fastml_out_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/FastML/{}/Dataset{}/{}".format(simulation_folder,dataset_number,sequence_input_type)


    elif name == "benchmark_randall_original_naming":
        #Highlight: In format6newick tree the leaves have been added an A in front of the number, otherwise the other programs do not like sequences with only numbers in the names
        file_origin = "/home/lys/Dropbox/PhD/DRAUPNIR/AncestralResurrectionStandardDataset"
        alignment_file_fasta_DNA =unaligned_file_fasta_DNA= "{}/RandallExperimentalPhylogenyDNASeqsLEAVES.fasta".format(file_origin)
        alignment_file_phylip_DNA = "{}/RandallExperimentalPhylogenyDNASeqsLEAVES.phylip".format(file_origin)
        alignment_file_fasta_PEP = "{}/benchmark_randall_original_naming_corrected_leaves_names.mafft".format(file_origin)
        alignment_file_phylip_PEP = "{}/benchmark_randall_original_naming_corrected_leaves_names.phylip".format(file_origin)

        if not os.path.exists(alignment_file_phylip_DNA):
            Convert_to_Format(alignment_file_fasta_DNA, "DNA", "phylip")
        if not os.path.exists(alignment_file_phylip_PEP):
            Convert_to_Format(alignment_file_fasta_PEP, "PROT", "phylip")

        codeml_working_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/PAML/CodeML/AncestralResurrectionStandardDataset/{}".format(sequence_input_type)
        codeml_out_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/PAML/CodeML/AncestralResurrectionStandardDataset/{}/PAML_ARS_codeml_{}.out".format(sequence_input_type,name)

        tree_file_newick6 = "{}/RandallBenchmarkTree_OriginalNaming_corrected_leaves_names.format6newick".format(file_origin)
        tree_file_newick7 = "{}/RandallBenchmarkTree_OriginalNaming_corrected_leaves_names.format7newick".format(file_origin)
        tree_file_newick = "{}/RandallBenchmarkTree_OriginalNaming.newick".format(file_origin)
        phylobayes_working_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/PhyloBayes/AncestralResurrectionStandardDataset/{}".format(sequence_input_type)
        iqtree_out_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/IQTree/AncestralResurrectionStandardDataset/{}".format(sequence_input_type)
        fastml_out_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/FastML/AncestralResurrectionStandardDataset/{}".format(sequence_input_type)
        fastml_working_dir = file_origin

    elif name.startswith("Coral"):
        file_origin = "/home/lys/Dropbox/PhD/DRAUPNIR/GPFCoralDataset"
        alignment_file_fasta_PEP = "{}/{}/{}_Aligned_Protein.fasta".format(file_origin,name,name)
        #alignment_file_fasta_DNA = "{}/{}/{}_Aligned_DNA.fasta".format(file_origin,name,name)
        alignment_file_fasta_DNA = "{}/{}/{}_CodonAlignment_DNA.fasta".format(file_origin,name,name)
        alignment_file_phylip_PEP = "{}/{}/{}_Aligned_Protein.phylip".format(file_origin, name, name)
        alignment_file_phylip_DNA = "{}/{}/{}_Aligned_DNA.phylip".format(file_origin, name, name)
        if not os.path.exists(alignment_file_phylip_DNA):
            Convert_to_Format(alignment_file_fasta_DNA, "DNA", "phylip")
        if not os.path.exists(alignment_file_phylip_PEP):
            Convert_to_Format(alignment_file_fasta_PEP, "PROT", "phylip")
        codeml_working_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/PAML/CodeML/{}/{}".format(name,sequence_input_type)
        codeml_out_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/PAML/CodeML/{}/{}/PAML_ARS_codeml_{}.out".format(name,sequence_input_type, name)
        tree_file_newick6 = "{}/{}/c90.format6newick".format(file_origin,name)
        tree_file_newick7 = "{}/{}/c90.format7newick".format(file_origin, name)
        tree_file_newick = "{}/{}/c90.newick".format(file_origin,name)
        phylobayes_working_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/PhyloBayes/{}/{}".format(name,sequence_input_type)
        iqtree_out_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/IQTree/{}/{}".format(name,sequence_input_type)
        fastml_out_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/FastML/{}/{}".format(name,sequence_input_type)
        fastml_working_dir = file_origin
    elif name.startswith("Cnidarian"):
        file_origin = "/home/lys/Dropbox/PhD/DRAUPNIR/GPFCoralDataset"
        alignment_file_fasta_PEP = "{}/Cnidarian_FP/Cnidarian_Aligned_PROTEIN.fasta".format(file_origin,name,name)
        #alignment_file_fasta_DNA = "{}/{}/{}_Aligned_DNA.fasta".format(file_origin,name,name)
        alignment_file_fasta_DNA = "{}/Cnidarian_FP/Cnidarian_Aligned_DNA.fasta".format(file_origin,name,name)
        alignment_file_phylip_PEP = "{}/Cnidarian_FP/Cnidarian_Aligned_PROTEIN.phylip".format(file_origin, name, name)
        alignment_file_phylip_DNA = "{}/Cnidarian_FP/Cnidarian_Aligned_DNA.phylip".format(file_origin, name, name)
        if not os.path.exists(alignment_file_phylip_DNA):
            Convert_to_Format(alignment_file_fasta_DNA, "DNA", "phylip")
        if not os.path.exists(alignment_file_phylip_PEP):
            Convert_to_Format(alignment_file_fasta_PEP, "PROT", "phylip")
        codeml_working_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/PAML/CodeML/{}/{}".format(name,sequence_input_type)
        codeml_out_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/PAML/CodeML/{}/{}/PAML_ARS_codeml_{}.out".format(name,sequence_input_type, name)
        tree_file_newick6 = "{}/Cnidarian_FP/c90.format6newick".format(file_origin,name)
        tree_file_newick7 = "{}/Cnidarian_FP/c90.format7newick".format(file_origin, name)
        tree_file_newick = "{}/Cnidarian_FP/c90.newick".format(file_origin,name)
        phylobayes_working_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/PhyloBayes/{}/{}".format(name,sequence_input_type)
        iqtree_out_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/IQTree/{}/{}".format(name,sequence_input_type)
        fastml_out_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/FastML/{}/{}".format(name,sequence_input_type)
        fastml_working_dir = file_origin
    elif name.endswith("_subtree"):
        file_origin = "/home/lys/Dropbox/PhD/DRAUPNIR/Douglas_SRC_Dataset/{}".format(name)
        alignment_file_fasta_PEP = "{}/{}_ALIGNED.mafft".format(file_origin, name.replace("_subtree",""))
        if sequence_input_type == "DNA": raise ValueError("Not DNA sequences available, only Protein")
        alignment_file_fasta_DNA  = alignment_file_phylip_DNA = None
        alignment_file_phylip_PEP = "{}/{}_ALIGNED.phylip-relaxed".format(file_origin, name.replace("_subtree",""))
        if not os.path.exists(alignment_file_phylip_PEP):
            Convert_to_Format(alignment_file_fasta_PEP, "PROT", "phylip-relaxed")
        codeml_working_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/PAML/CodeML/{}/{}".format(name,sequence_input_type)
        codeml_out_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/PAML/CodeML/{}/{}/PAML_ARS_codeml_{}.out".format(name, sequence_input_type, name)
        tree_file_newick6 = "{}/{}_ALIGNED.mafft.treefile".format(file_origin, name.replace("_subtree",""))
        tree_file_newick7 = "{}/{}_ALIGNED.format7newick".format(file_origin, name.replace("_subtree", ""))
        tree_file_newick = "{}/{}_ALIGNED.newick".format(file_origin, name.replace("_subtree", ""))

        phylobayes_working_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/PhyloBayes/{}/{}".format(name,sequence_input_type)
        fastml_out_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/FastML/{}/{}".format(name,sequence_input_type)
        fastml_working_dir = file_origin
        iqtree_out_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/IQTree/{}/{}".format(name,sequence_input_type)
    elif name in ["PKinase_PF07714","PF00400","PF00400_200","Douglas_SRC","SH3_pf00018_larger_than_30aa"]:
        file_origin = "/home/lys/Dropbox/PhD/DRAUPNIR/Mixed_info_Folder"
        alignment_file_fasta_PEP = "{}/{}.mafft".format(file_origin,name)
        if sequence_input_type == "DNA": raise ValueError("Not DNA sequences available, only Protein")
        alignment_file_fasta_DNA  = alignment_file_phylip_DNA = None
        alignment_file_phylip_PEP = "{}/{}.phylip".format(file_origin, name)
        if not os.path.exists(alignment_file_phylip_PEP):
            Convert_to_Format(alignment_file_fasta_PEP, "PROT", "phylip-relaxed")
        codeml_working_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/PAML/CodeML/{}/{}".format(name,sequence_input_type)
        codeml_out_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/PAML/CodeML/{}/{}/PAML_ARS_codeml_{}.out".format(name, sequence_input_type, name)
        tree_file_newick6 = "{}/{}.mafft.treefile".format(file_origin, name)
        tree_file_newick7 = "{}/{}.format7newick".format(file_origin, name)
        tree_file_newick = "{}/{}.newick".format(file_origin, name)

        phylobayes_working_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/PhyloBayes/{}/{}".format(name,sequence_input_type)
        fastml_out_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/FastML/{}/{}".format(name,sequence_input_type)
        fastml_working_dir = file_origin
        iqtree_out_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/IQTree/{}/{}".format(name,sequence_input_type)

    elif name in ["PF00096"]:
        file_origin = "/home/lys/Dropbox/PhD/DRAUPNIR/Dataset_PFAM"
        alignment_file_fasta_PEP = "{}/{}.fasta".format(file_origin,name)
        if sequence_input_type == "DNA": raise ValueError("Not DNA sequences available, only Protein")
        alignment_file_fasta_DNA  = alignment_file_phylip_DNA = None
        alignment_file_phylip_PEP = "{}/{}.phylip".format(file_origin, name)
        if not os.path.exists(alignment_file_phylip_PEP):
            Convert_to_Format(alignment_file_fasta_PEP, "PROT", "phylip-relaxed")
        codeml_working_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/PAML/CodeML/{}/{}".format(name,sequence_input_type)
        codeml_out_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/PAML/CodeML/{}/{}/PAML_ARS_codeml_{}.out".format(name, sequence_input_type, name)
        tree_file_newick6 = "{}/{}.fasta.treefile".format(file_origin, name)
        tree_file_newick7 = "{}/{}.format7newick".format(file_origin, name)
        tree_file_newick = "{}/{}.newick".format(file_origin, name)

        phylobayes_working_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/PhyloBayes/{}/{}".format(name,sequence_input_type)
        fastml_out_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/FastML/{}/{}".format(name,sequence_input_type)
        fastml_working_dir = file_origin
        iqtree_out_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/IQTree/{}/{}".format(name,sequence_input_type)

    elif name in ["aminopeptidase"]:
        file_origin = "/home/lys/Dropbox/PhD/DRAUPNIR/AminoPeptidase"
        alignment_file_fasta_PEP = "{}/2MAT_BLAST90.fasta".format(file_origin, name)
        if sequence_input_type == "DNA": raise ValueError("Not DNA sequences available, only Protein")
        alignment_file_fasta_DNA = alignment_file_phylip_DNA = None
        alignment_file_phylip_PEP = "{}/2MAT_BLAST90.phylip".format(file_origin, name)
        if not os.path.exists(alignment_file_phylip_PEP):
            Convert_to_Format(alignment_file_fasta_PEP, "PROT", "phylip-relaxed")
        codeml_working_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/PAML/CodeML/{}/{}".format(name,
                                                                                                         sequence_input_type)
        codeml_out_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/PAML/CodeML/{}/{}/PAML_ARS_codeml_{}.out".format(
            name, sequence_input_type, name)
        tree_file_newick6 = "{}/2MAT_BLAST90.fasta.treefile".format(file_origin, name)
        tree_file_newick7 = "{}/2MAT_BLAST90.format7newick".format(file_origin, name)
        tree_file_newick = "{}/2MAT_BLAST90.newick".format(file_origin, name)

        phylobayes_working_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/PhyloBayes/{}/{}".format(name,
                                                                                                            sequence_input_type)
        fastml_out_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/FastML/{}/{}".format(name,
                                                                                                sequence_input_type)
        fastml_working_dir = file_origin
        iqtree_out_dir = "/home/lys/Dropbox/PhD/DRAUPNIR/Benchmark_Results/IQTree/{}/{}".format(name,
                                                                                                sequence_input_type)



    else:
        print("Paths not available")
        raise NotImplementedError

    f = open("BenchmarkTimeTable.txt","a+")
    start = time.time()
    if program == "PAML":
        if sequence_input_type == "DNA":
            print("Using codons with PAML, have not found only nucleotides option (not even in BaseML)")
            Run_PAML(alignment_file_fasta_DNA, tree_file_newick6, codeml_out_dir, codeml_working_dir, use_codeml_aa=False,use_codeml_codons=True)
        elif sequence_input_type == "PROTEIN":
            Run_PAML(alignment_file_fasta_PEP,tree_file_newick6,codeml_out_dir,codeml_working_dir,use_codeml_aa=True,use_codeml_codons=False)
    elif program == "PhyloBayes":
        if sequence_input_type == "DNA":
            Run_PhyloBayes(alignment_file_phylip_DNA,tree_file_newick6,phylobayes_working_dir)
        elif sequence_input_type == "PROTEIN":
            Run_PhyloBayes(alignment_file_phylip_PEP, tree_file_newick6, phylobayes_working_dir)
    elif program == "FastML":
        if sequence_input_type == "DNA":
            Run_FastML(alignment_file_phylip_DNA,tree_file_newick6,fastml_working_dir,fastml_out_dir)
        elif sequence_input_type == "PROTEIN":
            Run_FastML(alignment_file_phylip_PEP, tree_file_newick6, fastml_working_dir, fastml_out_dir)
    elif program == "IQTree":
        if sequence_input_type == "DNA":
            Run_IQTree(alignment_file_fasta_DNA,tree_file_newick7,iqtree_out_dir)
        elif sequence_input_type == "PROTEIN":
            Run_IQTree(alignment_file_fasta_PEP,tree_file_newick7,iqtree_out_dir)
    else:
        print("Select a program or you are doing something wrong with your life")
    stop = time.time()
    f.write("{}\t{}\t{}\t{}\n".format(program,name, sequence_input_type,str(datetime.timedelta(seconds=stop-start)) ))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Benchmarking args")
    parser.add_argument('-dn', '--dataset-number', default=24, type=int, help='Dataset number')
    parser.add_argument('-input-type', '--input-type', default="DNA", type=str, help='Sequence input type, DNA or Protein')
    parser.add_argument('-program', '--benchmark-program', default=0, type=int, help='Benchmarking program, {"PAML":0,"PhyloBayes":1,"FastML":2,"IQTree":3} ')
    args = parser.parse_args()
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
                20:["simulations_CNP_1", 1, "CNP_simulations", "CNP_seq"],  # EvolveAGene4 CNP simulation # 1000 leaves, small alignment
                22:["PF01038_msa",None,None,None],
                23: ["simulations_insulin_1", 1, "Insulin_simulations", "Insulin_seq"],# EvolveAGene4 Insulin simulation #50 leaves
                24: ["simulations_insulin_2", 2, "Insulin_simulations", "Insulin_seq"],# EvolveAGene4 Insulin simulation #400 leaves
                25: ["simulations_PIGBOS_1", 1, "PIGBOS_simulations", "PIGBOS_seq"],# EvolveAGene4 PIGBOS simulation #300 leaves
                27: ["PF00400",None,None,None],
                28: ["aminopeptidase", None, None, None],
                29: ["PF01038_lipcti_msa_fungi", None, None, None],
                30: ["PF00096", None, None, None],
                31: ["PF00400_200", None, None, None]}

    name, dataset_number, simulation_folder, root_sequence_name = datasets[args.dataset_number]
    programs = {0:"PAML",1:"PhyloBayes",2:"FastML",3:"IQTree"}
    program = programs[args.benchmark_program]
    sequence_input_type = args.input_type
    print("Runnning dataset : {}".format(name))
    print("Using {} as input type".format(sequence_input_type))
    print("Using program {}".format(programs[args.benchmark_program]))
    #Run_BaliPhy(None) #For making trees
    main()




