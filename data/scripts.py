import sys
sys.path.append("..")
from lemma_baseline import qa_utils
from evaluation import util
import time
import numpy as np
import os
import subprocess
from collections import defaultdict

def count_conjunctions(fname):
    f = open(fname)
    num_conj = 0
    all_n = 0
    for l in f:
        if not l.startswith("predicate"):
            continue
        try:
            all_n += 1
            p = l.split()[1].split("#")[0][1:-1]
            print ("p: ", p)
            if is_conj(p):
                num_conj += 1
        except:
            pass
    print ("num_conj: ", num_conj, "from ", all_n)

def is_conj(p):
    ps = p.split(",")
    return ps[0]==ps[1]

def assign_types_file(fname,oname):
    #read the lines and form the graph
    start = time.time()
    lines = [line.rstrip() for line in open(fname)]
    print ("read lines: ", (time.time()-start))
    if len(lines)==0:
        return
    op = open(oname,"w")
    idx = 0
    op.write(lines[0]+"\n")

    types = lines[idx].split(",")[0].split(" ")[1].split("#")
    typesD = types[0]+"#"+types[1] #Direct
    typesR = types[1]+"#"+types[0] #Reverse

    if types[0]==types[1]:
        typesD = types[0]+"_1#"+types[0]+"_2" #Direct
        typesR = types[0]+"_2#"+types[0]+"_1" #Reverse


    idx += 1

    start = time.time()

    while idx<len(lines):
        line = lines[idx];
        #time to read a predicate
        pred = line[11:]
        #print "pred: ", pred

        if qa_utils.is_sorted(pred):
            pred = pred + "#" + typesD
        else:
            pred = qa_utils.swap(pred) + "#" + typesR

        op.write("predicate: "+pred+"\n")
        idx+=1
        op.write(lines[idx]+"\n")
        idx+=1

        #Now, we're on cos similarity
        #The neighbors

        while idx<len(lines) and not (lines[idx].startswith("predicate:")):
            line = lines[idx]

            if line == "":
                idx += 1
                op.write("\n")
                continue

            if "__" in line:
                #print "line was: ", line
                ii = line.index("__")+2;
                line = line[ii:]
                #print "now line: ", line

            #This means we have #cos sim, etc
            if (not (line.startswith("("))):
                op.write(lines[idx]+"\n")
                idx += 1
                continue

            #Now, we've got sth interesting!
            #print "line: ", line
            ss = line.split(" ")
            pred = ss[0]
            sim = ss[1]

            if qa_utils.is_sorted(pred):
                pred = pred + "#" + typesD
            else:
                pred = qa_utils.swap(pred) + "#" + typesR

            op.write(pred + " " + sim + "\n")


            idx += 1
    print ("process time: ", (time.time()-start))

def assign_types_files(in_dir,out_dir):
    files = os.listdir(in_dir)

    for f in files:
        if "_sim.txt" not in f or os.stat(in_dir+f).st_size == 0:
            continue
        print (f)
        assign_types_file(in_dir+f , out_dir + f)

def set_types_gens(fname):
    lines = open(fname).read().splitlines()

    for line in lines:
        noun = line.split("::")[0]
        types = qa_utils.get_hypernyms(noun)
        print (noun + "::"+ str(types))


def combine_feats(f1,f2):
    lines1 = open(f1).read().splitlines()
    lines2 = open(f2).read().splitlines()
    for idx,l in enumerate(lines1):
        l = l[:-1]
        l2 = lines2[idx]
        ss = l2.split("\t")
        f_ss = ss[-1][2:-1]
        l += " "+ f_ss +" ]"
        print (l)

def getsubsample(fname):
    lines = [line.rstrip() for line in open(fname)]
    lines = [line for line in lines if line.endswith("True")]
    lines = np.random.choice(lines,50,replace=False)
    for l in lines:
        print (l)

# a one-time function
def process_snli_output(fpath):
    f = open(fpath)
    N = 0
    all_ps = []
    for line in f:
        line = line.replace("[", "").replace("]", "").strip()
        ss = line.split()
        if (len(ss) == 4):
            pred = [np.float(x) for x in ss[1:]]
            pred -= np.max(pred)
            print (pred)
            e = np.exp(pred)
            print ("e:", e)
            s = sum(e)
            print (s)
            if s == 0:
                print ("s is 0")
                pred_l = np.argmax(pred)
                ps = [0] * 3
                ps[pred_l] = 1
            else:
                ps = e / s
            print (ps)
            all_ps.append([ps[0], ps[1], ps[2]])
            N += 1
    print (N)

    print ("all_ps: ")

    for ps in all_ps:
        print (str(ps[0]) + " " + str(ps[1]) + " " + str(ps[2]))

def convertLabelsMNLI(dir):
    filePaths = os.listdir(dir)
    dir2 = dir + "csv/"
    for p in filePaths:
        if not p.endswith(".txt"):
            continue
        if "mismatched" in p:
            pairId = 0
        else:
            pairId = 9847
        f = open(dir + "/" + p)
        f2 = open(dir2 + p[:-4] + ".csv", 'w')
        f2.write("pairID,gold_label\n")
        for l in f:
            l = l[:-1]
            label = l
            f2.write(str(pairId) + "," + label + "\n")
            pairId += 1
        f2.close()
        f.close()

def analyze_diff_snli(fname_preds_ens, fname_preds_nn, fname_orig):

    f2 = open(fname_orig)
    lines_orig = f2.read().splitlines()[1:]

    preds_ens = open(fname_preds_ens).read().splitlines()
    preds_nn = open(fname_preds_nn).read().splitlines()

    idx = 0
    accIdx = 0

    for l in lines_orig:
        ss = l.split("\t")
        if ss[0] == "-":
            idx += 1
            continue
        if ss[0] == "entailment":
            label = "0"
        elif ss[1] == "neutral":
            label = "1"
        else:
            label = "2"
        print (ss[5] + "#" + ss[6])
        ss = l.strip().split("\t")
        try:
            print (ss[0])
        except:
            pass

        if preds_ens[accIdx] != preds_nn[accIdx]:
            if preds_ens[accIdx] == label:
                print ("ours is correct")
            elif preds_nn[accIdx] == label:
                print ("nn is correct")

        print ("\n\n")

        accIdx += 1
        idx += 1

def read_stop_words():
    f = open(root + "/ent/stops.txt")
    ret = set()
    for l in f:
        l = l.strip()
        ret.add(l)
    return ret

#Convert our rels files to relsemb/gbooks format so that Xavier can process it
def convertFormatAll(folderAddr, myformat = "relsemb",removeStop=False, removeEventAndNeg = False, timeStamp=False):
    files = os.listdir(folderAddr)
    allTuplesAllTypes = None
    if myformat == "convE":
        allTuplesAllTypes = set()
    num_f = 0
    for f in files:
        # if f.endswith('relsG.txt'):
        # if num_f==1:
        #     break

        # if f.endswith('.mat') or f.endswith("_relation2id.txt") or f.endswith("_apairs") or f.endswith("_sim.txt"):
        if not f.endswith("_rels.txt"):
            continue
        print (f)
        num_f += 1
        convertFormat(folderAddr + f,myformat, allTuplesAllTypes,removeStop, removeEventAndNeg, timeStamp=timeStamp)
    if myformat == "convE":
        f = open(folderAddr + "/allTuples_ptyped_unique.txt", "w")
        for x in allTuplesAllTypes:
            f.write(x+"\n")
        f.close()

def makeNumbersExample():

    rs = ["elect","run"]
    ents = ["L-France","M-France","Mar-Col","Pet-Col"]

    x = np.zeros(shape=(2,4))
    x[0,1] = .9
    x[0,2] = .65
    x[1,0] = .6
    x[1,1] = .7
    x[1,2] = .7
    x[1,3] = .8

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            print (str(rs[i]) + " => " + str(ents[j]) + ": " + str(x[i, j] / np.sum(x[i, :])))

    for j in range(x.shape[1]):
        for i in range(x.shape[0]):
            print (str(ents[j]) + " => " + str(rs[i]) + ": " + str(x[i, j] / np.sum(x[:, j])))

def add_thing_types(fname):
    f = open(fname,'r')
    for line in f:
        line = line.strip()
        ss = line.split("\t")
        print (ss[0]+"\t"+ss[1]+"#thing_1#thing_2\t"+ss[2]+"\tthing\tthing")
    f.close()

#myformat: gbooks, relsemb, or mat
def convertFormat(fname,myformat = "relsemb", allTuplesAllTypes = None, removeStop=False, removeEventAndNeg = False, timeStamp=False):
    f = open(fname)

    if myformat=="gbooks":
        fname_out = fname[:-9] + "_relsG.txt"
    elif myformat == "convE":

        fname_out = fname[:-9] + "_convE.txt"
    elif myformat == "relsemb":
        fname_out = fname[:-9] + "_train2id.txt"
        fname_rels = fname[:-9] + "_relation2id.txt"
        fname_ents = fname[:-9] + "_entity2id.txt"

        f_out_rels = open(fname_rels, "w")
        f_out_ents = open(fname_ents, "w")

        out_lines = []
        rels_lines = []
        ent_lines = []
    else:
        fname_out = fname[:-4] + ".mat"
        fname_rels = fname[:-4] + "_relation2id.txt"
        fname_apairs = fname[:-4] + "_apairs.txt"
        f_out_rels = open(fname_rels, "w")
        f_out_apairs = open(fname_apairs, "w")
        mat = []

        rels_lines = []
        apair_lines = []

    print (fname_out)
    fout = open(fname_out,'w')


    ent2id = {}
    entPair2id = {}
    rels2id = {}

    #TODO: be careful, unique or not
    allTuples = set()
    # allTuples = []

    currentPred = ""
    ignorePreds = False

    types = fname[fname.rfind('/') + 1:-9].split('#')

    for l in f:

        l = l.strip()
        reversed_ents = False
        if l.startswith("types:"):#first line, let's continue!
            continue
        elif l.startswith("predicate:"):
            currentPred = l.split()[1]
            ss = currentPred.split("#")
            ignorePreds = ss[1].endswith("_2")#reverse order, let's ignore it!
            ignorePreds = ignorePreds or (removeStop and ss[0] in stops) or (removeStop and "__" in ss[0])
            # currentPred = ss[0]
            pred_types = [ss[1].replace("_1","").replace("_2",""), ss[2].replace("_1","").replace("_2","")]
            reversed_ents = pred_types[0]!=types[0]
        elif l=="":
            continue
        elif l.startswith("inv idx of"):
            break
        else:
            if ignorePreds:
                continue

            cidx = l.rfind(":")

            args = l[:cidx].split("#")
            # print args
            count = int(float(l[cidx+2:]))

            if myformat=="gbooks":
                fout.write(args[0]+"\t"+args[1]+"\t"+currentPred+"\t"+str(count)+"\n")
            elif myformat=="convE":
                if not reversed_ents:
                    thisTuple = (args[0]+"\t"+currentPred+"\t"+args[1]).replace(" ","_")
                else:
                    thisTuple = (args[1] + "\t" + currentPred + "\t" + args[0]).replace(" ", "_")

                if not timeStamp:
                    allTuples.add(thisTuple)
                else:
                    allTuples.add(thisTuple +"\t"+args[2])
                if not timeStamp:
                    allTuplesAllTypes.add(thisTuple + "\t"+ pred_types[0] + "\t" + pred_types[1]+"\t"+str(count))
                else:
                    allTuplesAllTypes.add(thisTuple + "\t" + pred_types[0] + "\t" + pred_types[1] + "\t" + str(count)+"\t"+args[2])

            else:
                print ("here current pred: ", currentPred)
                if currentPred not in rels2id:
                    if myformat=="relsemb" or myformat == "mat":
                        print ("appending rel: ", currentPred)
                        rels_lines.append(currentPred + "\t" + str(len(rels2id))+"\n")
                    rels2id[currentPred] = len(rels2id)
                    if myformat=="mat":
                        mat.append([])

                if args[0] not in ent2id:
                    if myformat=="relsemb":
                        ent_lines.append(args[0]+ "\t" + str(len(ent2id))+"\n")
                    ent2id[args[0]] = len(ent2id)
                if args[1] not in ent2id:
                    if myformat=="relsemb":
                        ent_lines.append(args[1] + "\t" + str(len(ent2id))+"\n")
                    ent2id[args[1]] = len(ent2id)

                entPair = args[0]+"#"+args[1]
                if not entPair in entPair2id:
                    entPair2id[entPair] = len(entPair2id)
                    if myformat=="mat":
                        apair_lines.append(entPair+"\n")

                if myformat=="mat":
                    mat[rels2id[currentPred]].append((entPair2id[entPair],count))

                # for j in range(count):
                if myformat=="relsemb":
                    out_lines.append(str(ent2id[args[0]]) + "\t" + str(ent2id[args[1]]) + "\t" + str(rels2id[currentPred])+"\n")

    if myformat=="convE":#unique
        for x in allTuples:
            fout.write(x+"\n")
        fout.close()

    if myformat == "relsemb":
        fout.write(str(len(out_lines))+"\n")
        fout.writelines(out_lines)
        fout.close()

        f_out_rels.write(str(len(rels_lines))+"\n")
        f_out_rels.writelines(rels_lines)
        f_out_rels.close()

        f_out_ents.write(str(len(ent_lines)) + "\n")
        f_out_ents.writelines(ent_lines)
        f_out_ents.close()
    elif myformat == "mat":
        #convert sparse matrix (mat) to dense matrix
        dmat = np.zeros(shape=(len(mat),len(entPair2id)))
        for i in range(len(mat)):
            for (j,c) in mat[i]:
                dmat[i,j] = c

        fout.write(str(dmat.shape[1])+" "+str(dmat.shape[0])+"\n")
        for i in range(dmat.shape[1]):
            for j in range(dmat.shape[0]):
                fout.write(str(int(dmat[j,i]))+" ")
            fout.write("\n")

        f_out_rels.write(str(len(rels_lines)) + "\n")
        f_out_rels.writelines(rels_lines)
        f_out_rels.close()

        f_out_apairs.write(str(len(entPair2id)) + "\n")
        f_out_apairs.writelines(apair_lines)
        f_out_apairs.close()

def compare_feats(path1,path2):
    f = open(path1)
    s1 = set()
    for line in f:
        ss = line.split(": ")
        line = ss[0]
        # print "a line: ", line
        s1.add(line)

    f.close()
    f = open(path2)
    for line in f:

        ss = line.split(": ")
        line = ss[0]
        # print line
        if line in s1:
            print ("both feature sets have: ", line.strip())


def find_all_gr_size(engG_dir_addr):
    files = os.listdir(engG_dir_addr)
    files = list(np.sort(files))

    f_out = open('sizes.txt','w')
    f_out_preds = open('NS_pred_sizes.txt','w')

    for f in files:
        if not "_rels" in f:
            continue
        print (f)
        #sth like person#location.txt
        output = subprocess.check_output("grep 'predicate:' "+ engG_dir_addr+f + " | wc -l", shell=True).strip()
        num_nodes = int(output)

        output = subprocess.check_output("grep 'inv idx of' " + engG_dir_addr + f + " | wc -l", shell=True).strip()
        num_aps = int(output)

        types = f[:-9]
        f_out.write(types + "\t" + str(num_nodes) + "\t" + str(num_aps) + "\n")

        f_in = open(engG_dir_addr + f)
        current_pred = None
        num_aps = 0
        for line in f_in:
            line = line.strip()
            if line.startswith("predicate:"):
                num_aps = 0
                current_pred = line[11:]
            elif line == "":
                f_out_preds.write(current_pred+"\t"+str(num_aps)+"\n")
            else:
                num_aps += 1



    f_out.close()

def writeEntGraph(simsFolder,types,pred2neighs):
    fnameTProp = simsFolder + types + "_sim.txt"

    op = open(fnameTProp, 'w')

    N = len(pred2neighs)
    op.write(types + " " + " num preds: " + str(N) + "\n")

    for pred in pred2neighs:
        neighs = pred2neighs[pred]
        neighs = sorted(neighs, key= lambda neigh: neigh[1],reverse=True)
        op.write("predicate: " + pred + "\n")
        op.write("num neighbors: " + str(len(neighs)) + "\n")
        op.write("\n")

        op.write("AMIEplus sims\n")
        for pred2, w in neighs:
            op.write(pred2 + " " + str(w) + "\n")

        op.write("\n")

    op.close()

    print("results written for: ", fnameTProp)


def getTypes2orderedtypes(types2orderedtypes,types):
    if types not in types2orderedtypes:
        types2orderedtypes[types] = types
        ss = types.split("#")
        types_reverse = ss[1]+"#"+ss[0]
        types2orderedtypes[types_reverse] = types
    return types2orderedtypes[types]

def convertAMIEPlusRules2entGraphs(rules_path):
    f = open(rules_path)

    types2orderedtypes = dict()
    types2pred2neighs = dict()


    for line in f:
        try:
            line = line.strip()
            ss = line.split("\t")
            rule = ss[0]
            ss_rule = rule.split()
            aligned = ss_rule[0]== ss_rule[4]
            p = ss_rule[1]
            q = ss_rule[5]
            sim = np.float(ss[3])

            type1 = p.split("#")[1].replace("_1", "").replace("_2", "")
            type2 = p.split("#")[2].replace("_1", "").replace("_2", "")

            type1_q = q.split("#")[1].replace("_1", "").replace("_2", "")
            type2_q = q.split("#")[2].replace("_1", "").replace("_2", "")

            types1 = type1+"#"+type2
            if aligned:
                types2 = type1_q+"#"+type2_q
            else:
                types2 = type2_q+"#"+type1_q

            if types1!=types2:
                print "not the same types: ", line
                continue


            types = getTypes2orderedtypes(types2orderedtypes,types1)
            if types not in types2pred2neighs:
                pred2neigh = dict()
                types2pred2neighs[types] = pred2neigh

            pred2neigh = types2pred2neighs[types]
            if not p in pred2neigh:
                pred2neigh[p] = list()



            if type1==type2:
                p_reverse = p.split("#")[0] + "#" + type1 + "_2#" + type1 + "_1"
                q_reverse = q.split("#")[0] + "#"+ type1+"_2#"+type1+"_1"
                if aligned:
                    pred2neigh[p].append([q, sim])
                    pred2neigh[p_reverse].append([q_reverse, sim])
                else:
                    pred2neigh[p].append([q_reverse, sim])
                    pred2neigh[p_reverse].append([q, sim])
            else:
                pred2neigh[p].append([q, sim])

            print (line, aligned)
        except:
            print "continuing: ", line
            continue

    simsFolder = root + "/" + "typedEntGraphDirAmiePlus/"
    if not os.path.exists(simsFolder):
        os.mkdir(simsFolder)
    for types,pred2neighs in types2pred2neighs.items():

        writeEntGraph(simsFolder,types,pred2neighs)

def converAMIEplusRulesToConst(rules_path):
    f = open(rules_path)
    for line in f:
        try:
            line = line.strip()
            ss = line.split("\t")
            rule = ss[0]
            ss_rule = rule.split()
            aligned = ss_rule[0] == ss_rule[4]
            p = ss_rule[1]
            q = ss_rule[5]
            if p=="" or q=="":
                print "bad line:", line
                continue
            sim = np.float(ss[3])
            aligned_str = ""
            if not aligned:
                aligned_str = "-"
            if sim>=0:
                print aligned_str+p + "$$" + q + "\t" + str(sim)
        except:
            # print "continuing: ", line
            continue


def convertBinaryToUnary(path,outpath, timeStamp=False):
    f = open(path)
    fout = open(outpath,'w')

    unaries2Count = defaultdict(int)

    for line in f:
        # print ("line: "+line)
        line = line.strip()
        try:

            ss = line.split("\t")
            arg1 = ss[0]
            arg2 = ss[2]
            type1 = ss[3]
            type2 = ss[4]
            count = ss[5]


            unary1,unary2 = util.getUnaryFrom_binary(ss[1])
            if not timeStamp:
                unaries2Count[unary1 + "\t" + arg1 + "\t" + type1] += np.int(count)
                unaries2Count[unary2 + "\t" + arg2 + "\t" + type2] += np.int(count)
            else:
                unaries2Count[unary1 + "\t" + arg1 + "#" + ss[6]+ "\t" + type1] += np.int(count)
                unaries2Count[unary2 + "\t" + arg2 + "#" + ss[6]+"\t" + type2] += np.int(count)
            # fout.write(unary1 + " " + arg1 + " " + type1 + " " + count+"\n")
            # fout.write(unary2 + " " + arg2 + " " + type2 + " " + count+"\n")
        except Exception as e:
            # traceback.print_exc()
            print ("bad line: ", line)
            continue

    for unary_rel,count in unaries2Count.items():
        fout.write(unary_rel + "\t" + str(count) + "\n")



root = "../../gfiles/"

convertAMIEPlusRules2entGraphs("../../../AMIE/rules_headcoverage_.01_min_init_supp_100.tsv")
# converAMIEplusRulesToConst("../../../AMIE/rules_headcoverage_.01.tsv")
# stops = read_stop_words()

# process_snli_output(root+"ent/test_probs.txt")
# convertLabelsMNLI(root+"multinli_1.0/predictions/")

# root2 = root+"predictions_train_snli_on_mnli_copy/"
# analyze_diff_snli(root2+"test_NN1_0.3_False_res.txt",root2+"test_NN1_1.0_False_res.txt",root2+"snli_1.0_test.txt")

# convertFormatAll(root+"typed_rels/",myformat="mat")
# convertFormatAll(root+"typedEntGrDir_aida_untyped_40_40/",myformat="mat")
# convertFormatAll(root + "typedEntGrDir_aida_untyped_10_10/")
# convertFormatAll(root + "typedEntGrDirC_CN_NBEE_NSBased_3_3_thth_thloc_time6_f20_GG_noGGThing/",myformat="convE",removeStop=False, removeEventAndNeg=False, timeStamp=False)
# compare_feats("/Users/javadhosseini/Desktop/compare/cc.txt","/Users/javadhosseini/Desktop/compare/dd.txt")
# makeNumbersExample()
# add_thing_types("/home/jhosseini/mnt2/pytorch/lpred/ConvE/data/FB15k_untyped/freebase_mtr100_mte100-test.txt")
# combine_feats(root+"ent/feats_aida_figer2.txt",root+"ent/feats_emb_java15.txt")
# getsubsample(root+"ent/all_new.txt")
# set_types_gens(root+"gens.txt")

# convertBinaryToUnary("../../../java/entGraph/embs/allTuples_ptyped_unique_day_3_0.txt","../../../java/entGraph/embs/allTuples_ptyped_unique_day_3_0_unary.txt", timeStamp=True)

# find_all_gr_size(root + 'typedEntGrDir_aida_figer_3_3_a/')
# find_all_gr_size(root + 'typedEntGrDir_aida_gen12_UT_hier_back3_2/')



# dirs = [root+"typedEntGrDir_test/",root+"typedEntGrDir_typed/"]
# assign_types_files(dirs[0],dirs[1])

# count_conjunctions(root+"typedEntGrDir_aida_figer/location#location.txt")
# test_time()
# test_time2()
#data = read_data()

# args = sys.argv[1:]
# if len(args)==0:
#     d = "typedEntGrDir"
# else:
#     d = args[0]

#typedEntGrDir_r
