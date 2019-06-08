import sys
sys.path.append("..")
from evaluation import util
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.metrics import confusion_matrix
from scipy.sparse import csr_matrix
import operator
import nltk
from nltk.corpus import stopwords as stw

NMAXTR = 5
NLEXFEATS = 600000
USE_ENT_SCORES = True#changing everything to 1 for a sanity check!
USE_IDENTIFIERS = False#MUST BE ALWAYS FALSE!
GLOBAL_FEATS = True

USE_LEXICAL_FEATS = False
SUPERVISED = True
THRESHOLD = 1
S_OR_M_NLI = False#True: snli, False: mnli

assert SUPERVISED or not USE_LEXICAL_FEATS

if GLOBAL_FEATS:
    feat_idx = 1
else:
    feat_idx = 0

def extract_feats_binary(line, predPairFeatsTyped, predPairFeats, reverse=False):

    ss = line.split("\t")

    if reverse and len(ss)>=2:
        ss = [ss[1],ss[0]]

    try:
        triples1 = ss[0].split("$$")
        triples2 = ss[1].split("$$")
    except:
        triples1 = []
        triples2 = []


    triples1 = [s for s in triples1 if s!=""]
    triples2 = [s for s in triples2 if s != ""]

    preds1 = [s.split(" ")[0] for s in triples1]
    preds2 = [s.split(" ")[0] for s in triples2]

    try:
        types1 = [[s.split(" ")[1].split("::")[1], s.split(" ")[2].split("::")[1]] for s in triples1]
        args1 = [[s.split(" ")[1].split("::")[0], s.split(" ")[2].split("::")[0]] for s in triples1]
        args2 = [[s.split(" ")[1].split("::")[0], s.split(" ")[2].split("::")[0]] for s in triples2]
    except:
        print "bad case: ", triples1
        print line
        types1 = None#Should happen!

    feats = []

    num_exact_pred = 0
    num_triple_slot12 = 0
    num_triple_slot1 = 0
    num_triple_slot2 = 0
    num_triple_noslot = 0


    max_sims_typed_tr2 = []#for each tr2, what is the highest entailing tr1!
    max_sims_typed_tr2_slot12 = []
    max_sims_typed_tr2_slot1 = []
    max_sims_typed_tr2_slot2 = []
    max_sims_typed_tr2_no_slot = []

    max_sim_typed = 0
    max_sim_typed_slot12 = 0
    max_sim_typed_slot1 = 0
    max_sim_typed_slot2 = 0
    max_sim_typed_no_slot = 0


    max_sims_tr2 = []  # for each tr2, what is the highest entailing tr1!
    max_sims_tr2_slot12 = []
    max_sims_tr2_slot1 = []
    max_sims_tr2_slot2 = []
    max_sims_tr2_no_slot = []

    max_sim = 0
    max_sim_slot12 = 0
    max_sim_slot1 = 0
    max_sim_slot2 = 0
    max_sim_no_slot = 0

    lexicalized_feats = []

    for idx2, tr2 in enumerate(triples2):

        max_sim_typed_tr2 = 0
        max_sim_typed_tr2_slot12 = 0
        max_sim_typed_tr2_slot1 = 0
        max_sim_typed_tr2_slot2 = 0
        max_sim_typed_tr2_no_slot = 0

        max_sim_tr2 = 0
        max_sim_tr2_slot12 = 0
        max_sim_tr2_slot1 = 0
        max_sim_tr2_slot2 = 0
        max_sim_tr2_no_slot = 0

        for idx1,tr1 in enumerate(triples1):

            if preds1[idx1] == preds2[idx2]:
                num_exact_pred += 1
                if args1[idx1][0] == args2[idx2][0] and args1[idx1][1] == args2[idx2][1]:
                    num_triple_slot12 += 1
                elif args1[idx1][0] == args2[idx2][0]:
                    num_triple_slot1 += 1
                elif args1[idx1][1] == args2[idx2][1]:
                    num_triple_slot2 +=1
                else:
                    num_triple_noslot += 1

            predPairTyped = preds1[idx1] + "#"+ preds2[idx2]+ "#True#" + types1[idx1][0]+"#"+types1[idx1][1]

            predSimTyped = 0


            #0 means no args, 1 means first arg, 2 means second arg, 3 means both args
            if args1[idx1][0] == args2[idx2][0] and args1[idx1][1] == args2[idx2][1]:
                argsShareIdx = 3
            elif args1[idx1][0]==args2[idx2][0]:
                argsShareIdx = 1
            elif args1[idx1][1] == args2[idx2][1]:
                argsShareIdx = 2
            else:
                argsShareIdx = 0

            # print predPairTyped
            if predPairTyped in predPairFeatsTyped:
                sims = predPairFeatsTyped[predPairTyped]
                if not USE_ENT_SCORES:
                    sims[feat_idx] = 0
                if sims[feat_idx] > THRESHOLD:
                    sims[feat_idx] = 1
                predSimTyped = sims[feat_idx]


                max_sim_typed_tr2 = np.max([max_sim_typed_tr2, sims[feat_idx]])
                if sims[feat_idx] != 1:
                    max_sim_typed = np.max([sims[feat_idx],max_sim_typed])
                if argsShareIdx == 3:
                    max_sim_typed_tr2_slot12 = np.max([sims[feat_idx],max_sim_typed_tr2_slot12])
                    if sims[feat_idx] != 1:
                        max_sim_typed_slot12 = np.max([sims[feat_idx],max_sim_typed_slot12])
                elif argsShareIdx == 1:
                    max_sim_typed_tr2_slot1 = np.max([sims[feat_idx], max_sim_typed_tr2_slot1])
                    if sims[feat_idx] != 1:
                        max_sim_typed_slot1 = np.max([sims[feat_idx], max_sim_typed_slot1])
                elif argsShareIdx == 2:
                    max_sim_typed_tr2_slot2 = np.max([sims[feat_idx], max_sim_typed_tr2_slot2])
                    if sims[feat_idx] != 1:
                        max_sim_typed_slot2 = np.max([sims[feat_idx], max_sim_typed_slot2])
                else:
                    max_sim_typed_tr2_no_slot = np.max([sims[feat_idx], max_sim_typed_tr2_no_slot])
                    if sims[feat_idx] != 1:
                        max_sim_typed_no_slot = np.max([sims[feat_idx], max_sim_typed_no_slot])
                # if sims[feat_idx]>max_sim_typed_tr2:

            cross_bigram = args1[idx1][0]+"-"+args1[idx1][1]+"#"+args2[idx2][0]+"-"+args2[idx2][1]
            cross_tuples = tr1+"#"+tr2
            cross_pred = predPairTyped+"#"+str(argsShareIdx)

            if reverse:
                cross_bigram += "#R"
                cross_tuples += "#R"
                cross_pred += "#R"

            if USE_LEXICAL_FEATS:
                lexicalized_feats.append((cross_bigram,predSimTyped))
                lexicalized_feats.append((cross_tuples, predSimTyped))
                lexicalized_feats.append((cross_pred, predSimTyped))

                if USE_IDENTIFIERS:
                    lexicalized_feats.append((cross_bigram+"#I", 1))
                    lexicalized_feats.append((cross_tuples + "#I", 1))
                    lexicalized_feats.append((cross_pred + "#I", 1))


            predSimUntyped = 0

            predPair = preds1[idx1] + "#" + preds2[idx2] + "#True";
            if predPair in predPairFeats:
                sims = predPairFeats[predPair]
                if not USE_ENT_SCORES:
                    sims[feat_idx] = 0
                predSimUntyped = sims[feat_idx]

                max_sim_tr2 = np.max([sims[feat_idx], max_sim_tr2])
                if sims[feat_idx] != 1:
                    max_sim = np.max([sims[feat_idx], max_sim])
                if argsShareIdx == 3:
                    max_sim_tr2_slot12 = np.max([sims[feat_idx], max_sim_tr2_slot12])
                    if sims[feat_idx] != 1:
                        max_sim_slot12 = np.max([sims[feat_idx], max_sim_slot12])
                elif argsShareIdx == 1:
                    max_sim_tr2_slot1 = np.max([sims[feat_idx], max_sim_tr2_slot1])
                    if sims[feat_idx] != 1:
                        max_sim_slot1 = np.max([sims[feat_idx], max_sim_slot1])
                elif argsShareIdx == 2:
                    max_sim_tr2_slot2 = np.max([sims[feat_idx], max_sim_tr2_slot2])
                    if sims[feat_idx] != 1:
                        max_sim_slot2 = np.max([sims[feat_idx], max_sim_slot2])
                else:
                    max_sim_tr2_no_slot = np.max([sims[feat_idx], max_sim_tr2_no_slot])
                    if sims[feat_idx] != 1:
                        max_sim_no_slot = np.max([sims[feat_idx], max_sim_no_slot])

            cross_pred = predPair+ "#" + str(argsShareIdx)

            if USE_LEXICAL_FEATS:
                lexicalized_feats.append((cross_bigram+"#U", predSimUntyped))
                lexicalized_feats.append((cross_tuples+"#U", predSimUntyped))
                lexicalized_feats.append((cross_pred, predSimUntyped))



        max_sims_typed_tr2.append(max_sim_typed_tr2)
        max_sims_typed_tr2_slot12.append(max_sim_typed_tr2_slot12)
        max_sims_typed_tr2_slot1.append(max_sim_typed_tr2_slot1)
        max_sims_typed_tr2_slot2.append(max_sim_typed_tr2_slot2)
        max_sims_typed_tr2_no_slot.append(max_sim_typed_tr2_no_slot)

        max_sims_tr2.append(max_sim_tr2)
        max_sims_tr2_slot12.append(max_sim_tr2_slot12)
        max_sims_tr2_slot1.append(max_sim_tr2_slot1)
        max_sims_tr2_slot2.append(max_sim_tr2_slot2)
        max_sims_tr2_no_slot.append(max_sim_tr2_no_slot)

    # feats.extend(max_sim_typed)
    max_sims_typed_tr2 = get_max_sims(max_sims_typed_tr2,NMAXTR)
    max_sims_typed_tr2_slot12 = get_max_sims(max_sims_typed_tr2_slot12, NMAXTR)
    max_sims_typed_tr2_slot1 = get_max_sims(max_sims_typed_tr2_slot1, NMAXTR)
    max_sims_typed_tr2_slot2 = get_max_sims(max_sims_typed_tr2_slot2, NMAXTR)
    max_sims_typed_tr2_no_slot = get_max_sims(max_sims_typed_tr2_no_slot, NMAXTR)

    max_sims_tr2 = get_max_sims(max_sims_tr2, NMAXTR)
    max_sims_tr2_slot12 = get_max_sims(max_sims_tr2_slot12, NMAXTR)
    max_sims_tr2_slot1 = get_max_sims(max_sims_tr2_slot1, NMAXTR)
    max_sims_tr2_slot2 = get_max_sims(max_sims_tr2_slot2, NMAXTR)
    max_sims_tr2_no_slot = get_max_sims(max_sims_tr2_no_slot, NMAXTR)

    #Now, append the max_sims
    feats.extend(max_sims_typed_tr2)
    feats.extend(max_sims_typed_tr2_slot12)
    feats.extend(max_sims_typed_tr2_slot1)
    feats.extend(max_sims_typed_tr2_slot2)
    feats.extend(max_sims_typed_tr2_no_slot)

    if GLOBAL_FEATS:
        feats.extend(max_sims_tr2)
        feats.extend(max_sims_tr2_slot12)
        feats.extend(max_sims_tr2_slot1)
        feats.extend(max_sims_tr2_slot2)
        feats.extend(max_sims_tr2_no_slot)

    feats.extend([max_sim_typed,max_sim_typed_slot12,max_sim_typed_slot1,max_sim_typed_slot2,max_sim_typed_no_slot])
    if GLOBAL_FEATS:
        feats.extend([max_sim, max_sim_slot12, max_sim_slot1, max_sim_slot2, max_sim_no_slot])
    feats.extend([num_exact_pred,num_triple_slot12,num_triple_slot1,num_triple_slot2,num_triple_noslot])

    feats.extend([len(triples1),len(triples2),len(triples1)*len(triples2),len(triples1)-len(triples2)])

    return feats,lexicalized_feats

def extract_feats_unary(line, unaryPairFeatsTyped, reverse = False):
    ss = line.split("\t")

    if reverse and len(ss)>=2:
        ss = [ss[1],ss[0]]

    try:
        triples1 = ss[0].split("$$")
        triples2 = ss[1].split("$$")
    except:
        triples1 = []
        triples2 = []

    triples1 = [s for s in triples1 if s!=""]
    triples2 = [s for s in triples2 if s != ""]

    # print "triples: ", triples1, triples2

    preds1 = [s.split(" ")[0] for s in triples1]
    preds2 = [s.split(" ")[0] for s in triples2]

    try:
        types1 = [[s.split(" ")[1].split("::")[1]] for s in triples1]
        # types2 = [[s.split(" ")[1].split("::")[1]] for s in triples2]
        args1 = [[s.split(" ")[1].split("::")[0]] for s in triples1]
        args2 = [[s.split(" ")[1].split("::")[0]] for s in triples2]
    except:
        print "bad case: ", triples1
        print line
        types1 = None

    feats = []

    num_exact_pred = 0
    num_triple_slot1 = 0
    num_triple_noslot = 0

    max_sims_typed_tr2 = []#for each tr2, what is the highest entailing tr1!
    max_sims_typed_tr2_slot1 = []
    max_sims_typed_tr2_no_slot = []

    max_sim_typed = 0
    max_sim_typed_slot1 = 0
    max_sim_typed_no_slot = 0

    lexicalized_feats = []

    for idx2, tr2 in enumerate(triples2):

        max_sim_typed_tr2 = 0
        max_sim_typed_tr2_slot1 = 0
        max_sim_typed_tr2_no_slot = 0

        for idx1,tr1 in enumerate(triples1):

            if preds1[idx1] == preds2[idx2]:
                num_exact_pred += 1
                if args1[idx1][0] == args2[idx2][0]:
                    num_triple_slot1 += 1
                else:
                    num_triple_noslot += 1

            unaryPairTyped = preds1[idx1] + "#"+ preds2[idx2]+ "#"+ types1[idx1][0]

            predSimTyped = 0

            #0 means no args, 1 means first arg
            if args1[idx1][0]==args2[idx2][0]:
                argsShareIdx = 1
            else:
                argsShareIdx = 0

            # print predPairTyped
            if unaryPairTyped in unaryPairFeatsTyped:
                sims = unaryPairFeatsTyped[unaryPairTyped]
                if not USE_ENT_SCORES:
                    sims[feat_idx] = 0
                if sims[feat_idx]>THRESHOLD:
                    sims[feat_idx] = 1
                # print "sims unary: ", sims, " for ", unaryPairTyped

                predSimTyped = sims[feat_idx]

                max_sim_typed_tr2 = np.max([max_sim_typed_tr2, sims[feat_idx]])
                if sims[feat_idx]!=1:
                    max_sim_typed = np.max([sims[feat_idx],max_sim_typed])
                if argsShareIdx==1:
                    max_sim_typed_tr2_slot1 = np.max([sims[feat_idx],max_sim_typed_tr2_slot1])
                    if sims[feat_idx] != 1:
                        max_sim_typed_slot1 = np.max([sims[feat_idx],max_sim_typed_slot1])
                else:
                    max_sim_typed_tr2_no_slot = np.max([sims[feat_idx], max_sim_typed_tr2_no_slot])
                    if sims[feat_idx] != 1:
                        max_sim_typed_no_slot = np.max([sims[feat_idx], max_sim_typed_no_slot])

            cross_unigram = args1[idx1][0] + "#" + args2[idx2][0]
            cross_tuples = tr1 + "#" + tr2
            cross_pred = unaryPairTyped + "#" + str(argsShareIdx)

            if reverse:
                cross_unigram += "#R"
                cross_tuples += "#R"
                cross_pred += "#R"


            if USE_LEXICAL_FEATS:

                if USE_IDENTIFIERS:
                    lexicalized_feats.append((cross_unigram + "#I", 1))
                    lexicalized_feats.append((cross_tuples + "#I", 1))
                    lexicalized_feats.append((cross_pred + "#I", 1))

                lexicalized_feats.append((cross_unigram, predSimTyped))
                lexicalized_feats.append((cross_tuples, predSimTyped))
                lexicalized_feats.append((cross_pred, predSimTyped))



        max_sims_typed_tr2.append(max_sim_typed_tr2)
        max_sims_typed_tr2_slot1.append(max_sim_typed_tr2_slot1)
        max_sims_typed_tr2_no_slot.append(max_sim_typed_tr2_no_slot)


    max_sims_typed_tr2 = get_max_sims(max_sims_typed_tr2,NMAXTR)
    max_sims_typed_tr2_slot1 = get_max_sims(max_sims_typed_tr2_slot1,NMAXTR)
    max_sims_typed_tr2_no_slot = get_max_sims(max_sims_typed_tr2_no_slot, NMAXTR)

    #Now, append the max_sims
    feats.extend(max_sims_typed_tr2)
    feats.extend(max_sims_typed_tr2_slot1)
    feats.extend(max_sims_typed_tr2_no_slot)
    feats.extend([max_sim_typed,max_sim_typed_slot1,max_sim_typed_no_slot])
    feats.extend([num_exact_pred,num_triple_slot1, num_triple_noslot])

    feats.extend([len(triples1),len(triples2),len(triples1)*len(triples2),len(triples1) - len(triples2)])

    return feats, lexicalized_feats

#exactly count similarities in an array, sorted by their values!
def get_max_sims(sims, count):
    sims = sorted(sims,reverse=True)
    if len(sims)>count:
        sims = sims[0:count]
    elif len(sims)<count:
        sims.extend([0]*(count-len(sims)))
    return sims

def get_sparse_matrix(feats_list, lexicalized_feats_list, lexical_map):
    row_idxes = []
    col_idxes = []
    data = []
    row_idx = 0
    for row_idx,feats in enumerate(feats_list):


        for col_idx,v in enumerate(feats):
            row_idxes.append(row_idx)
            col_idxes.append(col_idx)
            data.append(v)

        lexicalized_feats = lexicalized_feats_list[row_idx]
        for (x,v) in lexicalized_feats:
            if x in lexical_map:
                row_idxes.append(row_idx)
                col_idxes.append(lexical_map[x])
                data.append(v)


    return csr_matrix((data, (row_idxes, col_idxes)),shape=(len(feats_list),len(lexical_map)+len(feats_list[0])))


#(man,woman) .1, (man,woman) .6 = > (man,woman) .6
def make_unique_lex_feats(lexicalized_feats):
    lex_feats_map = {}

    for x,v in lexicalized_feats:
        if x not in lex_feats_map:
            lex_feats_map[x] = v
        else:
            lex_feats_map[x] = np.maximum(lex_feats_map[x],v)

    return lex_feats_map.items()


def extract_instances(fname, fname_unary, fname_orig, predPairFeatsTyped, predPairFeats, unaryPairFeatsTyped, unaryPairFeatsTyped_m, baseline_probs, lexical_map=None):

    feats_list = []
    lexicalized_feats_list = []
    labels = []

    f_orig = open(fname_orig)
    lines_orig = f_orig.read().splitlines()[1:]

    f = open(fname)
    f_unary = open(fname_unary)
    lines_unary = f_unary.read().splitlines()

    idx = 0
    b_idx = 0
    for l in f:
        # print l
        # if idx==1000:
        #     break#TODO: undo this
        ss = lines_orig[idx].split("\t")
        label = ss[0]

        if label=="-":
            idx += 1
            continue

        if label == "entailment":
            label = 0

        elif label == "neutral":
            label = 1
        else:
            label = 2

        labels.append(label)

        line = l[:-1]

        feats, lexicalized_feats = extract_feats_binary(line, predPairFeatsTyped, predPairFeats, False)
        feats_r, lexicalized_feats_r  = extract_feats_binary(line, predPairFeatsTyped, predPairFeats, True)
        feats_unary, lexicalized_feats_unary = extract_feats_unary(lines_unary[idx],unaryPairFeatsTyped, False)
        feats_unary_r, lexicalized_feats_unary_r = extract_feats_unary(lines_unary[idx],unaryPairFeatsTyped, True)
        # feats_unary_m = extract_feats_unary(lines_unary[idx], unaryPairFeatsTyped_m, False)
        # feats_unary_m_r = extract_feats_unary(lines_unary[idx], unaryPairFeatsTyped_m, True)
        feats.extend(feats_r)
        feats.extend(feats_unary)
        feats.extend(feats_unary_r)
        # feats.extend(feats_unary_m)
        # feats.extend(feats_unary_m_r)

        len1 = len(ss[5].split())
        len2 = len(ss[6].split())
        feats.append((len1-len2))

        # feats.extend(baseline_probs[b_idx], np.log(baseline_probs[b_idx]))

        lexicalized_feats.extend(lexicalized_feats_r)
        lexicalized_feats.extend(lexicalized_feats_unary)
        lexicalized_feats.extend(lexicalized_feats_unary_r)

        # print "line:", lines_orig[idx]
        # print "lexicalized feats: ", lexicalized_feats

        feats_list.append(feats)

        lexicalized_feats = make_unique_lex_feats(lexicalized_feats)

        lexicalized_feats_list.append(lexicalized_feats)
        idx += 1
        b_idx += 1

    N_unlexical_feats = len(feats_list[0])
    print "N unlex feats: ", N_unlexical_feats

    if lexical_map is None:
        #This is training call, build the map

        lex2count = {}

        for lexicalized_feats in lexicalized_feats_list:
            for x,_ in lexicalized_feats:
                if x not in lex2count:
                    lex2count[x] = 1
                else:
                    lex2count[x] += 1


        lex2count_sorted = sorted(lex2count.items(), key=operator.itemgetter(1) , reverse=True)

        idx = 0
        for (x,c) in lex2count_sorted:
            if idx==NLEXFEATS:
                break
            print (x,c)
            idx += 1

        least_count = 0
        if len(lex2count_sorted)>=NLEXFEATS:
            least_count = lex2count_sorted[NLEXFEATS-1][1]

        lexical_map = {}
        idx = 0
        for lexicalized_feats in lexicalized_feats_list:
            if idx%10000==0:
                print "lex map processed instances", idx
            for x,_ in lexicalized_feats:
                if lex2count[x] < least_count:
                    continue
                if x not in lexical_map:
                    lexical_map[x] = len(lexical_map)+N_unlexical_feats
                    if len(lexical_map)%10000==0:
                        print "lex map size: ", len(lexical_map), " ", x
            idx += 1
        print "built lexical map with size: ", len(lexical_map)


    assert len(feats_list)==len(labels)

    if USE_LEXICAL_FEATS:
        X = get_sparse_matrix(feats_list, lexicalized_feats_list, lexical_map)
    else:
        X = np.array(feats_list)

    Y = np.array(labels)

    return X,Y, lexical_map#, accepted_idxes

#2.3478771e-383e-45 => 2.3478771e-38 3e-45
def handle_e(s):
    if s.count('e')>1:
        ii = s.rfind('e')
        s = s[:ii-1]+" "+s[ii-1:]
    elif s.startswith("1.0"):
        s = "1.0 "+s[3:]
    elif s.startswith("0.0"):
        s = "0.0 "+s[3:]
    return s

def read_probs(fname):
    f = open(root + "ent/msnli_preds/"+fname)
    test_probs = []
    idx = 0
    for l in f:
        if idx==20000:
            break#TODO: undo this
        if l[:-1]=="1.085944e-251.03e-45":
            l = "1.085944e-25 1.0 3e-45"
        elif l[:-1]=="1.01e-451.0219008e-36":
            l = "1.0 1e-45 1.0219008e-36"

        l = l.strip()
        for i in range(10):
            dd = str(i)
            l = l.replace(dd + ".", " " + dd + ".")
        l = l.strip()
        while len(l.split()) != 3:
            l2 = ""
            for s in l.split():
                l2 += handle_e(s) + " "
            l = l2.strip()

        ps = [np.float(x) for x in l.split()]
        test_probs.append(ps)
        idx += 1

    test_probs = np.array(test_probs)
    return test_probs


def interpret_feats(lr_coefs_,lexical_map):

    lex_map_inv = {v:x for x,v in lexical_map.items()}

    N_unlex_feats = len(lr_coefs_[0]) - len(lexical_map)
    print "N unlex feats: ", N_unlex_feats

    print "interpreting lr coefs:"

    for cl in [0,1,2]:
        print "class ", cl
        max_arg_idxes = np.argsort(np.abs(lr_coefs_[cl]))[::-1]

        for i in max_arg_idxes[0:1000]:
            if i<N_unlex_feats:
                feat_name = str(i)
            else:
                feat_name = lex_map_inv[i]

            print "feat", feat_name, lr_coefs_[cl][i]

def get_acc_for_overlap(Y,Y_pred,overlapPercs,p):
    acc = 0.0
    N = 0.0
    for i,y in enumerate(Y):
        if overlapPercs[i]>p:
            continue
        N += 1
        if Y[i]==Y_pred[i]:
            acc += 1
    if N!=0:
        acc /= N
    return acc


def get_bigrams_from_sentences(fname_orig):
    stopwords = stw.words('english')

    f = open(fname_orig)
    f.readline()

    ret = []

    idx = 0
    for line in f:
        ss = line.split("\t")
        if ss[0]=="-":
            continue
        sent1 = ss[5]
        sent2 = ss[6]

        words1 = nltk.word_tokenize(sent1.decode('utf-8'))
        words2 = nltk.word_tokenize(sent2.decode('utf-8'))

        bigrams = {w1+"#"+w2 for w1 in words1 if w1 not in stopwords for w2 in words2 if w2 not in stopwords}
        ret.append(bigrams)
        idx += 1
    return ret


def get_acc_overlap_list(Y,Y_pred,fname_orig_train,fname_orig_test,ps,desc):

    train_bigrams = get_bigrams_from_sentences(fname_orig_train)
    test_bigrams = get_bigrams_from_sentences(fname_orig_test)

    train_bigrams_set = set()
    [train_bigrams_set.update(bs) for bs in train_bigrams]

    #compute overlap
    overlaps = []
    for bs in test_bigrams:
        intersect_len = len(bs.intersection(train_bigrams_set))
        this_overlap = 0
        if len(bs)!=0:
            this_overlap = np.float(intersect_len)/len(bs)
        # print this_overlap
        overlaps.append(this_overlap)


    print "overlap acc results for", desc

    ret = []

    for p in ps:
        acc = get_acc_for_overlap(Y,Y_pred,overlaps,p)
        ret.append(acc)

    return ret, overlaps

def write_report(cl, X, Y, fname_extractions, fname_extractions_unary, fname_orig, fname_orig_train, probs_NN, tune_lmbda, desc):

    print "##################################################"

    Y_pred = cl.predict(X)
    Y_pred_prob = cl.predict_proba(X)

    lines_extractions = open(fname_extractions).read().splitlines()
    f_unary = open(fname_extractions_unary)
    lines_unary = f_unary.read().splitlines()
    f2 = open(fname_orig)
    lines_orig = f2.read().splitlines()[1:]

    idx = 0
    accIdx = 0

    accIdx2Idx = {}

    for l in lines_extractions:
        ss = lines_orig[idx].split("\t")
        if ss[0] == "-":
            idx += 1
            continue
        print ss[5] + "#" + ss[6]
        ss = l.strip().split("\t")
        try:
            print ss[0]
            print ss[1]
        except:
            pass
        ss = lines_unary[idx].strip().split("\t")
        try:
            print ss[0]
            print ss[1]
        except:
            pass
        # if X_test[accIdx][0]!=0:
        #     print "interesting"
        print X[accIdx]
        print Y[accIdx]
        print Y_pred[accIdx]
        print Y_pred_prob[accIdx]
        print "\n\n"
        accIdx2Idx[accIdx] = idx
        accIdx += 1
        idx += 1

    print "report for:", desc
    print "3-way evaluation for", desc
    acc = accuracy_score(Y, Y_pred)
    print acc

    print "3-way classification report:"
    report = classification_report(Y, Y_pred)
    print report

    c = confusion_matrix(Y, Y_pred)
    print c

    print "2-way test evaluation: "

    Y_p = np.minimum(Y, 1)
    Y_pred_p = np.minimum(Y_pred, 1)

    acc = accuracy_score(Y_p, Y_pred_p)
    print acc

    print "2-way test classification report:"
    report = classification_report(Y_p, Y_pred_p)
    print report

    c = confusion_matrix(Y_train_p, Y_train_pred_p)

    print c

    print "now combining with snli results"

    Y_pred_prob = np.array(Y_pred_prob)

    if tune_lmbda!=-1:

        lmbdas = [1.0,tune_lmbda,0.0]

        ps = np.arange(.01, 1.01, .01)

        f_overlap = open("overlap_accs/" + desc.replace(" ", "_") + "_" + str(USE_LEXICAL_FEATS) + "_res.csv",'w')

        overlap_accs_list = []

        for idx,lmbda in enumerate(lmbdas):

            print "lambda:", lmbda
            comb_prob = lmbda * probs_NN[:len(Y_pred_prob)] + (1 - lmbda) * Y_pred_prob

            Y_pred = [np.argmax(comb_prob[i, :]) for i in range(len(comb_prob))]
            if idx==0:
                NN_predictions = Y_pred
                NN_comb_prob = comb_prob

            print "3-way evaluation: "
            acc = accuracy_score(Y, Y_pred)
            print acc

            print "3-way classification report:"
            report = classification_report(Y, Y_pred)
            print report

            print desc

            accs, overlaps = get_acc_overlap_list(Y, Y_pred, fname_orig_train, fname_orig, ps, desc)
            overlap_accs_list.append(accs)

            #save results for later use
            f = open("predictions/"+desc.replace(" ","_")+"_"+str(lmbda) + "_"+ str(USE_LEXICAL_FEATS) + "_res.txt",'w')
            if idx==1:
                print "effect of adding ent graph based classifier"
            if idx==2:
                print "pure ent based classifier vs NN"
            for y_idx,y in enumerate(Y_pred):

                if y==0:
                    label = "entailment"
                elif y==1:
                    label = "neutral"
                else:
                    label = "contradiction"

                f.write(label+"\n")

                if idx==1 or (idx==2 and overlaps[y_idx]<.45):
                    nn_y = NN_predictions[y_idx]
                    if y!=nn_y:
                        anyCorrect = False
                        if y==Y[y_idx]:
                            anyCorrect = True
                            print "ours is correct"
                        elif nn_y == Y[y_idx]:
                            anyCorrect = True
                            print "NN is correct"

                        if anyCorrect:
                            ss = lines_orig[accIdx2Idx[y_idx]].split("\t")
                            l = lines_extractions[accIdx2Idx[y_idx]]

                            print ss[5] + "#" + ss[6]

                            ss = l.strip().split("\t")
                            try:
                                print ss[0]
                                print ss[1]
                            except:
                                pass
                            ss = lines_unary[accIdx2Idx[y_idx]].strip().split("\t")
                            try:
                                print ss[0]
                                print ss[1]
                            except:
                                pass
                            # if X_test[accIdx][0]!=0:
                            #     print "interesting"
                            print X[y_idx]
                            print Y[y_idx]
                            print "overlap: ", overlaps[y_idx]
                            print "ours: ", y, "pure nn:", nn_y
                            print "ours: ", comb_prob[y_idx], NN_comb_prob[y_idx]
                            print "\n\n"

                # now, compare predictions with the pure NN!
            f.close()

        f_overlap.write("overlap,NN,mixed,entailment\n")
        for i,p in enumerate(ps):
            f_overlap.write(str(p)+","+str(overlap_accs_list[0][i])+","+str(overlap_accs_list[1][i])+","+str(overlap_accs_list[2][i])+"\n")

        f_overlap.close()

        print ""

    else:
        best_acc = -1
        for lmbda in np.arange(0, 1.01, .01):
            print "lambda:", lmbda
            comb_prob = lmbda * probs_NN[:len(Y_pred_prob)] + (1 - lmbda) * Y_pred_prob

            Y_pred = [np.argmax(comb_prob[i, :]) for i in range(len(comb_prob))]

            print "3-way evaluation: "
            acc = accuracy_score(Y, Y_pred)
            if (acc>best_acc):
                best_acc = acc
                tune_lmbda = lmbda
            print acc

            print "3-way classification report:"
            report = classification_report(Y, Y_pred)
            print report
        print desc
        print "best accuracy: ", best_acc, "lmbda: ", tune_lmbda

    return tune_lmbda


root = "../../gfiles/"

#read probs for pytorch example code

if S_OR_M_NLI:
    test_probs_snli = read_probs('test_probs_snli.txt')
    dev_probs_snli = read_probs('dev_probs_snli.txt')
else:
    test_probs_mnli_matched = read_probs('test_probs_mnli_matched_noT.txt')
    test_probs_mnli_mismatched = read_probs('test_probs_mnli_mismatched_noT.txt')
    dev_probs_mnli_matched = read_probs('dev_probs_mnli_matched_noT.txt')
    dev_probs_mnli_mismatched = read_probs('dev_probs_mnli_mismatched_noT.txt')

#read probs for decomposable model code
if S_OR_M_NLI:
    test_probs_snli_d = read_probs('test_probs_snli_d.txt')
    dev_probs_snli_d = read_probs('dev_probs_snli_d.txt')
else:#noT means no transfer. just train on mnli and test on mnli
    # test_probs_mnli_matched_d = read_probs('test_probs_mnli_matched_d_noT.txt')#TODO: these should be updated
    # test_probs_mnli_mismatched_d = read_probs('test_probs_mnli_mismatched_d_noT.txt')
    # dev_probs_mnli_matched_d = read_probs('dev_probs_mnli_matched_d_noT.txt')
    # dev_probs_mnli_mismatched_d = read_probs('dev_probs_mnli_mismatched_d_noT.txt')
    pass

fname_feats = root + "ent/" + "feats_cg_pr_msnli_all.txt"
fname_feats_unary = root + "ent/" + "feats_cg_pr_msnli_unary_all.txt"
# fname_feats_unary_m = root + "ent/" + "feats_cg_pr_snli_m_unary.txt"
predPairFeats, predPairFeatsTyped, predPairSumCoefs,predPairTypedExactFound = util.read_predPairFeats(fname_feats)
unaryPairFeatsTyped = util.read_unaryPairFeatsTyped(fname_feats_unary)
# unaryPairFeatsTyped_m = util.read_unaryPairFeatsTyped(fname_feats_unary_m)

print "ent scores extracted from ent graphs"

#read original files
if S_OR_M_NLI:
    fname_orig_train = root + "snli_1.0/" + "snli_1.0_train.txt"
    fname_orig_dev = root + "snli_1.0/" + "snli_1.0_dev.txt"
    fname_orig_test = root + "snli_1.0/" + "snli_1.0_test.txt"
else:
    fname_orig_train_mnli = root + "multinli_1.0/" + "multinli_1.0_train.txt"
    fname_orig_dev_mnli_matched = root + "multinli_1.0/" + "multinli_1.0_dev_matched.txt"
    fname_orig_dev_mnli_mismatched = root + "multinli_1.0/" + "multinli_1.0_dev_mismatched.txt"
    fname_orig_test_mnli_matched = root + "multinli_1.0/" + "multinli_0.9_test_matched_unlabeled.txt"
    fname_orig_test_mnli_mismatched = root + "multinli_1.0/" + "multinli_0.9_test_mismatched_unlabeled.txt"

#read binary extractions

if S_OR_M_NLI:
    fname_extractions_train = root + "snli_1.0/" + "snli_extractions_train.txt"
    fname_extractions_dev = root + "snli_1.0/" + "snli_extractions_dev.txt"
    fname_extractions_test = root + "snli_1.0/" + "snli_extractions_test.txt"
else:
    fname_extractions_train_mnli = root + "multinli_1.0/" + "multinli_extractions_train.txt"
    fname_extractions_dev_mnli_matched = root + "multinli_1.0/" + "multinli_extractions_dev_matched.txt"
    fname_extractions_dev_mnli_mismatched = root + "multinli_1.0/" + "multinli_extractions_dev_mismatched.txt"
    fname_extractions_test_mnli_matched = root + "multinli_1.0/" + "multinli_extractions_test_matched.txt"
    fname_extractions_test_mnli_mismatched = root + "multinli_1.0/" + "multinli_extractions_test_mismatched.txt"

#read unary extractions
if S_OR_M_NLI:
    fname_extractions_train_unary = root + "snli_1.0/" + "snli_extractions_train_unary.txt"
    fname_extractions_dev_unary = root + "snli_1.0/" + "snli_extractions_dev_unary.txt"
    fname_extractions_test_unary = root + "snli_1.0/" + "snli_extractions_test_unary.txt"
else:
    fname_extractions_train_mnli_unary = root + "multinli_1.0/" + "multinli_extractions_train_unary.txt"
    fname_extractions_dev_mnli_matched_unary = root + "multinli_1.0/" + "multinli_extractions_dev_unary_matched.txt"
    fname_extractions_dev_mnli_mismatched_unary = root + "multinli_1.0/" + "multinli_extractions_dev_unary_mismatched.txt"
    fname_extractions_test_mnli_matched_unary = root + "multinli_1.0/" + "multinli_extractions_test_unary_matched.txt"
    fname_extractions_test_mnli_mismatched_unary = root + "multinli_1.0/" + "multinli_extractions_test_unary_mismatched.txt"

#extract features for allcases

if S_OR_M_NLI:
    X_train, Y_train, lexical_map = extract_instances(fname_extractions_train, fname_extractions_train_unary, fname_orig_train, predPairFeatsTyped, predPairFeats, unaryPairFeatsTyped, None, None, None)
    X_dev, Y_dev, _ = extract_instances(fname_extractions_dev, fname_extractions_dev_unary, fname_orig_dev, predPairFeatsTyped, predPairFeats, unaryPairFeatsTyped, None, None, lexical_map)
    X_test, Y_test, _ = extract_instances(fname_extractions_test, fname_extractions_test_unary, fname_orig_test, predPairFeatsTyped, predPairFeats, unaryPairFeatsTyped, None, None, lexical_map)
else:
    X_train, Y_train, lexical_map = extract_instances(fname_extractions_train_mnli, fname_extractions_train_mnli_unary, fname_orig_train_mnli, predPairFeatsTyped, predPairFeats, unaryPairFeatsTyped, None, None, None)
    X_dev_mnli_matched, Y_dev_mnli_matched,_ = extract_instances(fname_extractions_dev_mnli_matched, fname_extractions_dev_mnli_matched_unary, fname_orig_dev_mnli_matched, predPairFeatsTyped, predPairFeats, unaryPairFeatsTyped, None, None, lexical_map)
    X_dev_mnli_mismatched, Y_dev_mnli_mismatched,_ = extract_instances(fname_extractions_dev_mnli_mismatched, fname_extractions_dev_mnli_mismatched_unary, fname_orig_dev_mnli_mismatched, predPairFeatsTyped, predPairFeats, unaryPairFeatsTyped, None, None, lexical_map)
    X_test_mnli_matched, Y_test_mnli_matched,_ = extract_instances(fname_extractions_test_mnli_matched, fname_extractions_test_mnli_matched_unary, fname_orig_test_mnli_matched, predPairFeatsTyped, predPairFeats, unaryPairFeatsTyped, None, None, lexical_map)
    X_test_mnli_mismatched, Y_test_mnli_mismatched,_ = extract_instances(fname_extractions_test_mnli_mismatched, fname_extractions_test_mnli_mismatched_unary, fname_orig_test_mnli_mismatched, predPairFeatsTyped, predPairFeats, unaryPairFeatsTyped, None, None, lexical_map)


print "features extracted"


cl = LogisticRegression(penalty='l1')
cl.fit(X_train,Y_train)

print "classes: ", cl.classes_

print "lr coefs: ", cl.coef_
interpret_feats(cl.coef_,lexical_map)


Y_train_pred = cl.predict(X_train)
print "3-way train evaluation: "
acc = accuracy_score(Y_train, Y_train_pred)
print acc

report = classification_report(Y_train, Y_train_pred)
print "3-way train classification report:"
print report


print "2-way train evaluation: "

Y_train_p = np.minimum(Y_train, 1)
Y_train_pred_p = np.minimum(Y_train_pred, 1)

print "2-way train classification results:"
report = classification_report(Y_train_p, Y_train_pred_p)
print report

acc = accuracy_score(Y_train_p, Y_train_pred_p)
print acc

#write reports for NN1
if S_OR_M_NLI:
    lmbda = write_report(cl,X_dev,Y_dev,fname_extractions_dev,fname_extractions_dev_unary,fname_orig_dev,fname_orig_train,dev_probs_snli, -1,'dev NN1')
    write_report(cl,X_test,Y_test,fname_extractions_test,fname_extractions_test_unary,fname_orig_test,fname_orig_train,test_probs_snli, lmbda, 'test NN1')
else:
    lmbda = write_report(cl,X_dev_mnli_matched,Y_dev_mnli_matched,fname_extractions_dev_mnli_matched,fname_extractions_dev_mnli_matched_unary,fname_orig_dev_mnli_matched,fname_orig_train_mnli,dev_probs_mnli_matched, -1, 'mnli dev matched NN1')
    write_report(cl, X_dev_mnli_matched, Y_dev_mnli_matched, fname_extractions_dev_mnli_matched, fname_extractions_dev_mnli_matched_unary, fname_orig_dev_mnli_matched,fname_orig_train_mnli, dev_probs_mnli_matched, lmbda, 'mnli dev dev matched NN1')
    write_report(cl,X_test_mnli_matched,Y_test_mnli_matched,fname_extractions_test_mnli_matched,fname_extractions_test_mnli_matched_unary,fname_orig_test_mnli_matched,fname_orig_train_mnli,test_probs_mnli_matched, lmbda, 'mnli test matched NN1')
    lmbda = write_report(cl,X_dev_mnli_mismatched,Y_dev_mnli_mismatched,fname_extractions_dev_mnli_mismatched,fname_extractions_dev_mnli_mismatched_unary,fname_orig_dev_mnli_mismatched,fname_orig_train_mnli,dev_probs_mnli_mismatched, -1, 'mnli dev mismatched NN1')
    write_report(cl, X_dev_mnli_mismatched, Y_dev_mnli_mismatched, fname_extractions_dev_mnli_mismatched, fname_extractions_dev_mnli_mismatched_unary, fname_orig_dev_mnli_mismatched,fname_orig_train_mnli, dev_probs_mnli_mismatched, lmbda, 'mnli dev dev mismatched NN1')
    write_report(cl,X_test_mnli_mismatched,Y_test_mnli_mismatched,fname_extractions_test_mnli_mismatched,fname_extractions_test_mnli_mismatched_unary,fname_orig_test_mnli_mismatched,fname_orig_train_mnli,test_probs_mnli_mismatched, lmbda, 'mnli test mismatched NN1')


#write reports for NN2
if S_OR_M_NLI:
    lmbda = write_report(cl,X_dev,Y_dev,fname_extractions_dev,fname_extractions_dev_unary,fname_orig_dev,fname_orig_train,dev_probs_snli_d, -1, 'dev NN2')
    write_report(cl,X_test,Y_test,fname_extractions_test,fname_extractions_test_unary,fname_orig_test,fname_orig_train,test_probs_snli_d, lmbda, 'test NN2')
else:
    # lmbda = write_report(cl,X_dev_mnli_matched,Y_dev_mnli_matched,fname_extractions_dev_mnli_matched,fname_extractions_dev_mnli_matched_unary,fname_orig_dev_mnli_matched,dev_probs_mnli_matched_d, -1, 'mnli dev matched NN2')
    # write_report(cl,X_test_mnli_matched,Y_test_mnli_matched,fname_extractions_test_mnli_matched,fname_extractions_test_mnli_matched_unary,fname_orig_test_mnli_matched,test_probs_mnli_matched_d, lmbda, 'mnli test matched NN2')
    # lmbda = write_report(cl,X_dev_mnli_mismatched,Y_dev_mnli_mismatched,fname_extractions_dev_mnli_mismatched,fname_extractions_dev_mnli_mismatched_unary,fname_orig_dev_mnli_mismatched,dev_probs_mnli_mismatched_d, -1, 'mnli dev mismatched NN2')
    # write_report(cl,X_test_mnli_mismatched,Y_test_mnli_mismatched,fname_extractions_test_mnli_mismatched,fname_extractions_test_mnli_mismatched_unary,fname_orig_test_mnli_mismatched,test_probs_mnli_mismatched_d, lmbda, 'mnli test mismatched NN2')
    pass
