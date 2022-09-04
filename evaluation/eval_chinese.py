import sys
import numpy as np
sys.path.append("..")
from graph import graph
from lemma_baseline import qa_utils_chinese
import evaluation.util_chinese
from lemma_baseline import chinese_baselines
from copy import deepcopy
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import os
from sklearn import svm
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
from constants.flags import opts
from ppdb import predict
from bert_baseline import bert_similarity
from bert_baseline import bert_template
import json

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.nan)

#this is the default postfix, but can be overwritten by --sim_suffix
f_post_fix = "_sim.txt"

#only used for binary graphs (not used by default)
grPostfix = "_binc_lm1_.001_reg_1.5_1_.2_.01_i20_HTLFRG_confAvg_nodisc.txt"

root = "../gfiles/"
sysargs = sys.argv[1:]
args = opts(sysargs)
debug = graph.debug = qa_utils_chinese.debug = chinese_baselines.debug = evaluation.util_chinese.debug = predict.debug = args.debug


def get_sum_simlar_feats(gr,p1s,q1s):
    feats = np.zeros(graph.Graph.num_feats)
    num_found = 0
    for p1 in p1s:
        for q1 in q1s:
            this_feats = gr.get_features(p1,q1)
            if this_feats is not None:
                if debug:
                    print("not None in get_sims: ", p1,q1, this_feats)
                feats += this_feats
                num_found += 1
    if not num_found:
        feats = None
    else:
        feats /= num_found
    return feats

#deprecated function
#coef1 is for the case without embeddings. coef2 is with embeddings
#Propagating to t1, t2 from other types
# def get_coefs(p1,q1,p,q,t1,t2,a,is_typed,args):

def equalType(p1,t1,t2):
    p1t = p1.replace("_1", "").replace("_2", "")
    p1ss = p1t.split("#")
    ret = t1 == p1ss[1] and t2 == p1ss[2]
    if ret==False:
        if debug:
            print("equalType false: ", p1,t1,t2)
    return ret


# p and q are the original ones. p1 and q1 are typed predicates
# It can be used for predPairFeats or predPairFeatsTyped
def add_feats_for_predPair(gr,p,q,a,t1,t2,p1,q1, p1s, q1s, predPairCounts, predPairSumCoefs, predPairSimSumCoefs, predPairFeats,predPairTypedExactFound,is_typed):

    feats = gr.get_features(p1,q1)  # get similarity scores between this type pair, [sims;orders]
    feats_sim = get_sum_simlar_feats(gr,p1s,q1s)  # most probably, p1s and q1s contain only one predicate, and this returns the same as feats.

    #This is for numNodes: num p, num q, num nodes

    if feats is not None or feats_sim is not None:
        no_exact_feats = False
        if debug:
            print("feats:", feats)
            print("feats_sim:", feats_sim)
        if feats is None:
            no_exact_feats = True  # means: feats is None, and feats_sim is not None.
            feats = np.zeros(graph.Graph.num_feats)

        if is_typed:
            predPair = p+"#"+q+"#"+str(a)+"#"+t1+"#"+t2
            if debug:
                print("featsT: ", feats)
        else:
            predPair = p+"#"+q+"#"+str(a)
        if debug:
            print("predPair:", predPair)
            print("p1,q1: ", p1, q1)
        if np.count_nonzero(feats)!=0 and not is_typed:
            if debug:
                print("adding predPair to counts")
            evaluation.util_chinese.addPred(predPair, predPairCounts)
        if predPair not in predPairFeats:
            predPairFeats[predPair] = (np.zeros(num_feats))
            #For similarity ones:
            if not is_typed:
                if debug:
                    print("setting predpairsums to zero")
                predPairSumCoefs[predPair] = 0
                predPairSimSumCoefs[predPair] = 0

        coef1,coef2 = 1, 1

        if is_typed and equalType(p1,t1,t2) and not no_exact_feats:
            if debug:
                print("adding to predPairTyped: ", predPair)
            predPairTypedExactFound.add(predPair)

        if debug:
            print("coef: ", coef1, " ", coef2)
        if is_typed:
            predPairFeats[predPair][0:graph.Graph.num_feats] = feats
            predPairFeats[predPair][graph.Graph.num_feats:2*graph.Graph.num_feats] = feats_sim

        else:
            if debug:
                print("setting predPairFeats")
            predPairFeats[predPair][0:graph.Graph.num_feats] += coef1 * feats
            predPairFeats[predPair][graph.Graph.num_feats:2*graph.Graph.num_feats] += coef2 * feats_sim
            if not no_exact_feats:
                if debug:
                    print("setting sumCoefs:")
                predPairSumCoefs[predPair] += coef1
            else:
                if debug:
                    print("adding coef only for sims: ", coef2)
            predPairSimSumCoefs[predPair] += coef2
    return feats is not None and feats[0] != 0 and not no_exact_feats  # feats[0] == 0 means feats used to be None


def add_feats_for_unaryPair(gr, u, v, u1,v1, t, unaryPairFeats, unaryPairSumCoefs):
    raise NotImplementedError
    feats_unary, coef1 = gr.get_features_unary(u1,v1)  # weighted average of all predicate pairs containing these halves of predicates (pred1.1) / (pred2.to.2)

    if feats_unary is not None:

        unaryPair = u + "#" + v + "#" + t
        if unaryPair not in unaryPairFeats:
            unaryPairFeats[unaryPair] = (np.zeros(graph.Graph.num_feats/2))#we don't have ranks for unary for simplicity

            unaryPairSumCoefs[unaryPair] = 0


        unaryPairFeats[unaryPair][0:graph.Graph.num_feats/2] += coef1 * feats_unary
        unaryPairSumCoefs[unaryPair] += coef1

    return feats_unary is not None


def form_samples_gr(gr, data, data_unary, predCounts, predPairCounts, predPairSumCoefs,predPairSimSumCoefs,predPairTypedSumCoefs, predPairTypedSimSumCoefs, predPairFeats, predPairFeatsTyped, predPairConnectedList,predPairConnectedWeightList, predPairTypedConnectedList, predPairTypedConnectedWeightList, predPairTypedExactFound, unaryPairFeatsTyped, unaryPairTypedSumCoefs, rels2Sims, args):
    if gr:
        types = gr.types
    else:
        raise AssertionError

    for (p,q,t1,t2,a) in data:
        ps = rels2Sims[p] if p in rels2Sims else [p]  # The most similar ones from the symmetric similarity measure
        qs = rels2Sims[q] if q in rels2Sims else [q]  # The most similar ones from the symmetric similarity measure
        # If you wanna exactly match the types, you should only take one of these!

        p1 = p + "#" + types[0] + "#" + types[1]
        p2 = p + "#" + types[1] + "#" + types[0]

        p1s = [pp + "#" + types[0] + "#" + types[1] for pp in ps]  # without .sim should be just adding brackets
        p2s = [pp + "#" + types[1] + "#" + types[0] for pp in ps]  # without .sim should be just adding brackets

        if a:  # if arguments are aligned.
            q1 = q + "#" + types[0] + "#" + types[1]
            q2 = q + "#" + types[1] + "#" + types[0]

            q1s = [qq + "#" + types[0] + "#" + types[1] for qq in qs]
            q2s = [qq + "#" + types[1] + "#" + types[0] for qq in qs]

        else:
            q1 = q + "#" + types[1] + "#" + types[0]
            q2 = q + "#" + types[0] + "#" + types[1]

            q1s = [qq + "#" + types[1] + "#" + types[0] for qq in qs]
            q2s = [qq + "#" + types[0] + "#" + types[1] for qq in qs]

        if p1 in gr.pred2Node or p2 in gr.pred2Node:  # add the (untyped) predicate into predCounts if it's in entailment graph (just to keep a record)
            evaluation.util_chinese.addPred(p, predCounts)

        if q1 in gr.pred2Node or q2 in gr.pred2Node:
            evaluation.util_chinese.addPred(q, predCounts)
        #gr,p,q,a,t1,t2,p1,q1, p1s, q1s, predPairCounts, predPairSumCoefs, predPairSimSumCoefs, predPairFeats,is_typed
        added = add_feats_for_predPair(gr,p,q,a,None,None,p1,q1,p1s,q1s,predPairCounts, predPairSumCoefs,predPairSimSumCoefs, predPairFeats,None,False)
        if added:
            if debug:
                print ("added: ", p1, " ", q1," ", a, " ", types)

        added2 = add_feats_for_predPair(gr,p,q,a,None,None,p2,q2,p2s,q2s,predPairCounts, predPairSumCoefs,predPairSimSumCoefs, predPairFeats,None,False)
        if added2:
            if debug:
                print ("added2: ", p2, " ", q2," ", a, " ", types)

        #Now, the typed ones!
        type_pair = None  # This is to find the order of types (if any) that matches the graph!
        if t1==t2:
            if types[0][:-2]==types[1][:-2] and types[0][:-2]==t1:  # the [:-2] probably comes from the _1 and _2 added for identical types
                type_pair = types[0] + "#" + types[1]
        else:
            type_pair1 = t1+"#"+t2
            type_pair2 = t2+"#"+t1
            if type_pair1 == types[0] + "#" + types[1]:
                type_pair = t1+"#"+t2  # We just check to see if graph types matches this predicate types. But still, should stick to t1, t2 not t2, t1!
            elif type_pair2 == types[0] + "#" + types[1]:
                type_pair = t1+"#"+t2

        if type_pair is not None:
            if debug:
                print("a type match: ",type_pair, (p,q,t1,t2,a))
            this_types = type_pair.split("#")
            p1 = p + "#" + this_types[0] + "#" + this_types[1]
            p1s = [pp + "#" + this_types[0] + "#" + this_types[1] for pp in ps]
            if a:
                q1 = q + "#" + this_types[0] + "#" + this_types[1]
                q1s = [qq + "#" + this_types[0] + "#" + this_types[1] for qq in qs]
            else:
                q1 = q + "#" + this_types[1] + "#" + this_types[0]
                q1s = [qq + "#" + this_types[1] + "#" + this_types[0] for qq in qs]

            added = add_feats_for_predPair(gr,p,q,a,t1,t2,p1,q1,p1s,q1s,predPairCounts,None, None, predPairFeatsTyped,predPairTypedExactFound,True)
            if added:
                print("added typed: ", p1, " ", q1," ", a, " ", types)

    ts = types  # This is to find the order of types (if any) that matches the graph!
    ts[0] = ts[0].replace("_1","").replace("_2","")
    ts[1] = ts[1].replace("_1","").replace("_2","")

    for (u, v, t) in data_unary:
        u1 = u + "#" + t
        v1 = v + "#" + t

        if t==ts[0] or t==ts[1]:
            added = add_feats_for_unaryPair(gr, u, v, u1,v1, t, unaryPairFeatsTyped, unaryPairTypedSumCoefs)
            # if added:
            #     print "added unary typed: ", u, v, t


def form_samples(fnames,fnames_unary,orig_fnames,engG_dir_addr,fname_feats=None, rels2Sims=None, args=None):
    if args and args.featIdx is not None:
        graph.Graph.featIdx = args.featIdx

    global num_feats
    num_feats = -1

    data_list = [evaluation.util_chinese.read_data(fnames[i], orig_fnames[i], args.CCG, args.typed, args.LDA) for i in range(len(fnames))]
    data_list_unary = []
    if fnames_unary:
        data_list_unary = [evaluation.util_chinese.read_data_unary(fnames_unary[i], args.typed) for i in range(len(fnames_unary))]

    predc_ds = []  # single predicate counts bucket, for evaluation data
    predPairc_DS = []  # predicate pairs counts bucket
    predPaircTyped_DS = []  # typed predicate pairs count bucket
    predPairc_Pos_DS = []  # positive (untyped) predicate pairs count bucket

    for data in data_list:
        (predCDS_, predPairsCDS_, predPairsTypedCDS_, predPairsC_Pos_DS_) = evaluation.util_chinese.getPredPairs(data)
        predc_ds.append(predCDS_)
        predPairc_DS.append(predPairsCDS_)
        predPaircTyped_DS.append(predPairsTypedCDS_)
        predPairc_Pos_DS.append(predPairsC_Pos_DS_)

    predPairTypedExactFound = set()

    if fname_feats is not None:
        predPairFeats, predPairFeatsTyped, predPairSumCoefs, predPairTypedExactFound = evaluation.util_chinese.read_predPairFeats(fname_feats, data_list)
        num_feats = evaluation.util_chinese.num_feats
        return data_list, predPairSumCoefs, predPairFeats,predPairFeatsTyped, None, None, None, None, predPairTypedExactFound

    # below fname_feats must be None

    data_agg = []  # turn type lists into single types, essentially just removing the brackets, since the lists should always be length 1
    for data in data_list:
        for (p,q,t1s,t2s,_,a,_) in data:
            for t_i in range(len(t1s)):
                t1 = t1s[t_i]
                t2 = t2s[t_i]
                data_agg.append((p,q,t1,t2,a))
    data_agg = set(data_agg)  # remove redundancy

    data_agg_unary = []
    for data_unary in data_list_unary:
        data_agg_unary.extend(data_unary)
    data_agg_unary = set(data_agg_unary)
    if debug:
        print("num unaryPairs: ", len(data_agg_unary))

    # Above reads in the evaluation data!

    # The following to be filled based on the graphs!
    predCounts = {}
    predPairCounts = {}
    predPairSumCoefs = {}
    predPairSimSumCoefs = {}
    predPairTypedSumCoefs = {}
    predPairTypedSimSumCoefs = {}
    predPairFeats = {}
    predPairFeatsTyped = {}
    unaryPairFeatsTyped = {}
    unaryPairTypedSumCoefs = {}

    predPairConnectedList = None
    predPairConnectedWeightList = None
    predPairTypedConnectedList = None
    predPairTypedConnectedWeightList = None

    files = os.listdir(engG_dir_addr)
    files = list(np.sort(files))
    num_f = 0

    for f in files:  # these are the entailment sims files!
        # if num_f == 100:#Use this for debugging!
        #     break
        gpath=engG_dir_addr+f

        if f_post_fix not in f or os.stat(gpath).st_size == 0:
            continue
        if debug:
            print("fname: ", f)
        num_f += 1

        if num_f % 50 == 0:
            print("num processed files: ", num_f)

        if read_sims:
            gr = graph.Graph(gpath=gpath, args = args)  # loads graph from the current sims file.
            gr.set_Ws()  # sets weights to gr.idxpair2score[(prem, hypo)]
        else:
            gr = None
        if num_feats==-1 and read_sims:
            num_feats = 2*gr.num_feats

        if read_sims:
            if debug:
                print("gr size: ", sys.getsizeof(gr), "num edge: ", gr.num_edges)
        if debug:
            print("reading TNFs: (BYPASSED!)")

        form_samples_gr(gr, data_agg, data_agg_unary, predCounts, predPairCounts, predPairSumCoefs, predPairSimSumCoefs, predPairTypedSumCoefs, predPairTypedSimSumCoefs,
                        predPairFeats, predPairFeatsTyped, predPairConnectedList,predPairConnectedWeightList,predPairTypedConnectedList,predPairTypedConnectedWeightList, predPairTypedExactFound,
                        unaryPairFeatsTyped, unaryPairTypedSumCoefs, rels2Sims,args)
        del gr

    for (idx,fname) in enumerate(fnames):
        if debug:
            print("stats for: " + fname)
            print("num all preds: ", len(predc_ds[idx]))
            print("num all predPairs: ", len(predPairc_DS[idx]))
            print("num all pos predPairs: ", len(predPairc_Pos_DS[idx]))
            print("num preds covered: ", np.count_nonzero( [(pred in predCounts) for pred in predc_ds[idx]] ))
            print("num all predPairs covered: ", np.count_nonzero( [(predpair in predPairCounts) for predpair in predPairc_DS[idx]] ))
            print("num pos predPairs covered: ", np.count_nonzero( [(predpair in predPairCounts) for predpair in predPairc_Pos_DS[idx]] ))
            print("predCounts: ")

        for p in predc_ds[idx]:
            if p in predCounts:
                if debug:
                    print(p + "\t" + str(predCounts[p]) + "\t" + str(predc_ds[idx][p]))
            else:
                if debug:
                    print(p + "\t" + "0" + "\t" + str(predc_ds[idx][p]))
        if debug:
            print("predPairCounts_Pos: ")
        for r in predPairc_Pos_DS[idx]:
            if r in predPairCounts:
                if debug:
                    print(r + "\t" + str(predPairCounts[r]) + "\t" + str(predPairc_DS[idx][r]))
            else:
                if debug:
                    print(r + "\t" + "0" + "\t" + str(predPairc_DS[idx][r]))

    if read_sims:
        predpair_set = []
        for predPairc in predPairc_DS:
            for predpair in predPairc:
                predpair_set.append(predpair)
        predpair_set = set(predpair_set)

        predpairTyped_set = []
        for predPairc in predPaircTyped_DS:
            for predpair in predPairc:
                predpairTyped_set.append(predpair)
        predpairTyped_set = set(predpairTyped_set)

        #divide the features!
        for r in unaryPairFeatsTyped:
            if unaryPairTypedSumCoefs[r] != 0:  # In the above case, it won't be zero
                unaryPairFeatsTyped[r] /= unaryPairTypedSumCoefs[r]

        # divide the unary features! (they used to be summed over all types, now we take the average -- Teddy)
        for r in predpair_set:
            if r in predPairSumCoefs:
                if predPairSumCoefs[r] != 0:  # In the above case, it won't be zero
                    predPairFeats[r][0:graph.Graph.num_feats] /= predPairSumCoefs[r]
                if predPairSimSumCoefs[r] != 0:
                    predPairFeats[r][graph.Graph.num_feats:2 * graph.Graph.num_feats] /= predPairSimSumCoefs[r]
        if debug:
            print("predPairFeats: ")
        if not os.path.isdir('feats'):
            os.mkdir('feats')
        f_feats = open('feats/feats_'+args.method+'.txt', 'w', encoding='utf8')
        for r in predpair_set:
            if r in predPairSumCoefs:
                line = r + "\t" + str(predPairSumCoefs[r]) + "\t" + str(predPairSimSumCoefs[r]) + "\t" + str(predPairFeats[r])
                if debug:
                    print(line)
                f_feats.write(line+'\n')
            else:
                predPairFeats[r] = deepcopy(np.zeros(num_feats))
                line = r + "\t" + "0\t0" + "\t" + str(np.zeros(num_feats))
                if debug:
                    print(line)
                f_feats.write(line + '\n')

        if debug:
            print("predPairFeatsTyped: ")
        for r in predpairTyped_set:
            if r in predPairFeatsTyped:
                line = r + "\t" + str(predPairFeatsTyped[r])
                if debug:
                    print(line)
                f_feats.write(line + '\n')
            else:
                predPairFeatsTyped[r] = deepcopy(np.zeros(num_feats))
                line = r + "\t" + str(np.zeros(num_feats))
                if debug:
                    print(line)
                f_feats.write(line + '\n')

        f_feats.write("predPairTypedExactFound:\n")
        for x in predPairTypedExactFound:
            f_feats.write(x + '\n')

        print(f"len of predPairExactFound: {len(predPairTypedExactFound)}")
        print("predPairExactFound: ", predPairTypedExactFound)

        # Writing unary features
        f_feats_unary = open('feats/feats_' + args.method + '_unary.txt', 'w', encoding='utf8')
        for r in unaryPairFeatsTyped:
            line = r + "\t" + str(unaryPairTypedSumCoefs[r]) + "\t" + str(unaryPairFeatsTyped[r])
            if debug:
                print(line)
            f_feats_unary.write(line + '\n')

    if debug:
        print("predPairConnectedList in fit predict:", predPairConnectedList)
    return data_list, predPairSumCoefs, predPairFeats,predPairFeatsTyped,predPairConnectedList,predPairConnectedWeightList,predPairTypedConnectedList, predPairTypedConnectedWeightList, predPairTypedExactFound


def get_typed_feats(data,predPairFeatsTyped):
    X_typed = []
    for (p,q,t1s,t2s,probs,a,l) in data:
        this_X = np.zeros(num_feats)
        for i in range(len(t1s)):
            this_X += predPairFeatsTyped[p+"#"+q+"#"+str(a)+"#"+t1s[i]+"#"+t2s[i]]*probs[i]
        X_typed.append(this_X)
    return X_typed


#works in both unsupervised and supervised settings to return the entailment scores
def fit_predict(data_list, predPairFeats,predPairFeatsTyped,predPairConnectedList,predPairTypedConnectedList,predPairTypedExactFound,args):

    [data_train, data_dev] = data_list
    assert predPairConnectedList is None
    if debug:
        print ("predPairConnectedList is None")
    Y_dev_TNF = None
    Y_dev_TNF_typed = None

    if not read_sims:
        return None,Y_dev_TNF,Y_dev_TNF_typed, None, None

    X_train = [list(deepcopy(predPairFeats[p+"#"+q+"#"+str(a)])) for (p,q,_,_,_,a,l) in data_train]

    X_train_typed = get_typed_feats(data_train,predPairFeatsTyped)

    if not args.useSims:
        X_train = [x[0:len(X_train[0])//2] for x in X_train]  # only the first half! -> the second half are symmetric similars! -- Teddy
        X_train_typed = [x[0:len(X_train_typed[0])//2] for x in X_train_typed]  # only the first half! -> the second half are symmetric similars! -- Teddy

    if debug:
        print("here shape: ", np.array(X_train).shape)

    # if not args.calcSupScores:
    #     [X_train[i].extend(X_train_typed[i]) for i in range(len(X_train))]
    # else:
    #     X_train = X_train_typed

    #X: avg, avg_rank, avg_emb, avg_rank_emb, avg_typed, avg_rank_typed, avg_emb_typed, avg_rank_emb_typed
    X_dev = [list(deepcopy(predPairFeats[p+"#"+q+"#"+str(a)])) for (p,q,_,_,_,a,l) in data_dev]
    X_dev_typed = get_typed_feats(data_dev,predPairFeatsTyped)

    if not args.useSims:
        X_dev = [x[0:len(X_dev[0])//2] for x in X_dev]#only the first half! The second half is based on those similarities.
        # The remaining first half can again be split in two, the first half of the first half is the original score, the second of the first half is the reciprocal ranking score.
        X_dev_typed = [x[0:len(X_dev_typed[0])//2] for x in X_dev_typed]#only the first half!

    assert not args.supervised
    if args.oneFeat:  # this flag has been set to ``True'' - Teddy
        if not args.saveMemory:
            if not args.useSims:
                f_idx = graph.Graph.featIdx
            else:  # use the non-exact predicate match
                f_idx = graph.Graph.num_feats + graph.Graph.featIdx
            if args.rankFeats:  # use ranking score
                f_idx += graph.Graph.num_feats//2
                if debug:
                    print('new feat idx for rank: ', f_idx)
        else:
            if args.useSims:
                f_idx = 2
            else:
                f_idx = 0

        if not args.exactType:
            if not args.rankDiscount:
                Y_dev_pred = [x[f_idx] for x in X_dev]
            else:
                Y_dev_pred = [x[f_idx]*x[f_idx+graph.Graph.num_feats//2] ** .5 for x in X_dev]
        else:
            if not args.rankDiscount:
                Y_dev_pred = [x[f_idx] for x in X_dev_typed]
            else:
                Y_dev_pred = [x[f_idx] * x[f_idx + graph.Graph.num_feats // 2] ** .5 for x in X_dev_typed]

            if args.backupAvg:  # if exact match is not found, backup with average.
                if not args.rankDiscount:
                    Y_dev_pred_backup = [x[f_idx] for x in X_dev]
                else:
                    # this is first average, then multiplied, but should be the other way around
                    Y_dev_pred_backup = [x[f_idx] * x[f_idx + graph.Graph.num_feats / 2] ** .5 for x in X_dev]
                Y_dev_pred2 = []
                for i in range(len(Y_dev_pred)):
                    l = Y_dev_pred[i]

                    (p, q, t1s, t2s, probs, a, _) = data_dev[i]

                    if l == 0:  # In practice, if it's zero, it's zero for everything!!! I tested this!
                        l = Y_dev_pred_backup[i]

                    Y_dev_pred2.append(l)

                Y_dev_pred = Y_dev_pred2

        if debug:
            print("nnz Y_dev_pred: ", np.count_nonzero(Y_dev_pred))

    elif args.wAvgFeats:

        assert not args.typed  # Because it's not implemented yet!

        ss = args.wAvgFeats.split()
        idxes = [np.int(x) for i,x in enumerate(ss) if i%2==0 ]
        weights = [np.float(x) for i, x in enumerate(ss) if i % 2 == 1]
        sum_weighs = sum(weights)

        def weighted_sum(x,idxes,weights):
            ret = 0
            for i,idx in enumerate(idxes):
                ret += x[idx]*weights[i]
            return ret/sum_weighs

        Y_dev_pred = [weighted_sum(x,idxes,weights) for x in X_dev]

    elif args.gAvgFeats:

        assert not args.typed  # Because it's not implemented yet!

        ss = args.wAvgFeats.split()
        idxes = [np.int(x) for i,x in enumerate(ss) if i%2==0 ]
        weights = [np.float(x) for i, x in enumerate(ss) if i % 2 == 1]
        sum_weighs = sum(weights)

        def weighted_sum(x,idxes,weights):
            ret = 0
            for i,idx in enumerate(idxes):
                ret += x[idx]*weights[i]
            return ret/sum_weighs

        Y_dev_pred = [weighted_sum(x,idxes,weights) for x in X_dev]

    else:
        raise Exception("terrible exception")

    return Y_dev_pred,Y_dev_TNF,Y_dev_TNF_typed, None, None


def eval(Y_pred,Y, write = True):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    assert len(Y) == len(Y_pred)
    for i in range(len(Y)):
        if Y_pred[i]==Y[i]:
            if Y_pred[i]==1:
                TP += 1
            else:
                TN += 1
        else:
            if Y_pred[i]==1:
                FP += 1
            else:
                FN += 1
    if write:
        print("TP: ", TP)
        print("FP: ", FP)
        print("TN: ", TN)
        print("FN: ", FN)
    if (TP+FP) == 0 or TP == 0:
        pr = 0
        rec = 0
        f1 = 0
    else:
        pr = float(TP)/(TP+FP)
        rec = float(TP)/(TP+FN)
        f1 = 2*pr*rec/(pr+rec)
    acc = float(TP+TN)/len(Y)
    if write:
        print("prec: ", pr)
        print("rec: ", rec)
        print("f1: ", f1)
        print("acc: ", acc)
        print(str(TP)+"\t"+str(FP)+"\t"+str(TN)+"\t"+str(FN)+"\t"+str(pr)+"\t"+str(rec)+"\t"+str(f1))

    return pr,rec, acc


def merge_results(Y_dev_pred, args):
    Y_dev_pred_merged = []
    mapping_fn = args.untyped_mapping_fn if args.use_untyped_mapping else args.typed_mapping_fn
    if len(mapping_fn) == 0:
        print(f"No mapping found! Returning the same Preds!")
        return Y_dev_pred
    if not args.use_untyped_mapping:
        with open(mapping_fn, 'r', encoding='utf8') as idx_fp:
            Y_dev_mapping = json.load(idx_fp)
    else:
        Y_dev_mapping = []
        with open(mapping_fn, 'r', encoding='utf8') as idx_fp:
            for idx_line in idx_fp:
                Y_dev_mapping.append(int(idx_line.strip()))
    assert len(Y_dev_mapping) == len(Y_dev_pred)

    for map_idx, cur_Y_pred in zip(Y_dev_mapping, Y_dev_pred):
        if map_idx == len(Y_dev_pred_merged):
            Y_dev_pred_merged.append(cur_Y_pred)
        else:
            assert map_idx == len(Y_dev_pred_merged) - 1
            if args.max_pooling:
                Y_dev_pred_merged[map_idx] = max(Y_dev_pred_merged[map_idx], cur_Y_pred)
            elif args.avg_pooling:
                Y_dev_pred_merged[map_idx] = float(Y_dev_pred_merged[map_idx] + cur_Y_pred) / 2
    return Y_dev_pred_merged


#It will use Y_dev_base and Y_dev_pred0 to form Y_dev_pred which will be used to find precisions and recalls
#It will also change Y_dev_TNFs so it will be used later
def final_prediction(data_dev, data_dev_CCG, predPairFeats, predPairFeatsTyped, predPairSumCoefs, Y_dev_lemma, Y_dev_exact, Y_dev_pred0,Y_dev_TNF0,Y_dev_TNF_typed0, lines_dev, root, args):

    def write_scores_from_stats(ofp, stats, with_thres=True):
        ofp.write(f"main auc: {stats['main_auc']}\n")
        for lid in range(len(stats['precrec']['precision'])):
            if with_thres:
                ofp.write(f"{stats['precrec']['precision'][lid]}\t{stats['precrec']['recall'][lid]}\t{stats['precrec']['thresholds'][lid]}\n")
            else:
                ofp.write(f"{stats['precrec']['precision'][lid]}\t{stats['precrec']['recall'][lid]}\n")
        return

    calc_bert_baseline = False
    max_pooling = True
    avg_pooling = False

    Y_dev_pred = []
    Y_dev = [l for (_,_,_,_,_,_,l) in data_dev]
    Y_dev_seen = []
    Y_dev_pred_seen = []

    Y_dev_merged = merge_results(Y_dev, args)
    Y_dev_lemma_merged = merge_results(Y_dev_lemma, args)
    Y_dev_exact_merged = merge_results(Y_dev_exact, args)

    # f_out_predpair_seen = open('predpair_seen.txt', 'w', encoding='utf8')

    if raw_fnames[1] and calc_bert_baseline:
        sim_calculator = bert_similarity.BertSimilarity_Calculator(model_name='bert')
        xlm_sim_calculator = bert_similarity.BertSimilarity_Calculator(model_name='xlm-roberta')
        #temp_calculator = bert_template.BertTemplate_Calculator()
        Y_dev_bert_sim = sim_calculator.calc_sim_file(raw_fnames[1])
        Y_dev_xlm_sim = xlm_sim_calculator.calc_sim_file(raw_fnames[1])
        Y_dev_bert_sim = merge_results(Y_dev_bert_sim, args)
        Y_dev_xlm_sim = merge_results(Y_dev_xlm_sim, args)
        #Y_dev_bert_relative_sim = sim_calculator.calc_sim_relative_file(raw_fnames[1])
        #Y_dev_bert_temp = temp_calculator.calc_sim_file(raw_fnames[1])
        assert len(Y_dev_merged) == len(Y_dev_bert_sim)
        assert len(Y_dev_merged) == len(Y_dev_xlm_sim)
        #assert len(Y_dev) == len(Y_dev_bert_temp)
        #assert len(Y_dev) == len(Y_dev_bert_relative_sim)
        Y_dev_bert_sim = [True if y_exact or y_lemma else y_pred for (y_exact, y_lemma, y_pred) in zip(Y_dev_exact_merged, Y_dev_lemma_merged, Y_dev_bert_sim)]
        Y_dev_xlm_sim = [True if y_exact or y_lemma else y_pred for (y_exact, y_lemma, y_pred) in zip(Y_dev_exact_merged, Y_dev_lemma_merged, Y_dev_xlm_sim)]
        fpr_bert_sim, tpr_bert_sim, thresholds_roc_bert_sim = metrics.roc_curve(Y_dev_merged, Y_dev_bert_sim)
        fpr_xlm_sim, tpr_xlm_sim, thresholds_roc_xlm_sim = metrics.roc_curve(Y_dev_merged, Y_dev_xlm_sim)
        #fpr_bert_temp, tpr_bert_temp, thresholds_roc_bert_temp = metrics.roc_curve(Y_dev, Y_dev_bert_temp)
        #fpr_bert_rel_sim, tpr_bert_rel_sim, thresholds_roc_bert_rel_sim = metrics.roc_curve(Y_dev, Y_dev_bert_relative_sim)
        auc_fpr_tpr_bert_sim = metrics.auc(fpr_bert_sim, tpr_bert_sim)
        auc_fpr_tpr_xlm_sim = metrics.auc(fpr_xlm_sim, tpr_xlm_sim)
        #auc_fpr_tpr_bert_temp = metrics.auc(fpr_bert_temp, tpr_bert_temp)
        #auc_fpr_tpr_bert_rel_sim = metrics.auc(fpr_bert_rel_sim, tpr_bert_rel_sim)
        (prec_bert_sim, recall_bert_sim, thresholds_bert_sim) = precision_recall_curve(Y_dev_merged, Y_dev_bert_sim)
        (prec_xlm_sim, recall_xlm_sim, thresholds_xlm_sim) = precision_recall_curve(Y_dev_merged, Y_dev_xlm_sim)
        #(prec_bert_temp, recall_bert_temp, thresholds_bert_temp) = precision_recall_curve(Y_dev, Y_dev_bert_temp)
        #(prec_bert_rel_sim, recall_bert_rel_sim, thresholds_bert_rel_sim) = precision_recall_curve(Y_dev, Y_dev_bert_relative_sim)
        try:
            main_auc_bert_sim = evaluation.util_chinese.get_auc(prec_bert_sim[:-1], recall_bert_sim[:-1])
        except Exception as e:
            print(e, file=sys.stderr)
            main_auc_bert_sim = 0.0
        try:
            main_auc_xlm_sim = evaluation.util_chinese.get_auc(prec_xlm_sim[:-1], recall_xlm_sim[:-1])
        except Exception as e:
            print(e, file=sys.stderr)
            main_auc_xlm_sim = 0.0
        #try:
        #    main_auc_bert_temp = evaluation.util_chinese.get_auc(prec_bert_temp[:-1], recall_bert_temp[:-1])
        #except Exception as e:
        #    print(e, file=sys.stderr)
        #    main_auc_bert_temp = 0.0
        #try:
        #    main_auc_bert_rel_sim = evaluation.util_chinese.get_auc(prec_bert_rel_sim[:-1], recall_bert_sim[:-1])
        #except Exception as e:
        #    print(e, file=sys.stderr)
        #    main_auc_bert_rel_sim = 0.0
        stats_json_bert_sim = {
            'roc': {'fpr': fpr_bert_sim.tolist(), 'tpr': tpr_bert_sim.tolist(), 'thresholds':thresholds_roc_bert_sim.tolist()},
            'roc_auc': float(auc_fpr_tpr_bert_sim),
            'precrec': {'precision': prec_bert_sim.tolist(), 'recall': recall_bert_sim.tolist(), 'thresholds': thresholds_bert_sim.tolist()},
            'main_auc': float(main_auc_bert_sim)
        }
        stats_json_xlm_sim = {
            'roc': {'fpr': fpr_xlm_sim.tolist(), 'tpr': tpr_xlm_sim.tolist(),
                    'thresholds': thresholds_roc_xlm_sim.tolist()},
            'roc_auc': float(auc_fpr_tpr_xlm_sim),
            'precrec': {'precision': prec_xlm_sim.tolist(), 'recall': recall_xlm_sim.tolist(),
                        'thresholds': thresholds_xlm_sim.tolist()},
            'main_auc': float(main_auc_xlm_sim)
        }
        #stats_json_bert_temp = {
        #    'roc': {'fpr': fpr_bert_temp.tolist(), 'tpr': tpr_bert_temp.tolist(), 'thresholds': thresholds_roc_bert_temp.tolist()},
        #    'roc_auc': float(auc_fpr_tpr_bert_temp),
        #    'precrec': {'precision': prec_bert_temp.tolist(), 'recall': recall_bert_temp.tolist(), 'thresholds': thresholds_bert_temp.tolist()},
        #    'main_auc': float(main_auc_bert_temp)
        #}
        #stats_json_bert_rel_sim = {
        #    'roc': {'fpr': fpr_bert_rel_sim.tolist(), 'tpr': tpr_bert_rel_sim.tolist(), 'thresholds': thresholds_roc_bert_rel_sim.tolist()},
        #    'roc_auc': float(auc_fpr_tpr_bert_rel_sim),
        #    'precrec': {'precision': prec_bert_rel_sim.tolist(), 'recall': recall_bert_rel_sim.tolist(), 'thresholds': thresholds_bert_rel_sim.tolist()},
        #    'main_auc': float(main_auc_bert_rel_sim)
        #}
        if debug:
            print("stats_json_bert_sim: ")
            print(stats_json_bert_sim)
            print("stats_json_xlm_sim: ")
            print(stats_json_xlm_sim)
            #print("stats_json_bert_temp: ")
            #print(stats_json_bert_temp)
            #print("stats_json_bert_rel_sim: ")
            #print(stats_json_bert_rel_sim)

    #Now do the final prediction!
    assert args.no_lemma_baseline
    assert args.no_constraints
    count_nonzero_edges_found_in_entgraph = 0
    for (idx, _) in enumerate(data_dev):
        (p_ccg,q_ccg,_,_,_,a_ccg,_) = data_dev_CCG[idx]
        if Y_dev_exact[idx]:
            pred = True
        elif Y_dev_lemma[idx]:
            pred = True
        elif Y_dev_pred0:  # means: the entailment graph based prediction score array exist! -- Teddy
            pred = Y_dev_pred0[idx]
            if pred < 0.999:
                count_nonzero_edges_found_in_entgraph += 1
        else:
            pred = False
        Y_dev_pred.append(pred)

        predPair = p_ccg+"#"+q_ccg+"#"+str(a_ccg)
        if predPairSumCoefs:
            predPairSeen = (predPair in predPairSumCoefs and predPairSumCoefs[predPair] > 0)
            if debug:
                print("is seen: ", predPair, predPairSeen)
            if predPairSeen:
                Y_dev_pred_seen.append(pred)
                Y_dev_seen.append(Y_dev[idx])

    print(f"count_nonzero_edges_found_in_entgraph: {count_nonzero_edges_found_in_entgraph}", file=sys.stderr)
    for (i,y) in enumerate(Y_dev_pred):
        if debug:
            print(lines_dev[i])
            print(y, " ", Y_dev[i])

    Y_dev_pred_merged = merge_results(Y_dev_pred, args)
    Y_dev_pred = Y_dev_pred_merged
    assert len(Y_dev_pred) == len(Y_dev_merged)

    fpr, tpr, thresholds = metrics.roc_curve(Y_dev_merged, Y_dev_pred)
    if args.write:
        s1 = root + out_dir
        #uncomment s2 and op_tp_fp if you want to have _roc.txt file
        # s2 = root + out_dir
        if not os.path.isdir(s1):
            os.mkdir(s1)
        # if not os.path.isdir(s2):
        #     os.mkdir(s2)
        op_pr_rec = open(s1 + args.method + ".txt", 'w', encoding='utf8')
        op_Y_pred = open(s1 + args.method + "_Y.txt", 'w', encoding='utf8')
        op_pr_rec_lemma = open(s1 + "lemma.txt", 'w', encoding='utf8')
        op_Y_pred_lemma = open(s1 + "lemma_Y.txt", 'w', encoding='utf8')
        op_Y_pred_exact = open(s1 + "exact_Y.txt", 'w', encoding='utf8')
        # op_tp_fp = open(s2 + args.method + "_roc.txt", 'w', encoding='utf8')
        if raw_fnames[1] and calc_bert_baseline:
            bert_sim_stats_fp = open(s1 + 'bert_sim_stats.json', 'w', encoding='utf8')
            xlm_sim_stats_fp = open(s1 + 'xlm_sim_stats.json', 'w', encoding='utf8')
            #bert_temp_stats_fp = open(s1 + 'bert_temp_stats.json', 'w', encoding='utf8')
            #bert_rel_sim_stats_fp = open(s1 + 'bert_rel_sim_stats.json', 'w', encoding='utf8')
            json.dump(stats_json_bert_sim, bert_sim_stats_fp, indent=4, ensure_ascii=False)
            json.dump(stats_json_xlm_sim, xlm_sim_stats_fp, indent=4, ensure_ascii=False)
            #json.dump(stats_json_bert_temp, bert_temp_stats_fp, indent=4, ensure_ascii=False)
            #json.dump(stats_json_bert_rel_sim, bert_rel_sim_stats_fp, indent=4, ensure_ascii=False)
            bert_sim_stats_fp.close()
            xlm_sim_stats_fp.close()
            #bert_temp_stats_fp.close()
            #bert_rel_sim_stats_fp.close()
            bert_sim_scores_fp = open(s1+'bert_sim_scores.txt', 'w', encoding='utf8')
            xlm_sim_scores_fp = open(s1+'xlm_sim_scores.txt', 'w', encoding='utf8')
            #bert_temp_scores_fp = open(s1+'bert_temp_scores.txt', 'w', encoding='utf8')
            #bert_rel_sim_scores_fp = open(s1+'bert_rel_sim_scores.txt', 'w', encoding='utf8')
            write_scores_from_stats(bert_sim_scores_fp, stats_json_bert_sim, with_thres=False)
            write_scores_from_stats(xlm_sim_scores_fp, stats_json_xlm_sim, with_thres=False)
            #write_scores_from_stats(bert_temp_scores_fp, stats_json_bert_temp, with_thres=False)
            #write_scores_from_stats(bert_rel_sim_scores_fp, stats_json_bert_rel_sim, with_thres=False)
            bert_sim_scores_fp.close()
            xlm_sim_scores_fp.close()
            #bert_temp_scores_fp.close()
            #bert_rel_sim_scores_fp.close()
            bert_sim_preds_fp = open(s1 + 'bert_sim_preds.txt', 'w', encoding='utf8')
            xlm_sim_preds_fp = open(s1 + 'xlm_sim_preds.txt', 'w', encoding='utf8')
            #bert_temp_preds_fp = open(s1 + 'bert_temp_preds.txt', 'w', encoding='utf8')
            #bert_rel_sim_preds_fp = open(s1 + 'bert_rel_sim_preds.txt', 'w', encoding='utf8')
            #for gold, bert_sim_pred, bert_temp_pred, bert_rel_sim_pred in zip(Y_dev, Y_dev_bert_sim, Y_dev_bert_temp, Y_dev_bert_relative_sim):
            for gold, bert_sim_pred, xlm_sim_pred in zip(Y_dev_merged, Y_dev_bert_sim, Y_dev_xlm_sim):
                bert_sim_preds_fp.write(str(gold) + " " + str(bert_sim_pred) + "\n")
                xlm_sim_preds_fp.write(str(gold) + " " + str(xlm_sim_pred) + "\n")
                #bert_temp_preds_fp.write(str(gold) + " " + str(bert_temp_pred) + "\n")
                #bert_rel_sim_preds_fp.write(str(gold) + " " + str(bert_rel_sim_pred) + '\n')
            bert_sim_preds_fp.close()
            xlm_sim_preds_fp.close()
            #bert_temp_preds_fp.close()
            #bert_rel_sim_preds_fp.close()

    auc_fpr_tpr = metrics.auc(fpr, tpr)
    if debug:
        print("auc_fpr_tpr: ", auc_fpr_tpr)

    auc = auc_fpr_tpr
    if debug:
        print ("tpr, fpr: ")
    for i in range(len(tpr)):
        try:
            if debug:
                print(tpr[i], " ", fpr[i], thresholds[i])
        except Exception as e:
            print(e, file=sys.stderr)
            pass

    if debug:
        print("num seen: ", len(Y_dev_seen), " vs ", len(Y_dev))

    (precision, recall, thresholds) = precision_recall_curve(Y_dev_merged, Y_dev_pred)
    assert len(precision[1:]) == len(recall[1:]) == len(thresholds)
    new_precision, new_recall, new_thresholds = [precision[0]], [recall[0]], []
    baseline_t_occured = False
    baseline_2_t_occured = False
    for p, r, t in zip(precision[1:], recall[1:], thresholds):
        if t < 0.998 or t == 1.:
            new_precision.append(p)
            new_recall.append(r)
            new_thresholds.append(t)
        elif t < 0.9998:
            if not baseline_t_occured:
                new_precision.append(p)
                new_recall.append(r)
                new_thresholds.append(t)
                baseline_t_occured = True
        #elif t >= 0.9998:
        #    if not baseline_2_t_occured:
        #        new_precision.append(p)
        #        new_recall.append(r)
        #        new_thresholds.append(t)
        #        baseline_2_t_occured = True
    assert new_thresholds[-1] == 1. or args.no_lemma
    precision = new_precision
    recall = new_recall
    thresholds = new_thresholds
    try:
        main_auc = evaluation.util_chinese.get_auc(precision[:-1], recall[:-1])
    except Exception as e:
        print(e, file=sys.stderr)
        main_auc = 0

    if args.write:
        op_pr_rec.write("auc: "+str(main_auc)+"\n")

    if debug:
        print("main_auc:", main_auc)
    for i in range(len(Y_dev_merged)):
        y_pred = Y_dev_pred[i]
        if y_pred==True:
            y_pred = 1
        elif y_pred==False:
            y_pred = 0
        op_Y_pred.write(str(Y_dev_merged[i])+" "+str(y_pred)+"\n")
        op_Y_pred_lemma.write(str(Y_dev_merged[i])+" "+str(Y_dev_lemma[i])+'\n')
        op_Y_pred_exact.write(str(Y_dev_merged[i])+" "+str(Y_dev_exact[i])+'\n')
    # util.get_confidence_interval(Y_dev, Y_dev_pred)
    if debug:
        print("avg pr score")
    a = metrics.average_precision_score(Y_dev_merged,Y_dev_pred)
    if debug:
        print(a)
    b = metrics.average_precision_score(Y_dev_merged, Y_dev_pred, average='micro')
    if debug:
        print(b)
    if debug:
        print("auc: ", auc)

    threshold = .16  # For Happy classification :) #But it will be set to threshold for precision ~ .76
    threshold_set = False
    if debug:
        print ("pr_rec:")
    for i in range(len(precision)):
        if args.write:
            if i > 0:
                op_pr_rec.write(str(precision[i])+ " "+ str(recall[i])+" "+str(thresholds[i-1])+"\n")
        try:
            # if not threshold_set and precision[i] > .85 and precision[i] < .86:
            if not threshold_set and .748 < precision[i] < .765:
                threshold = thresholds[i]
                threshold_set = True
            if debug:
                print(precision[i], " ", recall[i], thresholds[i])
        except Exception as e:
            if debug:
                print ("exception: ", e, '; ', precision[i],recall[i])
            pass

    if debug:
        print ("threshold set to: ", threshold)
    Y_dev_pred_binary = [y>threshold for y in Y_dev_pred]

    all_FPs = []
    all_FNs = []

    sample_f = open('samples.txt', 'w', encoding='utf8')

    if debug:
        if read_sims or args.instance_level:
            if predPairFeatsTyped:
                X_dev_typed = get_typed_feats(data_dev,predPairFeatsTyped)

            print("results:")
            for (idx,(p,q,t1s,t2s,probs,a,l)) in enumerate(data_dev):
                line_info = lines_dev[idx]+"\t"+p+"#"+q+"#"+str(a)+"\t"+str(t1s)+"#"+str(t2s)+"\t"
                if predPairFeats:
                    line_info += str(predPairFeats[p + "#" + q + "#" + str(a)]) + "\t" + str(X_dev_typed[idx]) + "\t" +\
                                 str(Y_dev_pred[idx])
                if Y_dev_pred_binary[idx] and Y_dev_merged[idx]:
                    conf_l = "TP"
                elif Y_dev_pred_binary[idx] and not Y_dev_merged[idx]:
                    all_FPs.append(line_info)
                    conf_l = "FP"
                elif not Y_dev_pred_binary[idx] and Y_dev_merged[idx]:
                    all_FNs.append(line_info)
                    conf_l = "FN"
                else:
                    conf_l = "TN"

                print(conf_l + " : " + lines_dev[idx])
                print("pred: ", Y_dev_pred[idx])
                if predPairFeats:
                    predPair = p+"#"+q+"#"+str(a)
                    print(predPairFeats[predPair])
                    # print predPairFeatsTyped[p+"#"+q+"#"+str(a)+"#"+t1+"#"+t2]
                    print(X_dev_typed[idx])
                    if conf_l=="FN" and predPairSumCoefs and (predPair not in predPairSumCoefs or predPairSumCoefs[predPair]==0):
                        print("unseen and FN")
                print(p+"#"+q+"#"+str(a))
                print(str(t1s)+"#"+str(t2s)+"\n")
                if args.LDA:
                    print(probs)

            if calc_bert_baseline:
                print("ours vs Bert's")
                for (idx,_) in enumerate(data_dev):
                    if Y_dev_pred_binary[idx] != Y_dev_bert_sim[idx]:
                        if Y_dev_pred_binary[idx] == Y_dev_merged[idx]:
                            print ("ours is correct: ", lines_dev[idx])
                        else:
                            print ("Bert's is correct: ", lines_dev[idx])

    print("ours final: ")
    eval(Y_dev_pred_binary,Y_dev_merged)

    op_pr_rec.close()
    op_Y_pred.close()
    op_pr_rec_lemma.close()
    op_Y_pred_lemma.close()
    op_Y_pred_exact.close()

    return Y_dev_pred


#These (until parameters) are fixed and won't change (too much)!
if args.outDir:
    out_dir = args.outDir+"/"
else:
    out_dir = 'results/pr_rec/'

if args.sim_suffix:
    f_post_fix = args.sim_suffix

assert (not args.dev or not args.test)
assert not args.snli or args.CCG
assert not (args.rankDiscount and args.rankFeats)

fnames_unary = None

if args.dev:
    if args.eval_range == 'full':
        fnames_CCG = [None, root + "chinese_ent/implications_dev_rels.txt"]
        fnames_oie = [None, None]
        orig_fnames = [None, root + "chinese_ent/implications_dev_translated.tsv"]
        raw_fnames = [None, root + "chinese_ent/implications_dev_translated_raw.tsv"]
    elif args.eval_range == 'nosame':
        fnames_CCG = [None, root + "chinese_ent/implications_dev_rels_nosame.txt"]
        fnames_oie = [None, None]
        orig_fnames = [None, root + "chinese_ent/implications_dev_translated_nosame.tsv"]
        raw_fnames = [None, root + "chinese_ent/implications_dev_translated_raw_nosame.tsv"]
    elif args.eval_range == 'google_PLM':
        fnames_CCG = [None, root + "chinese_ent/implications_dev_rels_google_PLM.txt"]
        fnames_oie = [None, None]
        orig_fnames = [None, root + "chinese_ent/implications_dev_translated_google_PLM.tsv"]
        raw_fnames = [None, root + "chinese_ent/implications_dev_translated_raw_google_PLM.tsv"]
    elif args.eval_range == 'bsl':
        fnames_CCG = [None, root + "chinese_ent/implications_dev_rels_bsl.txt"]
        fnames_oie = [None, None]
        orig_fnames = [None, root + "chinese_ent/implications_bsl_dev_translated.tsv"]
        raw_fnames = [None, root + "chinese_ent/implications_bsl_dev_translated_raw.tsv"]
    elif args.eval_range == 'orig':
        fnames_CCG = [None, root + "dev_ent_chinese/dev_all_rels.txt"]
        fnames_oie = [None, None]
        orig_fnames = [None, root + "dev_ent_chinese/dev_translated.tsv"]
        raw_fnames = [None, root + "dev_ent_chinese/dev_translated_raw.tsv"]
    elif args.eval_range == 'orig_alltype':
        fnames_CCG = [None, root + "dev_ent_chinese/dev_all_rels_alltype.txt"]
        fnames_oie = [None, None]
        orig_fnames = [None, root + "dev_ent_chinese/dev_translated_typed.tsv"]
        raw_fnames = [None, root + "dev_ent_chinese/dev_translated_raw_typed.tsv"]
        args.typed_mapping_fn = root + 'dev_ent_chinese/dev_typed_rel_levy_mapping.txt'
        args.untyped_mapping_fn = root + 'dev_ent_chinese/dev_rellevy_mapping.txt'
    elif args.eval_range == 'orig_exhaust':
        fnames_CCG = [None, root + "dev_ent_chinese/dev_exhaust_all_rels.txt"]
        fnames_oie = [None, None]
        orig_fnames = [None, root + "dev_ent_chinese/dev_exhaust_translated.tsv"]
        raw_fnames = [None, root + "dev_ent_chinese/dev_exhaust_translated_raw.tsv"]
        args.use_untyped_mapping = True
        args.untyped_mapping_fn = root + 'dev_ent_chinese/dev_exhaust_rellevy_mapping.txt'  # for non-alltype version of exhaust, typed mapping is untyped mapping.
    elif args.eval_range == 'orig_exhaust_jia':
        fnames_CCG = [None, root + "dev_ent_chinese/dev_exhaust_jia_all_rels.txt"]
        fnames_oie = [None, None]
        orig_fnames = [None, root + "dev_ent_chinese/dev_exhaust_jia_translated.tsv"]
        raw_fnames = [None, root + "dev_ent_chinese/dev_exhaust_jia_translated_raw.tsv"]
        args.use_untyped_mapping = True
        args.untyped_mapping_fn = root + 'dev_ent_chinese/dev_exhaust_jia_rellevy_mapping.txt'  # for non-alltype version of exhaust, typed mapping is untyped mapping.
    elif args.eval_range == 'orig_exhaust_alltype':
        fnames_CCG = [None, root + "dev_ent_chinese/dev_exhaust_all_rels_alltype.txt"]
        fnames_oie = [None, None]
        orig_fnames = [None, root + "dev_ent_chinese/dev_exhaust_translated_typed.tsv"]
        raw_fnames = [None, root + "dev_ent_chinese/dev_exhaust_translated_raw_typed.tsv"]
        args.typed_mapping_fn = root + 'dev_ent_chinese/dev_exhaust_typed_rel_levy_mapping.txt'
        args.untyped_mapping_fn = root + 'dev_ent_chinese/dev_exhaust_rellevy_mapping.txt'
    elif args.eval_range == 'orig_fineonly':
        fnames_CCG = [None, root + "dev_ent_chinese/dev_fineonly_all_rels_alltype.txt"]
        fnames_oie = [None, None]
        orig_fnames = [None, root + "dev_ent_chinese/dev_fineonly_translated_typed.tsv"]
        raw_fnames = [None, root + "dev_ent_chinese/dev_fineonly_translated_raw_typed.tsv"]
    elif args.eval_range == 'orig_fineonly_alltype':
        fnames_CCG = [None, root + "dev_ent_chinese/dev_fineonly_all_rels_alltype.txt"]
        fnames_oie = [None, None]
        orig_fnames = [None, root + "dev_ent_chinese/dev_fineonly_translated_typed.tsv"]
        raw_fnames = [None, root + "dev_ent_chinese/dev_fineonly_translated_raw_typed.tsv"]
        args.typed_mapping_fn = root + 'dev_ent_chinese/dev_fineonly_typed_rel_levy_mapping.txt'
        args.untyped_mapping_fn = root + 'dev_ent_chinese/dev_fineonly_rellevy_mapping.txt'
    elif args.eval_range == 'orig_exhaust_fineonly':
        fnames_CCG = [None, root + "dev_ent_chinese/dev_exhaust_fineonly_all_rels.txt"]
        fnames_oie = [None, None]
        orig_fnames = [None, root + "dev_ent_chinese/dev_exhaust_fineonly_translated.tsv"]
        raw_fnames = [None, root + "dev_ent_chinese/dev_exhaust_fineonly_translated_raw.tsv"]
    elif args.eval_range == 'orig_exhaust_fineonly_alltype':
        fnames_CCG = [None, root + "dev_ent_chinese/dev_exhaust_fineonly_all_rels_alltype.txt"]
        fnames_oie = [None, None]
        orig_fnames = [None, root + "dev_ent_chinese/dev_exhaust_fineonly_translated_typed.tsv"]
        raw_fnames = [None, root + "dev_ent_chinese/dev_exhaust_fineonly_translated_raw_typed.tsv"]
        args.typed_mapping_fn = root + 'dev_ent_chinese/dev_exhaust_fineonly_typed_rel_levy_mapping.txt'
        args.untyped_mapping_fn = root + 'dev_ent_chinese/dev_exhaust_fineonly_rellevy_mapping.txt'
    else:
        raise AssertionError
elif args.test:
    if args.eval_range == 'full':
        fnames_CCG = [None, root + "chinese_ent/implications_test_rels.txt"]
        fnames_oie = [None, None]
        orig_fnames = [None, root + "chinese_ent/implications_test_translated.tsv"]
        raw_fnames = [None, root + "chinese_ent/implications_test_translated_raw.tsv"]
    elif args.eval_range == 'nosame':
        fnames_CCG = [None, root + "chinese_ent/implications_test_rels_nosame.txt"]
        fnames_oie = [None, None]
        orig_fnames = [None, root + "chinese_ent/implications_test_translated_nosame.tsv"]
        raw_fnames = [None, root + "chinese_ent/implications_test_translated_raw_nosame.tsv"]
    elif args.eval_range == 'google_PLM':
        fnames_CCG = [None, root + "chinese_ent/implications_test_rels_google_PLM.txt"]
        fnames_oie = [None, None]
        orig_fnames = [None, root + "chinese_ent/implications_test_translated_google_PLM.tsv"]
        raw_fnames = [None, root + "chinese_ent/implications_test_translated_raw_google_PLM.tsv"]
    elif args.eval_range == 'bsl':
        fnames_CCG = [None, root + "chinese_ent/implications_test_rels_bsl.txt"]
        fnames_oie = [None, None]
        orig_fnames = [None, root + "chinese_ent/implications_bsl_test_translated.tsv"]
        raw_fnames = [None, root + "chinese_ent/implications_bsl_test_translated_raw.tsv"]
    elif args.eval_range == 'orig':
        fnames_CCG = [None, root + "test_ent_chinese/test_all_rels.txt"]
        fnames_oie = [None, None]
        orig_fnames = [None, root + "test_ent_chinese/test_translated.tsv"]
        raw_fnames = [None, root + "test_ent_chinese/test_translated_raw.tsv"]
    elif args.eval_range == 'orig_alltype':
        fnames_CCG = [None, root + "test_ent_chinese/test_all_rels_alltype.txt"]
        fnames_oie = [None, None]
        orig_fnames = [None, root + "test_ent_chinese/test_translated_typed.tsv"]
        raw_fnames = [None, root + "test_ent_chinese/test_translated_raw_typed.tsv"]
        args.typed_mapping_fn = root + 'test_ent_chinese/test_typed_rel_levy_mapping.txt'
        args.untyped_mapping_fn = root + 'test_ent_chinese/test_rellevy_mapping.txt'
    elif args.eval_range == 'orig_exhaust':
        fnames_CCG = [None, root + "test_ent_chinese/test_exhaust_all_rels.txt"]
        fnames_oie = [None, None]
        orig_fnames = [None, root + "test_ent_chinese/test_exhaust_translated.tsv"]
        raw_fnames = [None, root + "test_ent_chinese/test_exhaust_translated_raw.tsv"]
        args.use_untyped_mapping = True
        args.untyped_mapping_fn = root + 'test_ent_chinese/test_exhaust_rellevy_mapping.txt'  # for non-alltype version of exhaust, typed mapping is untyped mapping.
    elif args.eval_range == 'orig_exhaust_jia':
        fnames_CCG = [None, root + "test_ent_chinese/test_exhaust_jia_all_rels.txt"]
        fnames_oie = [None, None]
        orig_fnames = [None, root + "test_ent_chinese/test_exhaust_jia_translated.tsv"]
        raw_fnames = [None, root + "test_ent_chinese/test_exhaust_jia_translated_raw.tsv"]
        args.use_untyped_mapping = True
        args.untyped_mapping_fn = root + 'test_ent_chinese/test_exhaust_jia_rellevy_mapping.txt'  # for non-alltype version of exhaust, typed mapping is untyped mapping.
    elif args.eval_range == 'orig_exhaust_alltype':
        fnames_CCG = [None, root + "test_ent_chinese/test_exhaust_all_rels_alltype.txt"]
        fnames_oie = [None, None]
        orig_fnames = [None, root + "test_ent_chinese/test_exhaust_translated_typed.tsv"]
        raw_fnames = [None, root + "test_ent_chinese/test_exhaust_translated_raw_typed.tsv"]
        args.typed_mapping_fn = root + 'test_ent_chinese/test_exhaust_typed_rel_levy_mapping.txt'
        args.untyped_mapping_fn = root + 'test_ent_chinese/test_exhaust_rellevy_mapping.txt'
    elif args.eval_range == 'orig_fineonly':
        fnames_CCG = [None, root + "test_ent_chinese/test_fineonly_all_rels.txt"]
        fnames_oie = [None, None]
        orig_fnames = [None, root + "test_ent_chinese/test_fineonly_translated.tsv"]
        raw_fnames = [None, root + "test_ent_chinese/test_fineonly_translated_raw.tsv"]
    elif args.eval_range == 'orig_fineonly_alltype':
        fnames_CCG = [None, root + "test_ent_chinese/test_fineonly_all_rels_alltype.txt"]
        fnames_oie = [None, None]
        orig_fnames = [None, root + "test_ent_chinese/test_fineonly_translated_typed.tsv"]
        raw_fnames = [None, root + "test_ent_chinese/test_fineonly_translated_raw_typed.tsv"]
        args.typed_mapping_fn = root + 'test_ent_chinese/test_fineonly_typed_rel_levy_mapping.txt'
        args.untyped_mapping_fn = root + 'test_ent_chinese/test_fineonly_rellevy_mapping.txt'
    elif args.eval_range == 'orig_exhaust_fineonly':
        fnames_CCG = [None, root + "test_ent_chinese/test_exhaust_fineonly_all_rels.txt"]
        fnames_oie = [None, None]
        orig_fnames = [None, root + "test_ent_chinese/test_exhaust_fineonly_translated.tsv"]
        raw_fnames = [None, root + "test_ent_chinese/test_exhaust_fineonly_translated_raw.tsv"]
    elif args.eval_range == 'orig_exhaust_fineonly_alltype':
        fnames_CCG = [None, root + "test_ent_chinese/test_exhaust_fineonly_all_rels_alltype.txt"]
        fnames_oie = [None, None]
        orig_fnames = [None, root + "test_ent_chinese/test_exhaust_fineonly_translated_typed.tsv"]
        raw_fnames = [None, root + "test_ent_chinese/test_exhaust_fineonly_translated_raw_typed.tsv"]
        args.typed_mapping_fn = root + 'test_ent_chinese/test_exhaust_fineonly_typed_rel_levy_mapping.txt'
        args.untyped_mapping_fn = root + 'test_ent_chinese/test_exhaust_fineonly_rellevy_mapping.txt'
    else:
        raise AssertionError
else:
    raise AssertionError

# parameters

CCG = args.CCG
typed = args.typed
supervised = args.supervised
oneFeat = args.oneFeat  # as opposed to average features!
gpath = args.gpath
method = args.method
useSims = args.useSims
if gpath:
    print ("gpath: ", gpath)
if method is None:
    method = "x"
write = args.write
if debug:
    print ("args.tnf: ", args.tnf)
if args.instance_level:
    read_sims = False
else:
    read_sims = True

if debug:
    print ("args: ", CCG)

if gpath:
    engG_dir_addr = "../gfiles/" + gpath +"/"
    if debug:
        print ("dir_addr: ", engG_dir_addr)
else:
    engG_dir_addr = None

if CCG:
    fnames = fnames_CCG
    fname_feats = None
    sim_path = root + "ent/ccg.sim"
    if not gpath:
        if not args.featsFile:
            raise Exception("featsFile not provided")
        else:
            fname_feats = root + "ent/" + args.featsFile + ".txt"
else:
    fnames = fnames_oie
    fname_feats = None
    sim_path = root + "ent/oie.sim"
    if not gpath:
        if not args.featsFile:
            raise Exception("featsFile not provided")
        else:
            fname_feats = root + "ent/" + args.featsFile + ".txt"

rels2Sims = evaluation.util_chinese.read_rels_sim(sim_path, CCG, useSims)
#end parameters
#Form the samples (dev will contain the test if you use --test instead of --dev
# [_, Y_dev_lemma] = [chinese_baselines.predict_lemma_baseline(fname, args) for fname in orig_fnames]  # orig_fnames means
[_, Y_dev_lemma] = [chinese_baselines.predict_coarse_lemma_baseline(fname, args) for fname in orig_fnames]
if args.no_lemma:
    Y_dev_lemma = [False for x in Y_dev_lemma]
[_, Y_dev_exact] = [chinese_baselines.predict_exact_baseline(fname, args) for fname in raw_fnames]

#Do the training and prediction!
lines_dev = open(orig_fnames[1], encoding='utf8').read().splitlines()

if not args.instance_level:
    data_list, predPairSumCoefs, predPairFeats, predPairFeatsTyped, predPairConnectedList, \
    predPairConnectedWeightList, predPairTypedConnectedList, predPairTypedConnectedWeightList, predPairTypedExactFound = form_samples(fnames,
                                                                                                           fnames_unary,  # should be none
                                                                                                           orig_fnames,
                                                                                                           engG_dir_addr,
                                                                                                           fname_feats,  # should be none for full running
                                                                                                           rels2Sims,  # should be none
                                                                                                           args)
    Y_dev_pred0, Y_dev_TNF0, Y_dev_TNF_typed0, cl, scaler = fit_predict(data_list, predPairFeats,
                                                                        predPairFeatsTyped, predPairConnectedList,
                                                                        predPairTypedConnectedList,
                                                                        predPairTypedExactFound,
                                                                        args)
else:
    data_list = [evaluation.util_chinese.read_data(fnames[i], orig_fnames[i], args.CCG, args.typed, args.LDA) for i in
                 range(len(fnames))]
    Y_dev_pred0 = evaluation.util_chinese.read_instance_level_probs(fname_feats)
    assert len(Y_dev_pred0) == len(Y_dev_lemma)
    assert len(Y_dev_pred0) == len(Y_dev_exact)
    Y_dev_TNF0, Y_dev_TNF_typed0, cl, scaler, predPairSumCoefs, predPairFeats, predPairFeatsTyped = None, None, None, None, None, None, None

#do the final evaluation (either raw scores or graphs)
assert not args.snli
# In current setting, data_dev should be the same as data_dev_CCG! -- Teddy
data_dev = data_list[1]
data_dev_CCG = evaluation.util_chinese.read_data(fnames_CCG[1], orig_fnames[1], args.CCG, args.typed, args.LDA)
Y_dev = [l for (_,_,_,_,_,_,l) in data_dev]  # gold labels


Y_dev_lemma_merged = merge_results(Y_dev_lemma, args)
Y_dev_exact_merged = merge_results(Y_dev_exact, args)
Y_dev = merge_results(Y_dev, args)
print("baseline lemma eval:")
eval(Y_dev_lemma_merged,Y_dev)
print("baseline exact eval:")
eval(Y_dev_exact_merged,Y_dev)

if debug:
    print(Y_dev_pred0)

# here the data_dev and data_dev_CCG should be the same! -- Teddy
_ = final_prediction(data_dev,data_dev_CCG, predPairFeats ,predPairFeatsTyped, predPairSumCoefs, Y_dev_lemma, Y_dev_exact, Y_dev_pred0,Y_dev_TNF0,Y_dev_TNF_typed0, lines_dev, root, args)

assert Y_dev_TNF0 is None
