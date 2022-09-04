# coding=utf-8
from lemma_baseline import qa_utils_chinese
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
from graph import graph

debug = False


def getUnaryFrom_binary(pred):
    modifier = ""

    if "__" in pred:
        ss = pred.split("__")
        modifier = ss[0]
        pred = ss[1]

    ss = pred.split("#")
    pred = ss[0][1:-1]
    type1 = ss[1].replace("_1","").replace("_2","")
    type2 = ss[2].replace("_1","").replace("_2","")
    ss = pred.split(",")
    if len(ss)!=2:
        if debug:
            print ("bad pred: ", pred)
        return None

    unary1 = ss[0]+"#"+type1
    unary2 = ss[1]+"#"+type2

    if modifier!="":
        unary1 = modifier + "__" + unary1
        unary2 = modifier + "__" + unary2

    return unary1,unary2


def read_data(dpath, orig_dpath, CCG,typed,LDA):
    if dpath is None:  # added by Teddy
        return []
    f = open(dpath, encoding='utf8')

    if orig_dpath:
        lines_orig = open(orig_dpath, encoding='utf8').read().splitlines()
    else:
        lines_orig = None

    data = []
    for lidx, l in enumerate(f):
        line = l.replace("\n","")
        if lines_orig:
            line_orig = lines_orig[lidx]
        else:
            line_orig = None

        # line: (用于.1,用于.2) 药物::medicine 感染::disease	(治愈.1,治愈.2) 药物::medicine 感染::disease	False
        ss = line.split("\t")
        if len(ss)<3:
            if debug:
                print ("bad len problem: ", line)
            continue

        q_all = ss[0].split(" ")
        p_all = ss[1].split(" ")
        q = q_all[0]
        p = p_all[0]

        if len(p_all)>1 and typed and not LDA:
            try:
                t1 = p_all[1].split("::")[1]
                t2 = p_all[2].split("::")[1]
            except:
                t1 = "thing"
                t2 = "thing"
            t1s = [t1]
            t2s = [t2]
            probs = [1]
        elif typed and LDA:
            raise AssertionError
        else:
            #Not well formed
            t1 = "thing"
            t2 = "thing"
            t1s = [t1]
            t2s = [t2]
            probs = [1]

        #First, let's see if the args are aligned
        if CCG:
            a = True
            if line_orig:  # lazy way to check snli
                if len(q_all) > 1 and len(p_all) > 1:
                    a = qa_utils_chinese.aligned_args_rel(q_all, p_all)
        else:
            if line_orig:
                ss_orig = line_orig.split("\t")
                q_orig = ss_orig[0].split(",")
                p_orig = ss_orig[1].split(",")
                a = qa_utils_chinese.aligned_args([q_orig[0].strip(),"_",q_orig[2].strip()],[p_orig[0].strip(),"",p_orig[2].strip()])
                if a==-1:
                    a = qa_utils_chinese.aligned_args([p_orig[0].strip(),"",p_orig[2].strip()],[q_orig[0].strip(),"_",q_orig[2].strip()])
                    if a==-1:
                        raise Exception('HORRIBLE BUG!!!'+str(q)+" "+str(a))
            else:
                a = True  # means "aligned"
            raise AssertionError

        try:
            # again, no need to lemmatize
            q_arg1 = q_all[1].split("::")[0]
            q_arg2 = q_all[2].split("::")[0]

            p_arg1 = p_all[1].split("::")[0]
            p_arg2 = p_all[2].split("::")[0]

            if line_orig:
                if a:
                    if q_arg1 != p_arg1 or q_arg2 != p_arg2:
                        if debug:
                            print("not same args: ", line_orig)
                            print(line)
                else:
                    if q_arg1 != p_arg2 or q_arg2 != p_arg1:
                        if debug:
                            print("not same args: ", line_orig)
                            print(line)
        except Exception as e:
            if debug:
                print("problem: ", line)
                print(e)
            raise

        if ss[2].startswith("n") or ss[2]=="False":
            l = 0
        else:
            l = 1

        data.append((p,q,t1s,t2s,probs,a,l))  # t1s: [type_1]; t2s: [type_2]; probs: [1], a: aligned, l: True/False

    return data


def read_data_unary(dpath, is_typed):

    f = open(dpath, encoding='utf8')
    data = []

    idx = 0

    for l in f:
        line = l.strip()
        # if idx==100000:
        #     break
        # print line

        ss = line.split("\t")

        if len(ss)<2:
            if debug:
                print ("bad len problem unary: ", line)
            continue

        q_all = ss[0].split(" ")
        p_all = ss[1].split(" ")
        q = q_all[0]
        p = p_all[0]

        if (p.endswith(".1") and q.endswith(".1")) or (p.endswith(".2") and q.endswith(".2")):
            if is_typed:
                t1 = p_all[1].split("::")[1]
            else:
                t1 = "thing"
            #we don't care about the label, because it's not aligned with the original files
            # if ss[2].startswith("n") or ss[2]=="False":
            #     l = 0
            # else:
            #     l = 1
            #
            # print "label: ", l
            data.append((p,q,t1))
        idx += 1

    return data

def get_subsample(Y,Y_pred1,Y_pred2):
    n = len(Y)
    Y_p = []
    Y_pred1_p = []
    Y_pred2_p = []

    for i in range(n):
        r = np.random.randint(n)
        Y_p.append(Y[r])
        Y_pred1_p.append(Y_pred1[r])
        Y_pred2_p.append(Y_pred2[r])
    return Y_p, Y_pred1_p, Y_pred2_p

def get_subsample0(Y,Y_pred):
    n = len(Y)
    Y_p = []
    Y_pred_p = []

    for i in range(n):
        r = np.random.randint(n)
        Y_p.append(Y[r])
        Y_pred_p.append(Y_pred[r])
    return Y_p, Y_pred_p

def get_auc(precisions, recalls):
    xs = []
    ys = []
    assert len(precisions) == len(recalls)
    for i, (p, r) in enumerate(zip(precisions, recalls)):
        if p >= .5:  # AUC above precision of 0.5
            xs.append(r)
            ys.append(p)
    try:
        auc = metrics.auc(xs, ys)
    except ValueError as e:
        print(e)
        print(f"xs: {xs}")
        print(f"ys: {ys}")
        auc = 0.0
    # auc = metrics.auc(xs, ys, reorder=True)  # this reorder = True is probably deprecated?
    return auc

def get_confidence_interval(Y,Y_pred):
    aucs = []
    N = 1000;
    for i in range(N):
        Y_p,Y_pred_p = get_subsample0(Y,Y_pred)
        (precision, recall, _) = precision_recall_curve(Y_p, Y_pred_p)
        auc = get_auc(precision[:-1],recall[:-1])
        aucs.append(auc)
        print (auc)
    aucs = sorted(aucs)
    l = np.int(.025*N)
    r = np.int(.975*N)
    print (aucs)
    print ("interval: ", aucs[l], aucs[r])
    print (l, r)

def read_Ys(fname):
    f = open(fname, encoding='utf8')
    Y = []
    Y_pred = []
    for line in f:
        ss = line.split(" ")
        y = ss[0]
        y_pred = ss[1]
        print (y, y_pred)
        Y.append(int(y))
        Y_pred.append(float(y_pred))
    return Y,Y_pred


def statistical_significance(Y,Y_pred1,Y_pred2):
    N = 1000
    Ngood = 0
    precision_recall_curve(Y, Y_pred1)
    for i in range(N):
        Y_p, Y_pred1_p, Y_pred2_p = get_subsample(Y, Y_pred1,Y_pred2)
        (precisions1, recalls1, _) = precision_recall_curve(Y_p, Y_pred1_p)
        (precisions2, recalls2, _) = precision_recall_curve(Y_p, Y_pred2_p)
        auc1 = get_auc(precisions1[:-1], recalls1[:-1])
        auc2 = get_auc(precisions2[:-1], recalls2[:-1])
        print (auc1, auc2)
        if auc1>auc2:
            Ngood += 1
        print (Ngood, (i+1))

    print ("Ngood: ", Ngood)


def read_unaryPairFeatsTyped(fname):
    lines = open(fname, encoding='utf8').read().splitlines()

    unaryPairFeatsTyped = {}
    for line in lines:
        ss = line.split("\t")
        f_ss = (ss[2][1:len(ss[2])-1]).strip().split()
        unaryPairFeatsTyped[ss[0]] = np.array([np.float(f) for f in f_ss])

    return unaryPairFeatsTyped

def read_predPairFeats(fname, data_list=None):
    global num_feats
    lines = open(fname, encoding='utf8').read().splitlines()
    predPairFeats = {}
    predPairFeatsTyped = {}
    predPairSumCoefs = {}
    predPairTypedExactFound = set()

    first = True
    lIdx = 0
    for line in lines:
        try:
            lIdx += 1
            if line=="predPairTypedExactFound:":
                break
            ss = line.split("\t")
            is_typed = len(ss[0].split("#"))==5
            if len(ss[0].split("#"))>5:
                continue
            if len(ss) < 2:
                print ("small len: ", line)
                continue
            # print line
            if not is_typed:#Untyped
                # print line
                predPairSumCoefs[ss[0]] = np.float(ss[1])

                f_idx = len(ss)-1
                f_ss = ss[f_idx][1:-1].strip().split()

                # First time seeing features.
                if first:
                    num_feats = len(f_ss)
                    first = False
                    if debug:
                        print ("Feature file datums contain", num_feats, "features")
                if not first and len(f_ss) != num_feats:
                    if debug:
                        print ("ERROR IN PARSING" + line)
                    continue
                # predPairFeats[ss[0]] = [np.float(f)*predPairSumCoefs[ss[0]] for f in f_ss]
                # predPairFeats[ss[0]].extend([np.float(f) for f in f_ss])
                # print f_ss
                predPairFeats[ss[0]] = np.array([np.float(f) for f in f_ss])
                # print ss[0]
                # print predPairFeats[ss[0]][24]

                # print predPairFeats[ss[0]]
            else:#Typed
                # print line
                f_ss = (ss[1][1:len(ss[1])-1]).strip().split()
                #print line
                #print f_ss

                # First time seeing features.
                if first:
                    num_feats = len(f_ss)
                    first = False
                if not first and len(f_ss) != num_feats:
                    if debug:
                        print ("ERROR IN PARSING" + line)
                    continue
                predPairFeatsTyped[ss[0]] = np.array([np.float(f) for f in f_ss])
        except:
            continue


    if data_list is not None:

        #If something new is encountered, then set it to 0
        for data in data_list:
            for (p,q,t1s,t2s,probs,a,_) in data:
                ppair = p+"#"+q+"#"+str(a)
                if ppair not in predPairFeats:
                    predPairFeats[ppair] = np.zeros(num_feats) #list(deepcopy(graph.Graph.zeroFeats))
                    # predPairFeats[ppair].extend(list(deepcopy(graph.Graph.zeroFeats)))
                    predPairSumCoefs[ppair] = 0

                for t_i in range(len(t1s)):
                    t1 = t1s[t_i]
                    t2 = t2s[t_i]

                    ppairTyped = p+"#"+q+"#"+str(a)+"#"+t1+"#"+t2


                    if ppairTyped not in predPairFeatsTyped:
                        if debug:
                            print ("setting new: ", ppairTyped)
                        predPairFeatsTyped[ppairTyped] = np.zeros(num_feats) #list(deepcopy(graph.Graph.zeroFeats))

    for i in range(lIdx,len(line)):
        line = lines[i].strip()
        predPairTypedExactFound.add(line)

    graph.Graph.num_feats = num_feats/2

    return predPairFeats,predPairFeatsTyped,predPairSumCoefs,predPairTypedExactFound


def read_rels_sim(fpath, isCCG, useSims):
    if not useSims:
        return {}

    lines = open(fpath, 'r', encoding='utf8').read().splitlines()
    rel2Sims = {}
    for line in lines:
        ss = line.split("\t")
        ss = [x.strip() for x in ss]
        p = ss[0]
        modifier = ""
        if isCCG:
            ridx = p.rfind("__")
            if ridx!=-1:
                modifier = p[:ridx]
                p = p[ridx+2:]
        qs = []
        qss = []
        idx = 1
        while (idx<len(ss)):
            # if idx>40:
            #     break
            q = ss[idx]
            idx += 1
            sim = float(ss[idx])
            idx+=1
            # if sim<.90:
            #     continue

            if (isCCG):
                ridx = q.rfind("__")
                if ridx!=-1:
                    q = q[ridx+2:]
                if modifier!="":
                    q = modifier+"__"+q
                try:
                    if not qa_utils_chinese.same_CCG_args(p,q):# or not qa_utils_chinese.same_main_words(p,q,prepositions)
                        continue
                except:
                    if debug:
                        print ("exception for: ", q)
                    continue

            if q not in qs:
                qs.append(q)
                qss.append((q,sim))



        if isCCG and modifier != "":
            p = modifier+"__"+p
        if len(qs)==0 or qs[0]!=p:
            qs.insert(0,p)
            qss.append((p, 1))
        if debug:
            print ("p: ", p)
            print ("sims: ", qss)
        rel2Sims[p] = qs

    return rel2Sims

def remove_rank_feats(X_train):

    X_train1 = [list(x[0:len(X_train[0])/4]) for x in X_train]
    [X_train1[i].extend(x[len(X_train[0])/2:3*len(X_train[0])/4]) for i,x in enumerate(X_train)]
    return X_train

#X and Y lists. We have more negatives. We also remove 0s for positive samples!

def down_sample_negs(X,Y):

    X2 = []
    Y2 = []

    for i in range(len(Y)):
        if Y[i]==0 or np.count_nonzero(X[i])!=0:
            X2.append(X[i])
            Y2.append(Y[i])

    pair_recall = float(np.count_nonzero(Y2))/np.count_nonzero(Y)
    if debug:
        print ("pair recall: ", pair_recall)
    X = X2
    Y = Y2

    X2 = []
    Y2 = []

    for i in range(len(Y)):
        if Y[i]==1 or np.count_nonzero(X[i])!=0 or np.random.random()<pair_recall:
            X2.append(X[i])
            Y2.append(Y[i])

    X = X2
    Y = Y2
    return (X,Y)

def compute_corr_coeff(X, Y):
    N = X.shape[0]
    K = X.shape[1]
    X_all = np.zeros(shape=(N,K+1))
    X_all[0:N,0:K] = X
    X_all[:,K] = Y

    corr_coef = np.corrcoef(X_all,rowvar=0)

    if debug:
        print ("corr coef:")
        print (corr_coef[:,K])

def compute_pair_recalls(X,Y):

    pos_covered = 0
    for i in range(len(Y)):
        if Y[i]==1 and np.count_nonzero(X[i])!=0:
            pos_covered +=1

    pair_recall = float(pos_covered)/np.count_nonzero(Y)
    return pair_recall

def read_cos_feats(fname_cos_feats):
    ret = []
    lines = open(fname_cos_feats, encoding='utf8').read().splitlines()
    for l in lines:
        ss = l.split()
        ss = [float(s) for s in ss]
        ret.append(ss)
    return ret


def addPred(p, predCounts):
    if p not in predCounts:
        predCounts[p] = 0
    predCounts[p] = predCounts[p] + 1


def getPredPairs(data):
    preds = {}
    predPairs = {}
    predPairsTyped = {}
    predPairPos = {}
    for (p,q,t1s,t2s,_,a,l) in data:
        addPred(p,preds)
        addPred(p + "#" + q + "#"+str(a),predPairs)
        for t_i in range(len(t1s)):
            t1 = t1s[t_i]
            t2 = t2s[t_i]
            addPred(p + "#" + q + "#"+str(a)+"#"+t1+"#"+t2,predPairsTyped)
        if l==1:
            addPred(p + "#" + q + "#"+str(a),predPairPos)
    return preds, predPairs,predPairsTyped, predPairPos

def read_instance_level_probs(fname):
    ret = []
    for line in open(fname, encoding='utf8'):
        ret.append(float(line[:-1]))
    return ret