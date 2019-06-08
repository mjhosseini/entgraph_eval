import sys

import embed_reader as embed

sys.path.append("..")
from evaluation import util
import e_util
#Simple cos similarity
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def getEntailmentScores(ccgpath,fname_orig, er, isTyped = False):

    scores = []

    data = util.read_data(ccgpath,fname_orig,True,isTyped,False)

    for p, q, t1s, t2s, _, a, _ in data:
        if p!="" and q!="":
            if embMode=="glove":
                phrase1 = e_util.getPhraseFromCCGRel(p)
                phrase2 = e_util.getPhraseFromCCGRel(q)
            else:
                print "a is: ", a
                phrase1 = p
                phrase2 = q
                if isTyped:
                    t1 = t1s[0]
                    t2 = t2s[0]
                    if t1!=t2:
                        phrase1 += "#"+t1+"#"+t2
                        if a:
                            phrase2 += "#"+t1+"#"+t2
                        else:
                            phrase2 += "#" + t2 + "#" + t1
                    else:
                        phrase1 += "#" + t1+ "_1#" + t2+"_2"
                        phrase2 += "#" + t1+ "_1#" + t2+"_2" +"_reverse"

                    print p,q,t1s,t2s,a
                    print phrase1, phrase2


                elif embMode=="convE" and not a:
                    phrase2 += "_reverse"


            emb1 = er.getEmbeddingOfPhrase(phrase1)
            emb2 = er.getEmbeddingOfPhrase(phrase2)

            dim = len(emb1)
            emb1_nd = np.ndarray(shape=(1,dim))
            emb2_nd = np.ndarray(shape=(1, dim))
            emb1_nd[0,:] = emb1
            emb2_nd[0,:] = emb2

            # score = np.maximum(,0)
            score = (cosine_similarity(emb1_nd, emb2_nd)[0, 0] + 1)/2
            score = np.minimum(score,1)
        else:
            score = 0

        print "score: ", score
        scores.append(score)
    return scores


root = "../../gfiles/"

#python predict_entailment.py convE rels2emb_ConvE_NS_unt_10_10_1000.txt rels_NS_10_10 none dev_new_rels.txt dev_new.txt dev_convE_10_10_cos.txt
#example: python predict_entailment.py linkPred model_NS_10_10  rels_NS_10_10 NS_untyped_10_10 ber_all_rels.txt out_ber_transE.txt
#example: python predict_entailment.py linkPred model_NS_10_10  rels_NS_10_10 NS_untyped_10_10 trainTest_new_rels.txt trainTest_transE.txt
# python predict_entailment.py convE rels2emb_ConvE_NS_unt_20_20.txt rels_NS_10_10 NS_untyped_10_10 trainTest_new_rels.txt trainTest_transE.txt
# python predict_entailment.py convE rels2emb_ConvE_NS_unt_20_20_125.txt rels_NS_10_10 NS_untyped_10_10 trainTest_new_rels.txt trainTest_new.txt trainTest_convE_125_r.txt
if len(sys.argv) < 7:
    print "usage: predict_entailment.py <embMode> <emb-datafile> <all-rels-file> <relsFolder> <ent-examples-file> <ent-examples-orig-file> <ent-examples-file-out>"
    exit()


embMode = sys.argv[1]
embedPath = sys.argv[2]
relspath = sys.argv[3]
relsFolder = sys.argv[4]
fname_CCG = root+"ent/"+sys.argv[5]
fname_orig = root+"ent/"+sys.argv[6]
out_path = sys.argv[7]

if relspath!="" and relspath!="none":
    allRelsTotal = e_util.loadAllrelsTotal(relspath)
else:
    allRelsTotal = None

er = embed.EmbedReader(embedPath, 400000, embMode ,allRelsTotal, relsFolder)
# fname_CCG = root+"ent/trainTest_new_rels.txt"
# ber_all_rels.txt"

entScores = getEntailmentScores(fname_CCG, fname_orig, er, isTyped=False)

f = open(out_path,'w')

for s in entScores:
    f.write(str(s)+"\n")
f.close()
