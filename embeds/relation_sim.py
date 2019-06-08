import operator
import sys
from time import gmtime, strftime

import embed_reader as embed

sys.path.append("..")
from evaluation import util
import e_util

#Comment about prefixes such as Neg__(visit.1,visit.2)
#"glove": Nate's code completely ignores __ when computing the similar ones. But it still might write the __ files in the output.
#This is helpful as I can then just attach __ whenever I'm finding the similarities to one predicate!

#"linkPred": when you see normal preds (without __) at LHS, just accept normal preds at RHS.
#But when you see __ (e.g., plan__(visit.1,visit.2)), write it at LHS, but at the RHS, write the neighbors of
# the bare predicate (e.g., (visit.1,visit.2)).

mincosine = 0.6

# CCG on Newspike: 400k seems plenty
numrelations = 400000
# GBooks
#numrelations = 40000000

print "numrelations =", numrelations
print "mincosine =", mincosine

def loadAllrels_ol(ccgpath):
    all_rels_ol = []

    data = util.read_data(ccgpath,None,True,False,False)
    tmp_path = 'tmp_rels.txt'
    f = open(tmp_path,'w')

    for p, q, _, _, _, _, _ in data:
        if p!="":
            all_rels_ol.append(p)
        if q!="":
            all_rels_ol.append(q)
    all_rels_ol = set(all_rels_ol)
    for rel in all_rels_ol:
        f.write(rel+"\t1\n")

    return loadRelations(tmp_path)

#
# Loads all relations from the given file path, one relation per line.
# The file can be CCG or OpenIE relations.
# @returns A dictionary from input relation pattern to phrase string
#
def loadRelations(path):
    print "Loading relations from", path
    sys.stdout.flush()
    num = 0
    phrases = dict()
    with open(path) as fin:
        for line in fin:
            #print line
            #print line[0]
            #print line.find("NEG__")
            # CCG: "(told.2,told.3) 32723"
            if len(line) > 0 and (line[0]=='(' or line.find("__(")!=-1) and line.find(')') > -1:
                last = line.rindex(')') + 1
                pattern = line[0:last]
                phrase = e_util.getPhraseFromCCGRel(pattern)
                phrases[pattern] = phrase.lower().replace('_',' ')
                num += 1

            # OpenIE: "det broad education	det majority	be_the_need_of	10"
            elif line.find('\t') > -1:
                parts = line.split('\t')
                pattern = parts[2]
                count = parts[3]
                if int(count) > 20:  # don't load relations under 20 countx
                    phrase = e_util.getPhraseFromOpenIERel(pattern)
                    phrases[pattern] = phrase.lower().replace('_',' ')
                    num += 1                

            # OpenIE: "be_the_need_of"
            elif line.find(' ') == -1:
                pattern = line.rstrip()
                phrase = e_util.getPhraseFromOpenIERel(pattern)
                phrases[pattern] = phrase.lower().replace('_',' ') 

            else:
                print "ERROR, bad line:", line

            if num > numrelations:
                break
    return phrases


def printMostSimilar(allrels, testrels, allRels_ol):
    # Load up the phrase embeddings for all possible relations.
    print "************"
    print "Creating phrase embeddings from word embeddings."
    print "************"
    sys.stdout.flush()

    allembeds = dict()
    allembeds_ol = dict()
    N = 0


    if embMode== "glove":

        for pat in allRels_ol:
            if pat not in allembeds_ol:
                rel = allRels_ol[pat]
                allembeds_ol[pat] = er.getEmbeddingOfPhrase(rel)


        #Now, write the embeddings to file as it can be used later!
        f_out = open("rels2glove.txt",'w')
        for pat in allembeds_ol:
            f_out.write(pat+"\t"+str(list(allembeds_ol[pat]))+"\n")
        f_out.close()

    print "ol embeddings written"


    if embMode=="glove":
        allrelsvals = allrels.values()

    else:
        allrelsvals = allrels

    for rel in allrelsvals:
        if rel not in allembeds:
            allembeds[rel] = er.getEmbeddingOfPhrase(rel)
            N += 1
        if N % 10000 == 0:   # debug
            print "embedded:",N


    sys.exit(0)#TODO: be careful


    print "************"
    print "All phrase embeddings ready. Now finding most similar."
    print "************"

    # Loop over test items, and print the best matches!
    if embMode == "glove":
        testrelsItems = testrels.items()
    else:
        testrelsItems = testrels

    for item in testrelsItems:
        if embMode == "glove":
            #e.g., pat: (visit.1,visit.2) and phrase: visit
            (targetPattern, targetRel) = item
        else:
            (targetPattern, targetRel) = (item,item)
            if targetRel.find("__(") != -1:
                st = targetRel.find("__(")
                targetRel = targetRel[(st + 2):]

        try:
            tembed = er.getEmbeddingOfPhrase(targetRel)
        except:
            continue

        cosines = dict()
        for item2 in allrels:
            if embMode == "glove":
                (pat, rel) = (item2,allrels[item2])
            else:
                (pat, rel) = (item2, item2)
                if rel.find("__")!=-1:
                    continue


            cos = er.cosine(tembed, allembeds[rel])
            #print "COSINE: *%s*,*%s* - *%s*,*%s* = %.4f" % (targetPattern, targetRel, pat, rel, cos)
            # Don't save around low scores (efficiency reasons)
            if cos > mincosine:
                cosines[pat] = cos

        # Sort by highest cosine (expensive operation)
        sorted_x = sorted(cosines.items(), key=operator.itemgetter(1), reverse=True)

        # Print the top 30
        print "%s" % (targetPattern),
        for ii in xrange(0,30):
            if ii < len(sorted_x):
                print "\t%s\t%.3f" % (sorted_x[ii][0], sorted_x[ii][1]),
        print "\n",

        
# --------------------------------------------------
# MAIN
# --------------------------------------------------

if len(sys.argv) < 5:
    print "usage: relation_sim.py <embMode> <emb-datafile> <all-rels-file> <test-rels-file> <relsFolder>"
    exit()

# File paths.
embMode = sys.argv[1]
embedPath = sys.argv[2]
relspath = sys.argv[3]
testrelspath = sys.argv[4]
relsFolder = ""
if len(sys.argv)>5:
    relsFolder = sys.argv[5]

# Load the most frequent 50k word embeddings into memory.
print strftime("%Y-%m-%d %H:%M:%S", gmtime())
print "Loading word embeddings from file..."
allRelsTotal = e_util.loadAllrelsTotal(relspath)

root = "../../gfiles/"

fname_CCG = root+"ent/all_new_comb_rels.txt"
allRels_ol = loadAllrels_ol(fname_CCG)

er = embed.EmbedReader(embedPath, 400000, embMode,allRelsTotal, relsFolder)
print "After embedding loading"
print strftime("%Y-%m-%d %H:%M:%S", gmtime())

# Load all possible relations.
if embMode == "glove":
    allrels  = loadRelations(relspath)
    print "Loaded", len(allrels), "relations."
    sys.stdout.flush()
    testrels = loadRelations(testrelspath)
    print "Loaded", len(testrels), "relations."
else:
    allrels = allRelsTotal
    testrels = allRelsTotal
    # mincosine = 0 #Be careful
print strftime("%Y-%m-%d %H:%M:%S", gmtime())
sys.stdout.flush()

# Do it!
printMostSimilar(allrels, testrels, allRels_ol)
