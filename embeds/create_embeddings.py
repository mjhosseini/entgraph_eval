import numpy as np
import sys
import gzip

#
# create_embeddings.py <glove-file> <feats-file>
#
# Uses Javad's output feats file (feats_rels.txt) and grabs the words in the relations.
# It averages all the word embeddings of those words for each relation, and then computes
# the cosine similarity between the two phrases as a feature.
#
# TODO: it would be better to use the raw phrase from the Levy dataset, rather than the
#       relations in Javad's feature file. But I don't see how to align the Levy lines
#       with the feature relation lines.
#

glovePath = sys.argv[1];
levypath = sys.argv[2];
textmode = True
embeds = dict()
edim = 300

# Creates numpy vector from the given string line.
# Line format: <token> .2 .3 .1 -.6 -.3 .6 ...
def lineToVector(line):
    parts = line[line.find(' ')+1:].split(' ')
    embed = np.zeros(len(parts))
    ii = 0
    for part in parts:
        embed[ii] = float(part)
        ii += 1
    return embed

# Read the first N embeddings from the file
def loadEmbeddings(preload):
    global edim
    first = True
    num = 0
    with gzip.open(glovePath) as f:
        for line in f:
            unigram = line[0:line.find(' ')]            
            embeds[unigram] = lineToVector(line)
            #print unigram,"::",embed

            if first:
                edim = len(embeds[unigram])
                print "Dimension size:", edim
                first = False
            
            num += 1
            if num > preload:
                return

# Find the embedding for the given token.
def findEmbedding(token):
    if len(token) < 1:
        print "ERROR: empty token in findEmbedding()"
    else:
        with gzip.open(glovePath) as f:
            for line in f:
                unigram = line[0:line.find(' ')]
                if unigram == token:
                    #print "FOUND ON LINE:", line
                    embed = lineToVector(line)
                    embeds[unigram] = lineToVector(line)  # cache for later lookup
                    return embed
    # Didn't find it.
    print "Didn't find *", token, "*"
    return np.zeros(edim)


# Get the longest phrase from the relation.
# relstr: string of format '(smile.1,smile.like.2)'
# returns: 'smile like'
def getPhraseFromRel(relstr):
    #print relstr
    negated = False
    
    # Check for negation in front: "NEG_(smile.1,smile.like.2)"
    if relstr.startswith("NEG__"):
        relstr = relstr[5:]
        negated = True
    
    # Remove parentheses. Split on comma.
    pair = relstr[1:-1].split(",")
    if len(pair) != 2:
        print "ERROR Bad relation string: ", relstr
        return ""

    # Removing trailing .1 or .2
    pred1 = pair[0][0:pair[0].rfind('.')]
    pred2 = pair[1][0:pair[1].rfind('.')]

    # Replace periods with spaces. Return the longest!
    phrase1 = pred1.replace('.',' ')
    phrase2 = pred2.replace('.',' ')
    if len(phrase2) >= len(phrase1):  # keep the longest
        phrase1 = phrase2
    if negated:                       # check for negation
        phrase1 = "not " + phrase1
    return phrase1
    
#
# Create the embedding for this phrase (bag of words)
#
def getEmbedding(phrase):
    #print "getEmbedding:", phrase
    num = 0
    embed = np.zeros(edim)
    tokens = phrase.split(' ')
    for token in tokens:
        # we have it in memory
        if token in embeds:
            tokembed = embeds[token]
        # read the file
        else:
            #print "Didn't have", token
            tokembed = findEmbedding(token)
        #print "Adding", token
        #print tokembed
        #print len(embed),len(tokembed)
        embed = np.add(embed, tokembed)
        num += 1

    # Average over the number of tokens.
    if num > 1:
        embed = np.divide(embed, num)
    return embed


def cosine(v1, v2):
    sim = float(np.dot(v1,v2))
    mag1 = float(np.sqrt(v1.dot(v1)))
    mag2 = float(np.sqrt(v2.dot(v2)))

    # Sometimes a vector is all zeros, so just return cosine 0
    if mag1 == 0 or mag2 == 0:
        return 0.0

    cosine = sim/mag1/mag2

    # Sanity check
    if cosine > 1.0001:
        print "WOOPS TOO BIG COSINE =", cosine
        print v1
        print v2
        print "dot product:", sim
        print "magnitudes: ", mag1, mag2
        exit()

    return cosine



# --------------------------------------------------
# MAIN
# --------------------------------------------------

if len(sys.argv) < 3:
    print "usage: create_embeddings.py <glove-datafile> <levy-dataset>"
    exit()

# Load the most frequent 50k word embeddings into memory.
loadEmbeddings(50000)
print "After embedding loading, edim x=", edim

# Determine file format.
# (1) parsed relations
# (2) levy's original openie text
with open(levypath) as f:
    line = f.readline()
    if line.startswith("("):
        print "PARSED RELATION FEATURE-FILE MODE"
        textmode = False
    else:
        print "LEVY TEXT MODE"
        #print "Sorry: not implemented yet"
        #exit()

# Read the lines and make the features!
with open(levypath) as f:
    filename = "cosine-feats.txt"
    of = open(filename, 'w')

    lines = f.readlines()
    for line in lines:

        # (prevail.1,prevail.in.2)#(have.1,have.status.in.2)#True	0
        if not textmode:
            parts = line.split("#")
            if len(parts) != 3 and len(parts) != 5:
                print "ERROR Bad line:", line

            rel1 = parts[0]
            rel2 = parts[1]
            
            # Get each relation's longest phrase.
            phrase1 = getPhraseFromRel(rel1)
            phrase2 = getPhraseFromRel(rel2)
            #print "GOT:", phrase1, "-->", phrase2

        # african country, is located near, ethiopia	eritrea, federated with, ethiopia	n
        else:
            parts = line.split(",")
            if len(parts) != 5:
                print "ERROR Bad text line:", line

            phrase1 = parts[1].rstrip().strip()
            phrase2 = parts[3].rstrip().strip()
            print "line:", line.rstrip()
            print "\t",phrase1,"::",phrase2

        # Get the embedding for each phrase.
        embed1 = getEmbedding(phrase1)
        embed2 = getEmbedding(phrase2)

        # Output similarity
        sim = cosine(embed1, embed2)
        print sim
        of.write(str(sim) + "\n")

    of.close()
    print "Wrote cosine sims to", filename
