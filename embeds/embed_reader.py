import numpy as np

#
# Class to read word embeddings from file, and offer quick lookup.
#

class EmbedReader:

    #
    # filepath: path to the token/embedding file
    # num: the number of tokens to load into memory on startup
    # embMode: "glove" or "linkPred"
    def __init__(self, filepath, num, embMode = "glove",allrels=None, relsFolder=None):
        self.embMode = embMode
        self.filepath = filepath
        if embMode == "glove":
            self.embeds = dict()
            self.edim = 300
            self.loadGloveEmbeddings(filepath, num)
        elif embMode=="transE":
            self.allrels = allrels
            print ("all rels len: ", len(allrels))
            self.loadTransEEmbed(filepath, relsFolder)
        else:
            # self.allrels = allrels
            # print ("all rels len: ", len(allrels))
            self.loadConvEEmbed(filepath)

    
    # Creates numpy vector from the given string line.
    # Line format: <token> .2 .3 .1 -.6 -.3 .6 ...
    def lineToVector(self, line):
        parts = line[line.find(' ')+1:].split(' ')
        embed = np.zeros(len(parts))
        ii = 0
        for part in parts:
            embed[ii] = float(part)
            ii += 1
        return embed

    def loadConvEEmbed(self, filePath):

        self.phraseToEmbed = {}

        f = open(filePath)
        self.edim = -1
        for line in f:
            line = line.strip()
            ss = line.split("\t")
            if len(ss)!=2:
                continue
            rel = ss[0]

            embeds = [np.float(x) for x in ss[1][1:-1].split()]

            if self.edim == -1:
                self.edim = len(embeds)
                print ("edim: ", self.edim)
            self.phraseToEmbed[rel] = embeds

    def loadTransEEmbed(self, modelName, relsFolder):
        #TODO: implement this function to fill out the below dictionary!
        from transELoader import getRelatoinEmbeddings
        embs = getRelatoinEmbeddings(modelName, relsFolder)
        self.edim = len(embs[0])
        self.phraseToEmbed = {}
        assert len(self.allrels) == len(embs)
        for i in range(len(self.allrels)):
            if i%100==0:
                print (i)
            self.phraseToEmbed[self.allrels[i]] = embs[i]


    # Read the first N embeddings from the file
    def loadGloveEmbeddings(self, glovePath, preload):
        first = True
        num = 0
        fin = open(glovePath)
        for line in fin:
            unigram = line[0:line.find(' ')]            
            self.embeds[unigram] = self.lineToVector(line)
                #print unigram,"::",embed

            if first:
                self.edim = len(self.embeds[unigram])
                print ("Dimension size:", self.edim)
                first = False
                
            num += 1
            if num > preload:
                return
        fin.close()

    # Find the embedding for the given token.
    def getEmbedding(self, token):
        if len(token) < 1:
            print ("ERROR: empty token in findEmbedding()")
        elif token in self.embeds:
            return self.embeds[token]            

        # Didn't find it.
        print ("Didn't find *", token, "*")
        embed = np.zeros(self.edim)
        self.embeds[token] = embed  # cache to avoid repeated lookup
        return embed

    #
    # Create the embedding for this phrase (bag of words)
    #
    def getEmbeddingOfPhrase(self, phrase):

        if self.embMode != "glove":
            if phrase in self.phraseToEmbed:
                return self.phraseToEmbed[phrase]
            else:
                print "does not have: ", phrase
                return np.zeros(self.edim)
        else:
            num = 0
            embed = np.zeros(self.edim)

            tokens = phrase.split(' ')
            for token in tokens:
                tokembed = self.getEmbedding(token)
                embed = np.add(embed, tokembed)
                num += 1

            # Average over the number of tokens.
            if num > 1:
                embed = np.divide(embed, num)
            return embed

    def cosine(self, v1, v2):
        sim = float(np.dot(v1,v2))
        mag1 = float(np.sqrt(v1.dot(v1)))
        mag2 = float(np.sqrt(v2.dot(v2)))
        
        # Sometimes a vector is all zeros, so just return cosine 0
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        cosine = sim/mag1/mag2

        # Sanity check
        if cosine > 1.0001:
            print ("WOOPS TOO BIG COSINE =", cosine)
            print (v1)
            print (v2)
            print ("dot product:", sim)
            print ("magnitudes: ", mag1, mag2)
            exit()

        return cosine
