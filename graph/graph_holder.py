from sccGraph import *
debug = False

class GHolder:
    def __init__(self,pgragh, lmbda,gLines=None,lIdx=-1,types=None):
        self.types = types
        if gLines is not None:
            self.form_graphs_from_files(pgragh,lmbda,gLines,lIdx)
            return
        else:
            self.form_graphs(pgragh,lmbda)

    def form_graphs_from_files(self,pgraph,lmbda,gLines,lIdx):
        self.TNFs = []
        tnf = SCCGraph(pgraph, lmbda, gLines, lIdx)
        self.TNFs.append(tnf)
        self.lIdx = tnf.lIdx
        print "lidx: ", self.lIdx
        self.pred2Node = tnf.pred2Node

        N = len(tnf.node2sccIdx)
        self.node2comp = {}
        if debug:
            print ("N: ", N)

        for i in range(N):
            self.node2comp[i] = 0#

    def clean(self):
        for tnf in self.TNFs:
            del tnf.pgraph


    def write(self):
        for idx,tnf in enumerate(self.TNFs):
            print "subgraph #", idx
            tnf.write_SCC()




