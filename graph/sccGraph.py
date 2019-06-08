import sys
sys.path.append("..")
from gt_Graph import gt_Graph

class SCCGraph:

    fastInit = False

    def __init__(self, pgraph, lmbda,finalResLines=None,lIdx=-1):
        if pgraph:
            print "num nodes: ", len(pgraph.nodes)


        self.pgraph = pgraph
        if finalResLines is not None:
            self.pred2Node = {}
            self.read_from_file(finalResLines,lIdx)
            return

    def read_from_file(self,lines,lIdx):
        print "first line: ", lines[lIdx]
        self.lmbda = float(lines[lIdx].split()[1])
        self.idx2ArrayIdx = {}
        lIdx += 1

        self.node2sccIdx = []
        self.scc = gt_Graph()

        scc = self.scc
        node2comp = self.node2sccIdx

        scc_edges = set()

        currentNode = -1
        arrIdx = 0
        while lIdx < len(lines) and not lines[lIdx].startswith("lambda:"):

                if lines[lIdx]=="" or lines[lIdx]=="writing Done":
                    lIdx += 1
                    continue
                elif lines[lIdx].startswith("component"):
                    gt_Graph.gr_add_one_vertex(scc)#TODO: to be removed
                    currentNode += 1
                elif lines[lIdx].startswith(" => "):
                    neigh = int(lines[lIdx].strip().split()[1])
                    scc_edges.add((currentNode,neigh))
                else:
                    if self.pgraph:
                        nodeIdx = self.pgraph.pred2Node[lines[lIdx]].idx
                    else:
                        nodeIdx = arrIdx
                        self.pred2Node[lines[lIdx]] = arrIdx
                    self.idx2ArrayIdx[nodeIdx] = arrIdx#Necessary??
                    node2comp.append(currentNode)
                    arrIdx += 1

                lIdx += 1

        self.lIdx = lIdx

        gt_Graph.gr_add_edge_list(scc,list(scc_edges))

    @staticmethod
    def is_connected_scc(scc,node2comp,i,j):
        idx1 = node2comp[i]
        idx2 = node2comp[j]
        return idx1==idx2 or idx2 in scc.dedges[idx1]

    def write_SCC(self):

        SCC_g = self.scc

        for idx in range(SCC_g.num_vertices()):
            print "\ncomponent ", idx
            nodes_idxes = SCC_g.cnodes[idx]
            for iidx in nodes_idxes:
                node = self.pgraph.nodes[iidx]
                print node.id

            for neigh in SCC_g.get_out_neighbours(idx):
                neigh_idx = SCC_g.cnodes[neigh][0]
                id2 = self.pgraph.nodes[neigh_idx].id
                print " => ", neigh, " ", id2
        print "writing Done"
