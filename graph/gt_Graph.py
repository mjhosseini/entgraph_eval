import graph_tool.topology as topology
from graph_tool.all import *
import numpy as np

class gt_Graph(Graph):

    def __init__(self):
        Graph.__init__(self)
        self.cnodes = []
        self.dedges = {}


    @staticmethod
    def get_comps(G):
        labels, _ = topology.label_components(G)

        labels = labels.a
        nc = np.max(labels) + 1

        comps = [[] for _ in range(nc)]

        [comps[labels[i]].append(i) for i in range(len(labels))]
        return comps


    #Add N vertex
    @staticmethod
    def gr_add_vertex(G,N):
        G.add_vertex(N)
        for i in range(N):
            G.dedges[i] = set()

    @staticmethod
    def gr_add_one_vertex(G):
        G.add_vertex()
        i = G.num_vertices()-1
        G.dedges[i] = set()

    @staticmethod
    def gr_add_edge(G,i,j):
        G.add_edge(i,j)
        # if i not in G.dedges:
        #     G.dedges[i] = set()

        G.dedges[i].add(j)


    @staticmethod
    def gr_add_edge_list(G,l):
        G.add_edge_list(l)
        for e in l:
            i,j = e
            G.dedges[i].add(j)