"""
Reduce the size of the graph by first finding identifying the connected compoments 
and then doing the random walk sampling

Would like to experiment with several sampling methods
"""

from IPython.core.debugger import Pdb
from collections import defaultdict
import networkx as nx
import numpy
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import random



def trim_degrees(G, degree=1):
    G2 = G.copy()
    d = nx.degree(G)
    for n in G2.nodes():
        if d[n] <= degree: G2.remove_node(n)
    return G2


def create_hr(G):
   """
   Create heirarchical cluster of a graph G from distance matrix
   """
   # create shortest path matrix
   path_length = nx.all_pairs_shortest_path_length(G)
   return path_length

def read_graph(path, style='networkx'):
    """
    Read in the graph from the file and return a graph object of the type
    """
    # read in the file for analysis
    if style == 'snap':
        return None
    else: 
        G = nx.read_edgelist(path, delimiter=",", nodetype=str)
        #G = G.to_directed()
        # somehow does not work on directed graphs
        return G

def get_largest_component(G):
    """
    Returns the largest component in an undirected graph G
    
    Parameters
    ----------  
    G:            - undirected networkx.Graph 
    """

    H=nx.connected_components(G)
    list_of_comps = sorted(H, key = len, reverse=True)
    SG =  G.subgraph(list_of_comps[0])
    #print len(SG.nodes())
    return SG


def get_sampled_subgraph(G, size, start_node=None):
    """
    Returns a list of nodes sampled by the classic Random Walk (RW)
    
    Parameters
    ----------  
    G:            - networkx.Graph 
    start_node:   - starting node (if None, then chosen uniformly at random)
    size:         - the target sample length (int)
    """
    print "here"
    try:
        if type(G) != nx.Graph:
            raise nx.NetworkXException("G must be a simple undirected graph!") 
        
        if start_node==None:
            start_node = random.choice(G.nodes())

        v = start_node
        sample = [v]    
        while len(sample) < size:        
            v = random.choice(nx.neighbors(G, v))
            sample.append(v)
    except Exception as inst:
        print inst

    return G.subgraph(sample)


def write_graph(G):
    """
    Write the graph to the disk
    """
    nx.write_edgelist(G, "../data/sampled2.edgelist", data=False)


def main():
    G = read_graph('../data/clean.graph')
    
    #read in the graph file
    print "Number of nodes before trimming", len(G.nodes())
    #G = trim_degrees(G)
    print "Number of nodes after trimming", len(G.nodes())
    SG = get_largest_component(G)
    
    # get a graph of 10,000 nodes
    print type(SG) 
    SSG = get_sampled_subgraph(SG, size=30000)
    print "Number of nodes after random sampling", len(SSG.nodes())
    write_graph(SSG)


if __name__ == '__main__':
    main()