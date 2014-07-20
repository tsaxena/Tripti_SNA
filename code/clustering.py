"""
Would like to figure out the cliques in the large graph 
This would be beneficial in doing topic modeling

"""

from collections import defaultdict
import networkx as nx
import numpy
from scipy.cluster import hierarchy
from scipy.spatial import distance
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns


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

def get_components(G):
	H=nx.connected_components(G)
	return H


def main():
	G = read_graph('../data/clean.graph')
	#read in the graph file
	print "Number of nodes before trimming", len(G.nodes())
	#G = trim_degrees(G)
	print "Number of nodes after trimming", len(G.nodes())

	S = get_components(G)
	list_of_comps = sorted(S, key = len, reverse=True)
	
	SG =  G.subgraph(list_of_comps[0])
	print len(SG.nodes())

	nx.write_edgelist(SG, "subgraph.edgelist", data=False)

	# get a subgraph


if __name__ == '__main__':
	main()