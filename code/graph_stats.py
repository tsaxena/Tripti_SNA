
""" Read in the graph and analyse the properties of the graph
    Input: edgelist in format 
            # Edgelist looks like:
            # Source,Target
            # node1,node2
            # node3,node1
            # node1,node3
    Output: graph statistics
   
    Author : Tripti Saxena
"""

import networkx as nx    
from operator import itemgetter
from networkx.readwrite import json_graph
import json
import os.path  

def read_in_graph(filepath, format='csv'):
    """
    Read in the graph from the file and return a graph object of the type
    """
    if format == 'networkx':
        g_orig = nx.read_edgelist(filepath, create_using=nx.DiGraph())
        print "Read in edgelist file ", filepath
        print nx.info(g_orig)
        return g_orig
    else:
        print "Reading edgelist in csv format"

def save_to_jsonfile(filename, graph):
    g = graph
    g_json = json_graph.node_link_data(g) # node-link format to serialize
    json.dump(g_json, open(filename,'w'))
  
  
def calculate_degree(graph):
    print "Calculating degree..."
    g = graph
    deg = nx.degree(g)
    nx.set_node_attributes(g,'degree',deg)
    return g, deg
  
def calculate_indegree(graph):
    # will only work on DiGraph (directed graph)
    print "Calculating indegree..."
    g = graph
    indeg = g.in_degree()
    nx.set_node_attributes(g, 'indegree', indeg)
    return g, indeg
  
def calculate_outdegree(graph):
    # will only work on DiGraph (directed graph)
    print "Calculating outdegree..."
    g = graph
    outdeg = g.out_degree()
    nx.set_node_attributes(g, 'outdegree', outdeg)
    return g, outdeg
  
def calculate_betweenness(graph):
    print "Calculating betweenness..."
    g = graph
    bc=nx.betweenness_centrality(g)
    nx.set_node_attributes(g,'betweenness',bc)
    return g, bc
  
def calculate_eigenvector_centrality(graph):
    print "Calculating Eigenvector Centrality..."
    g = graph
    ec = nx.eigenvector_centrality(g)
    nx.set_node_attributes(g,'eigen_cent',ec)
    #ec_sorted = sorted(ec.items(), key=itemgetter(1), reverse=True)
    # color=nx.get_node_attributes(G,'betweenness')  (returns a dict keyed by node ids)
    return g, ec
  
def calculate_degree_centrality(graph):
    print "Calculating Degree Centrality..."
    g = graph
    dc = nx.degree_centrality(g)
    nx.set_node_attributes(g,'degree_cent',dc)
    degcent_sorted = sorted(dc.items(), key=itemgetter(1), reverse=True)
    for key,value in degcent_sorted[0:10]:
        print "Highest degree Centrality:", key, value
  
    return graph, dc
  
# 
  
def write_node_attributes(graph, filename):

    # utility function to let you print the node + various attributes in a csv format
    for node in graph.nodes(data=True):
    #    print graph.report_node_data(undir_g)
        node_idx, node_dict = node
        attrs = ','.join(str(v) for v in node_dict.values)
        print node #nx.get_node_attributes(graph, node_idx) #, ",", ",".join(vals)

def write_edge_attributes(graph, filepath, format='networkx', with_data=False):
    """
     Utility function to let you write an edgelist 
    """
    print "Writing edgelist to file..."
    if format == 'networkx':
        nx.write_edgelist(graph, filepath, data=with_data)
    else:
        print "generate csv"
    print "Done"

def report_node_data(graph, node=""):
    g = graph
    if len(node) == 0:
        print "Found these sample attributes on the nodes:"
        print g.nodes(data=True)[1]
    else:
        print "Values for node " + node
        print [d for n,d in g.nodes_iter(data=True) if n==node]
  
def run_analysis(dg):
    # my func will create a Digraph from node pairs.
    #dg = read_in_graph(path+graphfile) 
    #g = read_in_graph(path+smallfile) # my func will create a Digraph from node pairs.
    #g = read_json_file(path + inputjsonfile)
  
    print "Basic graph characteristics: \n"
    nx.info(dg)

    dg, deg = calculate_degree(dg)
    dg, indeg = calculate_indegree(dg)
    dg, outdeg = calculate_outdegree(dg)
    # # Taking forever, need to investigate
    #g, bet = calculate_betweenness(g)
    dg, eigen = calculate_eigenvector_centrality(dg)
    dg, degcent = calculate_degree_centrality(dg)

    return dg
  
    # # verify that the graph's nodes are carrying the attributes:
    #report_node_data(g, node=node_id)
  
   
    # # to do community partitions, must have undirected graph.
    #undir_g = g.to_undirected()
    
    # # find cliques 
    # # biggest clique found consists of 16 elements 
    #find_cliques(undir_g)
    
    # # Examine partitioning algo and potentially tweak.
    #undir_g, part = find_partition(undir_g)  

    # # draw_partition(undir_g, part)   # draws a matplotlib graph in grays
  
    # # show that the partition info is added to the nodes:
    #report_node_data(undir_g, node=node_id)
  
    # # trim what's saved to js file by taking only N nodes, with top values of a certain attribute...
    # eigen_sorted = sorted(eigen.items(), key=itemgetter(1), reverse=True)
    # for key, val in eigen_sorted[0:5]:
    #     print "highest eigenvector centrality nodes:", key, val
    # # for trimming, you want it reverse sorted, with low values on top.
    # eigen_sorted = sorted(eigen.items(), key=itemgetter(1), reverse=False)
  
    # small_graph = trim_nodes_by_attribute_for_remaining_number(undir_g, eigen_sorted, small_graph_size)
  
    # print nx.info(small_graph)

    
  
    
#def find_cliques(graph):
#     # returns cliques as sorted list
#     g = graph
#     cl = nx.find_cliques(g)
#     cl = sorted(list( cl ), key=len, reverse=True)
#     print "Number of cliques:", len(cl)
#     cl_sizes = [len(c) for c in cl]
#     print "Size of cliques:", cl_sizes
#     print "Writing first clique to file..."
#     clique_graph = graph.subgraph(cl[0])
#     nx.write_edgelist(clique_graph, "clique.edgelist", data=False)
#     return cl
  
# def find_partition(graph):
#     # code and lib from http://perso.crans.org/aynaud/communities/
#     # must be an undirected graph
#     g = graph
#     partition = community.best_partition( g )
#     print "Partitions found: ", len(set(partition.values()))
#     #to show members of each partition:
#     for i in set(partition.values()):
#         members = [nodes for nodes in partition.keys() if partition[nodes] == i]
#         print i, len(members)

#         # if i==0:
#         #      # write out the subgraph
#         #      community_graph = graph.subgraph(members)
#         #      #draw_graph(community_graph)
#         #      #nx.write_edgelist(community_graph, "community.edgelist", data=False)
#         #      #for member in members:
#         # #         print member, i
           
#     #print "Partition for node johncoogan: ", partition[node_id]
#     nx.set_node_attributes(g,'partition',partition)
#     return g, partition
  
# def draw_partition(graph, partition):
#     # requires matplotlib.pyplot, uncomment above
#     # uses community code and sample from http://perso.crans.org/aynaud/communities/ to draw matplotlib graph in shades of gray
#     g = graph
#     count = 0
#     size = float(len(set(partition.values())))
#     pos = nx.spring_layout(g)
#     for com in set(partition.values()):
#         count = count + 1
#         list_nodes = [nodes for nodes in partition.keys()
#                                     if partition[nodes] == com]
#         nx.draw_networkx_nodes(g, pos, list_nodes, node_size = 20,
#                                     node_color = str(count / size))
#     nx.draw_networkx_edges(g,pos, alpha=0.5)
#     plt.show()
  
  

