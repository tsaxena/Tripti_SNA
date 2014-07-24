"""
Reduce the size of the data
Strategies used
   1: Trimming the nodes based on the degree (works on both directed and undirected)
   2: Random Sampling # did not work that well (works on both directed and undirected)
   3. cliques (works only on undirected)
   4. communities (works only on undirected)
Also create a block model given the partitions

Todo: Would like to experiment with several sampling methods
"""
import networkx as nx
import community   # lib from http://perso.crans.org/aynaud/communities/
import matplotlib.pyplot as plt 
  
  
def find_cliques(graph):
    # returns cliques as sorted list
    print "Finding cliques..."
    g = graph
    cl = nx.find_cliques(g)
    cl = sorted(list( cl ), key=len, reverse=True)
    cl_sizes = [len(c) for c in cl]

    print "Cliques found: ", len(cl)
    return g, cl
  
def find_communities(graph):
    # code and lib from http://perso.crans.org/aynaud/communities/
    # must be an undirected graph
    g = graph
    partition = community.best_partition( g )
    part_members = {}
    print "Partitions found: ", len(set(partition.values()))
    #to show members of each partition:
    for i in set(partition.values()):
        members = [node for node in partition.keys() if partition[node] == i]
        part_members[i] = members

        # if i==0:
        #      # write out the subgraph
        #      community_graph = graph.subgraph(members)
        #      #draw_graph(community_graph)
        #      #nx.write_edgelist(community_graph, "community.edgelist", data=False)
        #      #for member in members:
        # #         print member, i
           
    #print "Partition for a node: ", partition['214839528530672']
    nx.set_node_attributes(g,'partition',partition)
    return g, partition, part_members

# get connected components
def find_connected_components(G):
    """
    Returns the largest component in an undirected graph G
    
    Parameters
    ----------  
    G:  undirected networkx.Graph 
    """

    H=nx.connected_components(G)
    comp_list = sorted(H, key = len, reverse=True)
    return g, comp_list

   
  
def draw_partition(graph, partition):
    # requires matplotlib.pyplot, uncomment above
    # uses community code and sample from http://perso.crans.org/aynaud/communities/ to draw matplotlib graph in shades of gray
    g = graph
    count = 0
    size = float(len(set(partition.values())))
    pos = nx.spring_layout(g)
    for com in set(partition.values()):
        count = count + 1
        list_nodes = [nodes for nodes in partition.keys()
                                    if partition[nodes] == com]
        nx.draw_networkx_nodes(g, pos, list_nodes, node_size = 20,
                                    node_color = str(count / size))
    nx.draw_networkx_edges(g,pos, alpha=0.5)
    plt.show()


def write_partition(partition, filename):
    print "Writing partition to file..."
    clique_graph = graph.subgraph(cl[0])
    nx.write_edgelist(clique_graph, "clique.edgelist", data=False)


def get_block_model(graph, partitions):
    """
    Returns a block model given a graph and partitions
    
    Parameters
    ----------  
    G:  networkx.Graph 
    partitions : 
    """


