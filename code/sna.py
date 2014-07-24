import numpy
import networkx as nx
import pandas as pd
import random
import graph_stats
import graph_partition
import graph_sample


genbase = '../data/generated/'
ingraphfile = 'edgelist.ntx'
outgraphfile = 'subgraphedgelist.ntx'
outnodedir = 'direct_nodes_info.json'
blockgraph = 'blockedgelist.ntx'
commfile = 'comm.csv'


# for now just reading in the subgraph
#smallfile = 'bizinfo_subgraph.edgelist' # reduction of the graph based on business info

"""
  Given a graph and an partitions, 
  returns a block diagram 
"""

def create_block_diagram(graph, member_dict):
    BM = nx.blockmodel(graph, member_dict.values())
    return BM


def write_comm_dictionary(part, filename):
    df = pd.DataFrame(part.items())
    
    print "Writing the community dictionary: ", df.shape
    df.to_csv(filename, index=False)



def main():

    ## 
    infilepath = genbase+ingraphfile
    dg = graph_stats.read_in_graph(infilepath, format='networkx')
    
    tdg = graph_sample.get_graph_sample(dg)
    tdg = graph_stats.run_analysis(tdg)
    
    # write the edgelist of this graph
    outedges = genbase+outgraphfile
    graph_stats.write_edge_attributes(tdg, outedges)
    #print nx.info(tdg)

    # write the nodes to a csv file with the following attributes
    #graph_stats.write_node_attributes(tdg, genbase + outnodedir)
    #graph_stats.save_to_jsonfile(genbase + outnodedir, tdg)


    # to do community partitions, must have undirected graph.
    # not sure why the size is not the same
    subgraphfile = genbase+outgraphfile
    sdg = graph_stats.read_in_graph(subgraphfile, format='networkx')
    
    # statistics based on undirected graphs
    undir_g = tdg.to_undirected()
    
    ### find cliques 
    #undir_g, cliques = graph_partition.find_cliques(undir_g)
    
    ### Examine partitioning algo and potentially tweak.
    undir_g, parts, part_members = graph_partition.find_communities(undir_g) 
    
    # write out the parts dictionary to the 
    print type(parts)
    write_comm_dictionary(parts, genbase+commfile)

    # Create a block model of the communities
    #print type(part_members[0]) 
    bg = create_block_diagram(undir_g, part_members)
    print nx.info(bg)
    graph_stats.write_edge_attributes(bg, genbase+blockgraph)


if __name__ == '__main__':
    main()