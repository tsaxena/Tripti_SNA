import networkx as nx

def trim_nodes_by_attribute_for_remaining_number(graph, attributelist, count):
    g = graph
    to_remove = len(graph.nodes()) - count - 1
    g.remove_nodes_from([x[0] for x in attributelist[0:to_remove]])
    print "Now graph has node count: ", len(g.nodes())
    return g
  
def trim_nodes_by_attribute_value(graph, attributedict, threshold):
    g = graph
    g.remove_nodes_from([k for k,v in attributedict.iteritems() if v <= threshold])
    return g

def trim_nodes_by_degree(graph, degree=1):
    graph_copy = graph.copy()
    d = nx.degree(graph)
    for n in graph_copy.nodes():
        if d[n] <= degree: graph_copy.remove_node(n)
    return graph_copy

def trim_nodes_by_random_walk(G, size, start_node=None):
    """
    Returns a list of nodes sampled by the classic Random Walk (RW)
    
    Parameters
    ----------  
    G:            - networkx.Graph 
    start_node:   - starting node (if None, then chosen uniformly at random)
    size:         - the target sample length (int)
    """
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


def get_graph_sample(graph, type= 'degree'):
    #trim the graph by removing nodes with degree==1

    if(type == 'degree'):     
        tdg =  trim_nodes_by_degree(graph)
        print "Number of nodes before trimming", nx.info(tdg)
        return tdg
    else:
        print "Invalid sampling type"
