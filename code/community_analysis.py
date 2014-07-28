""" This program analyzes the properites of the different communities 
"""

import pandas as pd 
import numpy as np 
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
import re
import operator
import pylab as pl
import seaborn
import time

#
base='../data/generated/'
infile='total_biz_info_with_catlabel.csv'
ingraphfile = 'subgraphedgelist.ntx'
origgraphfile = 'edgelist.ntx'
cliquebase = base + '/cliques/'
compbase = base + '/components/'
biconnbase = base + '/biconn/'


""" return the 5 categories that represent the community
"""
def get_categories(commdf):
    
    # combine two categories columnds in df
    tags = commdf.apply(combine, axis=1)

    # use count vectorizer to get the category frequency
    vec = CountVectorizer(tokenizer=tokenize)
    comm_data = vec.fit_transform(tags.values).toarray()
    vocab = vec.get_feature_names()
    dist = np.sum(comm_data, axis=0)

    # create a dictionary of tags and 
    cat_dist = {}
    for tag, count in zip(vocab, dist):
        cat_dist[tag] = count

    return cat_dist


def get_most_significant_categories(commdf, num=5):
    cat_dist = get_categories(commdf)
    sorted_dist = sort_dictionary(cat_dist)
    return sorted_dist[-num:]


# distribution of categories
# within the community
# this is done to manually inspect the category distribution

def plot_category_distribution(commid, cat_dist):
    
    vocab = cat_dist.keys()
    dist = cat_dist.values()

    # find the average distribution
    md = np.mean(dist)
    labels = [ v if cat_dist[v] > md else ' ' for v in vocab ]
    fig = pl.figure()
    ax = pl.subplot(111)
    width=0.8

    ax.bar(range(len(vocab)), dist, width=width)
    ax.set_xticks(np.arange(len(vocab)) + width/2)
    ax.set_xticklabels(labels, rotation=90)

    filename = '../figures/'+str(commid)+ 'cat_dist.pdf'
    pl.savefig(filename, bbox_inches='tight')

   
def get_density(subgraph):

    return nx.density(subgraph)


""" customized score
"""
def get_most_influential(subgraph, personalization, num=5):

    # personalization vector comes from the 
    # social presence
    pr_dict = nx.pagerank(subgraph)
    #most_influential = 
    #pr = sorted([(val, user) for user, val in pr.iteritems()], reverse=True)

    print pr #return [x[1] for x in pr[num]]


""" returns a clique subgrapg
"""
def get_community_cliques(commid, commdf, graph):

    # return first clique, all cliques
    # are actually the same

    undir_graph = graph.to_undirected()
    cl = nx.find_cliques(undir_graph)
    cl = sorted(list( cl ), key=len, reverse=True)
    
    #print "Number of cliques:", len(cl)
    #cl_sizes = [len(c) for c in cl]
    #print "Size of cliques:", cl_sizes

    if len(cl) > 0:
        
        g = graph.subgraph(cl[0])
        print g.edges()

        #copydf = commdf.loc[cl[0]]
        #print "Size of dataset extracted", copydf.shape[0]
        #print copydf[['name', 'category', 'category_labels']]
        #copydf.to_csv(cliquebase+'_'+ str(commid)+ '.csv')
    else:
        return None


""" returns strongly connected 
     component
"""
def get_community_scc(commid, commdf, graph):
    
    # return first clique, all cliques
    # are actually the same

    sccs = nx.strongly_connected_components(graph)
    for i,s in enumerate(sccs):
        if i == 0:
            print commid , "####", s
        #copydf = commdf.loc[cl[0]]
        #print "Size of dataset extracted", copydf.shape[0]
        #print copydf[['name', 'category', 'category_labels']]
        #copydf.to_csv(cliquebase+'_'+ str(commid)+ '.csv')



""" returns a list of pair of biconnected nodes 
     component
"""
def get_community_biconnections(commid, df, graph):
    
    print nx.info(graph)

    biconnected_nodes = []
    for e in graph.edges():
        a, b = e
        if graph.has_edge(b,a) and a != b:
            # check if already there in the list
            if (a,b) in biconnected_nodes or (b,a) in biconnected_nodes:
                pass
            else:
                biconnected_nodes.append((a,b))

    print "number of biconnected edges:", len(biconnected_nodes)

    source_nodes, target_nodes = zip(*biconnected_nodes)
    all_subgraph_nodes = set(source_nodes).union(set(target_nodes))
    print "Unique nodes in the biconnections", len(all_subgraph_nodes)

    # get the subgraph of all biconnected edges 
    # plot 
    dfname = biconnbase+ str(commid) + '_biz_info.csv'
    bicon_df = df.loc[all_subgraph_nodes]
    print bicon_df.shape
    bicon_df.to_csv(dfname)

    # subgraph generated from the coordinates
    sgname = biconnbase+ str(commid) + '_sg_edgelist.ntx'
    sg = graph.subgraph(list(all_subgraph_nodes))
    print nx.info(sg)
    nx.write_edgelist(sg, sgname, data=False)



def get_community_analytics(df, graph):

    communities = df['communityid'].unique()
    print "Number of communities: ", len(communities)

    # unified dictionary containing information
    # about all comunities
    community_data = { x:{} for x in communities}

    # for each community 
    commid = 7 # communities[0]
   
    print "Analysis for community: ", commid

    subdf = df[df['communityid'] == commid]

    subgraph = graph.subgraph(subdf.index.tolist())

    community_data[commid]['comm_size'] = subdf.shape[0]

    community_data[commid]['categories'] = get_most_significant_categories(subdf)

    community_data[commid]['most_connected'] = get_most_influential(subgraph)

    community_data[commid]['density'] = get_density(subgraph)

    community_data[commid]['clique'] = get_community_cliques(commid, subdf, subgraph)

    community_data[commid]['biconn'] = get_community_biconn(commid, subdf, subgraph)


    #get_community_biconnections(commid, subdf, subgraph)
    # #time.sleep(5.5)
  
    # # if community_modularity != None:
    # #     data[lvl][cid]['modularity'] = community_modularity[lvl]
    # # else:
    # #     data[lvl][cid]['modularity'] = None

    # #return data

#====================utils===============#
def combine(row):
  l = str(row['category_labels']).lower()
  c = str(row['category']).lower()
  return l[1: -1]+ ',' +c


def tokenize(text):
    REGEX = re.compile(r",\s*")
    return [tok.strip('"').lower() for tok in REGEX.split(text)]

def sort_dictionary(dictionary):
    sorted_x = sorted(dictionary.iteritems(), key=operator.itemgetter(1))
    return sorted_x

#===========


def main():
    # read in the total biz file
    df = pd.read_csv(base+infile)
    df = df.set_index('pageid')
    print df.columns

    # read in th original edgelist as a directed graph
    globalgraph= nx.read_edgelist(base+ingraphfile, create_using=nx.DiGraph(), nodetype=int)
    print nx.info(globalgraph)

    print "Community Analysis for all communities found -----"
    get_community_analytics(df, globalgraph)

    ## modularity and conductance of the map
    get_modularity()



if __name__ == '__main__':
    main()