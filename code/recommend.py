import pandas as pd 
import numpy as np
import networkx as nx
import math
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import CountVectorizer
import re
from sklearn.metrics.pairwise import cosine_similarity
import operator
import csv


""" returns a dictionary of dictionary 
    that calculates the distance between
    different 
"""
base='../data/generated/'
infile='total_biz_info_with_catlabel.csv'
ingraphfile = 'subgraphedgelist.ntx'
origgraphfile = 'edgelist.ntx'
cliquebase = base + '/cliques/'
compbase = base + '/components/'
biconnbase = base + '/biconn/'


def data_prep(infofile, graphfile):
    # read in the total biz file
    # Preparing the data files 
    df = pd.read_csv(infofile)

    #removing duplicate records
    df = df.groupby('pageid').first()
    print df.columns
    print df.index
    print df.shape
    print df.isnull().sum()
    df = df[df['latitude'] != 'N']
    print "Dropping loc, lat = N: ", df.shape
    df = df.dropna() #df[df['latitude'] != 'N']
    print "Dropping NA", df.shape #df.isnull().sum()


    # read in th original edgelist as a directed graph
    globalgraph= nx.read_edgelist(graphfile, create_using=nx.DiGraph(), nodetype=int)
    print "Original Graph:", nx.info(globalgraph)

    print "Keeping it consistent, removing all nodes not in database:"
    pageids = list(df.index)
    prunedglobalgraph = globalgraph.subgraph(pageids)
    print nx.info(prunedglobalgraph)
    return df, globalgraph



def get_distance_dict(filename):
    g = nx.read_edgelist(filename)
    print "Read in edgelist file ", filename
    print nx.info(g)
    path_length = nx.all_pairs_shortest_path_length(g)
    print len(path_length.keys())
    print path_length


def combine(row):
    l = str(row['category_labels']).lower()
    c = str(row['category']).lower()
    #n = str(row['name']).lower()
    return  l[1: -1] + ',' +c 

def tokenize(text):
    REGEX = re.compile(r",\s*")
    return [tok.strip('"').lower() for tok in REGEX.split(text)]


def sort_dictionary(dictionary):
    sorted_x = sorted(dictionary.iteritems(), key=operator.itemgetter(1))
    return sorted_x
  

def calculate_cosine_similarity(data):

    #data[np.isnan(data)] = 0.

    # base similarity matrix (all dot products)
    # replace this with data.dot(data.T).todense() if sparce
    similar = np.dot(data, data.T)
    
    # # squared magnitude of preference vectors (number of occurrences)
    square_mag = np.diag(similar)
    
    # # inverse squared magnitude
    inv_square_mag = 1. / square_mag
    
    # # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    inv_square_mag[np.isinf(inv_square_mag)] = 0.
    
    # # inverse of the magnitude
    inv_mag = np.sqrt(inv_square_mag)
    
    # # cosine similarity (elementwise multiply by inverse magnitudes)
    cos = similar * inv_mag
    cos = cos.T * inv_mag
    cos = 1. - cos

    #print "shape of the cosine matrix:", cos.shape
    return cos

def get_tsusers(graph):
    # size of the set
    nl =  graph.nodes()
    size = int(len(nl) * 0.01) 

    #randomly select 1% of the nodes to be on TownSquare 
    tsu = np.random.choice(nl, size)
    print tsu
    return tsu


class BizYouKnow(object):
    """Compute and store cosine similarity matrix, allow methods to operate on the stored matrix.
       Data Sources: block diagram (distance matrix)
                     biz info (combined biz,comm, node info) 
                     directed relationship graph(adj matrix)  
                     biz on townsqaured
    """


    def __init__(self, bizinfo, graph, tsusers):

        # datafile consisting of all business information
        # graph
        self.biz_db = bizinfo

        # a networkx directed graph of all the interactions
        self.global_dgraph = graph

        # dictionary of similarity matrices for each community
        self.num_of_communities = None
        
        # unified dictionary containing information
        # about all comunities
        self.community_data = None 

        ## list of users on ts
        self.tsusers = tsusers


    def precompute(self):

        """ compute the similarity matrix of the nodes with the same 
            community
        """
        print "Precomputing the similarity matrices......... \n"
        communities = self.biz_db['communityid'].unique() 
        self.num_of_communities = len(communities)
        self.community_data = { x:{} for x in communities}

        commid = 7  
        print "Precompute for community: ", commid
        self.calculate_community_datastructures(commid)


   
    def calculate_community_datastructures(self, comm_id):
        """ calculate the familiarity matrix for each community 
            and then append it to the list of matrices
        """
        # df of the community
        subdf = self.biz_db[self.biz_db['communityid'] == comm_id]

        # graph of the community
        subgraph = self.global_dgraph.subgraph(subdf.index.tolist())

        # calculate the familiarity matrix of the points
        # in the community
        # community_loc_sim_matrix = self.calculate_loc_sim_matrix(subdf, subgraph)

        self.community_data[comm_id]['df'] = subdf
        self.community_data[comm_id]['graph'] = subgraph
        self.community_data[comm_id]['sim_loc'] = self.calculate_loc_sim_matrix(subdf, subgraph)
        self.community_data[comm_id]['sim_cat'] = self.calculate_cat_sim_matrix_euclid(subdf, subgraph)
        self.community_data[comm_id]['sim_g'] = self.calculate_graph_distance_matrix(subdf, subgraph)
        self.community_data[comm_id]['pageidx'] = list(subdf.index)
        self.community_data[comm_id]['avg_loc'] = self.get_community_location(subdf)
        self.community_data[comm_id]['popularity'] = self.calculate_popularity(subdf, subgraph)
        print "Done"


    def calculate_loc_sim_matrix(self, df, graph):
        """ Given a dataset and subgraph corresponding 
            to a community retrieve a location similarity matrix
        """
        
        print "Calculating the location similarity matrix..."
        m, n = df.shape
        fam_mat = np.zeros((m,m))

        distdf = df[['latitude', 'longitude']].astype(float) 
        #print distdf.shape 

        # normalize the similarity matrix  
        data = (distdf.values - distdf.values.mean(axis=0))/distdf.values.std(axis=0)
        dist = cdist( data, data, metric='euclidean')  
        
        print "Done"
        print "Returning matrix: ", dist.shape
        print "Diagonal check:", dist.diagonal()
        return dist


    def calculate_cat_sim_matrix(self, df, graph):
        print "Calculating similarity matrix based on categories..."
        tags = df.apply(combine, axis=1)

        # use count vectorizer to get the category frequency
        vec = CountVectorizer(tokenizer=tokenize, stop_words = 'english')
        comm_data = vec.fit_transform(tags.values).toarray()
        print "shape of count matrix", comm_data.shape
        #print "type of the count vectorizer", type(comm_data)

        print "type of the array:", type(comm_data)
        #sim_mat = calculate_cosine_similarity(comm_data)
        print sim_mat.shape
        #print "vocab : ", vec.get_feature_names()
        #print "vocabulary", vec.vocabulary #get_feature_names()
        #dist = np.sum(comm_data, axis=0)
        #dist = cdist( data, data, metric='euclidean') 
        dist = 1 - cosine_similarity(comm_data)
        r = np.round(dist, 2)
        #r2 = np.round(sim_mat,2)
        #print "Done"
        #print "Returning matrix: ", sim_mat.shape
        #print "Diagonal check:", r2.diagonal()
        #print "Diagonal check", r.diagonal()
        #print "Row check:", r[0,:]
        return sim_mat

    def calculate_cat_sim_matrix_euclid(self, df, graph):
        print "Calculating similarity matrix based on categories..."
        tags = df.apply(combine, axis=1)

        # use count vectorizer to get the category frequency
        vec = CountVectorizer(tokenizer=tokenize, stop_words = 'english')
        data = vec.fit_transform(tags.values).toarray()
        print "shape of count matrix", data.shape
        #print "type of the count vectorizer", type(comm_data)

        # normalize the similarity matrix  
        # data = (distdf.values - distdf.values.mean(axis=0))/distdf.values.std(axis=0)
        dist = cdist( data, data, metric='euclidean')  
        
        print "Done"
        print "Returning matrix: ", dist.shape
        print "Diagonal check:", dist.diagonal()
        print dist
        return dist


    def calculate_graph_distance_matrix(self, df, graph):
        """ Given two business records, determine the familiarity score of the
            businesses 
        """
        print "Calculating graph distances between nodes..."
        #print nx.info(graph)
        path_length = nx.all_pairs_shortest_path_length(graph) 
        #print type(path_length)
        #print path_length[7530551900]
        return path_length


    def calculate_popularity(self, df, graph):
        """ Calculate popularity of the biz based on like score
            in_degree
        """
        # normalize and then add 
        #print df.columns
        sdf = df[['likecount', 'checkins', 'talking_about_count', 'were_here_count', 'indegree']]
        #print sdf.dtypes
        #sdf['pop'] = sdf.apply(cpop, axis=1)
        # normalize 
        data = (sdf.values - sdf.values.mean(axis=0))/sdf.values.std(axis=0)
        pop = np.sum(data, axis=1) 
        #print pop.shape, sdf.shape
        dictionary = dict(zip(list(sdf.index), pop))
        return dictionary


    def get_most_significant_categories(commdf, num=5):
        cat_dist = get_categories(commdf)
        sorted_dist = sort_dictionary(cat_dist)
        return sorted_dist[-num:]

    def get_community_location(self, df):
        distdf = df[['latitude', 'longitude']].astype(float) 
        avg_loc = distdf.values.mean(axis=0)
        print avg_loc


    def lookup_community(self, biz_id):
        if biz_id not in list(self.biz_db.index):
            return None
        
        user_attr = self.biz_db.loc[biz_id]
        return user_attr['communityid']
    
    def find_nearest_community(self, biz_id):
        """go through the communities in order of the distance 
        find one which has more than n people not already
        on """
        pass  

    
    def lookup_index(self, commid, biz_id):   
        return  self.community_data[commid]['pageidx'].index(biz_id)

    def lookup_pageid(self, commid, indx):    
        return  self.community_data[commid]['pageidx'][indx]


    def get_neighbor_recommendations(self, commid, userid, k):
        """ Given a dictionary of dictionary of graph distances 
            returns a list of ids which have 
            biconnections
        """

        #print "Neighbours reccomendations: "
        reccos = []
        
        G = self.community_data[commid]['graph']
        neighbourhood = G.neighbors(userid)
        #print neighbourhood

        # get all biconnected nodes
        biconns = []
        for n in neighbourhood:
            # user id is in the neighborhood of n
            if userid in G.neighbors(n) and userid != n:
                biconns.append(n)
        #print biconns

       
        pop_dict= {}
        for b in biconns:
            pop = self.community_data[commid]['popularity'][b]
            pop_dict[b] = pop

        # sort the dictionary
        for (b, p) in reversed(sort_dictionary(pop_dict)):
            reason = "biconn with pop:" + str(np.round(p,4))
            reccos.append((b, reason))
        
        # for neighbours not biconnected
        # rank them according to category similarity
        neighbor_reccos = {}
        rest_neighbors = list(set(neighbourhood) - set(biconns))
        
        # rank the neighbours according to similarity in category
        sim_cat_mat = self.community_data[commid]['sim_cat']
        sim_dict = {}
        uindx = self.lookup_index(commid, userid)
        for b in rest_neighbors:
            if b != userid:
                bindx = self.lookup_index(commid, b)
                sim = sim_cat_mat[uindx][bindx]
                sim_dict[b] = sim
        #print "category similarity", sort_dictionary(sim_dict)

        # sort the dictionary
        for (b, cs) in sort_dictionary(sim_dict):
            reason = "neighbor with category match (less is better):" + str(np.round(cs,4))
            reccos.append((b, reason))

        #return k reccomendations
        if len(reccos) >= k:
            return reccos[0:k]
        return reccos

    def get_triadic_recommendations(self, commid, userid, k):
        """ Given a graph give neighbor of neighbor recommendations 
        """
        # get the ego network of the user 2 step
        G = self.community_data[commid]['graph']
        EG = nx.ego_graph(G, userid, 2)
        
        
        nbrs = []
        for n in EG.neighbors(userid):
            if n != userid:
                for non in G.neighbors(n):
                    if non != userid and non in self.tsusers:
                        nbrs.append(n)


        #print "neighbors of neighbors on", nbrs
        # get neighbors and neighbors of neighbors
        # that is not you
        # neighbors = set(EG.neighbors) - set([userid])
        #nons = set(neighbors_of_neighbors)-set([userid])

        # for 
        reccos = []
        for n in nbrs:
            reason = "neighbor who likes a townsquare member"
            reccos.append((n, reason))
        
        if len(reccos) >= k:
            return reccos[0:k]
        return reccos

    
    def get_attribute_reccomendations(self, commid, userid, k):
        """ Returns a set of categorial and distance based recommendations
            taking the path into consideration
        """
        sim_loc = self.community_data[commid]['sim_loc']
        indx = self.lookup_index(commid, userid)
        #print "Index check:", indx , self.community_data[commid]['pageidx'][indx]
        
        sim_loc_vec = sim_loc[indx, :]
        
        # gives a list of indices for the sorted array 
        most_sim_index = np.argsort(sim_loc_vec) 

        #print type(most_sim_index)
        #print most_sim_index.shape
        
        # # get the page id from the list of indices
        dist_reccos = []
        for indx in most_sim_index:
            reason = "distance match: "+ str(sim_loc_vec[indx])
            dist_reccos.append((self.lookup_pageid(commid, indx), reason))
        
        # ignore the zeroth element
        return dist_reccos[1: k]
       


    def get_topk_recommendations(self, commid, userid, k):
       
        reccos = []
        # first get all the biconnected recommendations
        reccos = self.get_neighbor_recommendations(commid, userid, k)
        if len(reccos) >= k:
            return reccos

        # # if less than k then move on to neighbour of neighbour recommendations
        nr = len(reccos)
        rest = k - nr
        if rest > 0:
            reccos.extend(self.get_triadic_recommendations(commid, userid, rest))
      
        # # else rest of the reccomendations are distance based recommendations
        nr = len(reccos)
        rest = k - nr 
        if rest > 0:
            reccos.extend(self.get_attribute_reccomendations(commid, userid, rest))
        
        return reccos

 
    
    def recommend(self, biz_id, n=10):
        
        #print "Recommend top 5 businesses for :", biz_id
        # identify the comm id of the biz
        commid = self.lookup_community(biz_id) 
        reccos = []
        if commid!= None:
            comm_size = self.community_data[commid]['df'].shape[0]
            if comm_size > n:
                #print "Creating a neighbourhood using the community of size:", comm_size 
                #print "Comm id check: ", commid
                reccos =  self.get_topk_recommendations(commid, biz_id, n)
               
        #print "Finding the nearest community"
        #print "Nearest  community found :"
        #reccos = self.get_topk_recommendations(commid, biz_id)
        return reccos

       

    def recommend_for_all(self, commid):
        print "Reccomendations for all nodes in the community: "
        df = self.community_data[commid]['df']
        G = self.community_data[commid]['graph']
        all_reccos = []
        outdf = []
        print "Recommendations for nodes:", len(G.nodes())
        for n in G.nodes():
            # location of the node
            n_row = df.loc[n] 
            n_lat = n_row['latitude']
            n_lon = n_row['longitude']
            n_name = n_row['name']
            n_cat = n_row['category']
            outdf.append([n_lon, n_lat, n_name, n_cat, 'TN Member'])
            rl = self.recommend(n)
            # get the latitude and longitude of the node
            for i, r in rl:
                # 
                i_row = df.loc[i]
                i_lat = i_row['latitude']
                i_lon = i_row['longitude']
                i_name = i_row['name']
                i_cat = i_row['category']
                all_reccos.append([n,i, r])
                outdf.append([i_lon, i_lat, i_name, i_cat, r])

        #print all_reccos
        
        # write the data structures to a csv
        with open('../data/generated/reccos.csv', 'w') as fp:
             a = csv.writer(fp, delimiter=',')
             a.writerows(outdf)
       

    def __str__(self):
        return "Familiarity matrix of shape: {}".format(self.sim.shape)



def main():
    print "Getting the data for the recommender"
    df, graph = data_prep(base+infile, base+ingraphfile)
    tsusers = get_tsusers(graph)
    byk  = BizYouKnow(df, graph, tsusers)
    byk.precompute()
    byk.recommend_for_all(7)
    #byk.recommend(74465121112)
    
    #byk.recommend(7530551900)
    # something that does not exist
    #byk.recommend(7)
    #print df.loc[7530551900]['name']

if __name__=='__main__':
    main()