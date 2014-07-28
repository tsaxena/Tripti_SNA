import pandas as pd 
import numpy as np
import networkx as nx
import math
from scipy.spatial.distance import cdist


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
    print "How many location missing ", df[df['latitude'] == 'N'].shape
    print "How many location missing ", df[df['longitude'] == 'N'].shape
    df = df[df['latitude'] != 'N']
    print "Pruned out location: ", df.shape

    # read in th original edgelist as a directed graph
    globalgraph= nx.read_edgelist(graphfile, create_using=nx.DiGraph(), nodetype=int)
    print nx.info(globalgraph)

    print "Keeping it consistent, removing all nodes not in database:"
    pageids = list(df.index)
    prunedglobalgraph = globalgraph.subgraph(pageids)
    print nx.info(prunedglobalgraph)
    #print len(pageids)
    return df, globalgraph



def get_distance_dict(filename):
    g = nx.read_edgelist(filename)
    print "Read in edgelist file ", filename
    print nx.info(g)
    path_length = nx.all_pairs_shortest_path_length(g)
    print len(path_length.keys())
    print path_length

def calculate_cosine_similarity():

    data[np.isnan(data)] = 0.

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

    print "shape of the cosine matrix:", cos.shape
    print cos
    return cos


class BizYouKnow(object):
    """Compute and store cosine similarity matrix, allow methods to operate on the stored matrix.
       Data Sources: block diagram (distance matrix)
                     biz info (combined biz,comm, node info) 
                     directed relationship graph(adj matrix)  
                     biz on townsqaured
    """


    def __init__(self, bizinfo, graph):

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


    def precompute(self):

        """ compute the similarity matrix of the nodes with the same 
            community
        """
        communities = self.biz_db['communityid'].unique() 
        self.num_of_communities = len(communities)
        self.community_data = { x:{} for x in communities}

        print "Precompute the familiarity score for all communities"
        #for commid in self.community_data.keys():
        commid = 2   
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
        community_familiarity_matrix = self.calculate_familiarity_matrix(subdf, subgraph)

        self.community_data[comm_id]['df'] = subdf
        self.community_data[comm_id]['graph'] = subgraph
        self.community_data[comm_id]['sim'] = community_familiarity_matrix
        self.community_data[comm_id]['pageidx'] = list(subdf.index)



    def calculate_familiarity_matrix(self, df, graph):
        #create a similarity matrix
        m, n = df.shape
        fam_mat = np.zeros((m,m))

        #create an nodeid to integer index list
        pageids = list(df.index)
        print type(pageids)
        print len(pageids)

        print "Calculate the familiarity matrix"
        # for i, user1_id in enumerate(pageids[:-1]):
        #      for user2_id in pageids[i+1:]:
        #         user1_indx = pageids.index(user1_id)
        #         user2_indx = pageids.index(user2_id)
        #         familiarity_score = self.compute_familiarity_score(user1_id, user2_id, df, graph)
        #         fam_mat[user1_indx][user2_indx] = familiarity_score
        #         fam_mat[user2_indx][user1_indx] = familiarity_score
        #         print user1_id, user2_id, "Done !"

        # return fam_mat
        # taking only the distance values 

        distdf = df[['latitude', 'longitude']].astype(float) 
        print distdf.shape 

        # normalize   
        data = (distdf.values - distdf.values.mean(axis=0))/distdf.values.std(axis=0)
        

        #.............Euclidean distance .................................................
        dist = cdist( data, data, metric='euclidean')  # -> (nx, ny) distances
        return dist

    
    def fast_dist(a):
        mod_a = np.sqrt(inner1d(a ,a))
        norm_a = a / mod_a[:, None]
        out_fast = inner1d(norm_a[:, None, :], norm_a[None, :, :])
        return out_fast


    def compute_familiarity_score(self, user1_id, user2_id, df, graph):
        """ Given two business records, determine the familiarity score of the
            businesses 
        """
        # too slow 
        #rint user1_id, user2_id
        # user1_attr = df.loc[user1_id]
        # user2_attr = df.loc[user2_id]
        # familiarity_score = math.sqrt((float(user1_attr['latitude'])- float(user2_attr['latitude']))**2 +\
        #                               (float(user1_attr['longitude'])- float(user2_attr['longitude']))**2)

        "find biconnected nodes"

        "find cliques"

        ""
        
        return familiarity_score

    def lookup_community(self, biz_id):
        user_attr = self.biz_db.loc[biz_id]
        if user_attr.shape > 0:
            return user_attr['communityid']
        else:
            return None

    def lookup_index(self, commid, biz_id):   
        return  self.community_data[commid]['pageidx'].index(biz_id)

    def lookup_pageid(self, commid, indx):    
        return  self.community_data[commid]['pageidx'][indx]
        

    def recommend(self, biz_id, n=5):
        
        print "Given two biz_ids return n which are most likely to be familiar "
        # identify the comm id of the biz
        commid = self.lookup_community(biz_id) 
        if commid!= None:
            # look at the 
            print "Creating a neighbourhood using the community"
            print "Comm id check: ", commid
            sim_mat = self.community_data[commid]['sim']
            
            indx = self.lookup_index(commid, biz_id)
            print "Index check:", indx , self.community_data[commid]['pageidx'][indx]
            vec = sim_mat[indx, :]
            most_sim_index = np.argsort(vec)[0:n]
            most_sim_biz = [ self.lookup_pageid(commid, i) for i in most_sim_index]
            sim_biz_df = self.community_data[commid]['df'].loc[most_sim_biz]
            print sim_biz_df['name']

        else:
            print "Finding the nearest community"
        
        # get the similarity matrix of the closest community 
        # calculate similarity with the community
        # return the top 10 

        # most_sim_index = np.argsort(self.sim[self.lookup_index(nces_id),:])[0:n]
        # return self.data.iloc[most_sim_index]

    def __str__(self):
        return "Familiarity matrix of shape: {}".format(self.sim.shape)


def main():
    print "Getting the data for the recommender"
    df, graph = data_prep(base+infile, base+ingraphfile)
    byk  = BizYouKnow(df, graph)
    byk.precompute()
    byk.recommend(7530551900)
    print df.loc[7530551900]['name']

if __name__=='__main__':
    main()