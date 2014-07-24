# -*- coding: utf-8 -*-


""" Clean the data given the biz info and interaction graph 
    1. remove the nan entries 
    2. extract subgraph where all nodes are in biz info
    3. generate a biz directory 
   
    Author : Tripti Saxena
"""

import pandas as pd
import numpy as np
import os.path


genpath = '../data/generated/'
directory = 'biz.directory'
subgraph = 'edgelist'

inpath = '../data/orig/'
bizfile = 'fb_business_information.csv'
graphfile = 'fb_business_page_likes.csv'


def print_stats(df):
    print "[Dataframe Statistics...]"
    print "Shape of the dataframe: ", df.shape
    print "Number of null entries: ", df.isnull().sum()


def generate_biz_directory():
    """ create a business id, page id, name directory """
    bizdf = pd.read_csv(inpath+bizfile, escapechar='\\')
    
    # drop rows with empty column entries
    bizdf = bizdf.dropna()

    # add a column to calculate the gid
    bizdf['gid'] = np.arange(1, bizdf.shape[0]+1)

    # write it out
    print "[Generating the business directory...]"
    # print stats to make sure printing write
    print_stats(bizdf) 
    bizdf.to_csv(genpath+directory, index=False)
    


def generate_subgraph(format):
    """ keep only the entries in the graph that have biz 
        information """

    # get business information
    directorypath = genpath+directory
    if os.path.isfile(directorypath):
       
        bizdata = pd.read_csv( directorypath, escapechar='\\')

        #create a directory of page-id and object-ids
        tempdf = bizdata.set_index('pageid')
        tempdf = tempdf['objectid']
        dictionary = tempdf.to_dict()

        uncgraph = pd.read_csv(inpath+graphfile, escapechar='\\')
        uncgraph = uncgraph.dropna()
        uncgraph['likee_object_id'] = uncgraph.apply(lambda x: dictionary.get(x['likee_page_id']), axis=1)
        cgraph = uncgraph.dropna()
        cgraph = cgraph[['liker_page_id', 'likee_page_id']]
        cgraph.columns = ['Source', 'Target']

       
        print_stats(cgraph)
        if format == 'networkx' :
            print "[Generating a networkX graph...]" 
            cgraph.to_csv(genpath+subgraph+'.ntx', index=False, header=False, sep= ' ')
        else:
            print "[Generating a csv graph...]" 
            cgraph.to_csv(genpath+subgraph+'.csv', index=False)


    else:
        print "Either file is missing or is not readable"
    

def main():
    #write_business_directory()
    #
    generate_biz_directory()
    generate_subgraph('networkx')
    

if __name__=='__main__':
    main()