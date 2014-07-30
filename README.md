Businesses You May Know
===
Recommender engine for existing users of a social networking startup to invite other users to join the platform.


TownSquared is building private online communities for local businesses. The purpose of this project is to help increase the user base of TownSquared using a recommendation engine that prompts current users to invite other businesses they might know to join the platform.  I used the Facebook business to business likes to infer relationship between them and that is then used to build the recommendation engine. 

Data sources used included 
         Facebook profiles of businesses. 
         Business to Business likes on Facebook. 
    
Process

    Data pipeline
        clean_data.py
    SNA on B2B like graph, reduce the size of the graph by sampling.
        graph_sample.py
        graph_stats.py
        sna.py

    Find latent communities in the graph based on the structural properties of the graph.
        graph_partition.py
    
    Analyze each of the communities individually.
        community_analysis.py

    Recommend users based on a combination of distance similarity, category similarity, and graph distances.
        recommend.py
    

My first step was to perform social network analysis on the whole graph. I analyzed a number of ways to break the problem down by sampling and partitioning and found that structure based community detection gave the best results. The communities partition the graph such that the interactions within the communities can be analyzed at depth. I then build similarity matrices at the community level to perform recommendations. 