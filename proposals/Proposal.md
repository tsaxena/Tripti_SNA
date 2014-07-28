
Project 
=======


## Primary Project

Townsquared is building private online communities for local businesses. That
means they’re amassing a highly unique data set of owners of local restaurants,
small markets, pie stores, etc. The project is building a recommendation engine for TownSquared to increase its user base.


##Population of Interest
The core population of interest will be the small business owners that required a platform to interact with each other and solve community related problems together. We are interested in getting data about b2b interactions between these businesses. 


##Data
The primary source of data is Facebook, Factual and TownSquared. At present I have four files

+ fb_business_information.csv: This file contains general information about businesses that is retrieved from facebook, most of the attributes are self-explanatory. As far as the identifier is concerned its objectid which can be found in other CSV as well.

+ fb_business_page_likes.csv: This file contains information of page likes (in form of facebook ids) by a certain business, again identifier is objectid   

+ ts_ny_business_data.csv: This file contains general information of all the businesses in Townsquared database from NY state, although Townsqd is the source of this data but its actually fetched from Factual. Identifier in this file is objectid. 

+ cw_ny_business_data.csv: This file contains information from Cross Walk which is about the social media/directory pages of businesses on different website like Yelp, factual, facebook, twitter, linkedIn and etc. The identifier is objectid. 


## Project Stages and Steps.
###Stage - I: Build a recommender system for business referral to increase user base.

Steps involved in the project: 
1. Clean up the data – ensure that the likes are between business pages
2. Associate the Facebook pages with businesses in our native business database
OR create new businesses where we are confident that these are real businesses
3. Import it to gephi to do some basic analysis.
4. Create a map reduce algorithm taking into account different factors (location,
company type, social score) gleaned from the quantification of the network etc to
make a prediction of the top 10 businesses to recommend.
   

Questions to be answered by the first stage of the work are: 

###Stage - II: Build a graph database for long term analysis of the social network at TownSquared. Given time I would like to create a graph bases database for analysis of the information. 

+ Data Structures
The data will be stored primariliy in neo4j as a graph database to emphasize the connections between the entities. The major nodes will be
  - Business,
  - People 
 
 Business nodes will be populated with numerous attributes available. Denoting if businesses are already on TownSquare or not, and to figure out how people not on Townsquare can be invited to join. The edges will denote if there relationship between the different businesse. For now the only infromation available between the businesses is the 'like' information.
Other possible sources b2b interaction:
  - business websites.e.g. "we proudly serve starbucks"...kind of information will give us insight into who they interact with. 
  - location at a higher granularity: e.g. business owners in a mall or same building. How do you identify this level of co-location.  
  - company type. e.g. do similar businesses in the same area really like to connect with each other.

+ Initially, a number of simple metrics will be calculated for each company. The specific named methods have not been chosen.
    + Degree of connectedness,
    + Betweenness,
    + 'Density' of local network
    + Tightness of local network (complete triangles)
    + Change in network size over time

Questions to be answered: What are the qualities of the network – e.g. how many edges are there, what
kinds of businesses normally connected, do socially active businesses have more
b2b likes? Create a query tool that allows filtering to visualize nodes and
connections. Community detection problem inferred by connectivity.
