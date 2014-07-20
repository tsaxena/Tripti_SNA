import pandas as pd
import numpy as np

def get_business_id_dictionary():
	#read in the directory
	df = pd.read_csv('../data/try.out', index_col = 0)
	print df.head()

	directory = df['id'].to_dict()
	return directory
	

def get_interaction_graph():
	# recode the like nodes according to the directory
	df = pd.read_csv('../data/fb_business_page_likes.csv', escapechar='\\')
	df = df[['liker_page_id', 'likee_page_id']]
	df.columns = ['Source', 'Target']
	return df


def clean_graph(G):
	directory = get_business_id_dictionary()
	G['Source'] = G['Source'].apply(lambda x: directory.get(x))
	G['Target'] = G['Target'].apply(lambda x: directory.get(x))

	print "length of the original", G.shape
	G = G.dropna()
	print "Shape after dropping na", G.shape 
	#NLG['liker_page_id'] = NLG['liker_page_id'].astype('int')
	print G.dtypes

	#
	G = G.astype(int)
	return G

def write_clean_graph(G, name, style='csv'):
	if style == 'networkx':
		G.columns = ['Source', 'Target']
		G.to_csv(name, index=False, sep=',', header=False)
	elif style == 'snap':
		G.to_csv(name, index=False, sep=' ', header=False)
	else:
		G.to_csv(name, index=False)

def main():
	UG = get_interaction_graph()
	G = clean_graph(UG)
	#write_clean_graph(G, '../data/clean.graph')
	write_clean_graph(G, '../data/like_graph.snap', 'snap')
	

if __name__ == "__main__":
	main()