# -*- coding: utf-8 -*-
"""Create a business directory from the graph and business information"""
import pandas as pd
import numpy as np

def get_business_information():
	"""get the information about the businesses"""
	outdf = pd.read_csv('../data/fb_business_information.csv', escapechar='\\')
	outdf = outdf[[ 'pageid','idinformation','name']]
	outdf.columns = ['page_id', 'id', 'name']
	return outdf

def get_interaction_information():
	"""get the information about the businesses"""
	outdf = pd.read_csv('../data/fb_business_page_likes.csv', escapechar='\\')
	outdf = outdf[['likee_page_id', 'idinfo', 'pagename']]
	outdf.columns = ['page_id', 'id', 'name']
	return outdf

def get_merged_data():
	""" create a business id, page id, name directory """
	businesses = get_business_information()
	interactions = get_interaction_information()
	merged = businesses.append(interactions)	
	merged = merged.groupby('page_id').first()

	#renumber the page id's because need 32-bit id's
	#for snap to work
	merged['id'] = np.arange(1, merged.shape[0]+1)

	print merged.head()
	return merged 

def write_business_directory():
	""" create business directory """
	df = get_merged_data()
	df = df.dropna()
	df.to_csv('../data/try.out', index='page_id')
	print df.shape
	print df.isnull().sum()

def main():
	write_business_directory()


if __name__=='__main__':
	main()