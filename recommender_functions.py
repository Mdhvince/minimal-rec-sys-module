import numpy as np
import pandas as pd


def find_similar_items(item_id):
	'''
	INPUT:
	- item_id: an item id * (int)

	OUTPUT:
	- An array of the most similar items by title
	'''

	# find all item indices 
	item_idx = np.where(df_items[item_id_colname] == item_id)[0][0]
    
	# find the most similar item indices
	# to start I said they need to be the same for all content
	similar_id = np.where(dot_prod[item_idx] == np.max(dot_prod[item_idx]))[0]
    
	# pull the items titles based on the indices
	similar_items = np.array(df_items.iloc[similar_id, ][item_name_colname])
    
	return similar_items