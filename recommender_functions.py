import numpy as np
import pandas as pd

# create function with this in recommender.py
#movie_content = np.array(movies.iloc[:,4:])
#movie_content_transpose = np.transpose(movie_content)
#dot_prod = movie_content.dot(movie_content_transpose)

# arguments for make_recommendations() to feed find_similar_items():
# df_items(movies), item_id_colname, dot_prod, item_name_colname



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