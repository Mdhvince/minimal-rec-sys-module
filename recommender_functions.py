import numpy as np
import pandas as pd


#movie_content = np.array(movies.iloc[:,4:])
#movie_content_transpose = np.transpose(movie_content)
#dot_prod = movie_content.dot(movie_content_transpose)

def find_similar_items(item_id, df_items, dot_prod,
	item_id_colname='item_id', item_name_colname='item'):
	'''
	INPUT:
	- item_id: an item id * (int)
	- df_items: the item dataframe * (Pandas dataframe object)
	- dot_prod: the dot product matrix from the 
	create_dot_product_matrix() * (Numpy array)
	- item_id_colname: the name of the item id column * (str)
	Default: 'item_id'
	- item_name_colname: the name of the item name column * (str)
	Default: 'item'

	OUTPUT:
	- An array of the most similar items by title
	'''

	try:
		# find the row of each movie id
		item_idx = np.where(df_items[item_id_colname] == item_id)[0][0]
	    
		# find the most similar movie indices - to start I said they need to be the same for all content
		similar_idxs = np.where(dot_prod[item_idx] == np.max(dot_prod[item_idx]))[0]
	    
		# pull the movie titles based on the indices
		similar_items = np.array(df_items.iloc[similar_idxs, ][item_name_colname])

	except KeyError:
		print("ERORR !")
		print("Please modifie the item_id_colname argument and/or")
		print("the item_name_colname argument.")
    
	return similar_items