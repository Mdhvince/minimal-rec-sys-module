import numpy as np
import pandas as pd


def find_similar_items(item_id, df_items, item_name_colname, dot_prod):
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


def get_item_names(item_ids, df_items, item_id_colname, item_name_colname):
	'''
	INPUT:
	- item_ids: a list of item ids

	OUTPUT:
	-A list of items names that refere to the item ids given
	'''

	match_item_ids = df_items[item_id_colname].isin(item_ids)
	item_lst = list(df_items[match_item_ids][item_name_colname])
   
	return item_lst


def ranked_df(df_reviews, item_id_colname, rating_col_name, date_col_name):
	"""
	Input:
	- reviews dataframe
    
	Output:
	- ranked df of items based on popularity
	"""
    
	grouped_items = df_reviews.groupby(item_id_colname)
    
	# Get the ratings avg
	mean_ratings = grouped_items[rating_col_name].mean()

	# Get count of rating
	count_ratings = grouped_items[rating_col_name].count()
    
	# Get the date
	last_rating = grouped_items[date_col_name].max()
    
	# Create Dataframes
	mean_ratings = pd.DataFrame(mean_ratings)
	count_ratings = pd.DataFrame(count_ratings)
	last_rating = pd.DataFrame(last_rating)
    
	# Put all together
	mean_ratings['count_ratings'] = count_ratings
	mean_ratings['date'] = last_rating

	# sort by rating, if tie sort by count then date
	# include only items rated at least 5 times
	ranked_items = mean_ratings.sort_values([rating_col_name,
	 										 'count_ratings',
	 										 'date'], ascending=False)
	ranked_items = ranked_items[ranked_items['count_ratings'] > 4]
	ranked_items = ranked_items.reset_index()
    
	return ranked_items


def popular_recommendations(user_id, ranked_items, top_k):
	'''
	INPUT:
	- user_id: the user_id (str) of the individual you are making 
	recommendations for
	- top_k: an integer of the number recommendations you want back

	OUTPUT:
	top_items - a list of the top_k recommended items by item title in
	order best to worst
	'''

	top_items = list(ranked_items[item_id_colname])[:top_k]

	return top_items








