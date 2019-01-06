import numpy as np
import pandas as pd

def find_similar_user(user_id, df_reviews, user_id_colname, dot_prod_user):
	user_idx = np.where(df_reviews[user_id_colname] == user_id)[0][0]

	similar_id = np.where(dot_prod_user[user_idx] == np.max(dot_prod_user[user_idx]))[0]

	similar_users = list(np.array(df_reviews.iloc[similar_id, ][user_id_colname]))
	similar_users.remove(user_id)
    
	return similar_users


def find_similar_items(item_id, df_items, item_id_colname, dot_prod, window):

	# find item indice 
	item_idx = np.where(df_items[item_id_colname] == item_id)[0][0]
    
	# find the most similar item indices
	# to start I said they need to be the same for all content
	similar_id = np.where( (dot_prod[item_idx] <= np.max(dot_prod[item_idx])) & (dot_prod[item_idx] >= np.max(dot_prod[item_idx])-window) )[0]
    
	# pull the items titles based on the indices
	similar_items = np.array(df_items.iloc[similar_id, ][item_id_colname])
    
	return similar_items


def get_item_names(item_ids, df_items, item_id_colname, item_name_colname):

	match_item_ids = df_items[item_id_colname].isin(item_ids)
	item_lst = list(df_items[match_item_ids][item_name_colname])
   
	return item_lst


def ranked_df(df_reviews, item_id_colname, rating_col_name, date_col_name):
    
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


def popular_recommendations(user_id, ranked_items, item_id_colname, top_k):

	top_items = list(ranked_items[item_id_colname])[:top_k]

	return top_items








