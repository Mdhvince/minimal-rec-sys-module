import numpy as np
import pandas as pd

def find_similar_user(user_id, df_reviews, user_id_colname, dot_prod_user):
	user_idx = np.where(df_reviews[user_id_colname] == user_id)[0][0]

	similar_id = np.where(dot_prod_user[user_idx] == np.max(dot_prod_user[user_idx]))[0]

	similar_users = list(np.array(df_reviews.iloc[similar_id, ][user_id_colname]))

	if user_id in similar_users:
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


def user_user_cf(rec_user_user_ids, user_item_df, df_reviews, item_id_colname, item_name_colname):
	article_ids = []

	for user_id in rec_user_user_ids:
		for i in user_item_df.columns:
			if user_item_df.iloc[user_id][i] > 0:
				article_ids.append(i)

	article_names = list(set(df_reviews[df_reviews[item_id_colname].isin(article_ids)][item_name_colname]))

	return article_names


def ranked_df(df_reviews, item_id_colname, rating_col_name, date_col_name):
    

	# Get count of rating
	nb_ratings = dict(df_reviews.groupby(item_id_colname)[rating_col_name].count())
	df_reviews['count_ratings'] = df_reviews[item_id_colname].map(nb_ratings)

    
	# Get the ratings avg per items
	mean_rate_article = dict(df_reviews.groupby(item_id_colname)[rating_col_name].mean())
	df_reviews['mean_rate_article'] = df_reviews[item_id_colname].map(mean_rate_article)

	# Compute weighted rate
	v = df_reviews['count_ratings']
	m = df_reviews['count_ratings'].quantile(0.90)
	R = df_reviews['mean_rate_article']
	C = df_reviews[rating_col_name].mean()

	df_reviews['rank_score'] = (v/(v+m) * R) + (m/(m+v) * C)
	
	ranked_items = df_reviews.sort_values(by=['rank_score', date_col_name], ascending=False)
    
	return ranked_items


def popular_recommendations(user_id, ranked_items, item_id_colname, top_k):

	top_items = list(ranked_items.drop_duplicates(subset=[item_id_colname])[item_id_colname])[:top_k]

	return top_items








