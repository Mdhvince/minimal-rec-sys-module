import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def find_similar_user(user_id, df_reviews, user_id_colname, dot_prod_user):
	user_idx = np.where(df_reviews[user_id_colname] == user_id)[0][0]

	similar_id = np.where(dot_prod_user[user_idx] == np.max(dot_prod_user[user_idx]))[0]

	similar_users = list(np.array(df_reviews.iloc[similar_id, ][user_id_colname]))

	if user_id in similar_users:
		similar_users.remove(user_id)
    
	return similar_users


def find_similar_items(item_id, df_items, item_id_colname, tfidf_matrix):

	# Compute the cosine similarity matrix
	cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

	# Get the pairwsie similarity scores of all items with that item
	indice_item = df_items[df_items[item_id_colname] == item_id].index[0]
	sim_scores = list(enumerate(cosine_sim[indice_item]))

	# Sort from highest to lowest
	sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

	item_indices = [i[0] for i in sim_scores]

	similar_item_ids = list(df_items[item_id_colname].iloc[item_indices])
    
	return similar_item_ids


def get_item_names(item_ids, df_items, item_id_colname, item_name_colname):
    
	# ordered names
	item_lst = []
	for i in item_ids:
		name = tuple(df_items[df_items[item_id_colname] == i][item_name_colname])[0]
		item_lst.append(name)
    
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








