

# create function with this in recommender.py
#movie_content = np.array(movies.iloc[:,4:])
#movie_content_transpose = np.transpose(movie_content)
#dot_prod = movie_content.dot(movie_content_transpose)

# arguments for make_recommendations() to feed find_similar_items():
# df_items(movies), item_id_colname, dot_prod, item_name_colname

# arguments for make_recommendations() to feed get_item_names():
# df_items(movies), item_id_colname, item_name_colname

# arguments for make_recommendations() to feed ranked_df():
# df_reviews, item_id_colname, rating_col_name, date_col_name

# arguments for make_recommendations() to feed popular_recommendations():
# ranked_items(will be the name of the variable that store the ranked_df() in make rec,
# item_id_colname

import numpy as np
import pandas as pd
import recommender_functions as rf

class Recommender():

	def __init__(self, df_items, df_reviews,
				 user_id_colname='user_id', item_id_colname='item_id',
				 rating_col_name='rating', date_col_name='date'):
		"""
		Input:
		- df_items: Pandas datafram of items
		- df_reviews: Pandas datafram of items
		"""
		self.df_items = df_items
		self.df_reviews = df_reviews
		self.user_id_colname = user_id_colname
		self.item_id_colname = item_id_colname
		self.rating_col_name = rating_col_name
		self.date_col_name = date_col_name

		assert self.user_id_colname in self.df_reviews.columns, (
			'the user_id_colname given is not in your df_reviews'
			)
		assert self.user_id_colname in self.df_reviews.columns, (
			'the item_id_colname given is not in your df_reviews'
			)
		assert self.rating_col_name in self.df_reviews.columns, (
			'the rating_col_name given is not in your df_reviews'
			)
		assert self.date_col_name in self.df_reviews.columns, (
			'the date_col_name given is not in your df_reviews'
			)




	def fit(self, latent_features=12, learning_rate=0.0001, iters=100):

		self.latent_features = latent_features
		self.learning_rate = learning_rate
		self.iters = iters

		# Create user-item matrix
		user_item = self.df_reviews[[self.user_id_colname,
									 self.item_id_colname,
									 self.rating_col_name,
									 self.date_col_name
									 ]]

		self.user_item_df = (
			self.user_item.groupby(
					[self.user_id_colname, self.item_id_colname]
					)[self.rating_col_name].sum()
					.unstack()
					.reset_index()
					.set_index(self.user_id_colname)
					)
		self.user_item_mat = np.array(self.user_item_df)

		# Set up some useful values for later
		self.n_users = self.user_item_mat.shape[0]
		self.n_items = self.user_item_mat.shape[1]
		self.num_ratings = np.count_nonzero(~np.isnan(self.user_item_mat))

		#### FunkSVD ####

		# initialize the user and item matrices with random values
		user_mat = np.random.rand(self.n_users, self.latent_features)
		item_mat = np.random.rand(self.latent_features, self.n_items)

		sse_accum = 0

		print("Iterations \t\t Mean Squared Error ")

		for iteration in range(self.iters):
			old_sse = sse_accum
			sse_accum = 0

			for i in range(self.n_users):
				for j in range(self.n_items):

					# if the rating exists (so we train only on non-missval)
					if self.user_item_mat[i, j] > 0:
						# compute the error as the actual minus the dot 
						# product of the user and movie latent features
						diff = (
							self.user_item_mat[i, j]
							- np.dot(user_mat[i, :], item_mat[:, j])
						)
						# Keep track of the sum of squared errors for the 
						# matrix
						sse_accum += diff**2

						for k in range(self.latent_features):
							user_mat[i, k] += (
								self.learning_rate * (2*diff*item_mat[k, j])
							)

							item_mat[k, j] += (
								self.learning_rate * (2*diff*user_mat[i, k])
							)

			print(f"{iteration+1} \t\t {sse_accum/self.num_ratings} ")

		# Keep these matrices for later
		self.user_mat = user_mat
		self.item_mat = item_mat

		# Create ranked items
		self.ranked_items = rf.ranked_df(self.df_reviews)


	def predict_rating(self, user_id, item_id):

		try:
			self.user_ids_series = np.array(self.user_item_df.index)
			self.items_ids_series = np.array(self.user_item_df.columns)

			# User row and Movie Column
			user_row = np.where(self.user_ids_series == user_id)[0][0]
			item_col = np.where(self.items_ids_series == movie_id)[0][0]

			# Take dot product of that row and column in U and V 
			# to make prediction
			pred = np.dot(self.user_mat[user_row, :], self.item_mat[:, item_col])

			return pred
		except:
			print('Sorry but the prediction cannot be made because either')
			print('the movie or the user is not present in our database')
			return None


	def make_recommendations(self, _id, _id_type='movie', rec_num=5):

		pass
























