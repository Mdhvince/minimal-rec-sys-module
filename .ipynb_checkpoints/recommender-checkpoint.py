import numpy as np
import pandas as pd
import recommender_functions as rf

class Recommender():
	'''
    This Recommender uses FunkSVD to make predictions of exact 
    ratings. And uses either FunkSVD or a
    Knowledge Based recommendation (highest ranked) to make
    recommendations for users.  Finally, if given a item,
    the recommender will provide items that are most similar as
    a Content Based Recommender.
    '''

	def __init__(self, df_items, df_reviews, item_name_colname='item',
				 user_id_colname='user_id', item_id_colname='item_id',
				 rating_col_name='rating', date_col_name='date'):
		"""
		Input:
		- df_items: Pandas datafram of items
		- df_reviews: Pandas datafram of items
		"""
		self.df_items = df_items
		self.df_reviews = df_reviews
		self.item_name_colname = item_name_colname
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

		print('Create User-Item matrix...')

		# Create user-item matrix
		user_item = self.df_reviews[[self.user_id_colname,
									 self.item_id_colname,
									 self.rating_col_name,
									 self.date_col_name
									 ]]

		self.user_item_df = (
				user_item.groupby([self.user_id_colname,
								   self.item_id_colname])[self.rating_col_name]
														  .max().unstack()
				)
		self.user_item_mat = np.array(self.user_item_df)

		# Set up some useful values for later
		self.n_users = self.user_item_mat.shape[0]
		self.n_items = self.user_item_mat.shape[1]
		self.num_ratings = np.count_nonzero(~np.isnan(self.user_item_mat))
		self.user_ids_series = np.array(self.user_item_df.index)
		self.items_ids_series = np.array(self.user_item_df.columns)

		print('Train data with Funk Sigular Value Decomposition...')
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
						# product of the user and item latent features
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
		self.ranked_items = rf.ranked_df(self.df_reviews,
										 self.item_id_colname,
										 self.rating_col_name,
										 self.date_col_name)


	def predict_rating(self, user_id, item_id):

		try:
			# User row and Item Column
			user_row = np.where(self.user_ids_series == user_id)[0][0]
			item_col = np.where(self.items_ids_series == item_id)[0][0]

			# Take dot product of that row and column in U and V 
			# to make prediction
			pred = (
				np.dot(self.user_mat[user_row, :], self.item_mat[:, item_col])
			)

			return pred

		except:
			print('Sorry but the prediction cannot be made because either')
			print('the item or the user is not present in our database')

			return None


	def make_recommendations(self, _id, dot_prod,
							 _id_type='movie', rec_num=5):

		if _id_type == 'user':
			if _id in self.user_ids_series:
				message = 'Glad to see you again! recommended for you:\n'
				idx = np.where(self.user_ids_series == _id)[0][0]

				# predict items
				# take the dot product of that row and the V matrix
				preds = np.dot(self.user_mat[idx,:],self.item_mat)

				# pull the top items according to the prediction
				indices = preds.argsort()[-rec_num:][::-1]
				rec_ids = self.items_ids_series[indices]	
				rec_names = rf.get_item_names(rec_ids,
											  self.df_items,
											  self.item_id_colname,
											  self.item_name_colname)

			else:

				message = "Hey, you are new here, this is for you:\n"
				# if we don't have this user, give just top ratings back
				rec_names =rf.popular_recommendations(_id,
													  self.ranked_items,
													  rec_num)
		else:
			if _id in self.items_ids_series:
				message = 'Similar items for this rated item:\n'
				rec_names = (
					list(rf.find_similar_items(_id, 
											   self.df_items,
											   self.item_name_colname,
											   dot_prod))[:rec_num]
				)
			else:
				print("Please update the database with this item")

		return rec_ids, rec_names, message



