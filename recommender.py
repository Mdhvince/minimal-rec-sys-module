

# create function with this in recommender.py
#movie_content = np.array(movies.iloc[:,4:])
#movie_content_transpose = np.transpose(movie_content)
#dot_prod = movie_content.dot(movie_content_transpose)

# arguments for make_recommendations() to feed find_similar_items():
# df_items(movies), item_id_colname, dot_prod, item_name_colname

# arguments for make_recommendations() to feed get_item_names():
# df_items(movies), item_id_colname, item_name_colname

# arguments for make_recommendations() to feed ranked_df():
# df_reviews(reviews), item_id_colname, rating_col_name, date_col_name