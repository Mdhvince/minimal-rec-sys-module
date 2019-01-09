
# Recommendation engine module

## Motivation
Recommendation engines are one of the most widely used applications of machine learning techniques. I decided to create my own module completely reusable by other.
This Recommender uses FunkSVD to make predictions of exact ratings. And uses either FunkSVD or a Knowledge Based recommendation (highest ranked) to make recommendations for users.  Finally, if given a item,
the recommender will provide items that are most similar as a Content Based Recommender.

## File description
- movies_clean.csv: cleanned movies data to test my module
- reviews_clean.csv: cleanned reviews data to test my module
- recommender.py: class Recommender to use
- recommender_functions.py: functions used by the Recommender class (not used by user)

Full example provided my repo: <a href='https://github.com/Mdhvince/Recommendation_IBM_article/blob/master/rec_eng_ibm.ipynb'>recommender engine for article on IBM Watson Studio platform</a>

## Quick start

/!\ PLEASE MAKE SURE TO READ ALL THE DOCSTRING, THEY GIVES YOU VERY WELL EXPLAINATION ON EACH FUNCTIONS.

#### Read and clean your data
```
import numpy as np
import pandas as pd
import recommender as r

reviews_test = pd.read_csv('reviews_clean.csv')
movies_test = pd.read_csv('movies_clean.csv')
```
#### Create a Recommender object
```
rec = r.Recommender(df_items=movies_test,                       # df that contains all unique items with description and more
                    df_reviews=reviews_test,                    # df that contains interactions between users and items
                    user_item_df=user_item_df,                  # A user-item df
                    item_name_colname='movie',                  # The title column of the df (this can be use with the 1st df or the 2nd, that why I wanted the same name for both)
                    user_id_colname='user_id',                  # The name of the user id column
                    item_id_colname='movie_id',                 # The name of the item id column
                    rating_col_name='rating',                   # The rating column
                    date_col_name='date')                       # The date column
```
#### Fit the data
This function will train the data using a Funk Singular value decomposition, by creating a user matrix U (user by latent feature), an item matrix (latent feature by item) and a Sigma diagonal matrix with the shape (latent feature x latent feature) with the highest (more relevant) latent feature on the upper left and the lowest (less relevant) latent feature on the lower right.
```
rec.fit(iters=1)
```

#### Dot product matrix

To make recommendation: given a user id we want to find similar users to this user. Similarity are found by computing the dot product of users subset with its transpose, the more the result of an user-user pair is high, the more they have in common.
This dot_product_matrix will be use in the `make_recommendations()` function.
```
df_user_similarity = rec.user_item_df.reset_index().replace(np.nan, 0)
def prep_get_similar_user():
    user_content = np.array(df_user_similarity.iloc[:,1:])
    user_content_transpose = np.transpose(user_content)
    dot_prod = user_content.dot(user_content_transpose)
    return dot_prod

dot_product_matrix_user = prep_get_similar_user()
```


#### Make recommendations
According to the `_id_type` or the `_i` (exist or not), this functions will made recommendations using:
- Matrix Factorisation Funk SVD: if the _id_type='user' and the _id of the user exist in the database
- Collaborative Filtering User-Based: if the _id_type='user' and the _id of the user exist in the database
- Ranked Based: if the _id_type='user' and the _id of the user dosn't exist in the database (new user)
- Content-Based: if the _id_type='item' and the _id of the item exist in the database
- error message: if the item doesn't exist

To help you displaying result, you can use this helper function:
```
def display_recommendations(rec_ids, rec_names, message, rec_ids_users, rec_user_articles):
    
    if type(rec_ids) == type(None):
        print(f"{message}")
    
    else:
        dict_id_name = dict(zip(rec_ids, rec_names))
        
        if type(rec_ids_users) != type(None):
            print('Matrix Factorisation SVD:')
            print(f"\t{message}")
            
            for key, val  in dict_id_name.items():
                print(f"\t- ID items: {key}")
                print(f"\tName: {val}\n")

            print('CF User Based:')
            print('\tUser that are similar to you also seen:\n')
            for i in rec_user_articles[:5]:
                print(f"\t- {i}")
        else:
            print(f"\t{message}")
            dict_id_name = dict(zip(rec_ids, rec_names))
            for key, val  in dict_id_name.items():
                print(f"\t- ID items: {key}")
                print(f"\tName: {val}\n")
```
The you can simply run
```
rec_ids, rec_names, message, rec_ids_users, rec_user_articles = rec.make_recommendations(_id=3,
                                                                                         dot_prod_user= dot_product_matrix_user, #the matrix that you have created before
                                                                                         tfidf_matrix=tfidf_matrix, # the matrix that you should create to find similar movies
                                                                                         _id_type='user',
                                                                                         rec_num=5)
display_recommendations(rec_ids, rec_names, message, rec_ids_users, rec_user_articles)
```

Full example provided my repo: <a href='https://github.com/Mdhvince/Recommendation_IBM_article/blob/master/rec_eng_ibm.ipynb'>recommender engine for article on IBM Watson Studio platform</a>


## Interact with the project
Feel free to clone the repo and do your own recommendations, If you find something interesting that I not mentionned, comment or feel free to contact me.
Please report any bugs.

## Authors
Medhy Vinceslas

## License
Project under the <a href='https://choosealicense.com/licenses/cc0-1.0/'>CC0-1.0</a> License

## Acknowledgement
Thank you to the @udacity staff for giving me the opportunity to build my own recommendation engine module.



```python

```
