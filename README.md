
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
- recsys_main.ipynb: example of how to use the module

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
rec = r.Recommender(df_items=movies_test,
                    df_reviews=reviews_test,
                    item_name_colname='movie',
                    user_id_colname='user_id',
                    item_id_colname='movie_id',
                    rating_col_name='rating',
                    date_col_name='date')
```
#### Fit the data
This function will train the data using a Funk Singular value decomposition, by creating a user matrix U (user by latent feature), an item matrix (latent feature by item) and a Sigma diagonal matrix with the shape (latent feature x latent feature) with the highest (more relevant) latent feature on the upper left and the lowest (less relevant) latent feature on the lower right.
```
rec.fit(iters=1)
```

#### Dot product matrix
Then you need to create a dot product using a subset of your item dataframe, this subset contains only additionnal
feature like genre, years etc. but not informations about movie id, movie name, etc. The dot product is used to find
similar items.
```
def prep_get_similar_items():
    item_content = np.array(movies_test.iloc[:,4:])
    item_content_transpose = np.transpose(item_content)
    dot_prod = item_content.dot(item_content_transpose)
    return dot_prod

dot_product_matrix = prep_get_similar_items()
```

#### Make recommendations
Finally make recommendations using this dot product matrix. Recommendation can be made for existing user or movie using respectively FunkSVD or the dot product matrix (Content Based Recommendation). But also make recommendations for new user by displaying the most popular items (Ranked based recommendation).
```
rec_ids, rec_names, message = rec_loaded.make_recommendations(_id=10,
                                                              dot_prod=dot_product_matrix,
                                                              _id_type='user',
                                                              rec_num=5)
```
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
