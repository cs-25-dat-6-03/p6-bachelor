import numpy as np
import pandas as pd
from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate, train_test_split

# File handling
filepath = "dataset/"
movies = pd.read_csv(filepath + "movies.csv")
ratings = pd.read_csv(filepath + "ratings.csv")
movie_ratings = ratings.merge(movies, on="movieId")

print(ratings)

reader = Reader(rating_scale=(0.5, 5.0))

data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)

trainset, testset = train_test_split(data, test_size=.25)

model = SVD(n_factors=100)
model.fit(trainset)

user_id = 608
movie_id = 1

result = model.predict(user_id, movie_id)
print(f"\nPredicted rating for user {user_id} and movie {movie_id}: {result.est:.2f}")
