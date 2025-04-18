from surprise import Dataset, Reader
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise import accuracy
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split

# File handling
filepath = "dataset/"
movies_df = pd.read_csv(filepath + "movies.csv")
ratings_df = pd.read_csv(filepath + "ratings.csv")

df = pd.merge(ratings_df, movies_df[['movieId', 'genres']], on = 'movieId', how = 'left')

user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()
mlb = MultiLabelBinarizer()

df['userId'] = user_encoder.fit_transform(df['userId'])
df['movieId'] = movie_encoder.fit_transform(df['movieId'])

df = df.join(pd.DataFrame(mlb.fit_transform(df.pop('genres').str.split('|')), columns = mlb.classes_, index = df.index ))
     
df.drop(columns = "(no genres listed)", inplace = True)

train_df, test_df = train_test_split(df, test_size = 0.2)

reader = Reader(rating_scale = (0.5, 5))
data = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()

model_svd = SVD()
model_svd.fit(trainset)

predictions_svd = model_svd.test(trainset.build_anti_testset())
accuracy.rmse(predictions_svd)

def get_top_n_recommendations(user_id, n=5):
    user_movies = df[df['userId'] == user_id]['movieId'].unique()
    all_movies = df['movieId'].unique()
    movies_to_predict = list(set(all_movies) - set(user_movies))

    user_movie_pairs = [(user_id, movie_id, 0) for movie_id in movies_to_predict]
    predictions_cf = model_svd.test(user_movie_pairs)

    top_n_recommendations = sorted(predictions_cf, key = lambda x: x.est)[:n]

    for pred in top_n_recommendations:
       predicted_rating = pred.est
    print(predicted_rating)


    top_n_movie_ids = [int(pred.iid) for pred in top_n_recommendations]

    top_n_movies = movie_encoder.inverse_transform(top_n_movie_ids)

    return top_n_movies

user_id = 1
recommendations = get_top_n_recommendations(user_id)
top_n_movies_titles = movies_df[movies_df['movieId'].isin(recommendations)]['title'].tolist()
print(f"Top 5 Recommendations for User {user_id}:")
for i, title in enumerate(top_n_movies_titles, 1):
  print(f"{i}.{title}")