import pandas as pd
from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from collections import defaultdict


# Load your dataset
filepath = "dataset/"
df = pd.read_csv(filepath + "ratings.csv")
movies_df = pd.read_csv(filepath + "movies.csv")

output_file = "output.txt"

# Find the longest movie name
#longest_movie = movies_df.loc[movies_df['title'].str.len().idxmax()]

# Print the result
#print(f"\n\nLongest movie name: {longest_movie['title']}")
#print(f"Length: {len(longest_movie['title'])}")

user_id = 3  # Replace with the desired user ID
movie_ratings = df.merge(movies_df, on="movieId")
user_rated_movies = movie_ratings[movie_ratings['userId'] == user_id]

# Define a Reader - note that rating_scale is customizable
reader = Reader(rating_scale=(0.5, 5.0))

# Load the data from the DataFrame
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

# Split the dataset into train and test sets
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

# Initialize and train the SVD model
model = SVD()
model.fit(trainset)

# Predict ratings for the testset
predictions = model.test(testset)

# Optional: Evaluate accuracy
accuracy.rmse(predictions)

def get_top_n_with_titles(predictions, movies_df, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    
    # Sort predictions for each user and return top n
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    # Add movie titles
    top_n_titles = defaultdict(list)
    for uid, movies in top_n.items():
        for movie_id, rating in movies:
            title_row = movies_df[movies_df['movieId'] == movie_id]
            title = title_row['title'].values[0] if not title_row.empty else "Unknown"
            genres = title_row['genres'].values[0] if not title_row.empty else "Unknown"
            top_n_titles[uid].append((title, rating, genres))

    print(title_row['genres'].values[0])
    return top_n_titles

# Get top 10 recommendations with titles
top_n_titles = get_top_n_with_titles(predictions, movies_df, n=10)

# Show top movies for a user
#user_id = 3
print(f"Top recommendations for user {user_id}:\n")
for title, rating, genres in top_n_titles[user_id]:
    print(f"{title} - {genres} - Predicted Rating: {rating:.2f}")

with open(filepath + output_file, 'w') as file:
    file.write(f"Movies rated by user {user_id}:\n\n")
    file.write(user_rated_movies[['movieId', 'rating', 'title', 'genres']].to_string())
    file.write(f"\n\n\nTop recommendations for user {user_id}:\n\n")
    for title, rating, genres in top_n_titles[user_id]:
        file.write(f"{title} - {genres} - Predicted Rating: {rating:.2f}\n")