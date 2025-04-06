import pandas as pd

# File handling
filepath = "dataset/"
movies = pd.read_csv(filepath + "movies.csv")
ratings = pd.read_csv(filepath + "ratings.csv")

print(ratings.head())


# More data analysis
number_of_ratings = len(ratings)
number_of_unique_movies = ratings["movieId"].nunique()
number_of_unique_users = ratings["userId"].nunique()

#print(f"Number of ratings: {number_of_ratings}")
#print(f"Number of unique movieId's: {number_of_unique_movies}")
#print(f"Number of unique users: {number_of_unique_users}")
#print(f"Average number of ratings per user: {round(number_of_ratings/number_of_unique_users, 2)}")
#print(f"Average number of ratings per movie: {round(number_of_ratings/number_of_unique_movies, 2)}")


# Highest rated movies
movie_ratings = ratings.merge(movies, on="movieId")
highest_rated = movie_ratings["title"].value_counts()[0:10]

#print(highest_rated)


# Lowest and Highest average movie rating
mean_ratings = ratings.groupby("movieId")[["rating"]].mean()
lowest_rated = mean_ratings["rating"].idxmin()
highest_rated = mean_ratings["rating"].idxmax()

#print(movies[movies["movieId"] == lowest_rated])
#print(movies[movies["movieId"] == highest_rated])



