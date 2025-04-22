import pandas as pd

# File handling
filepath = "dataset/"
movies = pd.read_csv(filepath + "movies.csv")
ratings = pd.read_csv(filepath + "ratings.csv")

#print(movies.head())
#print(ratings.head())


# Columns rename
new_ratings = pd.read_csv(filepath + "ratings.csv")
new_ratings.columns = ["User ID", "Movie ID", "RatingS", "Timestamp"]

#print(new_ratings.head())


# Read CSV in chunks
counter = 1
for chunk in pd.read_csv(filepath + "ratings.csv", chunksize=1000):
    #print(chunk)
    if counter == 3:
        break
    counter+=1


# More data analysis
number_of_ratings = len(ratings)
number_of_unique_movies = ratings["movieId"].nunique()
number_of_unique_users = ratings["userId"].nunique()
number_of_unique_ratings = ratings["rating"].drop_duplicates().sort_values(ascending=[False])

#print(f"Number of ratings: {number_of_ratings}")
#print(f"Number of unique movieId's: {number_of_unique_movies}")
#print(f"Number of unique users: {number_of_unique_users}")
rating_counts = ratings["rating"].value_counts().sort_index(ascending=False)
for rating, count in rating_counts.items():
    print(f"{count} of {rating}")
exit(1)
#print(f"Average number of ratings per user: {round(number_of_ratings/number_of_unique_users, 2)}")
#print(f"Average number of ratings per movie: {round(number_of_ratings/number_of_unique_movies, 2)}")


# Highest rated movies
movie_ratings = ratings.merge(movies, on="movieId")
highest_rated = movie_ratings["title"].value_counts()[0:10]

#print(highest_rated)


# Sort by movieId instead of userId
sorted_ratings = movie_ratings.sort_values(by="movieId")

#print(sorted_ratings)


# Show only movieId 1
value1 = sorted_ratings.loc[sorted_ratings["movieId"] == 1]

#print(value1) 


# Lowest and Highest average movie rating
mean_ratings = movie_ratings.groupby("movieId")[["rating"]].mean()
lowest_rated = mean_ratings["rating"].idxmin()
highest_rated = mean_ratings["rating"].idxmax()

#print(movies[movies["movieId"] == lowest_rated])
#print(movies[movies["movieId"] == highest_rated])
