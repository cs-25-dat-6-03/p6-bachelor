import pandas as pd

# File handling
filepath = "dataset/"
movies = pd.read_csv(filepath + "movies.csv")
ratings = pd.read_csv(filepath + "ratings.csv")


# Removing "|" in genres, and turn it into a list of genres
movies["genres"] = movies["genres"].apply(lambda x: x.split("|"))
print(movies.head())