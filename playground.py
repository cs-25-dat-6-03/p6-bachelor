import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

# File handling
filepath = "dataset/" 
movies = pd.read_csv(filepath + "movies.csv")
ratings = pd.read_csv(filepath + "ratings.csv")
