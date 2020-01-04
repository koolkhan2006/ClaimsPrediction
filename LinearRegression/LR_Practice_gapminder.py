from scipy.spatial.distance import euclidean
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("*"*50)
print("Read Csv file")
print("*"*50)
df = pd.read_csv("gapminder.csv")
print(df.head())
print(df.shape)

print("*"*50)
print("One Hot encoding for the categorical data")
print("*"*50)
df = pd.get_dummies(df)
print(df)




