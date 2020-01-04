import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split as tts, GridSearchCV , RandomizedSearchCV

print("*"*50)
print("Read Csv file and check for missing values")
print("*"*50)
df = pd.read_csv("gapminder.csv")
print(df.isnull().sum())

print("*"*50)
print("Reading in the dataset and stored as df, Keeping a copy of original df as df_original")
print("*"*50)
df_original  = df.copy()
print(df.head())
print(df.shape)

print("*"*50)
print("Defining X and y. Basically you want to get predict life from the set of values")
print("*"*50)
X = df.drop(["life"],1)
y = df["life"]

print("*"*50)
print("Label Encoding the categorical values")
print("*"*50)
le = LabelEncoder()
print(X["Region"])
X["Region"]= le.fit_transform(X["Region"])
print(X["Region"])

print("*"*50)
print("Dividing the dataset into train and test")
print("*"*50)
X_train, X_test, y_train, y_test = tts(X,y, test_size = 0.25, random_state = 42)

print("*"*50)
print("Initializing Decision tree Regressor")
print("*"*50)
dtr = DecisionTreeRegressor(random_state=42)
dtr.fit(X_train,y_train)
print(dtr.score(X_test,y_test))

print("*"*50)
print("Initializing Decision tree Regressor with Hyperparameters")
print("*"*50)
dtr = DecisionTreeRegressor(random_state=42, max_depth=4, min_samples_split=0.12, criterion="friedman_mse")
dtr.fit(X_train,y_train)
print(dtr.score(X_test,y_test))

print("*"*50)
print("Get the feature importance directly from the DecisionTreeClassifier instance")
print("*"*50)
print(list(X))
print(dtr.feature_importances_)
df_feature_importance = pd.DataFrame()
df_feature_importance["Features"] = list(X)
df_feature_importance["Values"] = dtr.feature_importances_
print(df_feature_importance.sort_values(["Values"],ascending=False))

print("*"*50)
print("Score calculated above is the r2 score")
print("*"*50)
y_pred = dtr.predict(X_test)
print(r2_score(y_test,y_pred))

print("*"*50)
print("Mean Squared error score")
print("*"*50)
print(mean_squared_error(y_test,y_pred))
