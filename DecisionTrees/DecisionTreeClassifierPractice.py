import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import  classification_report, confusion_matrix
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
print("Defining X and y. Basically you want to get predict Region from the set of values")
print("*"*50)
X = df.drop(["Region"],1)
y = df["Region"]

print("*"*50)
print("Dividing the dataset into train and test")
print("*"*50)
X_train, X_test, y_train, y_test = tts(X,y, test_size = 0.25, random_state = 42)

print("*"*50)
print("Initializing Decision tree classifier")
print("*"*50)
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_train,y_train)
print(dtc.score(X_test,y_test))

print("*"*50)
print("Get the Classification report and Confusion matrix")
print("*"*50)
y_pred = dtc.predict(X_test)
print (classification_report(y_test,y_pred))
print (confusion_matrix(y_test,y_pred))

print("*"*50)
print("Get the feature importance directly from the DecisionTreeClassifier instance")
print("*"*50)
print(list(X))
print(dtc.feature_importances_)
df_feature_importance = pd.DataFrame()
df_feature_importance["Features"] = list(X)
df_feature_importance["Values"] = dtc.feature_importances_
print(df_feature_importance.sort_values(["Values"],ascending=False))

print("*"*50)
print("Initializing DecisionTreeClassifier with the Hyperparameters")
print("*"*50)
dtc = DecisionTreeClassifier(random_state=42, criterion="entropy", max_depth=10, min_samples_split=0.06,class_weight="balanced")
dtc.fit(X_train,y_train)
print(dtc.score(X_test,y_test))

print("*"*50)
print("Predicting the probability of y")
print("*"*50)
y_pred_proba = dtc.predict_proba(X_test)
print(y_pred_proba)