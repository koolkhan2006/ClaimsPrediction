import pandas as pd
import numpy as np
import warnings

from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler

print("*"*50)
print("Read Csv file")
print("*"*50)
df = pd.read_csv("train.csv")

print("*"*50)
print("Reading in the dataset and stored as df, Keeping a copy of original df as df_original")
print("*"*50)
df_original  = df.copy()
print(df_original.head())
print(df_original.shape)

print("*"*50)
print("Check count for missing values in each column")
print("*"*50)
print(df.isnull().sum())

print("*"*50)
print("Create a dataset and remove the last column")
print("*"*50)
X = df.iloc[:,:-1]
y =  df.iloc[:, 10]
print(X)
print(y)

print("*"*50)
print("Divide X into categorical and y")
print("*"*50)
X_number = X.select_dtypes(include = np.number)
X_Category = X.select_dtypes(exclude = np.number)
print(X_number.head())
print(X_Category.head())

print("*"*50)
print("Scale the numerical features")
print("*"*50)
scaler = StandardScaler()
X_number = pd.DataFrame(scaler.fit_transform(X_number), columns=list(X_number))
print(X_number.head())

print("*"*50)
print("One hot encoding the categorical values")
print("*"*50)
X_Category = pd.get_dummies(X_Category, drop_first=True)
print(X_Category.shape)

print("*"*50)
print("Concatinating numerical and categorical data frame")
print("*"*50)
X = pd.concat([X_number,X_Category],1)
print(X.shape)

print("*"*50)
print("Train test Split on this data")
print("*"*50)
lin_reg = LinearRegression()
X_train, X_test , y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=21)
lin_reg.fit(X,y)
# y_pred = lin_reg.predict(X_test)
print(lin_reg.score(X,y))