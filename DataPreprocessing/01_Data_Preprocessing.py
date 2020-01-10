import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
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
print(df_original.info())

print("*"*50)
print("Reading in the dataset and stored as df, Keeping a copy of original df as df_original")
print("*"*50)
cols_con = df.select_dtypes(['int64','float64']).drop('Claim',1)
cols_non_con = df.select_dtypes(['object'])

# n=1
# for j in cols_non_con.columns:
#     plt.figure(figsize=(40,10))
#     plt.subplot(3,2,n)
#     sns.countplot(df[j],hue=df['Claim'])
#     plt.title('Count plot for '+j,fontsize=20)
#     n+=1
#     plt.xticks(rotation=90,fontsize=12)
#     plt.show()

n = 1
plt.figure(figsize = (20,16))
for i in cols_con.columns:
    plt.subplot(3,2,n)
    sns.distplot(df[i])
    plt.title('Distribution for '+i,fontsize=20)
    n+=1
    plt.show()

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

