import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

from LogisticRegression import LogisticRegression

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

print("*"*50)
print("Countplot")
print("*"*50)
# n=1
# for j in cols_non_con.columns:
#     plt.figure(figsize=(40,10))
#     plt.subplot(3,2,n)
#     sns.countplot(df[j],hue=df['Claim'])
#     plt.title('Count plot for '+j,fontsize=20)
#     n+=1
#     plt.xticks(rotation=90,fontsize=12)
#     plt.show()

print("*"*50)
print("Distribution Plot")
print("*"*50)
# n = 1
# plt.figure(figsize = (20,16))
# for i in cols_con.columns:
#     plt.subplot(3,2,n)
#     sns.distplot(df[i])
#     plt.title('Distribution for '+i,fontsize=20)
#     n+=1
#     plt.show()

print("*"*50)
print("Boxplot")
print("*"*50)
# n=1
# plt.figure(figsize=(15,10))
# for box in cols_con.columns:
#     plt.subplot(2,2,n)
#     plt.boxplot(cols_con[box])
#     plt.title('Boxplot for '+box,fontsize=20)
#     n+=1
#     plt.show()

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
print("Check for the negative duration")
print("*"*50)
print(df[df['Duration'] < 0])

print('Mean DURATION for destination INDONESIA',df[(df['Agency'] == 'JZI') &(df['Agency Type'] == 'Airlines')&(df['Product Name'] == 'Basic Plan')&(df['Destination'] == 'INDONESIA')& (df['Claim'] ==0)& (df['Net Sales'] ==18.0)& (df['Commision (in value)'] ==6.3)]['Duration'].mean())
print('Mean DURATION for destination BANGLADESH',df[(df['Agency'] == 'JZI') &(df['Agency Type'] == 'Airlines') &(df['Product Name'] == 'Basic Plan')&(df['Destination'] == 'BANGLADESH')& (df['Claim'] ==0)& (df['Net Sales'] ==22.0)& (df['Commision (in value)'] ==7.7)]['Duration'].mean())
print('Mean DURATION for destination BRUNEI DARUSSALAM',df[(df['Agency'] == 'JZI') &(df['Agency Type'] == 'Airlines') &(df['Product Name'] == 'Basic Plan')&(df['Destination'] == 'BRUNEI DARUSSALAM')& (df['Claim'] ==0)& (df['Net Sales'] ==18.0)& (df['Commision (in value)'] ==6.3)]['Duration'].mean())

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

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 42,stratify = y)

model_lr = LogisticRegression(random_state=42,class_weight='balanced')
model_lr.fit(X_train,y_train)
y_pred_lr = model_lr.predict(X_test)
y_pred_lr_proba = np.array(pd.DataFrame(model_lr.predict_proba(X_test)).iloc[:,1])
model_lr.score(X_test,y_test)