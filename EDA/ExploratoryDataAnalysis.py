import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model.base import LinearClassifierMixin
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler
from scipy import stats
import statsmodels.api as sm
def standard_err(y_true,y_pred):
    gradient, intercept, r_value, p_value, std_err = stats.linregress(y_true,y_pred)
    return std_err

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
print("Apply KNN ")
print("*"*50)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 42)
knn = KNeighborsClassifier(n_neighbors=3,metric='euclidean')
knn.fit(X_train, y_train)
print(knn.score(X_test,y_test))

print("*"*50)
print("Apply KNN with cross validation and Gridsearch CV to get the best estimator")
print("*"*50)
print(cross_val_score(knn,X,y,cv=10))
knn = KNeighborsClassifier()
params = {"n_neighbors":np.arange(1,10,1), "metric":["euclidean", "minkowski", "jaccard", "cosine"]}
knn_cv = GridSearchCV(estimator=knn, param_grid=params, cv = 10)
knn_cv.fit(X,y)
print(knn_cv.best_params_)
print(knn_cv.best_estimator_)

print("*"*50)
print("Seaborn pairplot")
print("*"*50)
# sns.pairplot(df)

print("*"*50)
print("Seaborn heatmap")
print("*"*50)
# df.drop(["Gender","User ID"],1, inplace=True)
# sns.heatmap(df.corr(),annot=True)
# plt.show()

print("*"*50)
print("Histogram to check whether data is linear or not")
print("*"*50)
plt.hist(y)
plt.show()

print("*"*50)
print("There are no hyperparameters to control in Linear Regression. "
       "That comes with regularization. "
      "Seeing the features the line without any feature selection has taken")
print("*"*50)
# features = list(X)
# feature_weights = np.abs(lin_reg.coef_).tolist()
# d = dict(zip(features, feature_weights))
# d = pd.DataFrame(list(d.items()), columns=["features", "ranking"])
# d = d.sort_values(["ranking"], ascending=False)
# print(d)

print("*"*50)
print("Apply Stats model")
print("*"*50)
# X = sm.add_constant(X)
# model = sm.OLS(y,X).fit()
# print(model.summary())
# results_summary = model.summary()
# results_as_html = results_summary.tables[1].as_html()
# pval = pd.read_html(results_as_html, header=0, index_col=0)[0]
# print(pval['P>|t|'][pval['P>|t|']<=0.05])