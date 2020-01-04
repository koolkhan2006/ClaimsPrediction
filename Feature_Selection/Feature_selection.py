import pandas as pd
import numpy as np
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFECV, RFE, SelectKBest, f_classif
from sklearn.model_selection import train_test_split as tts, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from itertools import compress

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
print("Concatenating numerical and categorical data frame")
print("*"*50)
X= pd.concat([X_number,X_Category],1)
print(X.shape)

print("*"*50)
print("Remove the outliers")
print("*"*50)
# def remove_outlier(X,y,z):
#     scaler = StandardScaler()
#     scaled_X = pd.DataFrame(scaler.fit_transform(X), columns=list(X))
#     for column_name in list(X):
#         X = X.drop(scaled_X[scaled_X[column_name] <= -3].index)
#         X = X.drop(scaled_X[scaled_X[column_name] >= 3].index)
#         y = y.drop(scaled_X[scaled_X[column_name] <= -3].index)
#         y = y.drop(scaled_X[scaled_X[column_name] >= 3].index)
#         z = z.drop(scaled_X[scaled_X[column_name] <= -3].index)
#         z = z.drop(scaled_X[scaled_X[column_name] >= 3].index)
#         scaled_X = scaled_X.drop(scaled_X[scaled_X[column_name] <= -3].index)
#         scaled_X = scaled_X.drop(scaled_X[scaled_X[column_name] >= 3].index)
#         X = X.reset_index(drop=True)
#         y = y.reset_index(drop=True)
#         z = z.reset_index(drop=True)
#         scaled_X = scaled_X.reset_index(drop=True)
#     return(X,y,z)
# num,y,cat = remove_outlier(num,y,cat)

print("*"*50)
print("Apply Logistic Regression")
print("*"*50)
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X,y)
# y_pred = log_reg.predict(X_test_scale)
print(log_reg.score(X,y))

print("*"*50)
print("Apply Logistic Regression Cross validation")
print("*"*50)
log_regcv = LogisticRegressionCV()
log_regcv.fit(X,y)
print(log_regcv.score(X,y))

print("*"*50)
print("Apply Logistic Regression with class weight parameter to address the class imbalance issue."
      "After changing the class weight recall parameter of classification report has been increased.")
print("*"*50)
log_reg = LogisticRegression(random_state=42,class_weight="balanced",C=1000)
X_train, X_test, y_train, y_test = tts(X,y,test_size = 0.25, random_state = 42)
model = log_reg.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy_score(y_test,y_pred)
print (classification_report(y_test,y_pred))
print (confusion_matrix(y_test,y_pred))

print("*"*50)
print("Find the feature importance using the model coefficients")
print("*"*50)
features = list(X)
feature_weights = np.abs(log_regcv.coef_).tolist()[0]
d = dict(zip(features, feature_weights))
d = pd.DataFrame(list(d.items()), columns=["features", "ranking"])
d = d.sort_values(["ranking"], ascending=False)
lst_top_30_features = d[:30]
print(lst_top_30_features)

print("*"*50)
print("Find the ROC AUC score")
print("*"*50)
print (roc_auc_score(y_test,y_pred))

# print("*"*50)
# print("Statistical model for sm to check for P value(Wald test) and to get the Mc faddens Pseudo R2 score")
# print("*"*50)
# X_sm = sm.add_constant(X)
# model = sm.Logit(y,X_sm).fit()
# drop = list(X_sm)
# drop.remove("Age")
# X_new = X_sm[drop]
# model = sm.Logit(y,X_new).fit()
# model.summary()

print("*"*50)
print("Using SelectKBest library to get the best features using the statistical analysis for numerical and categorical input data and using score functions as f_classif"
      "because output is classifier")
print("*"*50)
skb = SelectKBest(score_func=f_classif, k = 10)
skb.fit(X,y)
print(skb.pvalues_)
features = list(X)
feature_pvalues = np.abs(skb.pvalues_).tolist()
d = dict(zip(features, feature_pvalues))
d = pd.DataFrame(list(d.items()), columns=["features", "P_values_ranking"])
d = d.sort_values(["P_values_ranking"], ascending=True) # asceneding is true because we need lowest p valus feature.
lst_top_30_features = d[:30]
print(lst_top_30_features)

print("*"*50)
print("Feature selection using the Recursive feature elimination")
print("*"*50)
log_reg = LogisticRegression()
# rfe_cv = RFECV(estimator=log_reg, cv=10)
rfe = RFE(estimator=log_reg, n_features_to_select=30)
rfe.fit(X,y)
boolean = rfe.get_support().tolist()
features_3 = list(compress(list(X), boolean))
print(features_3)

print("*"*50)
print("Using GridSearch CV to the best estimaor for Decision tree")
print("*"*50)
model2 = DecisionTreeClassifier(random_state=42)
params = {"criterion":["gini", "entropy"], "max_depth":np.arange(2,8),"min_samples_split":np.arange(0.01, 0.1, 0.01)}
model2_cv = GridSearchCV(estimator=model2, param_grid = params, cv = 5)
model2_cv.fit(X,y)
print(model2_cv.best_score_)
print(model2_cv.best_params_)
model2= model2_cv.best_estimator_
print(model2)
print(model2.feature_importances_)

print("*"*50)
print("Executing Random Forest classifier")
print("*"*50)
rfc = RandomForestClassifier(random_state=42, criterion = "gini", max_depth=7, min_samples_split=0.01, oob_score = True, n_estimators = 40, class_weight = "balanced")
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
rfc.score(X_test,y_test)
print (classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(rfc.feature_importances_)