import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import recall_score,accuracy_score,classification_report,f1_score,confusion_matrix,precision_score,roc_auc_score,roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler

print("*"*50)
print("Read Csv file")
print("*"*50)
df = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

print("*"*50)
print("Reading in the dataset and stored as df, Keeping a copy of original df as df_original")
print("*"*50)
df_original  = df.copy()
print(df_original.head())
print(df_original.shape)
print(df_original.info())

print("*"*50)
print("Divide X into categorical and y")
print("*"*50)
X1 = df_test.copy()
X_number1 = X1.select_dtypes(include = np.number)
X_number1 = X_number1.drop(["ID"],1)
X_Category1 = X1.select_dtypes(exclude = np.number)

print("*"*50)
print("Check count for missing values in each column")
print("*"*50)
print(df.isnull().sum())
print(df_test.isnull().sum())

print("*"*50)
print("Create a dataset and remove the last column")
print("*"*50)
X = df.iloc[:,:-1]
y =  df.iloc[:, 10]
print(X)
print(y)

print("Divide X into categorical and y")
print("*"*50)
X_number = X.select_dtypes(include = np.number)
X_number = X_number.drop(["ID"],1)
X_Category = X.select_dtypes(exclude = np.number)
print(X_number.head())
print(X_Category.head())

print("*"*50)
print("Scale the numerical features")
print("*"*50)
scaler = StandardScaler()
X_number = pd.DataFrame(scaler.fit_transform(X_number), columns=list(X_number))
print(X_number.head())
scaler = StandardScaler()
X_number1 = pd.DataFrame(scaler.fit_transform(X_number1), columns=list(X_number1))

print("*"*50)
print("One hot encoding the categorical values")
print("*"*50)
X_Category = pd.get_dummies(X_Category, drop_first=True)
X_Category1 = pd.get_dummies(X_Category1, drop_first=True)
print(X_Category.shape)
print(X_Category1.shape)

print("*"*50)
print("Concatinating numerical and categorical data frame")
print("*"*50)
X = pd.concat([X_number,X_Category],1)
X1 = pd.concat([X_number1,X_Category1],1)
print(X.shape)
print(X1.shape)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 42,stratify = y)

dt2 = DecisionTreeClassifier(criterion='entropy',max_depth=4,random_state = 12,class_weight='balanced')
dt2.fit(X_train,y_train)
dt_pred = dt2.predict(X_test)
dt_pred_proba = np.array(pd.DataFrame(dt2.predict_proba(X_test)).iloc[:,1])
dt2.score(X_test,y_test)

print('Report for decision tree model\n',classification_report(y_test,dt_pred))
print('ROC AUC score for decision tree model is ',roc_auc_score(y_test,dt_pred_proba))
print('precision Score for decision tree model is',precision_score(y_test,dt_pred,average='weighted'))
print('recall Score for decision model is',recall_score(y_test,dt_pred,average='weighted'))

# parameters={'max_depth':range(1,3),'criterion':['entropy','gini'],'n_estimators':[51,111,151]}
# rf_clf = RandomForestClassifier(class_weight='balanced',random_state=42)
# clf_model=GridSearchCV(estimator=rf_clf,param_grid=parameters,scoring='roc_auc')
# clf_model.fit(X_train,y_train)
# y_pred_rf_gs=clf_model.predict(X_test)
# clf_model.score(X_test,y_test)
#
# print(clf_model.best_params_)

# rf_clf = RandomForestClassifier(random_state = 42,criterion='entropy', max_depth=2,n_estimators= 151,class_weight='balanced')
# rf_clf.fit(X_train,y_train)
# y_pred_rf = rf_clf.predict(X_test)
# y_pred_rf_proba = np.array(pd.DataFrame(rf_clf.predict_proba(X_test)).iloc[:,1])
# rf_clf.score(X_test,y_test)
#
# print('Report for randomforest model\n',classification_report(y_test,y_pred_rf))
# print('ROC AUC score for randomforest model is ',roc_auc_score(y_test,y_pred_rf_proba))
# print('precision Score for randomforest model is',precision_score(y_test,y_pred_rf,average='weighted'))
# print('recall Score for randomforest model is',recall_score(y_test,y_pred_rf,average='weighted'))

# bagging_clf = BaggingClassifier(rf_clf, random_state=42,n_estimators=100,max_samples=12000)
# bagging_clf.fit(X_train,y_train)
# y_pred_bagging = bagging_clf.predict(X_test)
# y_pred_bagging_proba = np.array(pd.DataFrame(bagging_clf.predict_proba(X_test)).iloc[:,1])
# print(bagging_clf.score(X_test,y_test))
#
# print('Report for bagging model\n',classification_report(y_test,y_pred_bagging))
# print('ROC AUC score for bagging model is ',roc_auc_score(y_test,y_pred_bagging_proba))
# print('precision Score for bagging model is',precision_score(y_test,y_pred_bagging,average='weighted'))
# print('recall Score for bagging model is',recall_score(y_test,y_pred_bagging,average='weighted'))

# ada_clf = AdaBoostClassifier(base_estimator=dt2,random_state=42,learning_rate=0.3,n_estimators=150)
# ada_clf.fit(X_train,y_train)
# y_pred_adaboost = ada_clf.predict(X_test)
# y_pred_adaboost_proba = np.array(pd.DataFrame(ada_clf.predict_proba(X_test)).iloc[:,1])
# ada_clf.score(X_test,y_test)
#
# print('Report for adaboost model\n',classification_report(y_test,y_pred_adaboost))
# print('ROC AUC score for adaboost model is ',roc_auc_score(y_test,y_pred_adaboost_proba))
# print('precision Score for adaboost model is',precision_score(y_test,y_pred_adaboost,average='weighted'))
# print('recall Score for adaboost model is',recall_score(y_test,y_pred_adaboost,average='weighted'))

# ada_clf = AdaBoostClassifier(base_estimator=dt2,random_state=42,learning_rate=0.3,n_estimators=150)
# ada_clf.fit(X,y)
# y_pred_adaboost_final = ada_clf.predict(X1)
# y_pred_adaboost_proba_final = np.array(pd.DataFrame(ada_clf.predict_proba(X_test)).iloc[:,1])


rf_clf = RandomForestClassifier(random_state = 264,criterion='gini', max_depth=7,n_estimators= 151,class_weight='balanced')
bagging_clf = BaggingClassifier(rf_clf, random_state=42,n_estimators=100,max_samples=11800)
bagging_clf.fit(X,y)
y_valid = bagging_clf.predict(X1)

submission = pd.read_csv('sample_submission.csv')
print(submission['Claim'].shape)
submission['Claim'] = y_valid

output=pd.DataFrame(submission)
output.to_csv(r"results.csv",index=False)