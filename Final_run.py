import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, precision_score, recall_score
from sklearn.metrics import recall_score,accuracy_score,classification_report,f1_score,confusion_matrix,precision_score,roc_auc_score,roc_curve
from sklearn.metrics.scorer import roc_auc_scorer
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
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
print(model_lr.score(X_test,y_test))

print('Report for logistic model\n',classification_report(y_test,y_pred_lr))
print('precision Score for logistic model is',precision_score(y_test,y_pred_lr,average='weighted'))
print('recall Score for logistic model is',recall_score(y_test,y_pred_lr,average='weighted'))
print('confusion_matrix for logistic model is\n',confusion_matrix(y_test,y_pred_lr))

dt2 = DecisionTreeClassifier(criterion='entropy',max_depth=4,random_state = 12,class_weight='balanced')
dt2.fit(X_train,y_train)
dt_pred = dt2.predict(X_test)
dt_pred_proba = np.array(pd.DataFrame(dt2.predict_proba(X_test)).iloc[:,1])
dt2.score(X_test,y_test)

print('Report for decision tree model\n',classification_report(y_test,dt_pred))
print('ROC AUC score for decision tree model is ',roc_auc_score(y_test,dt_pred_proba))
print('precision Score for decision tree model is',precision_score(y_test,dt_pred,average='weighted'))
print('recall Score for decision model is',recall_score(y_test,dt_pred,average='weighted'))
print('confusion_matrix for decision model is\n',confusion_matrix(y_test,y_pred_lr))


parameters={'max_depth':range(1,3),'criterion':['entropy','gini'],'n_estimators':[51,111,151]}
rf_clf = RandomForestClassifier(class_weight='balanced',random_state=42)
clf_model=GridSearchCV(estimator=rf_clf,param_grid=parameters,scoring='roc_auc')
clf_model.fit(X_train,y_train)
y_pred_rf_gs=clf_model.predict(X_test)
clf_model.score(X_test,y_test)

print(clf_model.best_params_)

from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(random_state = 42,criterion='entropy', max_depth=2,n_estimators= 111,class_weight='balanced')
rf_clf.fit(X_train,y_train)
y_pred_rf = rf_clf.predict(X_test)
y_pred_rf_proba = np.array(pd.DataFrame(rf_clf.predict_proba(X_test)).iloc[:,1])
rf_clf.score(X_test,y_test)

print('Report for randomforest model\n',classification_report(y_test,y_pred_rf))
print('ROC AUC score for randomforest model is ',roc_auc_score(y_test,y_pred_rf_proba))
print('precision Score for randomforest model is',precision_score(y_test,y_pred_rf,average='weighted'))
print('recall Score for randomforest model is',recall_score(y_test,y_pred_rf,average='weighted'))

bagging_clf = BaggingClassifier(rf_clf, random_state=42,n_estimators=100,max_samples=12000)
bagging_clf.fit(X_train,y_train)
y_pred_bagging = bagging_clf.predict(X_test)
y_pred_bagging_proba = np.array(pd.DataFrame(bagging_clf.predict_proba(X_test)).iloc[:,1])
print(bagging_clf.score(X_test,y_test))

print('Report for bagging model\n',classification_report(y_test,y_pred_bagging))
print('ROC AUC score for bagging model is ',roc_auc_score(y_test,y_pred_bagging_proba))
print('precision Score for bagging model is',precision_score(y_test,y_pred_bagging,average='weighted'))
print('recall Score for bagging model is',recall_score(y_test,y_pred_bagging,average='weighted'))

parameters={'learning_rate':[0.1,0.3,0.6,0.8,1.0],'n_estimators':[50,100,150]}
ada_clf = AdaBoostClassifier(base_estimator=dt2,random_state=42)

clf_model_ada=GridSearchCV(estimator=ada_clf,param_grid=parameters,scoring='roc_auc')
clf_model_ada.fit(X_train,y_train)
y_pred_ada_gs=clf_model_ada.predict(X_test)
y_pred_ada_gs_proba = np.array(pd.DataFrame(clf_model_ada.predict_proba(X_test)).iloc[:,1])
clf_model_ada.score(X_test,y_test)

print(clf_model_ada.best_params_)

print('Report for adaboost model\n',classification_report(y_test,y_pred_ada_gs))
print('ROC AUC score for adaboost model is ',roc_auc_score(y_test,y_pred_ada_gs_proba))
print('precision Score for adaboost model is',precision_score(y_test,y_pred_ada_gs,average='weighted'))
print('recall Score for adaboost model is',recall_score(y_test,y_pred_ada_gs,average='weighted'))

# parameters={'learning_rate':[0.1,0.3,0.6,0.8,1.0],
#             'max_depth':range(1,4),'n_estimators':[50,100,150]}
# xgb_clf=XGBClassifier(random_state=42,scale_pos_weight = 34881/589)
#
# clf_model_xgb=GridSearchCV(estimator=xgb_clf,param_grid=parameters,scoring='roc_auc')
# clf_model_xgb.fit(X_train,y_train)
# y_pred_xgb_gs=clf_model_xgb.predict(X_test)
# y_pred_xgb_gs_proba = np.array(pd.DataFrame(clf_model_xgb.predict_proba(X_test)).iloc[:,1])
# clf_model_xgb.score(X_test,y_test)

precision_score_df = pd.DataFrame()
precision_score_df['Models'] = ['Logistic','DecisionTree','RandomForest','Bagging','AdaBoost']
precision_score_df['precision_score'] = [precision_score(y_test,y_pred_lr,average='weighted'),precision_score(y_test,dt_pred,average='weighted'),precision_score(y_test,y_pred_rf,average='weighted'),precision_score(y_test,y_pred_bagging,average='weighted'),precision_score(y_test,y_pred_ada_gs,average='weighted')]

print(precision_score_df)

plt.figure(figsize=(16,7))
plt.bar(precision_score_df['Models'],precision_score_df['precision_score'])
plt.title('precision_score',fontsize=15)
plt.xlabel('Models',fontsize=13)
plt.ylabel('Scores',fontsize=13)
plt.show()

recall_score_df = pd.DataFrame()
recall_score_df['Models'] = ['Logistic','DecisionTree','RandomForest','Bagging','AdaBoost']
recall_score_df['recall_score'] = [recall_score(y_test,y_pred_lr,average='weighted'),recall_score(y_test,dt_pred,average='weighted'),recall_score(y_test,y_pred_rf,average='weighted'),recall_score(y_test,y_pred_bagging,average='weighted'),recall_score(y_test,y_pred_ada_gs,average='weighted')]

plt.figure(figsize=(16,7))
plt.bar(recall_score_df['Models'],recall_score_df['recall_score'])
plt.title('recall_score',fontsize=15)
plt.xlabel('Models',fontsize=13)
plt.ylabel('Scores',fontsize=13)
plt.show()

print(recall_score_df)