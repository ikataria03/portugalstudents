# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction import DictVectorizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
from sklearn.model_selection import GridSearchCV, ParameterGrid

import numpy as np
import matplotlib.pyplot as plt

portuguese = pd.read_csv('student_portuguese_clean.csv')
portuguese

portuguese.corrwith(portuguese["final_grade"]).sort_values().reset_index(name="Correlation")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

#randomforest
all_portuguese_features = portuguese.drop(['grade_1', 'grade_2', 'final_grade'], axis=1)
features = all_portuguese_features
y_pred = portuguese['final_grade']
label_encoders = {}
for column in features.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    features[column] = label_encoders[column].fit_transform(features[column])
X_train, X_test, y_train, y_test = train_test_split(features, y_pred, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
rf_regressor = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
rf_regressor.fit(X_train, y_train)
predictions = rf_regressor.predict(X_test)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print()



#Gradient Boosting
!pip install xgboost
import xgboost as xgb

all_portuguese_features = portuguese.drop(['grade_1', 'grade_2', 'final_grade'], axis=1)
features = all_portuguese_features
y_pred = portuguese['final_grade']
continuous = [
    'age',
    'class_failures',
    'family_relationship',
    'free_time',
    'social',
    'weekday_alcohol',
    'weekend_alcohol',
    'health',
    'absences',
]
encoder = OneHotEncoder(sparse=False, drop='first')
encoded_cats = encoder.fit_transform(features[continuous])
encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(continuous))
features = features.drop(columns=continuous)
features = pd.concat([features, encoded_df], axis=1)
label_encoders = {}
for column in features.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    features[column] = label_encoders[column].fit_transform(features[column])
X_train, X_test, y_train, y_test = train_test_split(features, y_pred, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

xgb_reg = xgb.XGBRegressor(objective ='reg:squarederror',
                           colsample_bytree = 0.3,
                           learning_rate = 0.1,
                           max_depth = 5,
                           alpha = 10,
                           n_estimators = 100)
xgb_reg.fit(X_train, y_train)
predictions = xgb_reg.predict(X_test)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
print("RMSE: ", rmse)

portuguese['final_grade'].describe()

portugal = portuguese.copy()
final_grade_labels = ['fail', 'satisfactory', 'good', 'very good', 'excellent']
final_grade_bins = [0, 10, 14, 16, 18, 20]
portugal['final_grade_bin'] = pd.cut(portugal['final_grade'], bins=final_grade_bins, labels=final_grade_labels, right=False)
absences_labels = ['perfect', 'occasionally', 'frequent', 'excessive']
absences_bins = [0, 1, 5, 10, 35]
portugal['absences_bin'] = pd.cut(portugal['absences'], bins=absences_bins, labels=absences_labels, right=False)
portugal.columns

sns.histplot(data=portugal, x = 'final_grade_bin', stat='percent')
plt.title('Final Grades')
plt.xlabel('Final Grade Categories');

sns.countplot(data=portugal, x='absences_bin');

portugal = portuguese.copy()
final_grade_labels = ['fail', 'satisfactory', 'good', 'very good', 'excellent']
final_grade_bins = [0, 10, 14, 16, 18, 20]
portugal['final_grade_bin'] = pd.cut(portugal['final_grade'], bins=final_grade_bins, labels=final_grade_labels, right=False)
absences_labels = ['perfect', 'occasionally', 'frequent', 'excessive']
absences_bins = [0, 1, 5, 10, 35]
portugal['absences_bin'] = pd.cut(portugal['absences'], bins=absences_bins, labels=absences_labels, right=False)
#all_portuguese_features = portugal.drop(['grade_1', 'grade_2', 'final_grade', 'final_grade_bin'], axis=1)
#features = all_portuguese_features
y_pred = portugal['final_grade_bin']
continuous = [

]
#portugal = portugal[continuous]
#portugal = portugal.drop(['grade_1', 'grade_2', 'final_grade', 'final_grade_bin', 'absences', 'student_id', 'age', 'school'], axis=1)
portugal = portugal[['study_time', 'class_failures', 'mother_education', 'father_education', 'health', 'social', 'internet_access', 'family_support']]
encoder = OneHotEncoder(sparse=False, drop='first')
encoded_cats = encoder.fit_transform(portugal[continuous])
encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(continuous))
portugal = portugal.drop(columns=continuous)
portugal = pd.concat([portugal, encoded_df], axis=1)
label_encoders = {}
for column in portugal.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    portugal[column] = label_encoders[column].fit_transform(portugal[column])
X_train, X_test, y_train, y_test = train_test_split(portugal, y_pred, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = DecisionTreeClassifier(random_state = 42)
clf.fit(X_train, y_train)
print(accuracy_score(clf.predict(X_train), y_train))
print(accuracy_score(clf.predict(X_test), y_test))

portugal = portuguese.copy()
final_grade_labels = ['fail', 'pass']
final_grade_bins = [0, 10, 20]
portugal['final_grade_bin'] = pd.cut(portugal['final_grade'], bins=final_grade_bins, labels=final_grade_labels, right=False)
sns.histplot(data=portugal, x='final_grade_bin', stat='percent')
plt.title('Final Grades')
plt.xlabel('Final Grade Categories');

p = portugal[portugal['final_grade_bin'] == 'pass']
p

#EDA
sns.countplot(x = 'internet_access', data = p)
plt.title('Student\'s Accessibility to Internet');
plt.xlabel('Internet Access');

#EDA
sns.countplot(x = 'parent_status', data = p)
plt.title('Student\'s Parent Status');
plt.xlabel('Parent Status');

portugal = portuguese.copy()
final_grade_labels = [0, 1]
final_grade_bins = [0, 10, 20]
portugal['final_grade_bin'] = pd.cut(portugal['final_grade'], bins=final_grade_bins, labels=final_grade_labels, right=False)
absences_labels = ['perfect', 'occasionally', 'frequent', 'excessive']
absences_bins = [0, 1, 5, 10, 35]
portugal['absences_bin'] = pd.cut(portugal['absences'], bins=absences_bins, labels=absences_labels, right=False)
portugal['mom_higher_education'] = portugal['mother_education'].apply(lambda x: 1 if 'higher education' in x.lower() else 0)
portugal['dad_higher_education'] = portugal['father_education'].apply(lambda x: 1 if 'higher education' in x.lower() else 0)
y_pred = portugal['final_grade_bin']
continuous = [

]
portugal = portugal[['internet_access', 'family_support','social','school_support', 'parent_status', 'family_relationship', 'higher_ed']]
encoder = OneHotEncoder(sparse=False, drop='first')
encoded_cats = encoder.fit_transform(portugal[continuous])
encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(continuous))
portugal = portugal.drop(columns=continuous)
portugal = pd.concat([portugal, encoded_df], axis=1)
label_encoders = {}
for column in portugal.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    portugal[column] = label_encoders[column].fit_transform(portugal[column])
X_train, X_test, y_train, y_test = train_test_split(portugal, y_pred, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = DecisionTreeClassifier(random_state = 42)
clf.fit(X_train, y_train)
print(accuracy_score(clf.predict(X_train), y_train))
print(accuracy_score(clf.predict(X_test), y_test))

from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

precision_train = precision_score(y_train, y_pred_train)
f1_train = f1_score(y_train, y_pred_train)
recall_train = recall_score(y_train, y_pred_train)

precision_test = precision_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)

print("Precision on Training Set:", precision_train)
print("Precision on Test Set:", precision_test)

print("F1 Score on Training Set:", f1_train)
print("F1 Score on Testing Set:", f1_test)

print("Recall Score on Train Set:", recall_train)
print("Recall on Test Set:", recall_test)

