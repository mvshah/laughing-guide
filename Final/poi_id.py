#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, auc
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.feature_selection import SelectKBest
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from feature_format import featureFormat, targetFeatureSplit
import tester

with open("final_project_dataset.pkl", "r") as data_file:
    enron_data = pickle.load(data_file)
    
df = pd.DataFrame(enron_data)
df_new = df.transpose().reset_index()

df_new.columns
print(df_new.info())
print(df.shape)
print()
print("POI & Non POI information")
poi_non_poi = df_new.poi.value_counts()
poi_non_poi.index=['non-POI', 'POI']
poi_non_poi

#### Handle NAN values 

df_new.replace('NaN', np.nan, inplace=True)

print(df_new.info())
print("Amount of NaN values in the dataset: ", df_new.isnull().sum().sum())

df_new[['salary','deferral_payments', 'deferred_income','loan_advances', 'bonus','expenses','long_term_incentive','director_fees', 'other', 'total_payments', 'restricted_stock_deferred', 
'exercised_stock_options','restricted_stock','total_stock_value']] = df_new[['salary','deferral_payments', 'deferred_income','loan_advances', 'bonus','expenses', 
'long_term_incentive','director_fees', 'other', 'total_payments', 'restricted_stock_deferred', 
'exercised_stock_options','restricted_stock','total_stock_value']].fillna(value = 0)

#### Handle and removing outliers

def get_subset_by_IQR(df,column,index):
    q3 = df[column].quantile(0.95)
    iqr = (df[column] > q3)
    for c1, c2 in zip(df[iqr][index], df[iqr][column]):
        print("%-9s %s" % (c1, c2))

print("Outliers by salary: ")
get_subset_by_IQR(df_new,'salary','index')
print()
print("Outliers by bonus: ")
get_subset_by_IQR(df_new,'bonus','index')


#plt.plot(df_new['bonus'], df_new['salary'], );

#from scipy import stats
#df_new[(np.abs(stats.zscore(df_new)) < 3).all(axis=1)]
#df_new.plot.scatter(x = 'salary', y = 'bonus')
print(df_new['salary'].idxmax())
print(df_new['bonus'].idxmax())
## Since the index of TOTAL is 130, we will drop it
df_new.drop(130, inplace=True)

#df_new.plot.scatter(x = 'salary', y = 'bonus')

### Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

## These are the available features in the enron dataset
## Final features will be selected using SelectKBest algorithm in the later
## part of the project and if needed new features will be created

features_list = ['poi','salary', 'deferral_payments', 'total_payments', 
'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 
'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 
'long_term_incentive', 'restricted_stock', 'to_messages', 
'from_poi_to_this_person', 'from_messages', 
'from_this_person_to_poi', 'shared_receipt_with_poi','fraction_from_poi',
'fraction_to_poi', 'percent_of_stock','percent_of_salary'] 


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

df_new['fraction_from_poi'] = df_new['from_poi_to_this_person']/df_new['to_messages']
df_new['fraction_to_poi'] = df_new['from_this_person_to_poi']/df_new['from_messages']
df_new['percent_of_stock'] = df_new['bonus']/df_new['total_payments']
df_new['percent_of_salary'] = df_new['salary']/df_new['total_payments']

print(df_new.info())

df_new.fillna(value='NaN', inplace = True)

# back into dict
data_dict = df_new.to_dict(orient='index')
my_dataset = data_dict
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

transformed_features = StandardScaler().fit_transform(features)

pca = PCA()
#pca.fit(features)
pca_features = pca.fit_transform(transformed_features)

from sklearn.feature_selection import SelectKBest, f_classif, chi2

#ch2 = SelectKBest(chi2, k=2)
selector = SelectKBest(k = 18)
selected_features = selector.fit_transform(pca_features, labels)

from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import cross_val_score

rf_mod = RandomForestClassifier(n_estimators=100)
log_mod = LogisticRegression()
svm_mod = SVC()
clf_mod = DecisionTreeClassifier()
gnb_mod = GaussianNB()
knb_mod = KNeighborsClassifier()

print("The cross validation score for Random Forest : ")
print(cross_val_score(rf_mod, selected_features,labels))
print("The mean score for Random forest is :", cross_val_score(rf_mod, selected_features,labels).mean())
print()
print("The cross validation score for Logistic Regression : ")
print(cross_val_score(log_mod, selected_features,labels))
print("The mean score for Logistic Regression is :", cross_val_score(log_mod, selected_features,labels).mean())
print()
print("The cross validation score for Support Vector machine : ")
print(cross_val_score(svm_mod, selected_features,labels))
print("The mean score for Support Vector machine is :", cross_val_score(svm_mod, selected_features,labels).mean())
print(cross_val_score(clf_mod, selected_features,labels))
print()
print("The cross validation score for Decision Tree : ")
print("The mean score for Decision Tree is :", cross_val_score(clf_mod, selected_features,labels).mean())
print(cross_val_score(gnb_mod, selected_features,labels))
print()
print("The cross validation score for Gaussian Naive Bayes : ")
print("The mean score for Gaussian Naive Bayes is :", cross_val_score(gnb_mod, selected_features,labels).mean())
print()
print("The cross validation score for knn : ")
print(cross_val_score(knb_mod, selected_features,labels))
print("The mean score for knn is :", cross_val_score(knb_mod, selected_features,labels).mean())

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
from sklearn.metrics import fbeta_score, make_scorer
ftwo_scorer = make_scorer(fbeta_score, beta=2)

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


import warnings
warnings.filterwarnings("ignore")
svm = Pipeline([('scaler',StandardScaler()),('kbest',SelectKBest()), ('svm',svm.SVC())])

param_grid = ([{'svm__C': np.logspace(-2, 3, 5),
                'svm__gamma': [0.5],
                'svm__degree':[1,2],
                'svm__kernel': ['rbf','poly'],
                'svm__random_state':[np.random.RandomState(0)],
                'kbest__k':[14,15,16,17,18,20]}])

grid = GridSearchCV(svm, param_grid, scoring=ftwo_scorer)
svm_clf = grid.fit(features_train, labels_train).best_estimator_


pred = svm_clf.predict(features_test)
accuracy = accuracy_score(labels_test, pred)
precisionScore = precision_score(labels_test, pred)
recallScore =  recall_score(labels_test, pred)
F1 = 2 * (precisionScore * recallScore) / (precisionScore + recallScore)

#print('accuracy for support vector machine is ',accuracy)
#print ('precision for support vector machine is', precisionScore)
#print ('recall for support vector machine is', recallScore)
#print ('f1 score for support vector machine is', F1)

print()
print("The metrics for the test_Classifier are as follows - ")
tester.test_classifier(svm_clf, my_dataset, features_list)


from collections import OrderedDict
from operator import itemgetter

feature_list = features_list[1:]

best_features = {}

for i in range(20):
    best_features[feature_list[i]] = svm_clf.get_params()['kbest'].scores_[i]
    
d = OrderedDict(sorted(best_features.items(), key=itemgetter(1), reverse=True))
    
print(d)  


## Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings("ignore")

gnb = GaussianNB()
gnb = Pipeline([('scaler',StandardScaler()),('pca',PCA()), ('gnb',GaussianNB())])

param_grid = ([{'pca__n_components':[14,15,16]}])

grid = GridSearchCV(gnb, param_grid, scoring=ftwo_scorer)
gnb_clf = grid.fit(features_train, labels_train).best_estimator_


pred = gnb_clf.predict(features_test)
accuracy = accuracy_score(labels_test, pred)
precisionScore = precision_score(labels_test, pred)
recallScore =  recall_score(labels_test, pred)
F1 = 2 * (precisionScore * recallScore) / (precisionScore + recallScore)

#print('accuracy for Gaussian Naive Bayes is ',accuracy)
#print ('precision for Gaussian Naive Bayes is', precisionScore)
#print ('recall for Gaussian Naive Bayes is', recallScore)
#print ('f1 score for Gaussian Naive Bayes is', F1)

print()
print("The metrics for the test_Classifier are as follows - ")
tester.test_classifier(gnb_clf, my_dataset, features_list)
#print('accuracy for Gaussian Naive Bayes is ',accuracy)
#print ('precision for Gaussian Naive Bayes is', precisionScore)
#print ('recall for Gaussian Naive Bayes is', recallScore)
#print ('f1 score for Gaussian Naive Bayes is', F1)

print()
print("The metrics for the test_Classifier are as follows - ")
tester.test_classifier(gnb_clf, my_dataset, features_list)  

##k-nn
from sklearn.neighbors import KNeighborsClassifier

knn = Pipeline([('scaler',StandardScaler()),('kbest',SelectKBest()), ('knn', KNeighborsClassifier())])

param_grid = ([{'knn__n_neighbors':[1,3,5,7,9,11],
                'kbest__k':[14,16,18,20]}])

grid = GridSearchCV(knn, param_grid, scoring=ftwo_scorer)
knn_clf = grid.fit(features_train, labels_train).best_estimator_


pred = knn_clf.predict(features_test)
accuracy = accuracy_score(labels_test, pred)
precisionScore = precision_score(labels_test, pred)
recallScore =  recall_score(labels_test, pred)
F1 = 2 * (precisionScore * recallScore) / (precisionScore + recallScore)

#print('accuracy for k-nn Classifier is ',accuracy)
#print ('precision for k-nn Classifier is', precisionScore)
#print ('recall for k-nn Classifier is', recallScore)
#print ('f1 score for k-nn Classifier is', F1)

print()
print("The metrics for the test_Classifier are as follows - ")
tester.test_classifier(knn_clf, my_dataset, features_list)

from collections import OrderedDict
from operator import itemgetter

feature_list = features_list[1:]

best_features = {}

for i in range(14):
    best_features[feature_list[i]] = knn_clf.get_params()['kbest'].scores_[i]
    
d = OrderedDict(sorted(best_features.items(), key=itemgetter(1), reverse=True))
    
print(d) 

## Decision Tree 
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import randint

dtc = Pipeline([('scaler',StandardScaler()),('kbest',SelectKBest()), ('dtc', DecisionTreeClassifier())])

param_grid = ([{'dtc__class_weight': [None],
                'dtc__criterion': ['gini', 'entropy'],
                'dtc__min_samples_split': [2, 5, 10, 15, 20],
                'dtc__max_depth': [None, 2, 5, 10],
                'dtc__min_samples_leaf': [1, 5, 10],
                'dtc__random_state':[42],
                'kbest__k':[14,15,16,17,18,20]}])

grid = GridSearchCV(dtc, param_grid, scoring=ftwo_scorer)
dtc_clf = grid.fit(features_train, labels_train).best_estimator_


pred = dtc_clf.predict(features_test)
accuracy = accuracy_score(labels_test, pred)
precisionScore = precision_score(labels_test, pred)
recallScore =  recall_score(labels_test, pred)
F1 = 2 * (precisionScore * recallScore) / (precisionScore + recallScore)

#print('accuracy for Decision Tree Classifier is ',accuracy)
#print ('precision for Decision Tree Classifier is', precisionScore)
#print ('recall for Decision Tree Classifier is', recallScore)
#print ('f1 score for Decision Tree Classifier is', F1)

print()
print("The metrics for the test_Classifier are as follows - ")
tester.test_classifier(dtc_clf, my_dataset, features_list)

from collections import OrderedDict
from operator import itemgetter

feature_list = features_list[1:]

best_features = {}

for i in range(15):
    best_features[feature_list[i]] = knn_clf.get_params()['kbest'].scores_[i]
    
d = OrderedDict(sorted(best_features.items(), key=itemgetter(1), reverse=True))
    
print(d) 

### Comparing if adding newe features help performance or not
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import randint

features_list_old = ['poi','salary', 'deferral_payments', 'total_payments', 
'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 
'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 
'long_term_incentive', 'restricted_stock', 'to_messages', 
'from_poi_to_this_person', 'from_messages', 
'from_this_person_to_poi', 'shared_receipt_with_poi'] 

#data_old = featureFormat(my_dataset, features_list_old, sort_keys = True)
#labels_old, features_old = targetFeatureSplit(data_old)

#from sklearn.cross_validation import train_test_split
#features_train_old, features_test_old, labels_train_old, labels_test_old = train_test_split(features_old, labels_old, test_size=0.3, random_state=42)

dtc_old = Pipeline([('scaler',StandardScaler()),('kbest',SelectKBest()), ('dtc_old', DecisionTreeClassifier())])

param_grid = ([{'dtc_old__class_weight': [None],
                'dtc_old__criterion': ['gini', 'entropy'],
                'dtc_old__min_samples_split': [2, 5, 10, 15, 20],
                'dtc_old__max_depth': [None, 2, 5, 10],
                'dtc_old__min_samples_leaf': [1, 5, 10],
                'dtc_old__random_state':[42],
                'kbest__k':[14,15,16,17,18]}])

grid = GridSearchCV(dtc_old, param_grid, scoring=ftwo_scorer)
dtc_old_clf = grid.fit(features_train, labels_train).best_estimator_

#print('accuracy for Decision Tree Classifier is ',accuracy)
#print ('precision for Decision Tree Classifier is', precisionScore)
#print ('recall for Decision Tree Classifier is', recallScore)
#print ('f1 score for Decision Tree Classifier is', F1)

print()
print("The metrics for the test_Classifier without new features are as follows - ")
tester.test_classifier(dtc_old_clf, my_dataset, features_list_old)


### Best matrix

from sklearn.tree import DecisionTreeClassifier
from scipy.stats import randint

dtc = Pipeline([('scaler',StandardScaler()),('kbest',SelectKBest()), ('dtc', DecisionTreeClassifier())])

param_grid = ([{'dtc__class_weight': [None],
                'dtc__criterion': ['gini', 'entropy'],
                'dtc__min_samples_split': [2, 5, 10, 15, 20],
                'dtc__max_depth': [None, 2, 5, 10],
                'dtc__min_samples_leaf': [1, 5, 10],
                'kbest__k':[14,15,16,17,18,20]}])

grid = GridSearchCV(dtc, param_grid, scoring=ftwo_scorer)
clf = grid.fit(features_train, labels_train).best_estimator_


pred = dtc_clf.predict(features_test)
accuracy = accuracy_score(labels_test, pred)
precisionScore = precision_score(labels_test, pred)
recallScore =  recall_score(labels_test, pred)
F1 = 2 * (precisionScore * recallScore) / (precisionScore + recallScore)

#print("The accuracy for random forest is {}".format(accuracy_score(Y_test, rf_preds)))
#print("The accuracy for logistic regression is {}".format(accuracy_score(Y_test, log_preds)))
#print("The accuracy for svm is {}".format(accuracy_score(Y_test, svm_preds)))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

tester.dump_classifier_and_data(clf, my_dataset, features_list)