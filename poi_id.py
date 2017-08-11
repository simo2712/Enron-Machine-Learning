#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import pandas as pd
import matplotlib.pyplot
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import preprocessing
from numpy import mean

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
financial_features = ['salary', 'deferral_payments', 'total_payments',\
                       'loan_advances', 'bonus', 'restricted_stock_deferred',\
                       'deferred_income', 'total_stock_value', 'expenses',\
                       'exercised_stock_options','other', 'long_term_incentive',\
                       'restricted_stock', 'director_fees'] 
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages',\
                  'from_this_person_to_poi', 'shared_receipt_with_poi']
poi_label = ['poi']

# You will need to use more features
features_list = poi_label + financial_features +  email_features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
# Counting the number of data points and the number of poi/non-poi person
# Counting the number of features in the dataset 
print 'The total data points are: ' + str(len(data_dict))

count_poi = 0
count_non_poi = 0

for key in data_dict:
    if data_dict[key]['poi'] == True:
        count_poi += 1
    else:
        count_non_poi += 1
        
print 'The total of poi are: ' + str(count_poi)
print 'The total of non-poi are: ' + str(count_non_poi)

total_features = data_dict[data_dict.keys()[0]].keys()
print 'The total features are: '+ str(len(total_features))

#Creating a dict for the missing values
missing_features = {}
for element in total_features:
    missing_features[element] = 0

for person in data_dict:
    for element in data_dict[person]:
        if data_dict[person][element] == 'NaN':
            missing_features[element] += 1

df_missing_values = pd.DataFrame(missing_features.values(),
                                 index = missing_features.keys(),
                                 columns = ['NaN features'])
df_missing_values["% of NaN features"] = (df_missing_values['NaN features']/len(data_dict)*100).round(decimals = 2)
df_missing_values= df_missing_values.sort(columns=["NaN features","% of NaN features"], ascending = [False, False])
print df_missing_values

### Task 2: Remove outliers

my_dataset = data_dict

#Create Dataframe analyze better the data
my_dataset_df = pd.DataFrame(my_dataset)
my_dataset_df= my_dataset_df.T
my_dataset_df = my_dataset_df.replace('NaN', '')
my_dataset_df[['salary', 'deferral_payments', 'total_payments',
                       'loan_advances', 'bonus', 'restricted_stock_deferred',
                       'deferred_income', 'total_stock_value', 'expenses',
                       'exercised_stock_options','other', 'long_term_incentive',
                       'restricted_stock', 'director_fees','to_messages', 'from_poi_to_this_person', 'from_messages',
                  'from_this_person_to_poi', 'shared_receipt_with_poi']] = my_dataset_df[['salary', 'deferral_payments', 'total_payments',
                       'loan_advances', 'bonus', 'restricted_stock_deferred',
                       'deferred_income', 'total_stock_value', 'expenses',
                       'exercised_stock_options','other', 'long_term_incentive',
                       'restricted_stock', 'director_fees','to_messages', 'from_poi_to_this_person', 'from_messages',
                  'from_this_person_to_poi', 'shared_receipt_with_poi']].apply(pd.to_numeric)

#Boxplot salary
my_dataset_df.boxplot(column = 'salary')
matplotlib.pyplot.show()

print "Largest Values"
print my_dataset_df['salary'].nlargest(n=10)
print "--------------------"
print "Smallest Values"
print my_dataset_df['salary'].nsmallest(n=10)

my_dataset_df = my_dataset_df.drop('TOTAL')
my_dataset.pop( "TOTAL", 0 )
my_dataset_df.boxplot(column = 'salary')
matplotlib.pyplot.show()

print "Largest Values"
print my_dataset_df['salary'].nlargest(n=10)
print "--------------------"
print "Smallest Values"
print my_dataset_df['salary'].nsmallest(n=10)

#Boxplot bonus
my_dataset_df.boxplot(column = 'bonus')
matplotlib.pyplot.show()

print "Largest Values"
print my_dataset_df['bonus'].nlargest(n=10)
print "--------------------"
print "Smallest Values"
print my_dataset_df['bonus'].nsmallest(n=10)

#Scatter plot total payments and total stock values
my_dataset_df.plot(kind = 'scatter', x = 'total_payments', y = 'total_stock_value')
matplotlib.pyplot.show()

print "Largest Values"
print my_dataset_df['total_payments'].nlargest(n=10)
print "--------------------"
print "Smallest Values"
print my_dataset_df['total_payments'].nsmallest(n=10)

#Boxplot other financial feature  and the highest and smallest values
for feature in financial_features:
    if feature != 'bonus' and feature != 'salary' and feature != 'total_payments'\
    and feature != 'total_stock_value':
        my_dataset_df.boxplot(column = feature)
        matplotlib.pyplot.show()
    
        print "Largest Values"
        print my_dataset_df[feature].nlargest(n=5)
        print "--------------------"
        print "Smallest Values"
        print my_dataset_df[feature].nsmallest(n=5)


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

#Define function to do the fraction
def compute_ratio(total_payments, total_stock):
    """Ratio of poi_messages from total messages
    
    This function allows to divide the messages received and send to poi, by the
    number of the total messages received or send.
    
    Args:
        all_messages: all messages received or send
        poi_messages: messages send or receveid by poi   
    """
    if total_payments != 'NaN' and total_stock != 'NaN':
        ratio = total_payments/float(total_stock)
    else:
        ratio = 0
    return ratio

#Define two new variables
for person in my_dataset:
    total_payments = my_dataset[person]['total_payments']
    total_stock = my_dataset[person]['total_stock_value']
    ratio_pay_over_stock = compute_ratio(total_payments, total_stock)
    my_dataset[person]['ratio_pay_over_stock'] = ratio_pay_over_stock

    
new_features = features_list + ['ratio_pay_over_stock']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, new_features, sort_keys = True)
labels, features = targetFeatureSplit(data)


selector = SelectKBest(f_classif, k = 'all')
selector.fit_transform(features, labels)
scores = zip(new_features[1:],selector.scores_)
sorted_scores = sorted(scores, key = lambda x: x[1], reverse=True)


#Create a for loop to understand what is the best number of features

def select_features():
    for number in range(len(new_features)):
        kbest_features_list = poi_label + list(map(lambda x: x[0], sorted_scores))[0:number+1]
        data = featureFormat(my_dataset, kbest_features_list, sort_keys = True)
        labels, features = targetFeatureSplit(data)
        scaler = preprocessing.MinMaxScaler()
        features = scaler.fit_transform(features)
        print "Number of features: " + str(number+1)
        try_classifier(features,labels)

for i in sorted_scores[0:5]:
    print i   
kbest_features_list = poi_label + list(map(lambda x: x[0], sorted_scores))[0:5]   
print kbest_features_list

#Create new labels and features without the two new features
data = featureFormat(my_dataset, kbest_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

def validate_clf(grid_search, features, labels, params, iters=100):
    acc = []
    pre = []
    recall = []
    for i in range(iters):
        features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=i)
        grid_search.fit(features_train, labels_train)
        predictions = grid_search.predict(features_test)
        acc = acc + [accuracy_score(labels_test, predictions)] 
        pre = pre + [precision_score(labels_test, predictions)]
        recall = recall + [recall_score(labels_test, predictions)]
    print "accuracy: {}".format(mean(acc))
    print "precision: {}".format(mean(pre))
    print "recall:    {}".format(mean(recall))
    best_params = grid_search.best_estimator_.get_params()
    for param_name in params.keys():
        print("%s = %r, " % (param_name, best_params[param_name]))
        
# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.grid_search import GridSearchCV

def try_classifier(features, labels):
    #1. Naive Bayes
    from sklearn.naive_bayes import GaussianNB

    nb_clf = GaussianNB()
    nb_param = {}
    nb_grid_search = GridSearchCV(nb_clf, nb_param)

    print("Naive Bayes")
    validate_clf(nb_grid_search, features, labels, nb_param)


    #2. Random Forest
    from sklearn.ensemble import RandomForestClassifier

    rf_clf = RandomForestClassifier(n_estimators=10)
    rf_param = {"max_depth": [10, 20, 50],
                 "min_samples_split": [1,5,10],
                 "min_samples_leaf": [1,5,10],
                 "criterion": ["gini", "entropy"]}
    rf_grid_search = GridSearchCV(rf_clf, rf_param)

    print("Random Forest")
    validate_clf(rf_grid_search, features, labels, rf_param)

    #3. Support Vector Machines
    from sklearn import svm
    svm_clf = svm.SVC()
    svm_param = {'kernel':('linear', 'rbf', 'sigmoid'),\
                 'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                 'C': [0.1, 1, 10, 100, 1000]}
    svm_grid_search = GridSearchCV(estimator = svm_clf, param_grid = svm_param)

    print("SVM")
    validate_clf(svm_grid_search, features, labels, svm_param)

    #4. Logistic Regression
    from sklearn.linear_model import LogisticRegression
    lr_clf = LogisticRegression()
    lr_param = {'tol': [1, 0.1, 0.01, 0.001, 0.0001],
                'C': [0.1, 0.01, 0.001, 0.0001]}
    lr_grid_search = GridSearchCV(estimator = lr_clf, param_grid = lr_param)

    print("Logistic Regression")
    validate_clf(lr_grid_search, features, labels, lr_param)

#try_classifier(features,labels)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, kbest_features_list)