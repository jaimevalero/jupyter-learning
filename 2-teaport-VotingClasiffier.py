#TO-DO
#WORKING ON THIES
import pandas as pd

import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split

import sklearn.cross_validation
from scipy.spatial.distance import pdist, squareform
from sklearn.ensemble import VotingClassifier, RandomForestClassifier , ExtraTreesClassifier , GradientBoostingClassifier

from sklearn.feature_selection import SelectFwe, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
from sklearn.pipeline import Pipeline

from tpot.builtins import StackingEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

df = pd.read_csv('sources/cars.csv')

# Preprocessiing

# Replace 0s with median value
df['Horsepower']=df['Horsepower'].replace(0,df['Horsepower'].mean())
df['MPG']=df['MPG'].replace(0,df['MPG'].mean())

# Convert origin to numerical values

le = preprocessing.LabelEncoder()
le.fit(df['Origin'])
list(le.classes_)
labels = le.transform(df['Origin'])

# We add features
sample_df = pd.DataFrame()
sample_df['Horsepower'] = df['Horsepower']
sample_df['Weight']     = df['Weight']
sample_df['MPG']     = df['MPG']
sample_df['Displacement']     = df['Displacement']
sample_df['Acceleration']     = df['Acceleration']
sample_df['Model']     = df['Model']





# We hot encoding the cylinder columns
origin_dummies = pd.get_dummies( df['Cylinders'] )

# Add origin encoding
for origin_column in list(origin_dummies):
   sample_df[ origin_column ] = origin_dummies[ origin_column ]


X_train, X_test, y_train, y_test = train_test_split( sample_df, labels,train_size=0.7)



from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

#pipe_svc = Pipeline([('scl', StandardScaler()),
#            ('clf', SVC(random_state=1))])


#param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

#param_grid = [{'clf__C': param_range,
#               'clf__kernel': ['linear']},
#                 {'clf__C': param_range,
#                  'clf__gamma': param_range,
#                  'clf__kernel': ['rbf']}]

#gs = GridSearchCV(estimator=pipe_svc,
#                  param_grid=param_grid,
#                  scoring='accuracy',
#                  cv=10,
#                  n_jobs=2)

#gs = gs.fit(X_train, y_train)

#print(gs.best_score_)
#print(gs.best_params_)



#print('sssss')



# 0.74
clf1 = exported_pipeline = make_pipeline(
    StackingEstimator(estimator=ExtraTreesClassifier(max_features=0.8500000000000001, min_samples_leaf=2, min_samples_split=19, n_estimators=100)),
    GradientBoostingClassifier(learning_rate=1.0, max_depth=1, max_features=0.7000000000000001, min_samples_leaf=2, min_samples_split=8, n_estimators=100, subsample=1.0)
)

#078
clf2 = make_pipeline( SVC ( C = 1000.0, gamma = 0.01 ,kernel= 'rbf' ))
#0.86
clf3 = make_pipeline(
    SelectFwe(score_func=f_classif, alpha=0.04),
    RandomForestClassifier(criterion="entropy",  max_features=0.6000000000000001, min_samples_split=5, n_estimators=100)
)

# 0.82
#clf4 = exported_pipeline = make_pipeline(
#    StackingEstimator(estimator=LogisticRegression(C=1.0, dual=True)),
#    RandomForestClassifier(max_features=0.6000000000000001, min_samples_leaf=20, min_samples_split=18)
#)

#eclf1 = VotingClassifier(estimators=[
#         ('lr', clf1), ('rf', clf2), ('gnb', clf3), ('rnd', clf4)], voting='hard')
eclf1 = VotingClassifier(estimators=[
         ('lr', clf1), ('gnb', clf2), ('rnd', clf3)], voting='hard')
eclf1 = eclf1.fit(X_train, y_train)
print(eclf1.score(X_test, y_test))

model1 = clf1.fit(X_train, y_train)
print(model1.score(X_test, y_test))

model2 = clf2.fit(X_train, y_train)
print(model2.score(X_test, y_test))

model3 = clf3.fit(X_train, y_train)
print(model3.score(X_test, y_test))

#model4 = clf4.fit(X_train, y_train)
#print(model4.score(X_test, y_test))

#tpot = TPOTClassifier(generations=20, population_size=50, verbosity=2)
#tpot.fit(X_train, y_train)
#tpot.export('tpot_cars_pipeline.py')

#tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
#Best pipeline: GradientBoostingClassifier(RobustScaler(input_matrix), GradientBoostingClassifier__learning_rate=1.0, GradientBoostingClassifier__max_depth=5, GradientBoostingClassifier__max_features=0.25, GradientBoostingClassifier__min_samples_leaf=DEFAULT, GradientBoostingClassifier__min_samples_split=17, GradientBoostingClassifier__n_estimators=100, GradientBoostingClassifier__subsample=0.7)
# 0.770491803279

#tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2)
#Best pipeline: ExtraTreesClassifier(input_matrix, ExtraTreesClassifier__bootstrap=False, ExtraTreesClassifier__criterion=DEFAULT, ExtraTreesClassifier__max_features=0.45, ExtraTreesClassifier__min_samples_leaf=1, ExtraTreesClassifier__min_samples_split=7, ExtraTreesClassifier__n_estimators=DEFAULT)
#0.762295081967

#Sin MPG
#tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
#Best pipeline: ExtraTreesClassifier(input_matrix, ExtraTreesClassifier__bootstrap=DEFAULT, ExtraTreesClassifier__criterion=gini, ExtraTreesClassifier__max_features=0.45, ExtraTreesClassifier__min_samples_leaf=1, ExtraTreesClassifier__min_samples_split=6, ExtraTreesClassifier__n_estimators=DEFAULT)
#0.754098360656

# All features set
#tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
#Best pipeline: GradientBoostingClassifier(input_matrix, GradientBoostingClassifier__learning_rate=0.1, GradientBoostingClassifier__max_depth=6, GradientBoostingClassifier__max_features=0.35, GradientBoostingClassifier__min_samples_leaf=15, GradientBoostingClassifier__min_samples_split=16, GradientBoostingClassifier__n_estimators=100, GradientBoostingClassifier__subsample=0.75)
#0.819672131148

# All features set
# tpot = TPOTClassifier(generations=10, population_size=20, verbosity=2)
# Best pipeline: GradientBoostingClassifier(input_matrix, GradientBoostingClassifier__learning_rate=0.1, GradientBoostingClassifier__max_depth=9, GradientBoostingClassifier__max_features=0.3, GradientBoostingClassifier__min_samples_leaf=1, GradientBoostingClassifier__min_samples_split=17, GradientBoostingClassifier__n_estimators=100, GradientBoostingClassifier__subsample=0.85)
# 0.860655737705
