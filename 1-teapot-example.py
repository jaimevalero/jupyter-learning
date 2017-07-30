#TO-DO
#WORKING ON THIES
import pandas as pd

import numpy as np
from pandas import DataFrame
from sklearn.cross_validation import train_test_split
import sklearn.cross_validation
from scipy.spatial.distance import pdist, squareform
from tpot import TPOTClassifier
from tpot import TPOTRegressor

df = pd.read_csv('sources/cars.csv')

# Preprocessiing

# Replace 0s with median value
df['Horsepower']=df['Horsepower'].replace(0,df['Horsepower'].mean())
df['MPG']=df['MPG'].replace(0,df['MPG'].mean())

# Convert origin to numerical values
from sklearn import preprocessing
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



le = preprocessing.LabelEncoder()


tpot = TPOTClassifier(generations=7, population_size=15, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_cars_pipeline.py')

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

# All features set
# tpot = TPOTClassifier(generations=20, population_size=50, verbosity=2)
# Best pipeline: RandomForestClassifier(SelectFwe(input_matrix, SelectFwe__alpha=0.04), RandomForestClassifier__bootstrap=DEFAULT, RandomForestClassifier__criterion=entropy, RandomForestClassifier__max_features=0.6, RandomForestClassifier__min_samples_leaf=DEFAULT, RandomForestClassifier__min_samples_split=5, RandomForestClassifier__n_estimators=100)
# 0.868852459016
