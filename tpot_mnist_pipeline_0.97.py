import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'class' in the data file
###tpot_data = np.recfromcsv('./sources/cars.csv', delimiter=',', dtype=np.float64)
tpot_data = np.recfromcsv('./sources/cars.csv')
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_target, testing_target = \
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    StackingEstimator(estimator=LogisticRegression(C=1.0, dual=True)),
    RandomForestClassifier(max_features=0.6000000000000001, min_samples_leaf=20, min_samples_split=18)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

print (results)
