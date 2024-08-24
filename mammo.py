import pandas as pd
#import matplotlib as plt
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image
from six import StringIO
from sklearn import tree
from pydotplus import graph_from_dot_data



masses_data = pd.read_csv('../../MLCourse/mammographic_masses.data.txt', na_values=['?'], names=['BI-RADS', 'age', 'shape', 'margin', 'density', 'severity'])
masses_data.dropna(inplace=True)
df = masses_data

#print(type(df.age.describe()))
#print(df['age'].isnull())

all_features = masses_data[['age', 'shape', 'margin', 'density']].values
all_classes = masses_data[['severity']].values

feature_names = ['ages', 'shape', 'margin', 'density']

#print(type(all_features))
#print(all_features)

#print(all_features[1][0])

scaler = preprocessing.StandardScaler()
all_features_scaled = scaler.fit_transform(all_features)
#print(all_features_scaled)

## Decision Tree
np.random.seed(1234)

training_inputs, testing_inputes, training_classes, testing_classes = train_test_split(all_features_scaled, all_classes, train_size=0.75, random_state=1)
#print(len(training_inputs))
#print(training_inputs)

clf = DecisionTreeClassifier(random_state=1)

clf.fit(training_inputs, training_classes)

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=feature_names)
graph = graph_from_dot_data(dot_data.getvalue())

Image(graph.create_png())
graph.write_png("decision_tree.png")



