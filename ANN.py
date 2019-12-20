import pandas as pd
import numpy as np
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lb = LabelEncoder()
X.iloc[:, 1] = lb.fit_transform(X.iloc[:, 1])
X.iloc[:, 2] = lb.fit_transform(X.iloc[:, 2])
ohe = OneHotEncoder(categorical_features = [1])
X = ohe.fit_transform(X).toarray()
X = X[:, 1:]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from keras.models import Sequential
from keras.layers import Dense
from keras.backend import backend as K
from keras.layers import Dropout
classifier = Sequential()
classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'random_uniform', input_shape = (11,)))
classifier.add(Dropout(p = 0.1))
classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'random_uniform'))
classifier.add(Dropout(p = 0.1))
classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'random_uniform'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
            classifier = Sequential()
            classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'random_uniform', input_shape = (11,)))
            classifier.add(Dropout(p = 0.1))
            classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'random_uniform'))
            classifier.add(Dropout(p = 0.1))
            classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'random_uniform'))
            classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
            return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_