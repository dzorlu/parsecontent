"""
Extract TF-IDF features from the corpus
Train a Linear SVM model with hyperparameter search
Train a model for each class (info / event)
because the business logic has different cost imposed on false positives and false negatives
"""

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
	('clf', SVC(probability=True))
])

parameters = {
	'clf__kernel': ('rbf','sigmoid'),
	'clf__C': (1, 10, 100, 1000)
}


def train_ensemble(_data):
	X, y = _data
	grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
	grid_search.fit(X, y)
	return grid_search
