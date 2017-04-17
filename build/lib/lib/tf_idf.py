"""
Extract TF-IDF features from the corpus
Train a Logistic Regression with hyperparameter search
Train a model for class (event)
because the business logic has different cost imposed on false positives and false negatives
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
	('vect', CountVectorizer(stop_words='english', min_df=5)),
	('tfidf', TfidfTransformer()),
	('clf', SGDClassifier(loss='log')) # Logistic regression
])

parameters = {
	'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or unigrams and bigrams
	'tfidf__use_idf': (True, False),
	'tfidf__norm': ('l1', 'l2'),
	'clf__alpha': (0.001, 0.0001),
	'clf__penalty': ('l2', 'elasticnet'),
}


def train_tfidf_model(_data):
	X, y = _data
	grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
	grid_search.fit(X, y)
	return grid_search

