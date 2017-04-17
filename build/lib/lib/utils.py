import os

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from keras.layers import Embedding
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix

MAX_SEQUENCE_LENGTH = 100
MAX_NB_WORDS = 10000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 64
NB_EPOCHS = 20


def train_model(_model, X, y):
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VALIDATION_SPLIT, random_state=42)
	_y_train, _y_val = to_categorical(y_train), to_categorical(y_val)
	_model.fit(X_train, _y_train, validation_data=(X_val, _y_val), epochs=NB_EPOCHS, batch_size=BATCH_SIZE)
	return _model


def create_embedding_layer(word_index, embedding_matrix):
	# Set the trainable to False so that we can update the weights during training
	embedding_layer = Embedding(len(word_index) + 1,
		EMBEDDING_DIM,
		weights=[embedding_matrix],
		input_length=MAX_SEQUENCE_LENGTH,
		trainable=False)
	return embedding_layer


def optimize_business_metrics(prob, labels):
	threshold = np.arange(0, 1, 0.1)
	pred = prob[:, 1]
	total_cost = []
	for th in threshold:
		cost = 0
		_pred = (pred > th).astype(int)
		cm = confusion_matrix(labels, _pred).reshape(4,)
		print(th, cm)
		total_cost.append({th: np.round(np.sum(COST * cm))})
	return total_cost


def prediction_to_label(x):
	return np.array(list(map(lambda _x: "event" if _x == 1 else "info", x)))


def _extract_text(x):
	return x['facebookMessage'] if x['facebookMessage'] is not [] else x['caption']


def _remove_hashtags(x):
	return str(x).replace("#", " ").replace("@", " ")


def load_and_preprocess(file_path, train=True):
	"""
	Load and preprocess data 
	"""
	data = pd.read_csv(file_path, delimiter='`')
	# remove email and 'other' labels if training
	if train:
		data = data.query("dataType not in 'email'").query("label not in 'other'")
	text = data.apply(_extract_text, axis=1)
	text = text.apply(_remove_hashtags)
	data['text'] = text
	return data


def build_embedding_vector(glove_dir):
	"""
	Build index mapping words in the embeddings set to their embedding vector
	"""
	print('Indexing word vectors.')
	embeddings_index = {}
	f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()
	return embeddings_index


def create_word_embeddings(tokenizer, embeddings_index):
	'''
	Lookup table to map tokenizer words onto the embedding space
	'''
	word_index = tokenizer.word_index
	print('Found %s unique tokens.' % len(word_index))
	embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
	for word, i in word_index.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			# words not found in embedding index will be all-zeros.
			embedding_matrix[i] = embedding_vector
	return embedding_matrix
