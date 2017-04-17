import argparse
import pickle

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

import conv1d
import ensemble
import lstm
import tf_idf
from utils import *

MAX_SEQUENCE_LENGTH = 100
MAX_NB_WORDS = 10000
VALIDATION_SPLIT = 0.2

# Business Metrics to optimize - TN, FP, FN, TP
COST = 0.55, -2.25, -0.35, 0.55
DN = -0.1


def main():
	print("Load Data...")
	data = load_and_preprocess(INPUT_FILE_PATH)
	# training corpus
	# holdout 10% for calibration of model ensemble
	_label = data['label'].tolist()
	_data, _data_holdout, _label, _label_holdout = train_test_split(data, _label, test_size=0.1, random_state=42)
	# text data
	text = _data['text'].tolist()
	text_holdout = _data_holdout['text'].tolist()
	# label
	label = np.array(list(map(lambda x: 1 if 'event' in x else 0, _label)))
	label_holdout = np.array(list(map(lambda x: 1 if 'event' in x else 0, _label_holdout)))

	'''
	Training Data
	'''
	embeddings_index = build_embedding_vector(GLOVE_DIR)
	# fit the tokenizer
	tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
	tokenizer.fit_on_texts(text)
	# create sequence of training data
	sequences = tokenizer.texts_to_sequences(text)
	X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
	# create sequences of holdout data
	sequences_holdout = tokenizer.texts_to_sequences(text_holdout)
	X_holdout = pad_sequences(sequences_holdout, maxlen=MAX_SEQUENCE_LENGTH)
	# Create embedding matrix
	embedding_matrix = create_word_embeddings(tokenizer, embeddings_index)

	'''
	Model Training
	'''
	print("Training the model...")
	# NLP Models
	# TFIDF + Linear SVM
	fit_tfidf = tf_idf.train_tfidf_model((text, label))
	feature_tfidf = fit_tfidf.predict_proba(text)

	# Word Embeddings and Conv1D
	embedding_layer = create_embedding_layer(tokenizer.word_index, embedding_matrix)
	model_conv = conv1d.create_conv1d_model(embedding_layer)
	fit_conv1d = train_model(model_conv, X, label)
	feature_conv1d = fit_conv1d.predict(X)

	# Bi-directional LSTM
	embedding_layer = create_embedding_layer(tokenizer.word_index, embedding_matrix)
	model_lstm = lstm.create_lstm_model(embedding_layer)
	fit_lstm = train_model(model_lstm, X, label)
	feature_lstm = fit_lstm.predict(X)

	# Ensemble Model
	print("Traning the Ensemble Model...")
	feature = np.vstack((feature_tfidf[:, 1], feature_conv1d[:, 1], feature_lstm[:, 1])).T
	pholdout_tfidf = fit_tfidf.predict_proba(text_holdout)
	pholdout_lstm = fit_lstm.predict(X_holdout)
	pholdout_conv = fit_conv1d.predict(X_holdout)
	feature_ensemble = np.vstack((pholdout_tfidf[:, 1], pholdout_lstm[:, 1], pholdout_conv[:, 1])).T
	ensemble_model = ensemble.train_ensemble((feature_ensemble, label_holdout))

	# Save Models
	print("Saving Models...")
	fit_lstm.save(os.path.join(MODEL_PATH, "lstm_model.h5"))
	fit_conv1d.save(os.path.join(MODEL_PATH, "conv1d_model.h5"))
	tfidf_path = os.path.join(MODEL_PATH, "tfidf.p")
	with open(tfidf_path, "wb") as f:
		pickle.dump(fit_tfidf, f)
	tok_path = os.path.join(MODEL_PATH, "tokenizer.p")
	with open(tok_path, "wb") as f:
		pickle.dump(tokenizer, f)
	ensemble_path = os.path.join(MODEL_PATH, "ensemble.p")
	with open(ensemble_path, "wb") as f:
		pickle.dump(ensemble_model, f)
	print("Model Trained...")


if __name__ == '__main__':
	_parser = argparse.ArgumentParser()
	_parser.add_argument("--input-file", help="the full path to data to train the model on")
	_parser.add_argument("--model-path", help="the full path where the output will be persisted")
	_parser.add_argument("--glove-path", help="the full path where the word embedding live")
	args = _parser.parse_args()
	MODEL_PATH, INPUT_FILE_PATH, GLOVE_DIR = args.model_path, args.input_file, args.glove_path

	main()







