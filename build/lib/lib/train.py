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


def main(model_path, input_file_path, glove_dir):
  print("Load Data...")
  data = load_and_preprocess(input_file_path)
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
  embeddings_index = build_embedding_vector(glove_dir)
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

  # Word Embeddings and Conv1D
  embedding_layer = create_embedding_layer(tokenizer.word_index, embedding_matrix)
  model_conv = conv1d.create_conv1d_model(embedding_layer)
  fit_conv1d = train_model(model_conv, X, label)

  # Bi-directional LSTM
  embedding_layer = create_embedding_layer(tokenizer.word_index, embedding_matrix)
  model_lstm = lstm.create_lstm_model(embedding_layer)
  fit_lstm = train_model(model_lstm, X, label)

  # Ensemble Model
  print("Traning the Ensemble Model...")
  pholdout_tfidf = fit_tfidf.predict_proba(text_holdout)
  pholdout_lstm = fit_lstm.predict(X_holdout)
  pholdout_conv = fit_conv1d.predict(X_holdout)
  feature_ensemble = np.vstack((pholdout_tfidf[:, 1], pholdout_lstm[:, 1], pholdout_conv[:, 1])).T
  ensemble_model = ensemble.train_ensemble((feature_ensemble, label_holdout))

  # Save Models
  print("Saving Models...")
  fit_lstm.save(os.path.join(model_path, "lstm_model.h5"))
  fit_conv1d.save(os.path.join(model_path, "conv1d_model.h5"))
  tfidf_path = os.path.join(model_path, "tfidf.p")
  with open(tfidf_path, "wb") as f:
    pickle.dump(fit_tfidf, f)
  tok_path = os.path.join(model_path, "tokenizer.p")
  with open(tok_path, "wb") as f:
    pickle.dump(tokenizer, f)
  ensemble_path = os.path.join(model_path, "ensemble.p")
  with open(ensemble_path, "wb") as f:
    pickle.dump(ensemble_model, f)
  print("Model Trained...")


