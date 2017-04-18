"""
Bi-directional LSTM model
"""


from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional


def create_lstm_model(embedding_layer):
	model = Sequential()
	model.add(embedding_layer)
	model.add(Bidirectional(LSTM(64)))
	model.add(Dropout(0.5))
	model.add(Dense(2, activation='sigmoid'))
	model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
	return model
