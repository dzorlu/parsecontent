
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model

"""
"""
MAX_SEQUENCE_LENGTH = 100


def create_conv1d_model(embedding_layer):
	sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
	embedded_sequences = embedding_layer(sequence_input)
	x = Conv1D(128, 5, activation='relu')(embedded_sequences)
	x = Conv1D(128, 5, activation='relu')(x)
	x = MaxPooling1D(5)(x)
	x = Flatten()(x)
	x = Dense(128, activation='relu')(x)
	x = Dropout(0.5)(x)
	preds = Dense(2, activation='softmax')(x)
	model = Model(sequence_input, preds)
	model.compile(loss='categorical_crossentropy',
		optimizer='adam',
		metrics=['acc'])
	return model
