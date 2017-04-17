"""
Inference module.
args:
	--pass-to-human", "Pass low confidence scores to human for evaluation"
	--num-cases-human", "If pass-to-human is set to True, determine the number of cases that will be passed to humans"
	--input-file", "the full path to do inference"
	--model-path", help="the full path where the output will be persisted"
	--glove-path", the full path where the word embedding live"
	
If pass-to-human is set to True, 
"""
import argparse

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

import pickle
from utils import *

INFERENCE_CUTOFF = 0.8


def main():
	# Load the data
	print("Load the data..")
	data = load_and_preprocess(TEST_FILE_PATH,train=False)
	text = data['text'].tolist()

	# Load the model
	print("Load the models..")
	model_lstm = load_model(os.path.join(MODEL_PATH,'lstm_model.h5'))
	model_conv = load_model(os.path.join(MODEL_PATH, 'conv1d_model.h5'))
	tfidf_path = os.path.join(MODEL_PATH, "tfidf.p")
	with open(tfidf_path, "rb") as f:
		fit_tfidf = pickle.load(f)
	tok_path = os.path.join(MODEL_PATH, "tokenizer.p")
	with open(tok_path, "rb") as f:
		tokenizer = pickle.load(f)
	ensemble_path = os.path.join(MODEL_PATH, "ensemble.p")
	with open(ensemble_path, "rb") as f:
		model_ensemble = pickle.load(f)

	print("Inference..")
	sequences_holdout = tokenizer.texts_to_sequences(text)
	X = pad_sequences(sequences_holdout, maxlen=MAX_SEQUENCE_LENGTH)

	p_tfidf = fit_tfidf.predict_proba(text)
	p_lstm = model_lstm.predict(X)
	p_conv = model_conv.predict(X)
	feature_ensemble = np.vstack((p_tfidf[:, 1], p_lstm[:, 1], p_conv[:, 1])).T
	proba = model_ensemble.predict_proba(feature_ensemble)
	# We need a high precision. Hence, we set an inference cutoff.
	print("Making predictions...")
	prediction = (proba[:, 1] >= INFERENCE_CUTOFF).astype(int)
	labeled_prediction = prediction_to_label(prediction)
	# If we want to pass uncertain estimates to human, here is where we do it
	# by ranking the confidence of predictions and labelling low-confidence predictions as 'human'
	if PASS_LOW_CONFIDENCE_TO_HUMAN and NB_CASES_HUMAN_EVAL:
		print("Evaluating cases with low confidence")
		ensemble_decision = model_ensemble.decision_function(feature_ensemble)
		prediction_rank = abs(ensemble_decision).argsort()
		ix_humans = prediction_rank[:NB_CASES_HUMAN_EVAL]
		labeled_prediction[ix_humans] = 'human'
	# Save the data
	print("Writing the labeled predictions")
	pd.DataFrame(labeled_prediction, columns=['label']).to_csv(OUTPUT_FILE_PATH, index=False)


if __name__ == '__main__':
	_parser = argparse.ArgumentParser()
	_parser.add_argument("--pass-to-human", default=False, help="Pass low confidence scores to human for evaluation")
	_parser.add_argument("--num-cases-human", default=1000, help="Determine the number of cases that will be passed to humans")
	_parser.add_argument("--input-file", help="the full path to do inference")
	_parser.add_argument("--output-file", help="the full path for the output file")
	_parser.add_argument("--model-path", help="the full path where the output will be persisted")
	_parser.add_argument("--glove-path", help="the full path where the word embedding live")
	args = _parser.parse_args()
	PASS_LOW_CONFIDENCE_TO_HUMAN, NB_CASES_HUMAN_EVAL, MODEL_PATH, TEST_FILE_PATH, OUTPUT_FILE_PATH, GLOVE_DIR = \
		args.pass_to_human, args.num_cases_human, args.model_path, args.input_file, args.output_file, args.glove_path
	main()
