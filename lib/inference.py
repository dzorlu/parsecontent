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
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

import pickle
from .utils import *

INFERENCE_CUTOFF = 0.8


def main(pass_low_confidence_to_human,
         nb_cases_human_eval,
         model_path,
         test_file_path,
         output_file_path,
         glove_dir):

  # Load the data
  print("Load the data..")
  data = load_and_preprocess(test_file_path, train=False)
  text = data['text'].tolist()

  # Load the model
  print("Load the models..")
  model_lstm = load_model(os.path.join(model_path, 'lstm_model.h5'))
  model_conv = load_model(os.path.join(model_path, 'conv1d_model.h5'))
  tfidf_path = os.path.join(model_path, "tfidf.p")
  with open(tfidf_path, "rb") as f:
    fit_tfidf = pickle.load(f)
  tok_path = os.path.join(model_path, "tokenizer.p")
  with open(tok_path, "rb") as f:
    tokenizer = pickle.load(f)
  ensemble_path = os.path.join(model_path, "ensemble.p")
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
  if pass_low_confidence_to_human and nb_cases_human_eval:
    print("Evaluating cases with low confidence")
    ensemble_decision = model_ensemble.decision_function(feature_ensemble)
    prediction_rank = abs(ensemble_decision).argsort()
    ix_humans = prediction_rank[:nb_cases_human_eval]
    labeled_prediction[ix_humans] = 'human'
  # Save the data
  print("Writing the labeled predictions")
  pd.DataFrame(labeled_prediction, columns=['label']).to_csv(output_file_path, index=False)


