from dateutil.parser import *
import numpy as np
import pandas as pd


def _label_sequence_iter(label_ix):
    _seq = []
    _dat = {}
    for _lbl in label_ix:
        _dat.update({_lbl[0]: _seq.copy()})
        _seq += [_lbl[1]]
    return _dat


def create_label_sequences(data, label):
	"""
	This utility creates sequences of labels for each venue at a given point of time.
	For each id, I look at the past data points and construct a sequence
	labeling an 'event' type.
	For example:
	'55e2666b8da9a20003000c59': [1, 1, 1, 1],
	'55e2666b8da9a20003000c5a': [1, 1, 1],
	'55e2666b8da9a20003000c5c': [1, 1],
	'55e2666b8da9a20003000c5f': [1],
	"""
	venue_ts = data[['id', 'relevantTime', 'venueId']]
	venue_ts['label'] = label
	venue_ts['ts'] = venue_ts.apply(lambda x: parse(x['relevantTime']), axis=1)
	venue_ts['ts_rank'] = venue_ts.groupby('venueId')['ts'].rank()
	venue_ts = venue_ts.sort_values(['venueId', 'ts_rank'])
	venue_ts.index = venue_ts.id.tolist()
	label_sequences = {}
	for venue_id in set(venue_ts.venueId.tolist()):
		_venue = venue_ts.query("venueId == '{}'".format(venue_id))
		label_ix = _venue['label'].items()
		_label_sequence = _label_sequence_iter(label_ix)
		label_sequences.update(_label_sequence)
	return label_sequences


def create_feature(data, label):
	label_sequences = create_label_sequences(data, label)
	feature = []
	_iter = iter(label_sequences.items())
	for post in _iter:
		feature.append({'id': post[0], 'mean_event': np.mean(post[1])})
	feature = pd.DataFrame(feature)
	feature['label'] = label
	# Setting priors for posts from venues with no previous data.
	# This implies that if no previous post, treat it as 'info'
	return feature.dropna()


