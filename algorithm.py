from importlib import reload
from math import pi as PI
import os
from os import system
from random import random, seed
from sys import platform
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sys import modules
# import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from wfdb.io import rdrecord

SAMPLING_RATE = 125

def loadDataSample(nsamples=100, npoints=100000, nperiods=100, random_state=42, variation_parm=1):
	"""
	DESCRIPTION
	-----------
	

	PARAMETERS
	----------
	nsamples: 

	npoints: 

	nperiods: 

	seed: 

	variation_parm:


	RETURNS
	-------
	x: 

	"""

	seed(random_state)

	t = np.linspace(0, nperiods * PI, npoints)
	x = np.array([np.sin(t) + variation_parm * 0.01 * (random() - 0.5) for _ in range(nsamples)])
	return x	

# For Ethan:
tempdir = '/var/folders/jx/16bfc0jx3cb_ygmzkjfmcyv80000gn/T/tmpvnldi1kk/'


def plotOneSample(dataloc):
	pass


def getDataFiles(tempdir):
	filenames = []
	datadir = os.path.join(tempdir, 'physionet.org/files/ptbdb/1.0.0/')
	patientfiles = [e for e in os.listdir(datadir) if 'patient' in e]
	for patientfile in patientfiles:
		patientdir = os.path.join(datadir, patientfile)
		datafiles = [e for e in os.listdir(patientdir)]
		datafiles_base = [e.split('.')[0] for e in datafiles if e != 'index.html']
		for datafile in set(datafiles_base):
			dataloc = os.path.join(patientdir, datafile)
			filenames.append(dataloc)
	return filenames


def loadData(tempdir=None, nsamples=100, npoints=100000):
	"""
	# Ethan 

	DESCRIPTION
	-----------
	This function will download approximately 1.7 G in a temporary directory. I used the following logic when creating this function in order to ensure reproducibility. 

	# 1. create temp directory (tempfile)
	# 2. download data from url to temp file
	# --- command = wget -r -N -c -np https://physionet.org/files/ptbdb/1.0.0/
	# --- use os.system(command)
	# 3. return temp file name
	# 4. open and read the contents of the downloaded file
	# 5. append to numpy array

	Usage:
		data, tempdir = loadData()
		or 
		data, tempdir = loadData(tempdir)

	PARAMETERS
	----------
	tempdir: str (default=None)
		Path to temporary directory storing the data downloaded by this function. If not provided, the data will be redownloaded in a separate temporary directory. If this recently ran and the temporary directory file path is saved, pass it in as this argument. 
	nsamples: int (default=100)
		Number of samples to use at most from the downloaded data.
	npoitnts:
		Number of points to use for each sample. Samples with less points than this value will be excluded.
	RETURNS
	-------
	data:	
		Numpy ndarray representing the ECG data.
	tempdir:
		Path to temporary directory where the data is downloaded.
	"""

	# def loadDataHelper(data):
	# 	return data

	data = np.zeros((0, npoints))

	if not tempdir:
		tempdir = TemporaryDirectory().name
		try:
			if platform == "win32":
				system("wsl -r -N -c -np -P %s https://physionet.org/files/ptbdb/1.0.0/" % tempdir)
			if platform == "darwin":
				system("wget -r -N -c -np -P %s https://physionet.org/files/ptbdb/1.0.0/" % tempdir)
			else:
				print("I don't know what platform you have... Tell Ethan on Slack.")
				return None, None
		except Exception as e:
			print("[%s]:[%s] %s" % __name, __func__, e)
			print("Make sure if you are on Windows that you have wsl install. If you are on Mac, please install wget.")
			return None, None

	for dataloc in getDataFiles(tempdir):
		if data.shape[0] == nsamples:
			break
		rec = rdrecord(dataloc, channels=[0])
		x = rec.p_signal
		if x is None:
			x = rec.d_signal
		if x is None:
			continue
		if len(x) < npoints:
			continue
		x_t = np.transpose(x)
		data = np.vstack([data, x_t[:, 0:npoints]])

	return data, tempdir


def getSegments(data, n=100, ntrain_points=50000, ntest_points=100, random_state=42):
	"""
	DESCRIPTION
	-----------
	Points selected for ntrain_points and mtest_points need to be adjacent, but they do not necessarily need to be fetched from the beginning of the file.

	PARAMETERS
	----------
	data: (after preprocessed/selected)

	n: number of random samples to choose from data

	ntrain_points: int
		Number of points in training set
	mtest_points: int
		Number of points in the test set

	RETURNS
	-------
	tuple: 
		Tuple of samples separated into ntrain_points and mtest_points

	"""

	seed(random_state)

	npoints = data.shape[1]

	x_train = np.zeros((0, ntrain_points))
	x_test = np.zeros((0, ntest_points))

	data_size = 0

	for i in range(n):
		# TODO: Find a better way to ensure no NaN
		# while True: 
		# 	a = int(random() * (npoints - (ntrain_points + ntest_points)))
		# 	b = a + ntrain_points
		# 	c = b + ntest_points

		# 	data_train = data[i][a:b]
		# 	data_test = data[i][b:c]

		# 	if any(np.isnan(data_train)) or any(np.isnan(data_test)):
		# 		continue
		# 	break
		#   data_size += 1

		a = int(random() * (npoints - (ntrain_points + ntest_points)))
		b = a + ntrain_points
		c = b + ntest_points

		x_train = np.vstack([x_train, data[i][a:b]])
		x_test = np.vstack([x_test, data[i][b:c]])

	# TODO: Find a better way to ensure no NaN
	x_train = np.array([np.nan_to_num(e, nan=0, posinf=0, neginf=0) for e in x_train])
	x_test = np.array([np.nan_to_num(e, nan=0, posinf=0, neginf=0) for e in x_test])

	return x_train, x_test


def sparseRepresentation(x_train, x_test, batch_size=10, patches_per_img=50, alpha=0.01, beta=0.99, epsilon=1e-6, sparsity_coef=1, activity_epochs=300, epochs=30, num_layers=1):
	"""
	DESCRIPTION
	-----------
	# Use Isamu's module to do this...
	# This function accepts the first element of the tuple returned from getSegments
	
	PARAMETERS
	----------
	

	RETURNS
	-------
	
	"""

	from model import Activity, Dictionary, SparseModel, sparsity_loss, dictionary_loss

	data_size = x_train.shape[0]
	sample_size = x_train.shape[1]

	batch_size = 100
	num_batches = data_size // batch_size
	patches_per_img = 20
	dict_filter_size = sample_size // patches_per_img

	num_filters = 100

	alpha = 0.01 # learning rate
	beta = .99 # The amount to retain for A and B
	epsilon = 1e-6
	sparsity_coef = 1 # sparsity coef

	activity_epochs = 300
	epochs = 30
	num_layers = 1

	x_train_ = x_train.reshape([-1, dict_filter_size])
	x_test_ = x_test.reshape([-1, dict_filter_size]) 

	scaler = StandardScaler()
	scaler.fit(x_train_)

	x_train_ = scaler.transform(x_train_)
	x_test_ = scaler.transform(x_test_)

	print(x_train_.shape)
	print(batch_size)
	print(num_filters)
	print(dict_filter_size)
	print(activity_epochs)

	sparse_activity = Activity(batch_size=batch_size, units=num_filters, alpha=alpha, sparsity_coef=sparsity_coef)
	sparse_dictionary = Dictionary(units=num_filters, dict_filter_size=dict_filter_size, beta=beta)
	sparse_model = SparseModel(sparse_activity, sparse_dictionary, batch_size=batch_size, activity_epochs=activity_epochs, dict_filter_size=dict_filter_size, data_size=data_size, num_layers=num_layers)
	sparse_model.compile(sparsity_loss, dictionary_loss)

	sparse_model.fit(x_train_, epochs=epochs, batch_size=batch_size)

	return None


def score(x_pred, sparse_activity, sparse_dictionary):
	"""
	DESCRIPTION
	-----------
	1. weight_function would cause the score function called to be time dependent. 

	PARAMETERS
	----------
	

	RETURNS
	-------
	
	"""
	R = sparse_activity @ sparse_dictionary

	error = (x_pred - R) ** 2

	# sqrt(sum([(x_act[i] - x_pred[i]) ** 2 for i in range(x_pred)]))

	return error 



def buildModel(inputs, output_size, neurons, activ_func=activation_function, dropout=dropout, loss=loss, optimizer=optimizer):
	"""
	DESCRIPTION
	-----------
	This will simply return a compiled model

	PARAMETERS
	----------
	inputs: input data as an array

	n_outputs: number of predictions we want per sample

	neurons: number of neurons in the LSTM layer

	activ_func: activation function (e.g. tanh, sigmoid, etc.)

	dropout: value for the Dropout Layer

	optimizer: type of optimizer to use for model (e.g adam, rmsprop, etc.)

	RETURNS
	-------
	A compiled LSTM model and model summary (optional)
	
	"""

	model = Sequential()
  	model.add(LSTM(neurons, return_sequences=True, input_shape=(inputs.shape[1], inputs.shape[2]), activation=activ_func))
  	model.add(Dropout(dropout))
  	model.add(LSTM(neurons, return_sequences=True, activation=activ_func))
  	model.add(Dropout(dropout))
  	model.add(LSTM(neurons, activation=activ_func))
  	model.add(Dropout(dropout))
  	model.add(Dense(units=output_size))
  	model.add(Activation(activ_func))
  	model.compile(loss=loss, optimizer=optimizer, metrics=['mae'])
  	model.summary()

  	return model



def predictForecast(x_train, weights=None):
	"""
	DESCRIPTION
	-----------
	

	PARAMETERS
	----------
	

	RETURNS
	-------
	
	"""

	# Randomize those weights some how

	model = buildModel(x_train.shape)

	model.train(x_train)

	x_test = model.predict()
	
	model.save_weights('LSTM_pred_weights.h5')

	weights = model.load_weights('LSTM_pred_weights.h5')


	return x_test, weights


def runGeneticAlgorithm(x_train, x_test, nchildren=100, nepochs=20, alpha=0.05):
	"""
	DESCRIPTION
	-----------
	

	PARAMETERS
	----------
	

	RETURNS
	-------
	
	"""

	x_train_sparse = sparseRepresentation(x_train)

	weights = None

	ntop_weights = int(nchildren * alpha)

	for epoch in range(nepochs):

		child_weights = []
		child_scores = []

		for child in range(nchildren):

			weight_pred_segments = []
			scores_pred_segments = []

			# Randomize the weights for each child
			weights_random = None

			for j, data_segment in enumerate(data_segments):

				x_train = data_segment[0]
				x_act = data_segment[1]

				x_pred = predictForecast(x_train, weights=weights_random)

				weight_pred_segments.append(weights_random)
				scores_pred_segments.append(score(x_act, x_pred))

			child_weights.append(weight_pred_segments)
			child_scores.append(scores_pred_segments)

		best_weights = child_weights[np.argsort(np.array([np.mean(child_score) for child_score in child_scores]))[0:ntop_weights]]

		# Use top weights to create new weights

		# After each epoch create weights
		weights = None

		return weights


def main():
	"""
	DESCRIPTION
	-----------
	

	PARAMETERS
	----------
	

	RETURNS
	-------
	
	"""

	# data = loadData()
	data, tempfir = loadDataSample(tempdir=None)
	data_segment = getSegments(data, nsamples=100, ntrain_ponts=1000, ntest_points=200)
	results = runGeneticAlgorithm(x_train, x_test, nchildren=100, nepochs=20, alpha=0.05)


if __name__ == "__main__":
	pass
	#main()
