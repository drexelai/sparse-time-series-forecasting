import pandas as pd
import numpy as np
from math import pi as PI
from random import random, seed

import os
from os import system

from tempfile import TemporaryDirectory

SAMPLING_RATE = 125

def loadDataSample(nsamples=100, npoints=100000, nperiods=100, seed=42, variation_parm=1):
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

	# seed(42)

	t = np.linspace(0, nperiods * PI, npoints)
	x = np.array([np.sin(t) + variation_parm * 0.01 * (random() - 0.5) for _ in range(nsamples)])
	return x	


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
		system("wget -r -N -c -np -P %s https://physionet.org/files/ptbdb/1.0.0/" % tempdir)

	datadir = tempdir + '/physionet.org/files/ptbdb/1.0.0/'

	if not os.path.isdir(datadir):
		print("The given temporary directory does not contain the downloaded data. Please rerun the function without providing tempdir as an parameter.")
		return None, None

	patientfiles = [e for e in os.listdir(datadir) if 'patient' in e]

	for patientfile in patientfiles:
		patientdir = datadir + '/' + patientfile
		datafiles = [e for e in os.listdir(patientdir) if '.dat' in e]
		for datafile in datafiles:
			if data.shape[0] == nsamples:
				break
				#return data, tempdir
			dataloc = patientdir + '/' + datafile
			x = np.fromfile(dataloc)
			if len(x) < npoints:
				continue
			data = np.vstack([data, x[0:npoints]])

	return data, tempdir


def getSegments(data, n=100, ntrain_ponts=1000, mtest_points=200, seed=42):
	"""
	DESCRIPTION
	-----------
	Points selected for ntrain_ponts and mtest_points need to be adjacent, but they do not necessarily need to be fetched from the beginning of the file.


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
		Tuple of samples separated into ntrain_ponts and mtest_points

	"""

	x_train = None
	x_test = None

	return x_train, x_test


def sparseRepresentation(x_train):
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

	# from IsamusModule import buildDictionary

	x_train_sparse = [buildDictionary(e) for e in x_train]

	pass


def calcMSE(x_act, x_pred):
	"""
	DESCRIPTION
	-----------
	

	PARAMETERS
	----------
	

	RETURNS
	-------
	
	"""
	return sqrt(sum([(x_act[i] - x_pred[i]) ** 2 for i in range(x_pred)]))


def calcMAE(x_act, x_pred):
	pass


def calcRMSD(x_act, x_pred):
	pass


def score(x_act, x_pred, metric="mse", weight_function=None):
	"""
	DESCRIPTION
	-----------
	1. weight_function would cause the score function called to be time dependent. 

	PARAMETERS
	----------
	

	RETURNS
	-------
	
	"""
	assert x_act == x_pred, "[{}] It is expected that x_act [len={}] and x_pred [len={}] are the same length. Currently they are not.".format(__func__, len(x_act), len(x_pred))
	if metric.lower() == "mse":
		return calcMSE(x_act, x_pred)
	if metric.lower() == "mae":
		pass
	if metric.lower() == "rmsd":
		pass
	return None


def buildModel(input_shape, loss, ):
	"""
	DESCRIPTION
	-----------
	This will simply return a compiled model

	PARAMETERS
	----------
	

	RETURNS
	-------
	
	"""

	model = None # input_shape

	pass


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

	weights = None

	return x_test, weights


def runGeneticAlgorithm(data, nsamples=100, ntrain_ponts=1000, mtest_points=200, nchildren=100, nepochs=20, alpha=0.05):
	"""
	DESCRIPTION
	-----------
	

	PARAMETERS
	----------
	

	RETURNS
	-------
	
	"""

	# expect data_segments as a list of tuples containing pandas DataFrames which are all the same length of data between elements the list
	x_train, x_test = getSegments(data, n=nsamples, ntrain_ponts=ntrain_ponts, mtest_points=mtest_points)

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
	data = loadDataSample()
	results = runGeneticAlgorithm(data)


if __name__ == "__main__":
	pass
	#main()
