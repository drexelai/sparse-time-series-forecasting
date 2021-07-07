import pandas as pd
import numpy as np
from math import sqrt, pow

SAMPLING_RATE = 125

def load_data():
	pass


def preprocessData(data):
	pass


def getSegments(data, n=20, ntrain_ponts=1000, mtest_points=200):
	pass


def sparseRepresentation(data):
	pass


def scoreEuclidean(x_act, x_pred):
	return sqrt(sum([(x_act[i] - x_pred[i]) ** 2 for i in range(x_pred)]))


def score(x_act, x_pred, type="euclidean"):
	assert x_act == x_pred, "[{}] It is expected that x_act [len={}] and x_pred [len={}] are the same length. Currently they are not.".format(__func__, len(x_act), len(x_pred))
	return scoreEuclidean(x_act, x_pred)


def buildModel(input_shape):
	pass


def predictForecast(x_train, weights=None):

	# Randomize those weights some how

	model = buildModel(x_train.shape)

	model.train(x_train)

	x_test = model.predict()

	weights = None

	return x_test, weights


def runGeneticAlgorithm(data, nsamples=100, ntrain_ponts=1000, mtest_points=200, nchildren=100, nepochs=20, alpha=0.05):
	# expect data_segments as a list of tuples containing pandas DataFrames which are all the same length of data between elements the list
	data_segments = getSegments(data, n=nsamples, ntrain_ponts=ntrain_ponts, mtest_points=mtest_points)

	sparse_segments = [sparseRepresentation(data_segment) for data_segment in data_segments]

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
	data = load_data()
	data_preprocessed = preprocessData(data)
	results = learn(data_preprocessed)


if __name__ == "__main__":
	main()
