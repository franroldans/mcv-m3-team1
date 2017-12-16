import numpy as np


def predict_logistic_regression(weights, test_set):
    def sigmoid(scores):
        return 1 / (1 + np.exp(-scores))
    final_scores = np.dot(test_set, weights)
    return np.round(sigmoid(final_scores))
