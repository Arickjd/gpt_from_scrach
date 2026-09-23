import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        model_prediction = [] 
        w = weights
        for i in range(len(X)):
            y_pred = np.dot(X[i],w)
            model_prediction.append(round(y_pred,5))
        return model_prediction

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        sum_error = 0
        for pred,true in zip(model_prediction,ground_truth):
            sum_error += (pred - true)**2
        error = sum_error/len(model_prediction)
        return round(error[0],5)
