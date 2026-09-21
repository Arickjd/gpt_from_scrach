import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        den = 1 + np.exp(-z)
        output = 1 / den
        return np.round(output,5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise

        output = np.array([])
        for n in z:
            if n > 0:
                output = np.append(output, n) 
            else:
                output = np.append(output, 0) 
        return output
