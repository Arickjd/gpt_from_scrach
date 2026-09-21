import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        arg = z - max(z)
        den = 0
        for n in z:
            a = n- max(z)
            den = den + np.exp(a)

        sfmx = np.exp(arg) / den
        output =  [round(x, 4) for x in sfmx]
    
        return output
        
