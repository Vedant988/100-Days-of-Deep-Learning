# calculating backpropogation in "CATEGORICAL CROSS ENTROPY LOSS" (CCE)
"""
L1 = -[ y11*log(y^11) + y12*log(y^12) + y13*log(y^13) ]

to find,
dL1_dw11 = dL1_dy^11 * dy^11_dw11
dL1_dw12 = dL1_dy^12 * dy^11_dw12
dL1_dw13 = dL1_dy^13 * dy^11_dw13

dL1_dy^11 = -y11/y^11 
dL1_dy^12 = -y12/y^12 
dL1_dy^13 = -y13/y^13 
-----------------------> (-True/predicted)


lets analyse for batch of data:

applied One Hot Encoding 
true = [[1,0,0],
        [0,0,1],
        [1,0,0]]

predicted = [[0.7,0.4,0.1],
             [0.1,0.3,0.3],
             [0.4,0.1,0.6]]

dL1_dy = -[[1/0.7,0,0],
           [0,0,1/0.3],
           [1/0.4,0,0]]


"""

import numpy as np
labels = np.array([0, 2, 0])
true = np.eye(3)[labels]
print(f"true:\n{true}\n")

predicted = np.array([[0.7, 0.4, 0.1],
                      [0.1, 0.3, 0.3],
                      [0.4, 0.1, 0.6]])
print(f"predicted:\n{predicted}\n")

loss = -np.sum(true * np.log(predicted), axis=1)
print(f"loss:\n{loss}\n")

gradient = -true / predicted
print(f"dL1_dy:\n{gradient}\n")

# misconception    --> Greater magnitude of gradient ⇒ higher chance the prediction is correct
# Correct Intution --> Larger gradient magnitude means the model is more wrong about the correct class

# so here, smaller gredient magnitude means small update in BACKPROPOGATION
# larger gredient magnitude means Large Update in BACKPROPOGATION !!!