"""
## DISCLAIMER: NOT THE WAY USED FOR TRAINING NEURAL NETWORKS !!

for understanding GREDIENT DESCENT with RESP TO ** INPUTS **:
IMP: calculating GREDIENT DESCENT with resp to INPUTS is not the general way to train neural network !!
     Generally calculating GREDIENT DESCENT with resp to weight and Biases are considered Appropriate Way !!

forward propogagtion
z1 = w11*X1 + w12*X2 + w13*X3 + w14*X4 + b1
z2 = w21*X1 + w22*X2 + w23*X3 + w24*X4 + b2 
z3 = w31*X1 + w32*X2 + w33*X3 + w34*X4 + b3

activation function output
a1 = relu(z1)
a2 = relu(z2)
a3 = relu(z3)


Need to find:
** dl_d(input) == dl/dx **

dl_dX1, dl_dX2, dl_dX3, dl_dX4

we Know:
dl_d(weights) == X.T * dl_dz        #Correct

"""

import numpy as np
inputs = np.array([1,2,5,-3])

# 3-> hid_neuron, 4-> input_feature
weights=np.random.randn(3,4)

biases = np.array([-1.1,2.8,0.1])

def relu(z):
    return np.maximum(z,0)

def relu_derivative(z):
    return np.where(z>0,1,0)

learning_rate=0.001
epochs=100
for i in range(epochs):
    # Forward prop
    z = np.dot(weights,inputs)+biases
    a = relu(z)
    y = np.sum(a)
    loss = y**2


    # Backward Prop
    dL_dy = 2*y
    dy_da = np.ones_like(a)
    da_dz = relu_derivative(z)
    dL_dz = dL_dy * dy_da * da_dz

    # GREDIENT W.R.T weights and biases
    dL_dw = np.outer(dL_dz,inputs)
    dL_db = dL_dz

    # GREDIENT W.R.T *INPUTS*
    dL_dx = np.dot(dL_dz,weights)

    weights-=learning_rate*dL_dx
    biases-=learning_rate*dL_db

    if i%10==0:
        print(f"Loss:{loss}")
        print(f"dL/dx:{dL_dx}\n")

print(f"Output: {y}")

"""
terminal :
dL/dx:[-58.34618178  69.57964116 143.94533121  17.69235857]

Loss:114.725931150165
dL/dx:[-9.90781644 35.413938   20.39898314 -5.58353307]

Loss:59.59812132956246
dL/dx:[-4.9077224  17.54188501 10.1044006  -2.7657386 ]

Loss:36.45379664294226
dL/dx:[-2.90169617 10.37165848  5.97423778 -1.63524594]

Loss:24.46601637578157
dL/dx:[-1.90207666  6.79867509  3.91614337 -1.07191206]

Loss:17.427732812143386
dL/dx:[-1.33437932  4.76952988  2.74732394 -0.75198719]

Loss:12.93641539101668
dL/dx:[-0.98230592  3.51109866  2.02244784 -0.55357683]

Loss:9.895540401141389
dL/dx:[-0.74963582  2.67945583  1.54340854 -0.42245599]

Loss:7.743896815230948
dL/dx:[-0.58829648  2.10277362  1.21123055 -0.33153348]

Loss:6.168867358880321
dL/dx:[-0.47211609  1.68750502  0.97202933 -0.26606022]

Output: 2.2559061438711976


As this oberservation shows,
how much inputs should we change to minimize the loss (not general practice but used in some areas)
"""