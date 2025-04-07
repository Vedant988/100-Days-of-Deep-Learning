import numpy as np

weights=np.array([-2.0,-1.0,3.0])
bias=-1.0
inputs=np.array([-1.0,-2.0,2.0])
target_output=0.0
learning_rate=0.001

def relu(z):
    return np.maximum(0,z)

def relu_derivative(z):
    return np.where(z>0,1,0)

epochs=100
for iteration in range(epochs):
    # forward prop
    z1=np.dot(weights,inputs)+bias
    output=relu(z1)
    loss=(output-target_output)**2

    # backward prop
    dloss_doutput=2*(output-target_output)
    doutput_dlinear=relu_derivative(z1)
    dlinear_dweights=inputs
    dlinear_dbias=1

    dloss_dlinear=dloss_doutput*doutput_dlinear
    dloss_dweights=dloss_dlinear*dlinear_dweights
    dloss_dbias=dloss_dlinear*dlinear_dbias

    weights -= learning_rate*dloss_dweights
    bias -= learning_rate*dloss_dbias

    if iteration%10==0:
        print(f"Iteration: {iteration}, loss:{loss}")

print(f"Final Weights: {weights}")
print(f"Final Bias: {bias}")
print(f"Output: {output}")

        
