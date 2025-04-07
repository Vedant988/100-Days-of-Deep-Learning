import numpy as np

weights=np.random.randn(3,1)
bias=-1.0
inputs=np.random.randn(5,3)
target_output= np.zeros((5,1))
learning_rate=0.001

def relu(z):
    return np.maximum(0,z)

def relu_derivative(z):
    return np.where(z>0,1,0)

epochs=100
for iteration in range(epochs):
    # forward prop
    z1=np.dot(inputs,weights)+bias
    output=relu(z1)
    loss=np.mean((output-target_output)**2)

    # backward prop
    dloss_doutput=2*(output-target_output)/ inputs.shape[0]
    doutput_dlinear=relu_derivative(z1)
    dlinear_dweights=inputs
    dlinear_dbias=1

    dloss_dlinear=dloss_doutput*doutput_dlinear
    # dloss_dweights=dloss_dlinear*dlinear_dweights
    dloss_dweights=np.dot(inputs.T,dloss_dlinear)
    # dloss_dbias=dloss_dlinear*dlinear_dbias
    dloss_dbias=np.sum(dloss_dlinear)

    weights -= learning_rate*dloss_dweights
    bias -= learning_rate*dloss_dbias

    if iteration%10==0:
        print(f"Iteration: {iteration}, loss:{loss}")

print(f"Final Weights: {weights}")
print(f"Final Bias: {bias}")
print(f"Output: {output}")

        
