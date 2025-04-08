import numpy as np
inputs = np.array([1.0,2.0,-1.0])
weights=np.random.randn(4,3)
biases = np.array([-1.1,2.6,0.5,9.99])
print(f'weights:{weights}')

learning_rate=0.001
epochs=100

def relu(z):
    return np.maximum(z,0)

def relu_derivative(z):
    return np.where(z>0,1,0)

for i in range(epochs):
    z = np.dot(weights,inputs)+biases
    a = relu(z)
    y = np.sum(a)

    loss=y**2

    dl_dy=2*y
    dy_da=np.ones_like(a)
    
    da_dz=relu_derivative(z)
    dl_dz = dl_dy * dy_da * da_dz

    dl_dw = np.outer(dl_dz,inputs)
    dl_db = dl_dz
    
    weights = weights - learning_rate* dl_dw
    biases = biases - learning_rate* dl_db

    if i%10:
        print(f'epochs:{i}, loss:{loss}')

print(f'Output: {y}')