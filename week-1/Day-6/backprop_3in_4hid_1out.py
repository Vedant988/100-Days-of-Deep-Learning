import numpy as np
inputs = np.array([1.0,2.0,-1.0])
weights=np.random.randn(4,3)
biases = np.array([-1.1,2.6,0.5,9.99])
print(f'weights:{weights}')


"""
for understanding:

forward propogagtion
z1 = w11*X1 + w12*X2 + w13*X3 + b1
z2 = w21*X1 + w22*X2 + w23*X3 + b2 
z3 = w31*X1 + w32*X2 + w33*X3 + b3
z4 = w41*X1 + w42*X2 + w43*X3 + b4

activation function:
a1 = relu(z1)
a2 = relu(z2)
a3 = relu(z3)
a4 = relu(z4)

output:
y = a1+a2+a3+a4

loss:
loss = (y-0)^2

"""
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