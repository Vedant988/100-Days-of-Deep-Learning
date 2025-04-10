"""
General approach to train NN models
- as this change the model params (such as weights and biases) to minimalize the loss and train your model not DATASET haha..
"""
import numpy as np

# dl_dw = dl_dy * dy_da * da_dz * dz_dw
# assuming loss as MSE

"""
dl_dy = 2 * (y-y_true)
dy_da = 1
da_dz = relu_derivative(Z)
dz_dw = input11

dl_dz = dl_dy * dy_da * da_dz
dl_dz = 2* (y-y_true) * 1 * relu_derivative


""" 
z = np.array([[3,4,4.8],
              [2,-2.7,1],
              [-2.2,6,1.2]])

inputs=np.array([[2,6,-1.5],
                 [2.7,3.1,-1.1],
                 [2.2,3.8,-0.77]])

a = np.maximum(z,0)
y_true = np.ones_like(a)

dl_dy = 2* (a-y_true)/a.shape[0]
dy_da = 1
da_dz = np.where(z>0,1,0)

dl_dz = dl_dy * dy_da * da_dz

dl_dw_manual = np.zeros((3,3))
dl_db_manual = np.zeros((1,3))

for i in range(3):
    x_i=inputs[i].reshape(3,1)         
    dz_i = dl_dz[i].reshape(1,3)         
    dl_dw_manual += np.dot(x_i, dz_i)
    dl_db_manual += dz_i

print(f"Manual dl_dw: {dl_dw_manual}")
print(f"Manual dl_db: {dl_db_manual}\n")

dl_dw = np.dot(inputs.T,dl_dz)
dl_db = np.sum(dl_dz,axis=0,keepdims=True)


print(f"Matrix dl_dw: {dl_dw}")
print(f"Matrix dl_db: {dl_db}\n")
