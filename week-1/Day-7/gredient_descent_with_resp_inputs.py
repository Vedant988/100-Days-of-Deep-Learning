"""
Calculating dl_dx is not considered the way to Train any NN, as this doesnt help in training because x is not trainable
- Basically it shows how the loss would change if we could move inputs !!
- as we are not allowed to change input data !! as it part of your Dataset Not your Model !!


where Do we USE dl_dx:
- Adversial ML 
- Future Attribution (saliency Map - to check, visualize which part of image makes this image class predict i.e 'A')
- Unsupervised Learning (i.e. DeepDream, GAN latent Manipulation)
"""


import numpy as np

dl_dz = np.array([[1,2,3],
                  [3,4,5],
                  [4,5,6]])

weights = np.array([[3,3,3],
                    [1,1,1],
                    [2,4,5]])

dinput=np.dot(dl_dz,weights.T)
# dinput == dl_dx
print(f"weights:\n{weights}\n")
print(f"dl_dz matrix:\n{dl_dz}\n")
print(f"GREDIENT DESCENT WRT INPUTS:\n{dinput}\n")
