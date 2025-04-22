# in last code we found dL_da manually
# manual method calculating dL_dz1 = dL_da * da_dz1
# but calculating da_dz1 is much complex so we have,

# Direct formula for ---> dL_dz1 = (predictedm - ground_truth)

import numpy as np

# z = a1*w1​ + a2*w2 + b1 ("inputs" for the below function)
class Activation_softmax:
    def forward(self, inputs):
        exp_values=np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        self.output=exp_values/np.sum(exp_values,axis=1,keepdims=True)

class Loss_categorical:
    def calculate(self,predictions,y_true):
        samples=predictions.shape[0]
        clipped_preds=np.clip(predictions,1e-7,1-1e-7)
        correct_confidences=clipped_preds[range(samples),y_true]
        negative_log_likelihoods=-np.log(correct_confidences)
        return np.mean(negative_log_likelihoods)
    
class activation_softmax_loss_cross_entropy:
    def __init__(self):
        self.activation = Activation_softmax()
        self.loss = Loss_categorical()

    def forward(self,inputs,y_true):


    def backward(self,dvalues,y_true):