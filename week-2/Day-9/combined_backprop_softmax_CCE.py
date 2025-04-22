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
        self.activation.forward(inputs)
        self.output=self.activation.output
        print(f'a: This is the prediction for One-Hot-Encoding for class\n{self.output}')
        return self.loss.calculate(self.output,y_true)

    def backward(self,dvalues,y_true):
        samples=dvalues.shape[0]
        self.dinputs=dvalues.copy()
        self.dinputs[range(samples),y_true]-=1
        self.dinputs=self.dinputs/samples

# z values
logits = np.array([[2.0, 1.0, 0.1],     
                   [1.0, 3.0, 0.2],     
                   [0.2, 0.3, 0.5]])

y_true = np.array([0, 1, 2])
combined=activation_softmax_loss_cross_entropy()

loss=combined.forward(logits,y_true)
print("Loss:\n", loss)

combined.backward(combined.output,y_true)
print("\nGradients (dinputs):")
print(combined.dinputs)


"""
In Gredients,
-0.113 i.e. G[11] should closer to 1 
0.0808 i.e. G[12] should closer to 0
0.0328 i.e. G[13] should closer to 0
as  class 0 is ground truth ---> one Hot Encoding

----------> Negative Gredient PUSH them "Upward", while positive Push them "Down"
----------> which one are negative in gredient should be the one with correct class and should reach 1 in a1=softmax(z1)

"""


# misconception    --> Greater magnitude of gradient ⇒ higher chance the prediction is correct
# Correct Intution --> Larger gradient magnitude means the model is more wrong about the correct class

# so here, smaller gredient magnitude means small update in BACKPROPOGATION
# larger gredient magnitude means Large Update in BACKPROPOGATION !!!