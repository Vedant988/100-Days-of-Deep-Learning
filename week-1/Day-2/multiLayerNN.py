import numpy as np


class MultiLayer:
    def __init__(self,input_neuron,hidden_neuron,output_neuron,learning_rate=0.01):
        self.input_neuron=input_neuron
        self.output_neuron=output_neuron
        self.hidden_neuron=hidden_neuron
        self.learning_rate=learning_rate

        self.w1=np.random.randn(self.input_neuron,self.hidden_neuron)*0.01
        self.w2=np.random.randn(self.hidden_neuron,self.output_neuron)*0.01
        self.b1=np.zeros((1,self.hidden_neuron))
        self.b2=np.zeros((1,self.output_neuron))

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def relu(self,z):
        return np.maximum(0,z)
    
    def sigmoid_derivative(self,z):
        return z*(1-z)
    
    def relu_derivative(self,z):
        return(z>0).astype(float)
    
    def forward(self,X):
        self.z1=np.dot(X,self.w1)+self.b1
        self.a1=self.relu(self.z1)
        self.z2=np.dot(self.a1,self.w2)+self.b2
        self.a2=self.sigmoid(self.z2)
        return self.a2
    
    def backward(self,X,y):
        m=X.shape[0]

        
    




