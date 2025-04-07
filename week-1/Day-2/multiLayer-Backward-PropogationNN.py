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
        # print("\nBackward Propogation Starts..")
        m=X.shape[0]
        dz2=self.a2-y
        # print("\ndz2.shape:",dz2.shape)
        dw2=np.dot(self.a1.T,dz2)/m
        # print("\ndw2.shape:",dw2.shape)
        db2=np.sum(dz2,axis=0,keepdims=True)/m
        # print("\ndb2.shape:",db2.shape)

        da1=np.dot(dz2,self.w2.T)
        dz1=da1*self.relu_derivative(self.z1)
        dw1=np.dot(X.T,dz1)/m
        db1=np.sum(dz1,axis=0,keepdims=True)/m

        self.w2 -= self.learning_rate*dw2
        self.b2 -= self.learning_rate*db2
        self.w1 -= self.learning_rate*dw1
        self.b1 -= self.learning_rate*db1
        # print("\nBackward Propogation ENDS..")
        # print(f"\nw2: {self.w2}")
        # print(f"b2: {self.b2}")
        # print(f"w1: {self.w1}")
        # print(f"b1: {self.b1}")
        return True



    def train(self,X_train,y_train,epochs):
        for i in range(epochs):
            self.forward(X_train)
            # if i%100==0:
            #     print("\nz1.shape:",self.z1.shape)
            #     print("\na1.shape:",self.a1.shape)
            #     print("\nz2.shape:",self.z2.shape)
            #     print("\na2.shape:",self.a2.shape)
            self.backward(X_train,y_train)
            if i%100==0:
                loss=np.mean((self.a2-y_train)**2)
                print("\nEpoch:",i+1,"\t","loss:",loss)
                print(f"\n-------------- epoch: {i} completed --------------\n")
        return self.a2


if __name__=='__main__':
    X_train=np.random.rand(100,3)
    y_train=np.random.randint(0,2,(100,1))
    nn=MultiLayer(input_neuron=3,hidden_neuron=5,output_neuron=1,learning_rate=0.01)
    output = nn.train(X_train,y_train,epochs=100000)
    print(f"Output: {output[:5]}")
    # print(y_train)