import numpy as np
import random
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score


def sigmoid(z):
    return 1/(1+np.exp(-z))

class SimpleNN:
    def __init__(self,input_size,output_size):
        self.weights=np.random.randn(input_size,output_size)
        self.bias=np.random.randn(output_size)
        print("\nweights:",self.weights)
        print("\nbias:",self.bias)

    def forward(self,X):
        z = np.dot(X,self.weights)+self.bias 
        print("\nz:",z)
        final= sigmoid(z)
        print("\nfinal:",final)
        return final


iris=datasets.load_iris()
X=iris.data[:,:3]
y=iris.target
# as this is multiclass currently converting it to binary output 
yBinary = (y==0).astype(int)
# print(yBinary)
X_train,X_test,y_train,y_test = train_test_split(X,yBinary,test_size=0.3,random_state=22)
scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

nn = SimpleNN(3,1)
X_test_sample = X_train[:5]
y_test_sample = y_test[:5]
output=nn.forward(X_test_sample)

print("\nX_test_samples: ",X_test_sample)
print("\ny_test_sample:",y_test_sample)
print("\nOutput:",output)

predictions=(output>0.5).astype(int)
accuracy=accuracy_score(y_test_sample,predictions)
precision=precision_score(y_test_sample,predictions)
recall = recall_score(y_test_sample,predictions)
f1=f1_score(y_test_sample,predictions)

print("\nAccuracy:",accuracy)
print("\nPrecision:",precision)
print("\nRecall:",recall)
print("\nf1:",f1)

# nn = SimpleNN(3,1)
# X_input=np.array([[0.5,0.2,0.8],[0.7,0.2,0.4]])
# output = nn.forward(X_input)