"""
1) MSE - mean squared error
2) MAE - mean absolute error
3)


    Task Type	                  Preferred Loss Function

Regression	                 =   Mean Squared Error (MSE)
Binary Classification	     =   Binary Cross-Entropy (BCE)
Multiclass Classification	 =   Categorical Cross-Entropy (CCE)
Imbalanced Classification	 =   Focal Loss (variant of BCE)
"""
import numpy as np



# 1) MSE - primary goal is to predict continuous (regression) Values
X=np.random.rand(10,3)
print(X)

weights=np.array([1,-0.27,1.88])
print(weights)

bias=2.873
y_actual=X@weights+bias+np.random.randn(10)*0.01
print(y_actual)

predicted_weights=np.array([1.1,-0.11,1.78])
predicted_bias=1.22
y_predicted=X@predicted_weights+predicted_bias
mse=np.mean((y_actual-y_predicted)**2)

print("\nMean Squared Error:",mse)



# 2) MAE - primary goal Regression Task
"""
advantage-  1) less sensitive to outlier as it take absolute of error
            2) penalies all error equally, i.e. large or small error

disadvantage-   1) Not differentiable at Zero
                2) not useful when large error need to punish more than small errors!
"""
y_actual=np.array([3.2,5.8,8,9])
y_predicted=np.array([3,5,8.9,8.7])
mae = np.mean(np.abs(y_actual-y_predicted))
print("Mean Absolute Error:",mae)


# 3) huber loss
def huber_loss(y_actual,y_predicted,delta=1):
    """
    combination of both MAE + MSE
    useful when there are multiple outlier in data
    """
    errors=y_actual-y_predicted
    loss=[]
    for error in errors:
        if abs(error)<=delta:
            loss.append(0.5*error**2)
        else:
            loss.append(delta(abs(error)-(0.5*delta)))
        return np.mean(loss)
    
y_actual=np.array([3.2,5.8,8,9])
y_predicted=np.array([3,5,8.9,8.7])
huberLoss=huber_loss(y_actual,y_predicted)
print("Huber Loss:",huberLoss)


# 4) Binary Cross Entropy / log loss
"""
Used for Binary Classifications !!
"""
def bce_loss(y_actual,y_predicted,epsilon=1e-12):
    y_predicted=np.clip(y_predicted,epsilon,1.0-epsilon)
    loss=-np.mean(y_actual*np.log(y_predicted)+(1-y_actual)*np.log(1-y_predicted))
    return loss
y_actual = np.array([1,0,1,1,0])
y_predicted=np.array([0.7,0.9,0.3,0.55,0.14])
binary_cross_entropy_loss=bce_loss(y_actual,y_predicted)
print("binary_cross_entropy_loss:",binary_cross_entropy_loss)


# 5) Categorical Class Entropy - CCE loss
"""
multiple class Prediction
"""
def cce_loss(y_actual,y_pred,epsilon=1e-12):
    y_pred=np.clip(y_pred,epsilon,1-epsilon)
    loss= -np.sum(y_actual*np.log(y_pred))/y_actual.shape[0]
    return loss
y_actual=np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1]
])
y_predicted=np.array([
    [0.7,0.2,0.1],
    [0.1,0.9,0.8],
    [0.02,0.3,0.8]
])
categorical_class_entropy = cce_loss(y_actual,y_actual)
print("categorical_class_entropy:",categorical_class_entropy)

