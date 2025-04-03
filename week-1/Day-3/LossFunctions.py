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

print("\nmse:",mse)



# MAE - primary goal Regression Task
"""
advantage-  1) less sensitive to outlier as it take absolute of error
            2) penalies all error equally, i.e. large or small error

disadvantage-   1) Not differentiable at Zero
                2) not useful when large error need to punish more than small errors!
"""
y_actual=np.array([3.2,5.8,8,9])
y_predicted=np.array([3,5,8.9,8.7])

mae = np.mean(np.abs(y_actual-y_predicted))
print("\nMAE:",mae)

