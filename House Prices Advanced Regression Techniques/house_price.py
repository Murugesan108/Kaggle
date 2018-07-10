import pandas as py
import numpy as np
import os
from scipy.stats import skew
from sklearn.linear_model import Ridge,RidgeCV, LassoCV

os.getcwd()
os.chdir("C:\\Users\\murugesan.r\\Desktop\\Kaggle\\House Price")

os.listdir(os.getcwd())

house_train = py.read_csv("train.csv")
house_test = py.read_csv("test.csv")

house_train.shape
house_test.shape

house_train
house_train.head()

#Dimention
house_dim = house_train.shape #1460 81
house_train.dtypes

house_train.count()
#Features with large NA values ######################################################
relevant_cols = house_train.columns[~((len(house_train) - house_train.count()) > 1300)]
house_train.iloc[:,74].isnull().sum()
#
house_train = house_train[relevant_cols]
house_train.shape #1460 78
house_train.dtypes

#Correlation Matrix 

correlation_matrix = py.DataFrame.corr(house_train.iloc[:,0:house_dim[1]])
py.DataFrame.to_csv(correlation_matrix, "correlation matrix.csv")

#Plotting historgram to see the distribution #########################################
house_train.SalePrice.hist()

### Things to do:
#VIF
#Understanding important variable
#Get R squared and other validation terms
#Implement simple linear regression and submit


#To get all the values with NA values being dropped.
# house_train.SalePrice.dropna()
# house_train.SalePrice.skew() # To compute the Skewness

############## DATA PREPROCESSING ###############################################

# combine test and train together
req_columns = house_train.columns[1:house_train.shape[1]-1]

overall_data = py.concat((house_train[req_columns],house_test[req_columns]))
overall_data.shape

price = house_train.SalePrice
log_price = np.log1p(price)

price.hist()
log_price.hist()

#Check all the numeric columns if they have skewness in their distribution
numeric_cols = overall_data.dtypes[overall_data.dtypes != "object"].index

skew_rate = house_train[numeric_cols].apply(lambda x: skew(x.dropna()))
abnormal_col = skew_rate[(skew_rate > 0.75) | (skew_rate < -0.75)].index

overall_data[abnormal_col] = np.log1p(overall_data[abnormal_col])

overall_data = py.get_dummies(overall_data)
overall_data = overall_data.fillna(overall_data.mean())

#Get the trian and test data separately
#
train_X = overall_data[:1460]
train_Y = house_train.SalePrice

test_X = overall_data[1460:len(overall_data)]

########## Ridge Regression
#from sklearn.model_selection import cross_val_score

from sklearn.cross_validation import cross_val_score

#Negative sign is because "mean_squared_error" returns the values with flipped signs
def rmse(model):
    return (np.sqrt(-cross_val_score(model,train_X,train_Y,scoring = "mean_squared_error",cv = 10)))


alpha_values = [0.01,0.05,0.5,0.3,0.1,1,3,5,8,10,15,20,25,35,50,75]
#ridge_model = Ridge()

ridge_rmse = [rmse(Ridge(alpha = alpha_val)).mean() for alpha_val in alpha_values]

cv_ridge = py.Series(ridge_rmse, index = alpha_values)
cv_ridge.plot()

#One idea to try here is run Lasso a few times on
# boostrapped samples and see how stable the feature selection is.