import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

dataFrame = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')
#Set input to all but last column
input = dataFrame.drop(['logS'], axis = 1)
#Select Last Column
output = dataFrame.logS

#Create test and training sets by splitting the data
input_train, input_test, output_train, output_test = train_test_split(input, output, test_size = .2, random_state = 36)

#Try ML using a linear regression
lr = LinearRegression()
lr.fit(input_train, output_train)


output_lr_train_pred = lr.predict(input_train)
output_lr_test_pred = lr.predict(input_test)

lr_train_mse = mean_squared_error(output_train, output_lr_train_pred)
lr_train_rSquared = r2_score(output_train, output_lr_train_pred)

lr_test_mse = mean_squared_error(output_test, output_lr_test_pred)
lr_test_rSquared = r2_score(output_test, output_lr_test_pred)

print(lr_train_mse)

linRegResults = pd.DataFrame(['Linear Regression', lr_train_mse, lr_train_rSquared, lr_test_mse, lr_test_rSquared]).transpose()
linRegResults.columns = ['Method', 'Training MSE', 'Training R-Squared', 'Testing MSE', 'Testing R-Squared']

print(linRegResults)

#Try ML using Random Forest

#RandomForestRegressor is used for when output (y) variable comprises of numerical values
#If you want to use categorial variables, use RandomForestClassifier

randomForest = RandomForestRegressor(max_depth = 2, random_state = 19)
randomForest.fit(input_train, output_train)

output_randomForest_train_pred = randomForest.predict(input_train)
output_randomForest_test_pred = randomForest.predict(input_test)

randomForest_train_mse = mean_squared_error(output_train, output_randomForest_train_pred)
randomForest_train_r2 = r2_score(output_train, output_randomForest_train_pred)

randomForest_test_mse = mean_squared_error(output_test, output_randomForest_test_pred)
randomForest_test_r2 = r2_score(output_test, output_randomForest_test_pred)

randomForestResults = pd.DataFrame(['Random Forest', randomForest_train_mse, randomForest_train_r2, randomForest_test_mse, randomForest_test_r2]).transpose()
randomForestResults.columns = ['Method', 'Training MSE', 'Training R-Squared', 'Testing MSE', 'Testing R-Squared']

print(randomForestResults)
