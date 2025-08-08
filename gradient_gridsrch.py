#Gradient Boosted Trees optimized by Grid Search

#Import Libraries
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV



#Read csv
df = pd.read_csv('DQN1 Dataset.csv')



#Load features into a dataframe
df = df[['temp', 'pm2.5', 'no2','co2','humidity','dew','solarradiation','windspeed','healthRiskScore']].copy()



#Load predictor variables
X = df.iloc[:, 0:8]

#Load target values
y = df.iloc[:,8]

#Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=17, test_size=0.2)


#Instantiate a gradient booster
gbr = GradientBoostingRegressor(criterion='squared_error', learning_rate=0.1, random_state=42,n_estimators=300)

#Train the model
gbr.fit(X_train, y_train)

#Predict scores
y_pred = gbr.predict(X_test)

#Evaluate and print coefficient of determination
print('R2:', gbr.score(X_test,y_test))

#Evaluate MAE
mape = np.mean(np.abs((y_test - y_pred) / np.abs(y_test)))

#Find and print MAPE
print('Mean Absolute Percentage Error (MAPE):', round(mape * 100, 2))

#Initialize regularization parameters
param_grid = {
    'max_depth': [5,7,8],
    'max_features': [4,6,8]
}

#Initiliaze grid search optimizer
grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=5, scoring='r2')

#Train and find optimal regularizers
grid_search.fit(X_train, y_train)

#Run predictions
optimized_pred = grid_search.predict(X_test)


#Evaluate and print coefficient of determination
print('R2 optimized:', grid_search.score(X_test,y_test))


#Evaluate MAE
mape = np.mean(np.abs((y_test - optimized_pred) / np.abs(y_test)))

#Find and print MAPE
print('MAPE optimized:', round(mape * 100, 2))






