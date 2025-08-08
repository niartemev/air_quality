#Bagging tree regressor optimized by Grid Search

#Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor


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

#Intitliaze a bagging regressor with decision tree as base estimator
bagger = BaggingRegressor(estimator=DecisionTreeRegressor())

#Train the model
bagger.fit(X_train, y_train)

#Evaluate and output coef. of determination
print('R2:', bagger.score(X_test,y_test))

#Run predictions
y_pred = bagger.predict(X_test)

#Evaluate MAE
mape = np.mean(np.abs((y_test - y_pred) / np.abs(y_test)))

#Evaluate MAPE and print
print('Mean Absolute Percentage Error (MAPE):', round(mape * 100, 2))

#Intitliaze list of regularizers for optimization
param_grid = {
    'estimator__max_features': [0.5,0.7,0.85,0.9,1.0],
    'estimator__max_depth': [5,7,10,12,14],
    'n_estimators':[300]
}

#Intitialize Grid Search optimizer
grid_search = GridSearchCV(estimator=bagger, param_grid=param_grid, cv=10, scoring='r2')

#Find optimal parameters and train model
grid_search.fit(X_train, y_train)

#Run predictions
optimized_pred = grid_search.predict(X_test)

#Evaluate coef. of determination & print output
print('R2 optimized:', grid_search.score(X_test,y_test))

#Evaluate MAE
mape = np.mean(np.abs((y_test - optimized_pred) / np.abs(y_test)))

#Evaluate MAPE & print output
print('MAPE optimized:', round(mape * 100, 2))





