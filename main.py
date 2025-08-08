#import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np

#read csv file
df = pd.read_csv('DQN1 Dataset.csv')
#read relevant columns into a dataframe
df = df[['temp', 'pm2.5', 'no2','co2','humidity','dew','solarradiation','windspeed','healthRiskScore']].copy()
#set x (predictors) to the first 8 columms
X = df.iloc[:, 0:8]
#set y to healthRiskScore (predicted value)
y = df.iloc[:,8]

#split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=0.2)

#initilize a random forest regressor with default parameters
rf = RandomForestRegressor()
#train the model
rf.fit(X_train, y_train)

#predict scores of the test subset
pred = rf.predict(X_test)

#calculate and print coefficient of determination
print("Coefficient of determination: " + str(rf.score(X_test, y_test)))
#determine mean absolute percentage
mape = np.mean(np.abs((y_test - pred) / np.abs(y_test)))
#print and round MAPE
print('Mean Absolute Percentage Error (MAPE):', round(mape * 100, 2))

