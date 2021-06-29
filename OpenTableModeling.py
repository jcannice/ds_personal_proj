"""
Created on Thur Jun 24
@author: Joe Cannice
*** adapted from Ken Jee ***

Project: OpenTable
Py: Model Building
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import *
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# import data frame
bay_df = pd.read_csv('bay_df.csv').dropna()

print(bay_df.columns)

# select feature columns
model_df = bay_df[['Rating', 'City', 'Review Count', 'Promoted',
                   'Price', 'Cuisine', 'Median Household Income (USD)']]

# convert to dummy variables
dummy_df = pd.get_dummies(model_df)

# train test split
X = dummy_df.drop('Rating', axis=1)
y = dummy_df.Rating.values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# multiple linear regression by OLS
X_sm = X = sm.add_constant(X) #add constant to fit line

model = sm.OLS(y, X_sm)
model.fit().summary()

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# lr model produces very poor cross val scores
cross_val_score(estimator=lr_model, X=X_train, y=y_train, scoring = 'neg_mean_absolute_error', cv=3)
 
# lasso regression to account for abundance of one hot features
lasso = Lasso()
lasso.fit(X_train, y_train)
np.mean(cross_val_score(lasso, X_train, y_train, scoring = 'neg_mean_absolute_error', cv=3))

# find best alpha
alpha = []
error = []

for i in range(1, 100):
    alpha.append(i/1000)
    lasso_a = Lasso(alpha=i/1000)
    error.append(np.mean(cross_val_score(lasso_a, X_train, y_train, scoring = 'neg_mean_absolute_error', cv=3)))
    
plt.plot(alpha, error)

alpha_max = alpha[np.argmax(error)] 

# random forest
rf = RandomForestRegressor(random_state=42)
np.mean(cross_val_score(rf, X_train, y_train, scoring = 'neg_mean_absolute_error', cv=3))

# select best rf through GridSearchCV
params = {'n_estimators': range(10, 50, 10), 'criterion': ('mse', 'mae'), 'max_features': ('auto', 'sqrt', 'log2')}
gs = GridSearchCV(rf, params, scoring = 'neg_mean_absolute_error', cv=3)
gs.fit(X_train, y_train)

best_s = gs.best_score_
best_e = gs.best_estimator_

# test ensembles
pred_lr = lr_model.predict(X_test)
pred_lasso = lasso.predict(X_test)
pred_rf = best_e.predict(X_test)

mean_absolute_error(y_test, pred_lr)
mean_absolute_error(y_test, pred_lasso)
mean_absolute_error(y_test, pred_rf)


# export model to pickle
import pickle
filename = 'model.sav'
pickl = {'model': best_e}
pickle.dump(pickl, open(filename, 'wb'))

# check model works taking input
with open(filename, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']
    
model.predict(X_test.iloc[0].values.reshape(1, -1))

list(X_test.iloc[0])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    