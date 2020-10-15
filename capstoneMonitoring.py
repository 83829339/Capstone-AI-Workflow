#!/usr/bin/env python
"""
example performance monitoring script
"""

import os, sys, pickle, re
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

from scipy.stats import wasserstein_distance

from capstoneModelBuildingSelection import *

DATA_DIR = "C:\\Architecture\\IA\\Curso Python\\Capstone\\ai-workflow-capstone-master\\cs-train"
MODEL_DIR = "models"
MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = "supervised learing model for time-series"

def get_latest_train_data():
    """
    load the data used in the latest training
    """

    data_file = os.path.join("models",'latest-train.pickle')

     
    models = [f for f in os.listdir(os.path.join(".","models")) if re.search("test",f)]

    if len(models) == 0:
        raise Exception("Models with prefix '{}' cannot be found did you train?".format("test"))


    with open(data_file,'rb') as tmp:
        data = pickle.load(tmp)


    print('data is:', data)


    return(data)

    
def get_monitoring_tools(X,y):
    """
    determine outlier and distance thresholds
    return thresholds, outlier model(s) and source distributions for distances
    NOTE: for classification the outlier detection on y is not needed

    """

    
    
    """
    preprocessor = get_preprocessor()
    preprocessor = preprocessor.fit(X)
    X_pp = preprocessor.transform(X)
    
    
    if test:
        n_samples = int(np.round(0.3 * X.shape[0]))
        subset_indices = np.random.choice(np.arange(X.shape[0]),n_samples,
                                          replace=False).astype(int)
        mask = np.in1d(np.arange(y.size),subset_indices)
        y=y[mask]
        X=X[mask]
        dates=dates[mask]
"""

    
    ## Perform a train-test split
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
    #                                                    shuffle=True, random_state=42)
    ## train a random forest model
    param_grid_rf = {
    'rf__criterion': ['mse','mae'],
    'rf__n_estimators': [10,15,20,25]
    }
 
    xpipe = Pipeline(steps=[('scaler', StandardScaler()),
                              ('rf', RandomForestRegressor())])

    #grid = GridSearchCV(xpipe, param_grid=param_grid_rf, cv=5, iid=False, n_jobs=-1)
    #grid.fit(X_train, y_train)
    #y_pred = grid.predict(X_test)


    xpipe.fit(X, y)
    
    bs_samples = 1000
    outliers_X = np.zeros(bs_samples)
    wasserstein_X = np.zeros(bs_samples)
    wasserstein_y = np.zeros(bs_samples)
    
    for b in range(bs_samples):
        """
        n_samples = int(np.round(0.30 * X.shape[0]))
        subset_indices = np.random.choice(np.arange(X.shape[0]),n_samples,replace=True).astype(int)
        #y_bs=y[subset_indices]
        #X_bs=X_pp[subset_indices,:]
        #X_bs=X_train[subset_indices,:]
        mask = np.in1d(np.arange(y.size),subset_indices)
        y_bs=y[mask]
        X_bs=X[mask]
        
        test1 = xpipe.predict(X_bs)
        wasserstein_X[b] = wasserstein_distance(X_train,X_bs)
        wasserstein_y[b] = wasserstein_distance(y_train,y_bs)
        outliers_X[b] = 100 * (1.0 - (test1[test1==1].size / test1.size))
        
        """

        n_samples = int(np.round(0.30 * X.shape[0]))
        subset_indices = np.random.choice(np.arange(X.shape[0]),n_samples,replace=True).astype(int)
        mask = np.in1d(np.arange(y.size),subset_indices)
        #print('subset_indices y mask:',subset_indices,mask)
        y_bs=y[subset_indices]
        X_bs=X[mask]
    
        test1 = xpipe.predict(X_bs)
        wasserstein_y[b] = wasserstein_distance(y,y_bs.flatten())
        outliers_X[b] = 100 * (1.0 - (test1[test1==1].size / test1.size))


    ## determine thresholds as a function of the confidence intervals
    outliers_X.sort()
    outlier_X_threshold = outliers_X[int(0.975*bs_samples)] + outliers_X[int(0.025*bs_samples)]

    wasserstein_y.sort()
    wasserstein_y_threshold = wasserstein_y[int(0.975*bs_samples)] + wasserstein_y[int(0.025*bs_samples)]
    
    to_return = {"outlier_X": np.round(outlier_X_threshold,1),
                 "wasserstein_y":np.round(wasserstein_y_threshold,2),
                 "clf_X":xpipe,
                 "latest_X":X,
                 "latest_y":y}
    return(to_return)


if __name__ == "__main__":

    ## get latest training data
    data = get_latest_train_data()
    y = data['y']
    X = data['X']

    ## get performance monitoring tools
    pm_tools = get_monitoring_tools(X,y)
    print("outlier_X",pm_tools['outlier_X'])
    #print("wasserstein_X",pm_tools['wasserstein_X'])
    print("wasserstein_y",pm_tools['wasserstein_y'])
    
    print("done")
    
