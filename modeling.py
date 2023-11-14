# imports 
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt

#statistical metrics:
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# model imports:
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, f_regression, SequentialFeatureSelector

#--------------- evalute function ------------------------------#
def evaluate_reg(y, yhat):
    '''
    based on two series, y_act, y_pred, (y, yhat), we
    evaluate and return the root mean squared error
    as well as the explained variance for the data.
    
    returns: rmse (float), rmse (float)
    '''
    rmse = mean_squared_error(y, yhat, squared=False)
    r2 = r2_score(y, yhat)
    return rmse, r2


def evaluate_baseline(y_true):
    """
    Evaluate a baseline model using mean prediction.

    Parameters:
    - y_true: True target values

    Returns:
    - eval_df: DataFrame containing evaluation metrics for the baseline model
    """
    baseline = y_true.mean()
    baseline_array = np.repeat(baseline, y_true.shape[0])
    
    baseline_rmse = np.sqrt(mean_squared_error(y_true, baseline_array))
    baseline_r2 = r2_score(y_true, baseline_array)
    
    eval_df = pd.DataFrame([{
        'model': 'baseline',
        'rmse': baseline_rmse,
        'r2': baseline_r2
    }])
    
    return eval_df

def add_eval_df(model_name, rmse, r2, eval_df):
    """
    Add model evaluation metrics to an existing DataFrame.

    Parameters:
    - model_name: Name of the model
    - rmse: Root Mean Squared Error
    - r2: R-squared
    - eval_df: Existing DataFrame to which the model metrics will be added

    Returns:
    - eval_df: Updated DataFrame with the new model metrics
    """
    new_row = pd.DataFrame([{
        'model': model_name,
        'rmse': rmse,
        'r2': r2
    }])
    
    eval_df = pd.concat([eval_df, new_row], ignore_index=True)
    
    return eval_df

# create validation df 
def evaluate_validate(model_name, y_true, y_pred):
    """
    Evaluate a model's predictions and create a DataFrame with the results.

    Parameters:
    - model_name: Name of the model
    - y_true: True target values
    - y_pred: Predicted values from the model

    Returns:
    - model_df: DataFrame containing model evaluation metrics
    """
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    
    val_df = pd.DataFrame([{
        'model': model_name,
        'val_rmse': rmse,
        'val_r2': r2
    }])
    
    return val_df







#------------------------------Visual Functions-----------------------#



