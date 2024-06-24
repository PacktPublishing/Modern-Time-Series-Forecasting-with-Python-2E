import random
import warnings
import pandas as pd
import numpy as np
#from darts import TimeSeries
from statsforecast.models import *
from statsforecast.core import StatsForecast

from src.utils.ts_utils import _remove_nan_union


def sse(y_true: np.ndarray, y_pred: np.ndarray):
    y_true, y_pred = _remove_nan_union(y_true, y_pred)
    return np.sum((y_true - y_pred) ** 2)


def block_shuffle(x, num_blocks):
    sh_array = np.array_split(x, num_blocks)
    random.shuffle(sh_array)
    return np.concatenate(sh_array)

def _remove_nan_union(y_true: np.ndarray, y_pred: np.ndarray):
    """Remove NaN values from both y_true and y_pred arrays to ensure they are aligned."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return y_true[mask], y_pred[mask]


def expand_to_df(x, start_date='2000-01-01', freq='30min', unique_id='default_id'):
    #x.rename(columns = {'energy_consumption':target_col}, inplace = True)
    data_arrays = np.array(x).T
    dates = pd.date_range(start=start_date, periods=data_arrays.shape[0], freq=freq)
    df = pd.DataFrame(data_arrays, columns=['y','shuffled_y'])
    df['unique_id'] = unique_id  # Add a column with the unique identifier
    df['ds'] = dates  # Add the datetime as a regular column named 'ds'
    return df

def nixtla_backtest(x, model, block_size, backtesting_start, n_folds, freq):
    temp_array = block_shuffle(x, num_blocks = len(x) // block_size )
    x = np.vstack([x, temp_array])

    # Define the output dictionary for SSE values
    sse_results = {}
    temp = expand_to_df(x, freq = freq)

    # Assuming cross_val_cols are columns to be validated
    cross_val_cols = ['y', 'shuffled_y']
    sf = StatsForecast( 
            models = [model], 
            freq = freq )
    
    for col_name in cross_val_cols:
        
        history = int(len(temp) * backtesting_start)
        #print(history)
        crossvalidation_df = sf.cross_validation(
                    df = temp,
                    h = history // n_folds,
                    step_size = history // n_folds,
                    n_windows = n_folds,
                    target_col = col_name )
        #print(crossvalidation_df.head())
        #print(col_name)
        sse_results[f'sse_{col_name}'] = sse(crossvalidation_df[col_name], crossvalidation_df[model.__class__.__name__])
        #print(sse_results)
    return sse_results

def _remove_nan_union(y_true: np.ndarray, y_pred: np.ndarray):
    """Remove NaN values from both y_true and y_pred arrays to ensure they are aligned."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return y_true[mask], y_pred[mask]
   
def sse(y_true: np.ndarray, y_pred: np.ndarray):
    y_true, y_pred = _remove_nan_union(y_true, y_pred)
    return np.sum((y_true - y_pred) ** 2)   

def kaboudan_metric(x, model, block_size , backtesting_start , n_folds , freq ):
    scores = nixtla_backtest(x, model, block_size, backtesting_start, n_folds, freq)
    return 1 - (scores['sse_y'] / scores['sse_shuffled_y'])

def modified_kaboudan_metric(x, model, block_size , backtesting_start , n_folds , freq ):
    scores = nixtla_backtest(x, model, block_size, backtesting_start, n_folds, freq)
    return np.clip(1 - np.sqrt(scores['sse_y'] / scores['sse_shuffled_y']), 0, None)

# def _backtest(model, x, backtesting_start, n_folds):
#     history_len = int(len(x) * backtesting_start)
#     train_x = x[:history_len]
#     test_x = x[history_len:]
#     blocks = np.array_split(test_x, n_folds)
#     metric_l = []
#     for i, block in enumerate(blocks):
#         x_ = TimeSeries.from_values(train_x)
#         with warnings.catch_warnings():
#             warnings.filterwarnings("ignore", category=FutureWarning)
#             model.fit(x_)
#         y_pred = model.predict(len(block))
#         metric_l.append(sse(block, np.squeeze(y_pred.data_array().values)))
#         if i < len(blocks) - 1:
#             train_x = np.concatenate([train_x, block])
#     return np.mean(metric_l) if len(metric_l) > 1 else metric_l[0]


# def kaboudan_metric(x, model, block_size=5, backtesting_start=0.5, n_folds=1):
#     sse_before = _backtest(model, x, backtesting_start, n_folds)
#     x_shuffled = block_shuffle(x, num_blocks=len(x) // block_size)
#     sse_after = _backtest(model, x_shuffled, backtesting_start, n_folds)
#     return 1 - (sse_before / sse_after)



# def modified_kaboudan_metric(x, model, block_size=5, backtesting_start=0.5, n_folds=1):
#     sse_before = _backtest(model, x, backtesting_start, n_folds)
#     x_shuffled = block_shuffle(x, num_blocks=len(x) // block_size)
#     sse_after = _backtest(model, x_shuffled, backtesting_start, n_folds)
#     return np.clip(1 - np.sqrt(sse_before / sse_after), 0, None)
