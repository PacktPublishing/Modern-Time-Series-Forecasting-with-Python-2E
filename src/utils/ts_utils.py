import numpy as np
from functools import partial
from src.decomposition.seasonal import _detrend
from typing import Callable, Dict, List

def make_stationary(x: np.ndarray, method: str="detrend", detrend_kwargs:dict={}):
    """Utility to make time series stationary

    Args:
        x (np.ndarray): The time series array to be made stationary
        method (str, optional): {"detrend","logdiff"}. Defaults to "detrend".
        detrend_kwargs (dict, optional): These kwargs will be passed on to the detrend method
    """
    if method=="detrend":
        detrend_kwargs["return_trend"] = True
        stationary, trend = _detrend(x, **detrend_kwargs)
        def inverse_transform(st, trend):
            return st+trend
        return stationary, partial(inverse_transform, trend=trend)
    elif method == "logdiff":
        stationary = np.log(x[:-1]/x[1:])
        def inverse_transform(st, x):
            _x = np.exp(st)
            return _x*x[1:]
        return stationary, partial(inverse_transform, x=x)

#from darts import TimeSeries
#from darts.metrics.metrics import _get_values_or_raise
#from darts.metrics import metrics as dart_metrics
from datasetsforecast.losses import *
from typing import Optional, Tuple, Union, Sequence, Callable, cast
from src.utils.data_utils import is_datetime_dtypes
import pandas as pd



def _remove_nan_union(array_a: np.ndarray,
                      array_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the two inputs arrays where all elements are deleted that have an index that corresponds to
    a NaN value in either of the two input arrays.
    """

    isnan_mask = np.logical_or(np.isnan(array_a), np.isnan(array_b))
    return np.delete(array_a, isnan_mask), np.delete(array_b, isnan_mask)

def _zero_to_nan(series: Union[pd.Series, "pl.Expr"]) -> Union[pd.Series, "pl.Expr"]:
    if isinstance(series, pd.Series):
        res = series.replace(0, np.nan)
    else:
        res = pl.when(series == 0).then(float("nan")).otherwise(series.abs())
    return res

def forecast_bias_NIXTLA(
    df: pd.DataFrame,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
) -> Union[float, np.ndarray]:
    """Forecast Bias (FB)

    Forecast bias measures the percentage of over or under forecasting by the model.
    If the bias is negative, it means the model is underforecasting, meaning the 
    forecast is less than actuals. If the bias is positive, it means the model 
    is overforecasting, meaning the forecast is greater than the actuals."""

    # Select only the necessary columns before grouping
    columns_to_sum = models + [target_col, id_col]
    temp_df = df[columns_to_sum]

    # Group the data by 'id_col' and sum the required columns
    temp = temp_df.groupby(id_col, observed=True).sum()

    # Calculate the forecast bias
    res = (
        temp[models]
        .sub(temp[target_col], axis=0)  # Subtract the target column from the models predictions
        .div(_zero_to_nan(temp[target_col].abs()), axis=0)  # Normalize by the absolute values of the target column
        .fillna(0)*100  # Fill NA with zero
    )

    # Set the index name and reset the index to make 'id_col' a column again
    res.index.name = id_col
    res = res.reset_index()

    return res

# recreated DARTS function to remove dependency

def _get_values(
    vals: np.ndarray, stochastic_quantile: Optional[float] = 0.5
) -> np.ndarray:
    """
    Returns a deterministic or probabilistic numpy array from the values of a time series.
    For stochastic input values, return either all sample values with (stochastic_quantile=None) or the quantile sample
    value with (stochastic_quantile {>=0,<=1})
    """
    if vals.shape[2] == 1:  # deterministic
        out = vals[:, :, 0]
    else:  # stochastic
        if stochastic_quantile is None:
            out = vals
        else:
            out = np.quantile(vals, stochastic_quantile, axis=2)
    return out


def forecast_bias(actual_series: Union[ np.ndarray],
        pred_series: Union[ np.ndarray],
        intersect: bool = True,
        *,
        reduction: Callable[[np.ndarray], float] = np.mean,
        inter_reduction: Callable[[np.ndarray], Union[float, np.ndarray]] = lambda x: x,
        n_jobs: int = 1,
        verbose: bool = False) -> Union[float, np.ndarray]:
    """ Forecast Bias (FB).

    Given a time series of actual values :math:`y_t` and a time series of predicted values :math:`\\hat{y}_t`
    both of length :math:`T`, it is a percentage value computed as

    .. math:: 100 \\cdot \\frac{\\sum_{t=1}^{T}{y_t}
              - \\sum_{t=1}^{T}{\\hat{y}_t}}{\\sum_{t=1}^{T}{y_t}}.

    If any of the series is stochastic (containing several samples), the median sample value is considered.

    Parameters
    ----------
    actual_series
        The `TimeSeries` of actual values.
    pred_series
        The `TimeSeries` of predicted values.
    intersect
        For time series that are overlapping in time without having the same time index, setting `intersect=True`
        will consider the values only over their common time interval (intersection in time).
    reduction
        Function taking as input a `np.ndarray` and returning a scalar value. This function is used to aggregate
        the metrics of different components in case of multivariate TimeSeries instances.
    inter_reduction
        Function taking as input a `np.ndarray` and returning either a scalar value or a `np.ndarray`.
        This function can be used to aggregate the metrics of different series in case the metric is evaluated on a
        TimeSeries. Defaults to the identity function, which returns the pairwise metrics for each pair
        of `TimeSeries` received in input. Example: `inter_reduction=np.mean`, will return the average of the pairwise
        metrics.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a TimeSeries is
        passed as input, parallelising operations regarding different TimeSerie`. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Raises
    ------
    ValueError
        If :math:`\\sum_{t=1}^{T}{y_t} = 0`.

    Returns
    -------
    float
        The Forecast Bias (OPE)
    """
    assert type(actual_series) is type(pred_series), "actual_series and pred_series should be of same type."
    if isinstance(actual_series, np.ndarray):
        y_true, y_pred = actual_series, pred_series
    else:
        y_true = actual_series
        y_pred = pred_series
    #     y_true, y_pred = _get_values_or_raise(actual_series, pred_series, intersect)
    #y_true, y_pred = _remove_nan_union(y_true, y_pred)
    y_true_sum, y_pred_sum = np.sum(y_true), np.sum(y_pred)
    # raise_if_not(y_true_sum > 0, 'The series of actual value cannot sum to zero when computing OPE.', logger)
    return ((y_true_sum - y_pred_sum) / y_true_sum) * 100.

def cast_to_series(df):
    is_pd_dataframe = isinstance(df, pd.DataFrame)    
    if is_pd_dataframe: 
        if df.shape[1]==1:
            df = df.squeeze()
        else:
            raise ValueError("Dataframes with more than one columns cannot be converted to pd.Series")
    return df

def metrics_adapter(metric_func, actual_series,
        pred_series,
        insample = None,
        m: Optional[int] = 1,
        intersect: bool = True,
        reduction: Callable[[np.ndarray], float] = np.mean,
        inter_reduction: Callable[[np.ndarray], Union[float, np.ndarray]] = lambda x: x,
        n_jobs: int = 1,
        verbose: bool = False):
    """
    This function is designed to adapt various time series metrics functions to ensure they are compatible with the expected input types, handling both numpy arrays and pandas Series. It standardizes input time series data before passing them to the specified metric function, providing a flexible interface for time series analysis.
    Parameters
    ----------
    metric_func (Callable): 
        The metric function to be applied. It should accept at least two arguments: the actual series and the predicted series. Optional parameters like `insample` for MASE can also be passed if supported by the function.
    actual_series (np.ndarray or pd.Series): 
        The actual values of the time series.
    pred_series (np.ndarray or pd.Series): 
        The predicted values of the time series.
    insample (np.ndarray or pd.Series, optional): 
        The in-sample data needed for some metrics like MASE. Default is None.
    m (Optional[int]): 
        An optional integer parameter that might be needed for some metrics, like seasonal MASE, to specify the seasonality. Default is 1.
    intersect (bool): 
        If True, aligns the actual and predicted series on their intersection based on their time indices before applying the metric. Default is True.
    reduction (Callable[[np.ndarray], float]): 
        A function to reduce the computed metric across dimensions in case of multivariate time series. Default is `np.mean`.
    inter_reduction (Callable[[np.ndarray], Union[float, np.ndarray]]): 
        A function to reduce or transform the intermediate computed metrics across multiple time series or ensemble predictions. Defaults to an identity function, which returns the metrics as-is.
    n_jobs (int): 
        Number of parallel jobs to run when the metric function supports parallel execution. Defaults to 1 (sequential execution).
    verbose (bool): 
        If True, additional details regarding the computation process may be printed. Useful for debugging or tracking computation progress. Defaults to False.
    Returns
    ----------
    ValueError: 
        If the time index is required (e.g., for MASE) and not available in the series inputs.
    AssertionError: 
        If the input series types do not match or expected conditions for parameters are not met.
 
    """
    actual_series, pred_series = cast_to_series(actual_series), cast_to_series(pred_series)
    if insample is not None:
        insample = cast_to_series(insample)
    assert type(actual_series) is type(pred_series), f"actual_series({type(actual_series)}) and pred_series({type(pred_series)}) should be of same type."
    if insample is not None:
        assert type(actual_series) is type(insample), "actual_series and insample should be of same type."
    is_nd_array = isinstance(actual_series, np.ndarray)
    is_pd_series = isinstance(actual_series, pd.Series)
    
    if is_pd_series:
        is_datetime_index = is_datetime_dtypes(actual_series.index) and is_datetime_dtypes(pred_series.index)
        if insample is not None:
            is_datetime_index = is_datetime_index and is_datetime_dtypes(insample.index)
    else:
        is_datetime_index = False
    if metric_func.__name__ == "mase":
        if not is_datetime_index:
            raise ValueError("MASE needs pandas Series with datetime index as inputs")
    
    if metric_func.__name__ == "mase":
        #return metric_func(actual_series=actual_series, pred_series=pred_series, insample=insample, m=m, intersect=intersect, reduction=reduction, inter_reduction=inter_reduction, n_jobs=n_jobs, verbose=verbose)
        return metric_func(actual_series, pred_series, insample)

    else:
        #return metric_func(actual_series=actual_series, pred_series=pred_series, intersect=intersect, reduction=reduction, inter_reduction=inter_reduction, n_jobs=n_jobs, verbose=verbose)
        return metric_func(actual_series, pred_series)

def mae(actuals, predictions):
    return np.nanmean(np.abs(actuals-predictions))

def mse(actuals, predictions):
    return np.nanmean(np.power(actuals-predictions, 2))

def mase(actuals, predictions, insample):
    """
    Calculate the Mean Absolute Scaled Error (MASE).
    
    Parameters:
    actuals : np.ndarray
        Actual observed values corresponding to the predictions.
    predictions : np.ndarray
        Predicted values.
    insample : np.ndarray
        In-sample data to calculate the scaling factor based on a naive forecast.

    Returns:
    float
        The MASE metric.
    """
    # Calculate MAE of predictions
    mae_predictions = np.nanmean(np.abs(actuals - predictions))
    
    # Shift the insample data to create a simple naive forecast
    naive_forecast = np.roll(insample, 1)
    # Assuming the first element is not a valid forecast
    naive_forecast[0] = np.nan 
    
    # Calculate MAE of the naive forecast
    mae_naive = np.nanmean(np.abs(insample - naive_forecast))
    
    # Calculate MASE
    mase_value = mae_predictions / mae_naive
    return mase_value


def forecast_bias_aggregate(actuals, predictions):
    return 100 * (np.nansum(predictions) - np.nansum(actuals)) / np.nansum(actuals)


# Average Length that wors with utilsforecast (Nixtla)
def average_length(
    df: pd.DataFrame,
    models: List[str],
    level: int,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> pd.DataFrame:
    """Average Length of y_hat_lo and y_hat_hi.

    Parameters
    ----------
    df : pandas or polars DataFrame
        Input dataframe with id, times, actuals and predictions.
    models : list of str
        Columns that identify the models predictions.
    level : int
        Confidence level used for intervals.
    id_col : str (default='unique_id')
        Column that identifies each serie.
    target_col : str (default='y')
        Column that contains the target.

    Returns
    -------
    pandas or polars Dataframe
        dataframe with one row per id and one column per model.
    """
    if isinstance(df, pd.DataFrame):
        out = np.empty((df.shape[0], len(models)))
        for j, model in enumerate(models):
            out[:, j] = df[f"{model}-hi-{level}"] - df[f"{model}-lo-{level}"]
        res = (
            pd.DataFrame(out, columns=models, index=df.index)
            .groupby(df[id_col], observed=True)
            .mean()
        )
        res.index.name = id_col
        res = res.reset_index()
    else:
        raise NotImplementedError("Only pandas DataFrames are supported for now.")
    return res


def level_to_quantiles(l):
    if l > 1:
        l = l / 100
    return [round(q, 2) for q in [0.5 - l * 0.5, 0.5 + l * 0.5]]


def error_rate_to_quantiles(alpha):
    return [round(q, 2) for q in [alpha / 2, 1 - alpha / 2]]
