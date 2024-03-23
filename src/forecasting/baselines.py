import numpy as np
from darts.models.forecasting.forecasting_model import LocalForecastingModel
from darts import TimeSeries


class NaiveMovingAverage(LocalForecastingModel):
    def __init__(self, window):
        """Naive Moving Average Model
        This model predicts the moving average of the last 'window' values.
        """
        super().__init__()
        self.window = window

    def __str__(self):
        return "Naive Moving Average"

    def fit(self, series: TimeSeries):
        super().fit(series)
        # No explicit action needed on fit for a naive model, but you could calculate
        # and store the moving average here if you prefer
        self.series = series

    def predict(self, n: int, num_samples: int = 1):
        super().predict(n, num_samples)
        # Create a forecast by extending the series with the mean of the last 'window' values
        last_window_values = self.series.values()[-self.window:]
        forecast_values = np.mean(last_window_values) * np.ones(n)
        return self._build_forecast_series(forecast_values)

    def supports_multivariate(self) -> bool:
        """Indicates whether the model supports multivariate time series."""
        return False

    # Depending on the version of Darts you are using, you may also need to implement
    # the following method if it's declared as abstract in LocalForecastingModel:
    def _supports_range_index(self) -> bool:
        """
        Indicates whether the model supports range index.
        """
        return True