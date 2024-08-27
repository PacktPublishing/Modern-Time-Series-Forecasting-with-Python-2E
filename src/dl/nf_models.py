from neuralforecast.common._base_windows import BaseWindows
import torch
import torch.nn as nn
from typing import Tuple
from neuralforecast.losses.pytorch import MAE

class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1].repeat(1, (self.kernel_size - 1) // 2)
        end = x[:, -1:].repeat(1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x)
        return x


class SeriesDecomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.MovingAvg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.MovingAvg(x)
        res = x - moving_mean
        return res, moving_mean


class DNonLinear(BaseWindows):
    def __init__(
        self,
        # Inhereted hyperparameters with no defaults
        h,
        input_size,
        # Model specific hyperparameters
        moving_avg_window=3,
        dropout=0.1,
        # Inhereted hyperparameters with defaults
        exclude_insample_y=False,
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = -1,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: int = None,
        windows_batch_size=1024,
        inference_windows_batch_size=-1,
        start_padding_enabled: bool = False,
        step_size: int = 1,
        scaler_type: str = "identity",
        random_seed: int = 1,
        num_workers_loader: int = 0,
        drop_last_loader: bool = False,
        **trainer_kwargs,
    ):
        if moving_avg_window % 2 == 0:
            raise Exception("moving_avg_window should be uneven")
        super(DNonLinear, self).__init__(
            h=h,
            input_size=input_size,
            exclude_insample_y=exclude_insample_y,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=max_steps,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            windows_batch_size=windows_batch_size,
            inference_windows_batch_size=inference_windows_batch_size,
            start_padding_enabled=start_padding_enabled,
            step_size=step_size,
            num_lr_decays=num_lr_decays,
            learning_rate=learning_rate,
            early_stop_patience_steps=early_stop_patience_steps,
            scaler_type=scaler_type,
            random_seed=random_seed,
            drop_last_loader=drop_last_loader,
            num_workers_loader=num_workers_loader,
            **trainer_kwargs,
        )
        # Model specific hyperparameters
        self.moving_avg_window = moving_avg_window
        self.dropout = dropout
        # Model components
        self.decomp = SeriesDecomp(self.moving_avg_window)
        self.non_linear_block = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.input_size, 100),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(100, self.h),
        )
        self.linear_trend = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.input_size, self.h),
        )
        self.seasonality = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.input_size, self.h),
        )

    def forward(self, windows_batch):
        # Parse windows_batch
        insample_y = windows_batch[
            "insample_y"
        ].clone()  # --> (batch_size, input_size)
        seasonal_init, trend_init = self.decomp(
            insample_y
        )  # --> (batch_size, input_size)
        # Non-linear block
        non_linear_part = self.non_linear_block(
            insample_y
        )  # --> (batch_size, horizon)
        # Linear trend block
        trend_part = self.linear_trend(trend_init)  # --> (batch_size, horizon)
        # Seasonality block
        seasonal_part = self.seasonality(
            seasonal_init
        )  # --> (batch_size, horizon)
        # Combine the components
        forecast = (
            trend_part + seasonal_part + non_linear_part
        )  # --> (batch_size, horizon)
        # Map the forecast to the domain of the target
        forecast = self.loss.domain_map(forecast)
        return forecast
