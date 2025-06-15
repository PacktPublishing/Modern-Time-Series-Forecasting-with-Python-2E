from typing import Optional
import numpy as np
import pandas as pd
from scipy.optimize import brentq


class ConformalPrediction:
    def __init__(
        self,
        model: str,
        level: float,
        alias: str = None,
    ):
        self.model = model
        level = float(level)
        assert level > 1 and level < 100, "level should be between 1 and 100"
        self.alpha = (100 - level) / 100
        if level.is_integer():
            level = int(level)
        self.level = level
        self.alias = alias
        self.method = "Vanilla Conformal Prediction"
        self._mthd = "CP"

    def calculate_scores(self, Y_calib_df: pd.DataFrame) -> pd.DataFrame:
        Y_calib_df = Y_calib_df.copy()
        Y_calib_df["calib_scores"] = np.abs(Y_calib_df["y"] - Y_calib_df[self.model])
        return Y_calib_df

    def get_quantile(self, Y_calib_df: pd.DataFrame) -> dict:
        def get_qhat(Y_calib_df):
            n_cal = len(Y_calib_df)
            q_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
            return np.quantile(
                Y_calib_df["calib_scores"].values, q_level, method="higher"
            )

        return Y_calib_df.groupby("unique_id").apply(get_qhat).to_dict()

    def fit(self, Y_calib_df: pd.DataFrame):
        Y_calib_df = self.calculate_scores(Y_calib_df)
        self.q_hat = self.get_quantile(Y_calib_df)
        return self

    def get_prediction_interval_names(self) -> list:
        return [
            f"{self.alias or self.model}-{self._mthd}-lo-{self.level}",
            f"{self.alias or self.model}-{self._mthd}-hi-{self.level}",
        ]

    def calc_prediction_interval(self, Y_test_df: pd.DataFrame, q_hat: dict) -> tuple:
        return (
            Y_test_df[self.model] - Y_test_df["unique_id"].map(q_hat),
            Y_test_df[self.model] + Y_test_df["unique_id"].map(q_hat),
        )

    def predict(self, Y_test_df: pd.DataFrame) -> pd.DataFrame:
        Y_test_df = Y_test_df.copy()
        lo, hi = self.get_prediction_interval_names()
        Y_test_df[lo], Y_test_df[hi] = self.calc_prediction_interval(
            Y_test_df, self.q_hat
        )
        return Y_test_df
    
    @property
    def method_name(self):
        return f"{self.method} ({self._mthd})"


class ConformalizedQuantileRegression(ConformalPrediction):
    def __init__(
        self,
        model: str,
        level: Optional[float] = None,
        alias: str = None,
    ):
        super().__init__(model, level, alias)
        self.method = "Conformalized Quantile Regression"
        self._mthd = "CQR"
        self.lower_quantile_model = f"{model}-lo-{level}"
        self.upper_quantile_model = f"{model}-hi-{level}"

    def calculate_scores(self, Y_calib_df: pd.DataFrame) -> pd.DataFrame:
        Y_calib_df = Y_calib_df.copy()
        lower_bounds = Y_calib_df[self.lower_quantile_model]
        upper_bounds = Y_calib_df[self.upper_quantile_model]
        Y_calib_df["calib_scores"] = np.maximum(
            lower_bounds - Y_calib_df["y"], Y_calib_df["y"] - upper_bounds
        )
        return Y_calib_df

    def calc_prediction_interval(self, Y_test_df: pd.DataFrame, q_hat: dict) -> tuple:
        return (
            Y_test_df[self.lower_quantile_model] - Y_test_df["unique_id"].map(q_hat),
            Y_test_df[self.upper_quantile_model] + Y_test_df["unique_id"].map(q_hat),
        )


class ConformalizedUncertaintyEstimates(ConformalPrediction):
    def __init__(
        self,
        model: str,
        uncertainty_model: str,
        level: Optional[float] = None,
        alias: str = None,
    ):
        super().__init__(model, level, alias)
        self.method = "Conformalized Uncertainty Intervals"
        self._mthd = "CUE"
        self.uncertainty_model = uncertainty_model

    def calculate_scores(self, Y_calib_df: pd.DataFrame) -> pd.DataFrame:
        Y_calib_df = Y_calib_df.copy()
        uncertainty = Y_calib_df[self.uncertainty_model]
        Y_calib_df["calib_scores"] = (
            np.abs(Y_calib_df["y"] - Y_calib_df[self.model]) / uncertainty
        )
        return Y_calib_df

    def calc_prediction_interval(self, Y_test_df: pd.DataFrame, q_hat: dict) -> tuple:
        uncertainty = Y_test_df[self.uncertainty_model]
        return (
            Y_test_df[self.model] - uncertainty * Y_test_df["unique_id"].map(q_hat),
            Y_test_df[self.model] + uncertainty * Y_test_df["unique_id"].map(q_hat),
        )


class WeightedConformalPredictor:
    def __init__(
        self,
        conformal_predictor: ConformalPrediction,
        K: int,
        weight_strategy: str,
        custom_weights: list = None,
        decay_factor: float = 0.5,
    ):
        """
        Initialize the weighted predictor.

        Parameters:
        - conformal_predictor: An instance of a conformal prediction class (e.g., ConformalPrediction).
        - K: Size of the sliding window for the most recent scores.
        - use_uniform_weights: Boolean to decide if uniform weights should be used.
        - custom_weights: User-defined weights (K) for the sliding window.
        - decay_factor: If provided, use exponential decay with this factor for weights.
        """
        # add weighted tag to the method
        conformal_predictor.method += " Weighted"
        conformal_predictor._mthd += "_Wtd"
        self.conformal_predictor = conformal_predictor
        self.K = K
        assert weight_strategy in [
            "uniform",
            "custom",
            "decay",
        ], "Invalid weight strategy"
        if weight_strategy == "custom":
            assert (
                custom_weights is not None
            ), "Custom Weights should be provided for custom weight_strategy"
        self.weight_strategy = weight_strategy
        if custom_weights is not None:
            custom_weights = np.array(custom_weights)
            assert (
                custom_weights.shape[0] == K and custom_weights.ndim == 1
            ), "Custom Weights should be of dimensions K"
        self.custom_weights = custom_weights
        self.decay_factor = decay_factor
        self.scores = []

    def _calculate_weight(self, total_timesteps: int) -> np.array:
        """Calculate the weight for the most recent scores using the specified strategy."""
        if self.weight_strategy == "uniform":
            return np.array([1 / total_timesteps] * total_timesteps)
        elif self.weight_strategy == "custom":
            return self.custom_weights
        elif self.weight_strategy == "decay":
            return np.array(
                [self.decay_factor**i for i in range(total_timesteps)][::-1]
            )
        else:
            raise ValueError(
                "Custom weights or decay factor must be provided if use_uniform_weights is set to False."
            )

    @staticmethod
    def get_weighted_quantile(scores, weights: np.array, alpha: float) -> float:
        """Calculate the weighted quantile using Brent's method."""

        def critical_point_quantile(q):
            return (weights * (scores <= q)).sum() - (1 - alpha)

        return brentq(critical_point_quantile, np.min(scores), np.max(scores))

    def fit(self, Y_calib_df: pd.DataFrame):
        """Fit the model on calibration data."""
        self.calib_df = self.conformal_predictor.calculate_scores(
            Y_calib_df.sort_values(["unique_id", "ds"])
        )

    def predict(self, Y_test_df: pd.DataFrame) -> pd.DataFrame:
        """Predict intervals in one shot using the entire calibration data."""
        Y_test_df = Y_test_df.sort_values(["unique_id", "ds"])

        # Group calibration data by unique_id
        grouped_calib = self.calib_df.groupby("unique_id")

        # Calculate quantiles for each unique_id
        self.q_hat = {}
        for unique_id, group in grouped_calib:
            # Take the last K timesteps
            group = group.iloc[-self.K :]
            scores = group["calib_scores"].values

            # Calculate weights based on the last K timesteps
            total_timesteps = len(scores)
            weights = self._calculate_weight(total_timesteps)
            normalized_weights = weights / weights.sum()

            # Calculate quantile for the current unique_id
            quantile = self.get_weighted_quantile(
                scores, normalized_weights, self.conformal_predictor.alpha
            )
            self.q_hat[unique_id] = quantile

        # Calculate prediction intervals using the underlying conformal predictor's method
        lo, hi = self.conformal_predictor.get_prediction_interval_names()
        Y_test_df[lo], Y_test_df[hi] = (
            self.conformal_predictor.calc_prediction_interval(Y_test_df, self.q_hat)
        )

        return Y_test_df
    
    @property
    def method_name(self):
        return self.conformal_predictor.method_name


class OnlineWeightedConformalPredictor:
    def __init__(
        self,
        conformal_predictor: ConformalPrediction,
        K: int,
        weight_strategy: str,
        custom_weights: list = None,
        decay_factor: float = 0.5,
    ):
        """
        Initialize the online weighted predictor.

        Parameters:
        - conformal_predictor: An instance of a conformal prediction class (e.g., ConformalPrediction).
        - K: Size of the sliding window for the most recent scores.
        - use_uniform_weights: Boolean to decide if uniform weights should be used.
        - custom_weights: User-defined weights (K) for the sliding window.
        - decay_factor: If provided, use exponential decay with this factor for weights.
        """
        conformal_predictor.method += " Weighted Online"
        conformal_predictor._mthd += "_Wtd_O"
        self.conformal_predictor = conformal_predictor
        self.K = K
        assert weight_strategy in [
            "uniform",
            "custom",
            "decay",
        ], "Invalid weight strategy"
        if weight_strategy == "custom":
            assert (
                custom_weights is not None
            ), "Custom Weights should be provided for custom weight_strategy"
            custom_weights = np.array(custom_weights)
            assert (
                custom_weights.shape[0] == K and custom_weights.ndim == 1
            ), "Custom Weights should be of dimensions K"
        self.weight_strategy = weight_strategy
        self.custom_weights = custom_weights
        self.decay_factor = decay_factor
        self.scores = []
        self.calib_df = None

    def _calculate_weight(self, total_timesteps: int) -> np.array:
        """Calculate the weight for the most recent scores using the specified strategy."""
        if self.weight_strategy == "uniform":
            return np.array([1 / total_timesteps] * total_timesteps)
        elif self.weight_strategy == "custom":
            return self.custom_weights
        elif self.weight_strategy == "decay":
            return np.array(
                [self.decay_factor**i for i in range(total_timesteps)][::-1]
            )
        else:
            raise ValueError(
                "Custom weights or decay factor must be provided if use_uniform_weights is set to False."
            )

    @staticmethod
    def get_weighted_quantile(scores, weights: np.array, alpha: float) -> float:
        """Calculate the weighted quantile using Brent's method."""

        def critical_point_quantile(q):
            return (weights * (scores <= q)).sum() - (1 - alpha)

        return brentq(critical_point_quantile, np.min(scores), np.max(scores))

    def fit(self, Y_calib_df: pd.DataFrame):
        """Initialize the calibration with provided data."""
        self.calib_df = self.conformal_predictor.calculate_scores(Y_calib_df)
        self.calib_df = self._keep_last_k(self.calib_df, self.K)

    def predict_one(self, current_test):
        """Predict intervals for all unique_ids at a single time step."""
        current_test = current_test.copy()
        unique_ids = current_test["unique_id"].unique()
        predictions = []

        for unique_id in unique_ids:
            group = self.calib_df[self.calib_df["unique_id"] == unique_id].iloc[
                -self.K :
            ]

            if not group.empty:
                scores = group["calib_scores"].values
                total_timesteps = len(scores)

                # Calculate weights based on the last K timesteps
                weights = self._calculate_weight(total_timesteps)
                normalized_weights = weights / weights.sum()

                # Calculate quantile for the current unique_id
                quantile = self.get_weighted_quantile(
                    scores, normalized_weights, self.conformal_predictor.alpha
                )
                self.q_hat = {unique_id: quantile}

                # Calculate prediction intervals using the underlying conformal predictor's method
                lo, hi = self.conformal_predictor.get_prediction_interval_names()
                lower, upper = self.conformal_predictor.calc_prediction_interval(
                    current_test, self.q_hat
                )
                current_test[lo] = lower.values
                current_test[hi] = upper.values

            predictions.append(current_test[current_test["unique_id"] == unique_id])

        return pd.concat(predictions)

    @staticmethod
    def _keep_last_k(group, K: int):
        if (group.groupby("unique_id")["ds"].count() > K).any():
            return group.groupby("unique_id").tail(K)
        return group

    def update(self, new_data: pd.DataFrame):
        """Update the calibration data with new observations for all unique_ids at a single time step."""
        new_scores = self.conformal_predictor.calculate_scores(new_data)
        self.calib_df = self._keep_last_k(
            pd.concat([self.calib_df, new_scores]).reset_index(drop=True), self.K
        )

    def offline_predict(self, Y_test_df: pd.DataFrame):
        """Simulate online prediction using the fit, predict_one, and update functions."""
        Y_test_df = Y_test_df.copy()
        predictions = []

        time_steps = Y_test_df["ds"].unique()

        for ds in time_steps:
            current_test = Y_test_df[Y_test_df["ds"] == ds]
            prediction = self.predict_one(current_test)
            predictions.append(prediction)
            self.update(prediction)

        return (
            pd.concat(predictions)
            .reset_index(drop=True)
            .sort_values(["unique_id", "ds"])
        )
    
    @property
    def method_name(self):
        return self.conformal_predictor.method_name


import numpy as np
import pandas as pd

class OnlineAdaptiveConformalInference:
    def __init__(
        self,
        conformal_predictor: ConformalPrediction,
        gamma: float = 0.005,
        update_method: str = "simple",
        momentum_bw: float = 0.95,
        per_unique_id: bool = True,
    ):
        """
        Initialize the Adaptive Conformal Prediction wrapper.

        Parameters:
        - conformal_predictor: An instance of a conformal prediction class (e.g., ConformalPrediction).
        - gamma: Learning rate for updating alpha in the adaptive method.
        - update_method: Method for updating alpha ("simple" or "momentum").
        - momentum_bw: Bandwidth for the momentum update method.
        - per_unique_id: Boolean indicating whether alpha should be maintained separately for each unique_id.
        """
        conformal_predictor.method += " Adaptive"
        conformal_predictor._mthd += "_ACI"
        self.conformal_predictor = conformal_predictor
        self.alpha = conformal_predictor.alpha
        self.gamma = gamma
        self.update_method = update_method
        self.momentum_bw = momentum_bw
        self.per_unique_id = per_unique_id

        # Initialize alphas for each unique_id if per_unique_id is True
        self.alphat = {}
        self.alpha_trajectory = {}
        self.adapt_err_seq = {}
        self.predictions = {}

        # Initialize global alpha if per_unique_id is False
        if not self.per_unique_id:
            self.alphat_global = self.alpha
            self.alpha_trajectory_global = []
            self.adapt_err_seq_global = []
            self.predictions_global = None

    def fit(self, Y_calib_df: pd.DataFrame):
        """
        Fit the conformal predictor model with calibration data.
        """
        self.calib_df = self.conformal_predictor.calculate_scores(Y_calib_df)
        self.scores_by_id = (
            self.calib_df.groupby("unique_id")["calib_scores"].apply(list).to_dict()
        )
        if self.per_unique_id:
            # Initialize alpha values for each unique_id
            unique_ids = self.calib_df["unique_id"].unique()
            for uid in unique_ids:
                self.alphat[uid] = self.alpha
                self.alpha_trajectory[uid] = []
                self.adapt_err_seq[uid] = []
                self.predictions[uid] = None
        return self

    def update_alpha(self, unique_id: str, adapt_err: int):
        """
        Update the alpha value for a specific unique_id based on the chosen update method.
        """
        if self.per_unique_id:
            self.alpha_trajectory[unique_id].append(self.alphat[unique_id])
            self.adapt_err_seq[unique_id].append(adapt_err)

            if self.update_method == "simple":
                self.alphat[unique_id] = self.alphat[unique_id] + self.gamma * (
                    self.alpha - adapt_err
                )
            elif self.update_method == "momentum":
                w = np.flip(
                    self.momentum_bw
                    ** np.arange(1, len(self.adapt_err_seq[unique_id]) + 1)
                )
                w /= np.sum(w)
                self.alphat[unique_id] = self.alphat[unique_id] + self.gamma * (
                    self.alpha - np.sum(np.array(self.adapt_err_seq[unique_id]) * w)
                )
        else:
            self.alpha_trajectory_global.append(self.alphat_global)
            self.adapt_err_seq_global.append(adapt_err)

            if self.update_method == "simple":
                self.alphat_global = self.alphat_global + self.gamma * (
                    self.alpha - adapt_err
                )
            elif self.update_method == "momentum":
                w = np.flip(
                    self.momentum_bw
                    ** np.arange(1, len(self.adapt_err_seq_global) + 1)
                )
                w /= np.sum(w)
                self.alphat_global = self.alphat_global + self.gamma * (
                    self.alpha - np.sum(np.array(self.adapt_err_seq_global) * w)
                )

    def predict_one(self, current_test: pd.DataFrame) -> pd.DataFrame:
        """
        Predict intervals for all unique_ids at a single time step.
        """
        current_test = current_test.copy()
        unique_ids = current_test["unique_id"].unique()
        predictions = []

        for unique_id in unique_ids:
            group_scores = self.scores_by_id.get(unique_id, [])
            if group_scores:
                # Determine the appropriate alpha to use
                alpha = (
                    self.alphat[unique_id]
                    if self.per_unique_id
                    else self.alphat_global
                )

                # Calculate quantile for the current unique_id
                self.q_hat = {
                    unique_id: np.quantile(group_scores, 1 - alpha, method="higher")
                }

                # Calculate prediction intervals using the underlying conformal predictor's method
                lo, hi = self.conformal_predictor.get_prediction_interval_names()
                lower, upper = self.conformal_predictor.calc_prediction_interval(
                    current_test, self.q_hat
                )
                current_test[lo] = lower.values
                current_test[hi] = upper.values

                # Store the most recent prediction for later use in update
                if self.per_unique_id:
                    self.predictions[unique_id] = (lower.values[0], upper.values[0])
                else:
                    self.predictions_global = (lower.values[0], upper.values[0])

            predictions.append(current_test[current_test["unique_id"] == unique_id])

        return pd.concat(predictions)

    def update(self, new_data: pd.DataFrame):
        """
        Update the calibration data and alpha values with new observations for all unique_ids at a single time step.
        """
        new_scores = self.conformal_predictor.calculate_scores(new_data)
        for unique_id, score in zip(
            new_scores["unique_id"], new_scores["calib_scores"]
        ):
            if unique_id in self.scores_by_id:
                self.scores_by_id[unique_id].append(score)
            else:
                self.scores_by_id[unique_id] = [score]

            # Retrieve stored predictions and calculate adapt_err
            if self.per_unique_id:
                lower, upper = self.predictions[unique_id]
                actual_y = new_data.loc[new_data["unique_id"] == unique_id, "y"].values[0]
                adapt_err = int(actual_y < lower or actual_y > upper)
                self.update_alpha(unique_id, adapt_err)
            else:
                lower, upper = self.predictions_global
                actual_y = new_data["y"].values[0]
                adapt_err = int(actual_y < lower or actual_y > upper)
                self.update_alpha(None, adapt_err)

    def offline_predict(self, Y_test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate online prediction using the fit, predict_one, and update functions.
        """
        Y_test_df = Y_test_df.copy()
        predictions = []

        time_steps = Y_test_df["ds"].unique()

        for ds in time_steps:
            current_test = Y_test_df[Y_test_df["ds"] == ds]
            prediction = self.predict_one(current_test)
            predictions.append(prediction)
            self.update(current_test)

        return (
            pd.concat(predictions)
            .reset_index(drop=True)
            .sort_values(["unique_id", "ds"])
        )

    def get_trajectory(self, unique_id: str = None) -> np.array:
        """
        Returns the trajectory of alpha over time for a specific unique_id or globally.
        """
        if self.per_unique_id and unique_id is not None:
            return np.array(self.alpha_trajectory.get(unique_id, []))
        else:
            return np.array(self.alpha_trajectory_global)

    def get_errors(self, unique_id: str = None) -> np.array:
        """
        Returns the sequence of adaptive errors for a specific unique_id or globally.
        """
        if self.per_unique_id and unique_id is not None:
            return np.array(self.adapt_err_seq.get(unique_id, []))
        else:
            return np.array(self.adapt_err_seq_global)
        
    @property
    def method_name(self):
        return self.conformal_predictor.method_name



# import pandas as pd

# # Parameters
# n_cal = 50  # Number of calibration points per unique_id
# n_test = 20  # Number of test points per unique_id
# n_unique_ids = 3  # Number of unique_ids
# level = 90  # Confidence level for prediction intervals

# # Generate unique_ids
# unique_ids = [f'H{i+1}' for i in range(n_unique_ids)]

# # Generate synthetic calibration data
# calibration_data = []
# for unique_id in unique_ids:
#     ds = np.arange(1, n_cal + 1)
#     y = np.random.normal(loc=500, scale=50, size=n_cal)
#     lstm_qr_median = y + np.random.normal(loc=0, scale=30, size=n_cal)
#     lstm_qr_lo = lstm_qr_median - np.random.normal(loc=100, scale=10, size=n_cal)
#     lstm_qr_hi = lstm_qr_median + np.random.normal(loc=100, scale=10, size=n_cal)
#     calibration_data.append(pd.DataFrame({
#         'unique_id': unique_id,
#         'ds': ds,
#         'y': y,
#         'LSTM_QR': lstm_qr_median,
#         'LSTM_QR-lo-90': lstm_qr_lo,
#         'LSTM_QR-hi-90': lstm_qr_hi
#     }))

# calibration_df = pd.concat(calibration_data).reset_index(drop=True)

# # Generate synthetic test data
# test_data = []
# for unique_id in unique_ids:
#     ds = np.arange(n_cal + 1, n_cal + n_test + 1)
#     y = np.random.normal(loc=500, scale=50, size=n_test)
#     lstm_qr_median = y + np.random.normal(loc=0, scale=30, size=n_test)
#     lstm_qr_lo = lstm_qr_median - np.random.normal(loc=100, scale=10, size=n_test)
#     lstm_qr_hi = lstm_qr_median + np.random.normal(loc=100, scale=10, size=n_test)
#     test_data.append(pd.DataFrame({
#         'unique_id': unique_id,
#         'ds': ds,
#         'y': y,
#         'LSTM_QR': lstm_qr_median,
#         'LSTM_QR-lo-90': lstm_qr_lo,
#         'LSTM_QR-hi-90': lstm_qr_hi
#     }))

# test_df = pd.concat(test_data).reset_index(drop=True)

# conformal_predictor = ConformalPrediction(model='LSTM_QR', level=90)

# # Test WeightedConformalPredictor

# weighted_predictor = WeightedConformalPredictor(conformal_predictor, K=5, use_uniform_weights=False, decay_factor=None, custom_weights=[0.1, 0.2, 0.3, 0.2, 0.1])

# # Fit the weighted predictor on the calibration data
# weighted_predictor.fit(calibration_df)

# # Predict intervals on the test data
# prediction_intervals = weighted_predictor.predict(test_df)

# # Display the prediction intervals
# print(prediction_intervals)


# # Test Online weighted conformal predictor
# online_weighted_predictor = OnlineWeightedConformalPredictor(
#     conformal_predictor,
#     K=5,
#     use_uniform_weights=False,
#     decay_factor=None,
#     custom_weights=[0.1, 0.2, 0.3, 0.2, 0.1]
# )

# # Fit the online weighted predictor on the calibration data
# online_weighted_predictor.fit(calibration_df)

# # Simulate online prediction
# prediction_intervals = online_weighted_predictor.offline_predict(test_df)

# print(prediction_intervals)


# # Test Online weighted conformal predictor
# online_weighted_predictor = OnlineAdaptiveConformalInference(
#     conformal_predictor,
#     gamma=0.01,
#     update_method="simple",
#     momentum_bw=0.95,
#     per_unique_id=True
# )

# # Fit the online weighted predictor on the calibration data
# online_weighted_predictor.fit(calibration_df)

# # Simulate online prediction
# prediction_intervals = online_weighted_predictor.offline_predict(test_df)

# print(prediction_intervals)
