import ConfigSpace
import numpy as np
import pandas as pd
import typing
import random

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm


class SequentialModelBasedOptimization(object):

    def __init__(self, config_space, random_ratio=0.3):
        """
        Initializes empty variables for the model, the list of runs (capital R), and the incumbent
        (theta_inc being the best found hyperparameters, theta_inc_performance being the performance
        associated with it)
        """
        self.config_space = config_space  # configuration space
        self.runs = (
            []
        )  # run list, stores all evaluated (configuration, performance) tuples
        self.theta_inc = None  # the best found hyperparameters
        self.theta_inc_performance = None  # the best performance
        self.scaler = StandardScaler()
        self.model = GaussianProcessRegressor()
        self.random_ratio = random_ratio  # random_ratio: The proportion of random sampling (0.3 means 30% random, 70% Bayesian)
        self.iteration = 0

    def initialize(
        self, capital_phi: typing.List[typing.Tuple[typing.Dict, float]]
    ) -> None:
        """
        Initializes the model with a set of initial configurations, before it can make recommendations
        which configurations are in good regions. Note that we are minimising (lower values are preferred)

        :param capital_phi: a list of tuples, each tuple being a configuration and the performance (typically,
        error rate)
        """
        # Save all initial (configuration, performance) to self.runs
        self.runs = list(capital_phi)

        # Find the configuration with the best (lowest) performance as the current incumbent
        if len(self.runs) > 0:
            # Sort by performance (second element), take the minimum
            best_idx = np.argmin([run[1] for run in self.runs])
            best_run = self.runs[best_idx]
            self.theta_inc = best_run[0]
            self.theta_inc_performance = best_run[1]
        else:
            self.theta_inc = None
            self.theta_inc_performance = None

    def fit_model(self) -> None:
        """
        Fits the internal surrogate model on the complete run list.
        """
        if len(self.runs) < 2:  # Need at least 2 points to train
            return

        # Extract configurations and performances
        configs = [run[0] for run in self.runs]
        performances = [run[1] for run in self.runs]

        # Validate performance data for NaN values
        performances_array = np.array(performances)
        if np.isnan(performances_array).any():
            print("Warning: NaN values detected in performances, replacing with median")
            median_perf = np.nanmedian(performances_array)
            performances_array = np.nan_to_num(performances_array, nan=median_perf)
            performances = performances_array.tolist()

        # Convert configurations to numeric matrix (fit scaler during training)
        X = self._configs_to_matrix(configs, fit_scaler=True)
        y = np.array(performances)

        # Final validation before training
        if np.isnan(X).any() or np.isnan(y).any():
            print("Warning: NaN values detected in training data, replacing with 0")
            X = np.nan_to_num(X, nan=0.0)
            y = np.nan_to_num(y, nan=0.0)

        # Train the model
        try:
            self.model.fit(X, y)
        except Exception as e:
            print(f"Error fitting model: {e}")
            print("Attempting to fit with cleaned data...")
            # Additional cleaning if needed
            X = np.nan_to_num(X, nan=0.0)
            y = np.nan_to_num(y, nan=0.0)
            self.model.fit(X, y)

    def select_configuration(self) -> ConfigSpace.Configuration:
        """
        Determines which configurations are good, based on the internal surrogate model.
        Note that we are minimizing the error, but the expected improvement takes into account that.
        Therefore, we are maximizing expected improvement here.

        :return: A size n vector, same size as each element representing the EI of a given
        configuration
        """

        self.iteration += 1
        # Hybrid strategy: decide whether to use Bayesian optimization or random sampling according to the specified ratio
        if random.random() < self.random_ratio:
            # Random sampling
            print(f"Iteration {self.iteration}: Using random sampling")
            return self.config_space.sample_configuration()

        # Bayesian optimization
        print(f"Iteration {self.iteration}: Using Bayesian optimization")

        # Generate multiple candidate configurations
        num_candidates = 100  # Generate 100 candidate configurations
        candidates = []
        for _ in range(num_candidates):
            candidates.append(self.config_space.sample_configuration())

        # Convert candidate configurations to numeric matrix (don't fit scaler during prediction)
        X_candidates = self._configs_to_matrix(candidates, fit_scaler=False)

        # Calculate expected improvement
        ei_values = self.expected_improvement(
            model_pipeline=self.model,
            f_star=self.theta_inc_performance,
            theta=X_candidates,
        )

        # Select the configuration with the largest EI
        best_idx = np.argmax(ei_values)
        best_config = candidates[best_idx]

        return best_config

    def _configs_to_matrix(self, configs: list, fit_scaler: bool = False) -> np.array:
        """
        Convert a list of configurations to a numeric matrix

        :param configs: list of configurations
        :param fit_scaler: whether to fit the scaler (only True during training)
        :return: numeric matrix
        """
        # Convert to DataFrame and encode
        configs_df = pd.DataFrame([dict(config) for config in configs])
        configs_encoded = pd.get_dummies(configs_df)

        # Handle NaN values by filling with 0
        configs_encoded = configs_encoded.fillna(0)

        # Check for any remaining NaN values
        if configs_encoded.isnull().any().any():
            print(
                "Warning: NaN values detected in configuration matrix, replacing with 0"
            )
            configs_encoded = configs_encoded.fillna(0)

        # Only fit scaler during training, otherwise use transform
        if fit_scaler:
            X = self.scaler.fit_transform(configs_encoded)
            # Store the training columns for consistent encoding
            self._training_columns = configs_encoded.columns.tolist()
        else:
            # Ensure we have the same columns as during training
            if not hasattr(self, "_training_columns"):
                # If no training columns stored, fit on current data
                X = self.scaler.fit_transform(configs_encoded)
                self._training_columns = configs_encoded.columns.tolist()
            else:
                # Align columns with training data
                configs_encoded = configs_encoded.reindex(
                    columns=self._training_columns, fill_value=0
                )
                X = self.scaler.transform(configs_encoded)

        # Final check for NaN values after scaling
        if np.isnan(X).any():
            print("Warning: NaN values detected after scaling, replacing with 0")
            X = np.nan_to_num(X, nan=0.0)

        return X

    @staticmethod
    def expected_improvement(
        model_pipeline, f_star: float, theta: np.array
    ) -> np.array:
        """
        Acquisition function that determines which configurations are good and which
        are not good.

        :param model_pipeline: The internal surrogate model (should be fitted already)
        :param f_star: The current incumbent (theta_inc)
        :param theta: A (n, m) array, each column represents a hyperparameter and each row
        represents a configuration
        :return: A size n vector, same size as each element representing the EI of a given
        configuration
        """
        # Check for NaN values in input
        if np.isnan(theta).any():
            print("Warning: NaN values detected in theta, replacing with 0")
            theta = np.nan_to_num(theta, nan=0.0)

        # Check for NaN values in f_star
        if np.isnan(f_star):
            print("Warning: NaN value detected in f_star, using 0")
            f_star = 0.0

        # ei stands for "expected improvement". In Bayesian optimization, each candidate configuration point will have an ei value,
        # so ei is actually a vector whose length equals the number of configurations being evaluated. Each ei[i] represents the expected
        # improvement for candidate configuration theta[i].
        # The formula for EI (for each candidate point) is:
        #     EI = (f_star - mu) * Φ(z) + sigma * φ(z)
        #     where z = (f_star - mu) / sigma
        #     Φ(z) is the cumulative distribution function (CDF) of the standard normal distribution,
        #     φ(z) is the probability density function (PDF) of the standard normal distribution.

        # Get the predicted mean (mu) and standard deviation (sigma) for each theta point (both are vectors)
        mu, sigma = model_pipeline.predict(theta, return_std=True)

        # Handle NaN values in predictions
        if np.isnan(mu).any() or np.isnan(sigma).any():
            print("Warning: NaN values detected in model predictions, replacing with 0")
            mu = np.nan_to_num(mu, nan=0.0)
            sigma = np.nan_to_num(sigma, nan=1e-6)

        # Calculate the improvement (vector)
        improvement = f_star - mu

        # 1e-9 is 0.000000001 (a very small number to avoid division by zero)
        sigma = np.maximum(sigma, 1e-9)

        # Calculate the standardized improvement z (vector)
        z = improvement / sigma

        # Calculate EI for each theta point (vector)
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)

        # Handle NaN values in EI calculation
        if np.isnan(ei).any():
            print("Warning: NaN values detected in EI calculation, replacing with 0")
            ei = np.nan_to_num(ei, nan=0.0)

        # Ensure EI is non-negative, result is still a vector
        ei = np.maximum(ei, 0.0)

        return ei

    def update_runs(self, run: typing.Tuple[typing.Dict, float]):
        """
        After a configuration has been selected and ran, it will be added to the run list
        (so that the model can be trained on it during the next iterations).

        :param run: A tuple (configuration, performance) where performance is error rate
        """
        # Add new run result
        self.runs.append(run)

        # Update the best configuration if the new result is better
        if self.theta_inc_performance is None or run[1] < self.theta_inc_performance:
            self.theta_inc = run[0]
            self.theta_inc_performance = run[1]

        # Retrain the model
        self.fit_model()
