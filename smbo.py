import ConfigSpace
import numpy as np
import pandas as pd
import typing

from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm


class SequentialModelBasedOptimization(object):

    def __init__(self, config_space):
        """
        Initializes empty variables for the model, the list of runs (capital R), and the incumbent
        (theta_inc being the best found hyperparameters, theta_inc_performance being the performance
        associated with it)
        """
        self.config_space = config_space  # configuration space
        self.runs = []  # run list, stores all evaluated (configuration, performance) tuples
        self.theta_inc = None  # the best found hyperparameters
        self.theta_inc_performance = None  # the best performance
        self.scaler = StandardScaler()
        self.model = GaussianProcessRegressor(random_state=42)
        # self.model = Pipeline([
        #     ("scaler", StandardScaler()),
        #     ("gp", GaussianProcessRegressor(kernel=RBF(), random_state=42))
        # ])

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

        # Convert configurations to numeric matrix
        X = self._configs_to_matrix(configs)
        y = np.array(performances)

        # Train the model
        self.model.fit(X, y)

    def select_configuration(self) -> ConfigSpace.Configuration:
        """
        Determines which configurations are good, based on the internal surrogate model.
        Note that we are minimizing the error, but the expected improvement takes into account that.
        Therefore, we are maximizing expected improvement here.

        :return: A size n vector, same size as each element representing the EI of a given
        configuration
        """
        if self.model is None:
            # If the model is not trained, randomly select a configuration
            return self.config_space.sample_configuration()

        # Generate candidate configurations
        candidates = [self.config_space.sample_configuration() for _ in range(100)]

        # Convert candidate configurations to numeric matrix
        X_candidates = self._configs_to_matrix(candidates)

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

    def _configs_to_matrix(self, configs: list) -> np.array:
        """
        Convert a list of configurations to a numeric matrix

        :param configs: list of configurations
        :return: numeric matrix
        """
        # Convert to DataFrame and encode
        configs_df = pd.DataFrame([dict(config) for config in configs])
        configs_encoded = pd.get_dummies(configs_df)

        X = self.scaler.fit_transform(configs_encoded)

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
        # Get predicted mean and standard deviation
        mu, sigma = model_pipeline.predict(theta, return_std=True)

        # Calculate improvement (f_star - mu), since we are minimizing
        improvement = f_star - mu

        # Avoid division by zero
        sigma = np.maximum(sigma, 1e-9)

        # Calculate standardized improvement
        z = improvement / sigma

        # Calculate expected improvement (EI)
        # EI = improvement * Φ(z) + sigma * φ(z)
        # where Φ is the CDF of the standard normal distribution, φ is the PDF
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)

        # Ensure EI is non-negative
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
