import ConfigSpace

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.impute import SimpleImputer
from scipy.stats import spearmanr
import numpy as np


class SurrogateModel:

    def __init__(self, config_space):
        self.config_space = config_space
        self.df = None
        self.model = None

    def fit(self, df, test_size=0.2, random_state=42):
        """
        Receives a data frame, in which each column (except for the last two) represents a hyperparameter, the
        penultimate column represents the anchor size, and the final column represents the performance.

        :param df: the dataframe with performances
        :param test_size: proportion of data to use as holdout set (default 0.2)
        :param random_state: random seed for train-test split
        :return: Does not return anything, but stores the trained model in self.model
        """

        # 1. Store the original data
        self.df = df.copy()

        # 2. Feature columns (hyperparameters + anchor_size)
        X = df[df.columns[:-1]].copy()  # hyperparameters + anchor_size
        y = df[df.columns[-1]].copy()  # score column

        # 3. Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.X_test = X_test
        self.y_test = y_test

        # 4. Automatically identify and encode categorical features and convert boolean features to numeric (bool will be automatically converted to 0/1)
        X_train_encoded = pd.get_dummies(X_train, dummy_na=True)

        # 5. Handle missing values in numerical features using mean imputation.
        num_cols = X_train_encoded.select_dtypes(include=["number"]).columns
        self.num_cols_ = num_cols.tolist()  # Save numeric column names for prediction
        imputer = SimpleImputer(strategy="mean")
        X_train_encoded[num_cols] = imputer.fit_transform(X_train_encoded[num_cols])

        # Build a simple random forest regressor
        # Common parameters for RandomForestRegressor:
        # n_estimators: number of trees in the forest, default is 100
        # criterion: function to measure the quality of a split, default for regression is "squared_error"
        # max_depth: maximum depth of the tree, default is None (i.e., unlimited)
        # min_samples_split: minimum number of samples required to split an internal node, default is 2
        # min_samples_leaf: minimum number of samples required to be at a leaf node, default is 1
        # max_features: number of features to consider when looking for the best split, default is "auto"
        # random_state: random seed for reproducibility
        # n_jobs: number of jobs to run in parallel, -1 means using all processors
        # For example:
        self.model = RandomForestRegressor(
            n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
        )
        self.model.fit(X_train_encoded, y_train)

        # Record the column order and the imputer used during training
        self.feature_columns_ = X_train_encoded.columns.tolist()
        self.imputer_ = imputer

    def predict(self, theta_new):
        """
        Predicts the performance of a given configuration theta_new

        :param theta_new: a dict, where each key represents the hyperparameter (or anchor)
        :return: float, the predicted performance of theta new (which can be considered the ground truth)
        """
        # Convert theta_new to a DataFrame (single row)
        X_new = pd.DataFrame([theta_new])

        # Apply one-hot encoding in the same way as during training
        X_new_encoded = pd.get_dummies(X_new, dummy_na=True)

        # reindex ensures that the columns in X_new_encoded match the feature order used during model training.
        # If any expected feature columns are missing in the new data, they are added and filled with 0 to avoid inconsistency errors.
        X_new_encoded = X_new_encoded.reindex(
            columns=self.feature_columns_, fill_value=0
        )

        # Use the same imputer to transform the numerical columns
        # Only use the numeric columns that were present during training
        X_new_encoded[self.num_cols_] = self.imputer_.transform(
            X_new_encoded[self.num_cols_]
        )

        # Use the trained model to make predictions
        pred = self.model.predict(X_new_encoded)

        # Return the predicted value (single float)
        return float(pred[0])

    def evaluate(self):
        """
        Evaluates the surrogate model on the holdout set.
        Computes Spearman correlation, MSE, R², and MAE.

        :return: dict with evaluation metrics
        """
        if not hasattr(self, "X_test") or self.X_test is None:
            raise ValueError(
                "Model must be trained with test_size > 0 to evaluate on holdout set"
            )

        # Prepare test data in the same way as training data
        X_test_encoded = pd.get_dummies(self.X_test, dummy_na=True)
        X_test_encoded = X_test_encoded.reindex(
            columns=self.feature_columns_, fill_value=0
        )
        X_test_encoded[self.num_cols_] = self.imputer_.transform(
            X_test_encoded[self.num_cols_]
        )

        # Make predictions
        y_pred = self.model.predict(X_test_encoded)
        y_true = self.y_test.values

        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mae = np.mean(np.abs(y_true - y_pred))

        # Calculate Spearman correlation
        spearman_corr, spearman_pvalue = spearmanr(y_true, y_pred)

        results = {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R²": r2,
            "Spearman Correlation": spearman_corr,
            "Spearman P-value": spearman_pvalue,
            "Number of test samples": len(y_true),
        }

        return results
