import ConfigSpace

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.impute import SimpleImputer


class SurrogateModel:

    def __init__(self, config_space):
        self.config_space = config_space
        self.df = None
        self.model = None

    def fit(self, df):
        """
        Receives a data frame, in which each column (except for the last two) represents a hyperparameter, the
        penultimate column represents the anchor size, and the final column represents the performance.

        :param df: the dataframe with performances
        :return: Does not return anything, but stores the trained model in self.model
        """

        # 1. Store the original data
        self.df = df.copy()

        # 2. Feature columns (hyperparameters + anchor_size)
        X = df[df.columns[:-1]].copy()  # hyperparameters + anchor_size
        y = df[df.columns[-1]].copy()  # score column

        # 3. Automatically identify and encode categorical features and convert boolean features to numeric (bool will be automatically converted to 0/1)
        X_encoded = pd.get_dummies(X, dummy_na=True)

        # 4.Handle missing values in numerical features using mean imputation.
        num_cols = X_encoded.select_dtypes(include=["number"]).columns
        imputer = SimpleImputer(strategy="mean")
        X_encoded[num_cols] = imputer.fit_transform(X_encoded[num_cols])

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
        self.model.fit(X_encoded, y)

        # Record the column order and the imputer used during training
        self.feature_columns_ = X_encoded.columns.tolist()
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
        num_cols = X_new_encoded.select_dtypes(include=["number"]).columns
        X_new_encoded[num_cols] = self.imputer_.transform(X_new_encoded[num_cols])

        # Use the trained model to make predictions
        pred = self.model.predict(X_new_encoded)

        # Return the predicted value (single float)
        return float(pred[0])
