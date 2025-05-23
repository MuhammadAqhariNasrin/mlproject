import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
import pickle


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "best_model.pkl")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def evaluate_model(self, model, X_test, y_test, X_full, y_full):
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        cv_scores = cross_val_score(model, X_full, y_full, cv=5, scoring='r2')
        cv_r2 = round(np.mean(cv_scores), 2)

        return {
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 2),
            "CV R2": cv_r2
        }

    def tune_model(self, model, param_grid, X, y):
        try:
            rs = RandomizedSearchCV(model, param_grid, cv=3, n_iter=10, n_jobs=-1, scoring='r2', random_state=42)
            rs.fit(X, y)
            return rs.best_estimator_
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array: np.ndarray, test_array: np.ndarray, target_column: str):
        try:
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Define models and their tuning grids
            models = {
                "Random Forest": (RandomForestRegressor(random_state=42), {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5, 10]
                }),
                "Gradient Boosting": (GradientBoostingRegressor(random_state=42), {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 5]
                }),
                "XGBoost": (XGBRegressor(random_state=42, verbosity=0), {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 5]
                }),
                "CatBoost": (CatBoostRegressor(verbose=0, random_state=42), {
                    "iterations": [100, 200],
                    "depth": [4, 6],
                    "learning_rate": [0.01, 0.1]
                }),
                "AdaBoost": (AdaBoostRegressor(random_state=42), {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1, 1.0]
                }),
                "Decision Tree": (DecisionTreeRegressor(random_state=42), {
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5, 10]
                }),
                "Linear Regression": (LinearRegression(), {}),
                "Ridge": (Ridge(), {
                    "alpha": [0.1, 1.0, 10.0]
                }),
                "Lasso": (Lasso(), {
                    "alpha": [0.01, 0.1, 1.0]
                })
            }

            results = []
            best_model = None
            best_score = float("-inf")

            for name, (model, params) in models.items():
                logging.info(f"Tuning model: {name} for target: {target_column}")
                if params:
                    tuned_model = self.tune_model(model, params, X_train, y_train)
                else:
                    tuned_model = model.fit(X_train, y_train)

                metrics = self.evaluate_model(tuned_model, X_test, y_test, X_train, y_train)
                results.append({"Model": name, **metrics})

                if metrics["R2"] > best_score:
                    best_score = metrics["R2"]
                    best_model = tuned_model

            # Save best model
            os.makedirs(os.path.dirname(self.config.trained_model_file_path), exist_ok=True)
            model_path = self.config.trained_model_file_path.replace("best_model", f"best_model_{target_column.replace(' ', '_')}")
            with open(model_path, "wb") as f:
                pickle.dump(best_model, f)
                logging.info(f"Saved best model for '{target_column}' to: {model_path}")

            return pd.DataFrame(results)

        except Exception as e:
            raise CustomException(e, sys)
