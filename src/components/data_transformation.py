import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer_object(self, df: pd.DataFrame, target_column: str) -> ColumnTransformer:
        try:
            # Identify input features
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            numerical_cols = df.select_dtypes(include=["int64", "float64"]).drop(columns=[target_column]).columns.tolist()

            logging.info(f"[{target_column}] Categorical Columns: {categorical_cols}")
            logging.info(f"[{target_column}] Numerical Columns: {numerical_cols}")

            num_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
            cat_pipeline = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

            preprocessor = ColumnTransformer([
                ("num", num_pipeline, numerical_cols),
                ("cat", cat_pipeline, categorical_cols)
            ])

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str, target_column: str):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            if target_column not in train_df.columns or target_column not in test_df.columns:
                raise ValueError(f"Target column '{target_column}' not found in both datasets.")

            preprocessor = self.get_data_transformer_object(train_df, target_column)

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # Save preprocessor separately for each target
            prep_path = self.config.preprocessor_obj_file_path.replace("preprocessor", f"preprocessor_{target_column.replace(' ', '_')}")
            os.makedirs(os.path.dirname(prep_path), exist_ok=True)
            with open(prep_path, "wb") as f:
                pickle.dump(preprocessor, f)
                logging.info(f"Saved preprocessor for target: {target_column}")

            train_arr = np.c_[X_train_transformed, y_train.values]
            test_arr = np.c_[X_test_transformed, y_test.values]

            return train_arr, test_arr, prep_path

        except Exception as e:
            raise CustomException(e, sys)
