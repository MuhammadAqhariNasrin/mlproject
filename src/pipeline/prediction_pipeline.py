import os
import sys
import pandas as pd
import pickle
from src.exception import CustomException
from src.logger import logging


class PredictPipeline:
    def __init__(self, target: str):
        self.target = target.lower().replace(" ", "_")  # e.g. "math score" -> "math_score"
        self.model_path = os.path.join("artifacts", f"best_model_{self.target}.pkl")
        self.preprocessor_path = os.path.join("artifacts", f"preprocessor_{self.target}.pkl")

    def load_object(self, file_path):
        try:
            with open(file_path, "rb") as file:
                return pickle.load(file)
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, input_data: dict):
        try:
            logging.info(f"Predicting for target: {self.target}")

            # Load model and preprocessor
            model = self.load_object(self.model_path)
            preprocessor = self.load_object(self.preprocessor_path)

            # Convert input dict to DataFrame
            df = pd.DataFrame([input_data])  # single-row DataFrame

            # Transform input
            transformed_data = preprocessor.transform(df)

            # Make prediction
            prediction = model.predict(transformed_data)

            return round(float(prediction[0]), 2)

        except Exception as e:
            raise CustomException(e, sys)
