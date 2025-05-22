import os
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def run_pipeline():
    try:
        # Step 1: Data Ingestion
        print("\n📥 Starting data ingestion...")
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()

        # Step 2: Initialize components
        transformer = DataTransformation()
        trainer = ModelTrainer()

        # Step 3: Loop over target columns
        targets = ['math score', 'reading score', 'writing score']
        for target in targets:
            print(f"\n🔄 Processing target: {target}")
            train_arr, test_arr, prep_path = transformer.initiate_data_transformation(
                train_path, test_path, target_column=target
            )

            print(f"✅ Transformation complete for '{target}'. Preprocessor saved to: {prep_path}")

            results = trainer.initiate_model_trainer(train_arr, test_arr, target_column=target)
            print(f"\n📊 Results for '{target}':\n{results}")

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")

if __name__ == "__main__":
    run_pipeline()
