import os
import sys

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

print("📂 sys.path includes:")
for p in sys.path:
    print("  ", p)

try:
    from src.pipeline.prediction_pipeline import PredictPipeline
    print("✅ Import worked: PredictPipeline loaded!")
except ModuleNotFoundError as e:
    print("❌ Import failed:", e)
