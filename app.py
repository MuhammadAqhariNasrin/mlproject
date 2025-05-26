import os
import sys

# Ensure root folder is in sys.path so src/ works
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from flask import Flask, render_template, request
from src.pipeline.prediction_pipeline import PredictPipeline

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def predict_score():
    prediction = None

    if request.method == "POST":
        target = request.form.get("target")

        # Step 1: collect all inputs as strings
        data = {
            "gender": request.form.get("gender"),
            "race/ethnicity": request.form.get("race_ethnicity"),
            "parental level of education": request.form.get("parent_education"),
            "lunch": request.form.get("lunch"),
            "test preparation course": request.form.get("test_prep"),
            "math score": request.form.get("math_score"),
            "reading score": request.form.get("reading_score"),
            "writing score": request.form.get("writing_score")
        }

        # Step 2: drop the field we're trying to predict
        data.pop(target, None)

        # Step 3: convert the remaining scores to float
        for key in ["math score", "reading score", "writing score"]:
            if key in data:
                try:
                    data[key] = float(data[key])
                except (ValueError, TypeError):
                    data[key] = 0.0  # fallback if field is blank or invalid

        # Step 4: run prediction
        pipeline = PredictPipeline(target=target)
        prediction = pipeline.predict(data)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
