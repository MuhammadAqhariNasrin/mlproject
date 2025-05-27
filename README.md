
# 🎓 Student Score Predictor

A machine learning-powered web application to predict a student's **Math**, **Reading**, or **Writing** score based on their input details. This app uses a trained model behind the scenes and presents a clean web form interface built using Flask and Bootstrap.

> ✅ Trained offline using Scikit-learn, XGBoost, and CatBoost  
> ✅ Predicts score instantly  
> ✅ Ready to deploy on AWS Elastic Beanstalk

---

## 🔍 Features

- Predict one of: **Math**, **Reading**, or **Writing** score
- Bootstrap-styled web interface
- Handles preprocessing and model inference
- Built-in input validation
- Compatible with AWS Elastic Beanstalk deployment
- Models saved as `.pkl` files for reuse without retraining

---



## 📁 Folder Structure

student-score-predictor/
├── app.py # Main Flask app
├── app.wsgi # WSGI entry point for Elastic Beanstalk
├── requirements.txt # Python dependencies
├── templates/
│ └── index.html # HTML form (Bootstrap styled)
├── src/
│ └── pipeline/
│ └── prediction_pipeline.py # ML logic
├── artifacts/ # Trained models + preprocessors (.pkl files)
├── .ebextensions/
│ └── python.config # Beanstalk configuration
├── README.md # You're reading it!


---

## 🛠️ How to Run Locally

1. 📦 Clone this repo

```bash
git clone https://github.com/your-username/student-score-predictor.git
cd student-score-predictor

2. 🐍 Create a virtual environment

python -m venv venv
# Activate it:
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate


3. 📥 Install dependencies

pip install -r requirements.txt


4. ▶️ Run the app

python app.py


Then open your browser at: http://127.0.0.1:5000

🌐 How to Deploy on AWS Elastic Beanstalk

1. Make sure your files include:

app.py

app.wsgi

.ebextensions/python.config

requirements.txt

.pkl model files in artifacts/


2. Create a ZIP file:
zip -r student-score-app.zip *

3. Go to Elastic Beanstalk Console, create a Python environment, and upload your zip.

 

