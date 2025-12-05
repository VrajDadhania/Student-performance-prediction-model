# from flask import Flask, render_template

# app = Flask(__name__)

# @app.route('/predict')
# def main():
#     return render_template('index.html')

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask,request,render_template
import os
import pickle
import pandas as pd
# from src.pipeline.predict_pipeline import CustomData,PredictPipeline

app = Flask(__name__)

MODEL_FILE_PATH = 'linear_regression_model.pkl'

# Global variable to hold the loaded model object
model = None
# Keep a short in-memory history of recent predictions
history = []
with open(MODEL_FILE_PATH, 'rb') as f:
    model = pickle.load(f)

parental_mapping = {
    'highschool': 1,
    'phd': 2,
    'bachelors': 3,
    'masters': 4
}

extracurricular_mapping = {
    'yes': 1,
    'no': 0
}

gender_mapping = {
    'male': 1,
    'female': 0
}

internet_mapping = {
    'yes': 1,
    'no': 0
}

@app.route('/')
def index():
    return render_template('index.html', history=history)


@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        
        try:
            data = {
                'subject': request.form['subject'],
                'parental_education': request.form['parental_education'],
                'extracurricular': request.form['extracurricular'],
                'attendance_rate': float(request.form['attendance_rate']),
                'gender': request.form['gender'],
                'internet_access': request.form['internet_access'],
                'study_hours': float(request.form['study_hours']),
                'past_score': float(request.form['past_score'])
            }

            model_features = {
                'Gender': gender_mapping[data['gender']],
                'Study_Hours_per_Week': data['study_hours'],
                'Attendance_Rate': data['attendance_rate'],
                'Past_Exam_Scores': data['past_score'],
                'Parental_Education_Level': parental_mapping[data['parental_education']],
                'Internet_Access_at_Home': internet_mapping[data['internet_access']],
                'Extracurricular_Activities': extracurricular_mapping[data['extracurricular']],    
            }


            df = pd.DataFrame([model_features])
            prediction = model.predict(df)[0]

            rounded_score = int(round(prediction))
    
            if rounded_score > 97:
                rounded_score = 97
            else:
                rounded_score = max(0, rounded_score)

            # Track a small rolling history (most recent first)
            history.insert(0, {
                'subject': data['subject'],
                'parental_education': data['parental_education'],
                'extracurricular': data['extracurricular'],
                'attendance_rate': data['attendance_rate'],
                'gender': data['gender'],
                'internet_access': data['internet_access'],
                'study_hours': data['study_hours'],
                'past_score': data['past_score'],
                'prediction': rounded_score
            })
            if len(history) > 10:
                history.pop()

            return render_template('result.html', prediction=rounded_score)
            
        except KeyError as e:
            return f"Error: Missing form field - {e}", 400
            
    return "This page only accepts form submissions.", 405
    

if __name__ == '__main__':
    app.run(debug=True)