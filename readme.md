# Student Performance Prediction Model

## What is this

This project implements a simple web-application to predict students’ academic performance using a trained machine-learning model.  
It uses a pre-trained regression model (stored as `linear_regression_model.pkl`) to estimate student performance based on input features.  
The project uses a web UI (HTML templates) to accept user inputs, and a backend built using Python and Flask to serve predictions.  

## Project Structure

- `main.py` — the main Flask application script.  
- `linear_regression_model.pkl` — serialized (trained) ML model used for predictions.  
- `templates/` — HTML templates for the web interface.  
- `static/` — (optional) static files (CSS/JS) if used by UI.  
- (Optional) A notebook file included for exploratory analysis or retraining.  

## Requirements

- Python 3.x  
- Flask  
- scikit-learn  
- (plus any other dependencies you may have used — e.g. pandas, numpy)  

## How to Run

1. (Recommended) Create and activate a virtual environment:

    ```bash
    python -m venv venv  
    source venv/bin/activate   # On Windows: `venv\Scripts\activate`
    ```

2. Install required packages:

    ```bash
    pip install flask scikit-learn
    ```

3. Start the application:

    ```bash
    python main.py
    ```

4. Open your web browser and go to:

    ```
    http://127.0.0.1:5000
    ```

5. Fill in the student’s information in the form, submit — the app will display the predicted performance.  

## (Optional) How It Works Internally

- On form submission, the Flask backend collects input values.  
- These inputs are fed into the pre-trained regression model loaded from `linear_regression_model.pkl`.  
- The model returns a predicted performance score.  
- The result is rendered back to the user via HTML page/template.  

## Notes

- This project is intended as a demo / educational tool — predictions depend on trained model quality and chosen input features.  
- If you retrain the model or use a different dataset, update `linear_regression_model.pkl` accordingly.  
- Feel free to extend the project (improve UI, add input validation, support more features, deploy on cloud, etc.).  
