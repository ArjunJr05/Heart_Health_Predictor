from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
@app.route('/', methods = ['GET'])
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST','GET'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Validate input data
        required_fields = [
            'age', 'sex', 'chest_pain_type', 'ekg_results', 'max_hr',
            'exercise_angina', 'st_depression', 'slope_of_st',
            'num_vessels', 'thallium'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({'status': 'error', 'message': f'Missing field: {field}'}), 400
        
        # Create DataFrame with the correct feature order
        input_data = pd.DataFrame([[
            float(data['age']),
            int(data['sex']),  # 1 = male, 0 = female
            int(data['chest_pain_type']),  # 1-4
            int(data['ekg_results']),  # 0-2
            float(data['max_hr']),  # thalach
            int(data['exercise_angina']),  # 1 = yes, 0 = no
            float(data['st_depression']),  # max HR ST depression
            int(data['slope_of_st']),  # 1-3
            int(data['num_vessels']),  # number of vessels fluro
            int(data['thallium'])  # 3 = normal, 6 = fixed, 7 = reversible
        ]], columns=[
            'Age', 'Sex', 'Chest pain type', 'EKG results', 'Max HR', 
            'Exercise angina', 'ST depression', 'Slope of ST', 
            'Number of vessels fluro', 'Thallium'
        ])
        
        # Scale the input data
        scaled_data = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(scaled_data)
        probability = np.max(model.predict_proba(scaled_data))
        
        # Return result
        return jsonify({
            'status': 'success',
            'prediction': prediction[0],
            'probability': float(probability)
        })
        
    except ValueError as e:
        return jsonify({'status': 'error', 'message': f'Invalid input data: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'An error occurred: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print("Starting Flask server for Heart Disease Prediction...")
    app.run(debug=True)