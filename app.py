from flask import Flask, render_template, jsonify, request
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import dummy_data_gen
import training_inference
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_logs', methods=['GET'])
def get_logs():
    try:
        # Read the first 20 rows of each log file
        analytics_logs = pd.read_csv('analytics_logs.csv').head(20).to_dict('records')
        backend_logs = pd.read_csv('backend_logs.csv').head(20).to_dict('records')
        datadog_logs = pd.read_csv('datadog_mobile_logs.csv').head(20).to_dict('records')
        
        return jsonify({
            'status': 'success',
            'analytics_logs': analytics_logs,
            'backend_logs': backend_logs,
            'datadog_logs': datadog_logs
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/get_tickets', methods=['GET'])
def get_tickets():
    try:
        # Read training and test tickets
        training_tickets = pd.read_csv('jira_tickets_training.csv').to_dict('records')
        test_tickets = pd.read_csv('jira_tickets_cv.csv').to_dict('records')
        
        return jsonify({
            'status': 'success',
            'training_tickets': training_tickets,
            'test_tickets': test_tickets
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        # Run the dummy_data_gen script to generate data
        print("Starting data generation...")
        dummy_data_gen.start()
        print("Data generation completed")
        
        # Fix file names to match what the application expects
        print("Fixing file names...")
        print("File names fixed")
        
        # Check if files were created
        expected_files = [
            'datadog_mobile_logs.csv',
            'backend_logs.csv',
            'analytics_logs.csv',
            'jira_tickets_training.csv',
            'jira_tickets_cv.csv'
        ]
        
        missing_files = [f for f in expected_files if not os.path.exists(f)]
        if missing_files:
            return jsonify({
                'status': 'error',
                'message': f'Missing expected files: {", ".join(missing_files)}'
            }), 500
        
        # Return success with file information
        file_info = {}
        for file in expected_files:
            if os.path.exists(file):
                file_size = os.path.getsize(file)
                file_info[file] = f"{file_size / 1024:.2f} KB"
        
        return jsonify({
            'status': 'success',
            'message': 'Data generation completed successfully',
            'files': file_info
        })
    except Exception as e:
        print(f"Error in simulation: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error during simulation: {str(e)}'
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if required files exist
        required_files = [
            'datadog_mobile_logs.csv',
            'backend_logs.csv',
            'analytics_logs.csv',
            'jira_tickets_training.csv',
            'jira_tickets_cv.csv'
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            return jsonify({
                'status': 'error',
                'message': f'Missing required files: {", ".join(missing_files)}'
            }), 500
        
        # Run the training and prediction
        print("Starting training and prediction...")
        
        # Train the model
        df_cv_predictions = training_inference.start()
        
        # # Make predictions
        # df_cv_predictions = training_inference.make_predictions(clf, dv, le)
        
        # # Get model performance
        # performance = training_inference.get_model_performance(clf, dv, le)
        
        # Convert predictions to a list of dictionaries
        predictions = df_cv_predictions[0].to_dict('records')
        clf = df_cv_predictions[1]
        dv = df_cv_predictions[2]
        le = df_cv_predictions[3]
        performance = training_inference.get_model_performance(clf, dv, le)
        
        return jsonify({
            'status': 'success',
            'message': 'Prediction completed successfully',
            'predictions': predictions,
            'performance': performance
        })
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error during prediction: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001) 