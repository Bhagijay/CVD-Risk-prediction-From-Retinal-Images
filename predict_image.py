"""
Heart Disease Detection from Retinal Images
This module provides functionality to analyze retinal images and predict various health metrics
using a deep learning model. It serves as a standalone prediction tool that can be used
independently or integrated into the main application.

Author: [Your Name]
Date: [Current Date]
"""

import os
import numpy as np
# Suppress TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
import warnings
warnings.filterwarnings('ignore')  # Suppress general Python warnings

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

def predict_retinal_image(image_path):
    """
    Analyzes a retinal image to predict various health metrics using a deep learning model.
    
    This function performs the following steps:
    1. Loads a pre-trained deep learning model
    2. Preprocesses the input retinal image:
       - Resizes to 224x224 pixels (model's expected input size)
       - Normalizes pixel values to [0,1] range
    3. Makes predictions for multiple health metrics
    4. Performs risk assessment based on predicted values
    
    Parameters:
    -----------
    image_path : str
        Path to the retinal image file to be analyzed
        
    Returns:
    --------
    None
        Prints predictions and risk assessment directly to console
        
    Health Metrics Predicted:
    - Age
    - Systolic Blood Pressure (SBP)
    - Diastolic Blood Pressure (DBP)
    - BMI (Body Mass Index)
    - HbA1c (Glycated Hemoglobin)
    """
    
    # Load the trained model from disk
    model_path = 'retinal_heart_risk_model.h5'
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    try:
        # Suppress tensorflow output during model loading for cleaner console output
        import logging
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        
        # Load the pre-trained model
        model = load_model(model_path)
        
        # Image preprocessing steps
        image = Image.open(image_path)
        processed_image = image.resize((224, 224))  # Resize to model's expected input size
        processed_image = img_to_array(processed_image) / 255.0  # Normalize pixel values
        processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
        
        # Generate predictions using the model
        predictions = model.predict(processed_image, verbose=0)
        
        # Extract individual predictions for each health metric
        predicted_age = predictions[0][0]
        predicted_sbp = predictions[0][1]
        predicted_dbp = predictions[0][2]
        predicted_bmi = predictions[0][3]
        predicted_hba1c = predictions[0][4]
        
        # Display prediction results
        print("\nPrediction Results:")
        print("-" * 30)
        print(f"Predicted Age: {predicted_age:.1f} years")
        print(f"Predicted Systolic Blood Pressure: {predicted_sbp:.1f} mmHg")
        print(f"Predicted Diastolic Blood Pressure: {predicted_dbp:.1f} mmHg")
        print(f"Predicted BMI: {predicted_bmi:.1f}")
        print(f"Predicted HbA1c: {predicted_hba1c:.1f}%")
        
        # Risk Assessment Logic
        # Calculate risk score based on multiple factors
        risk_score = 0
        if predicted_sbp > 120 or predicted_dbp > 75:  # Blood pressure threshold
            risk_score += 2
        if predicted_bmi > 21:  # BMI threshold
            risk_score += 1
        if predicted_hba1c > 5.3:  # HbA1c threshold
            risk_score += 2
            
        # Display risk assessment results
        print("\nRisk Assessment:")
        print("-" * 30)
        if risk_score >= 4:
            print("High Risk - Please consult a healthcare professional")
        elif risk_score >= 2:
            print("Medium Risk - Regular monitoring recommended")
        else:
            print("Low Risk - Continue maintaining healthy lifestyle")
            
    except Exception as e:
        print(f"Error processing image: {str(e)}")

# Entry point for command-line execution
if __name__ == "__main__":
    # Get image path from user input
    image_path = input("Enter the path to your retinal image: ")
    if os.path.exists(image_path):
        predict_retinal_image(image_path)
    else:
        print(f"Error: Image file not found at {image_path}") 