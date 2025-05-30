# Heart Disease Detection using Retinal Images

A deep learning-based system for detecting heart disease risk through retinal image analysis. This project combines computer vision techniques with machine learning to analyze retinal images and predict potential heart disease risks.

## Project Overview

This project implements a novel approach to heart disease risk detection using retinal image analysis. The system utilizes deep learning and computer vision techniques to identify patterns and features in retinal images that may indicate cardiovascular health risks.

## Features

- **Image Processing**: Advanced image processing techniques to extract meaningful features from retinal images
- **Feature Extraction**: Calculates important metrics including:
  - Mean, Standard Deviation, and Variance
  - Entropy
  - Contrast, Correlation, Energy, and Homogeneity
- **Deep Learning Model**: Convolutional Neural Network (CNN) for accurate prediction
- **Web Interface**: User-friendly Django web application for easy interaction
- **Real-time Analysis**: Quick processing and prediction of heart disease risk

## Project Structure

```
Heart_Disease_Detection_using_retinal_images/
├── finalyear_project/        # Django project settings
├── myapp/                    # Main Django application
├── media/                    # Uploaded images storage
├── static/                   # Static files (CSS, JS, images)
├── templates/               # HTML templates
├── retinal_disease_classification_data/  # Dataset directory
├── requirements.txt         # Project dependencies
├── manage.py               # Django management script
├── predict_image.py        # Image prediction module
├── test_imports.py         # Import testing script
├── test_database.py        # Database testing script
├── check_versions.py       # Dependency version checker
├── retinal_heart_risk_model.h5  # Trained model
└── best_model.h5           # Best performing model weights
```

## Dataset

The project uses the DRIVE (Digital Retinal Images for Vessel Extraction) dataset from Kaggle:
https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction

## System Architecture

1. **Input Layer**: 
   - Retinal image capture
   - Image preprocessing and enhancement
   - Image size normalization
   - Color space conversion

2. **Processing Layer**:
   - Feature extraction using CNN
   - Image analysis with OpenCV
   - Pattern recognition
   - Deep learning model inference

3. **Output Layer**:
   - Risk assessment calculation
   - Results visualization
   - User-friendly web interface presentation

## Implementation Details

### Deep Learning Model
- Architecture: Convolutional Neural Network (CNN)
- Framework: TensorFlow/Keras
- Input Shape: 224x224x3 (RGB images)
- Training: Transfer learning with pre-trained weights

### Web Application
- Framework: Django
- Frontend: HTML5, CSS3, JavaScript
- Image Processing: OpenCV, scikit-image
- Database: SQLite

## Setup Instructions


1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run database migrations:
```bash
python manage.py makemigrations
python manage.py migrate
```

4. Run the Django server:
```bash
python manage.py runserver
```

## Technologies Used

- **Backend**:
  - Python 3.8+
  - Django 3.2+
  - OpenCV 4.5+
  - NumPy 1.19+
  - scikit-image 0.18+
  - TensorFlow 2.5+

- **Frontend**:
  - HTML5
  - CSS3
  - JavaScript
  - Bootstrap 5

## Testing

The project includes several test scripts:
- `test_imports.py`: Validates all required dependencies
- `test_database.py`: Tests database connections and operations
- `check_versions.py`: Verifies compatibility of installed packages

## Future Improvements

1. Implementation of additional deep learning models
2. Enhanced image preprocessing techniques
3. Real-time video analysis capability
4. Mobile application development
5. Integration with medical record systems

