import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image

# Define paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'retinal_heart_risk_model.h5')
TEST_DIR = os.path.join(os.path.dirname(__file__), 'test_images')

# Create test directory if it doesn't exist
if not os.path.exists(TEST_DIR):
    os.makedirs(TEST_DIR)
    print(f"Created test directory at {TEST_DIR}")
    print("Please add some test images to this directory before running the script again.")
    exit(0)

def preprocess_image(image_path):
    """Preprocess a single image for prediction."""
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize to match training size
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def get_risk_level(probability):
    """Convert probability to risk level."""
    if probability < 0.3:
        return 'Low'
    elif probability < 0.6:
        return 'Medium'
    else:
        return 'High'

def predict_image(model, image_path):
    """Make predictions for a single image."""
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)[0]
    
    # Define condition names
    conditions = [
        'Hypertension',
        'Diabetes',
        'Smoking',
        'Obesity',
        'Heart Disease'
    ]
    
    # Create dictionary of predictions with risk levels
    results = {}
    for condition, prob in zip(conditions, predictions):
        results[condition] = {
            'probability': float(prob),
            'risk_level': get_risk_level(prob)
        }
    return results

def display_results(image_path, predictions):
    """Display the image and prediction results."""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Display image
    img = Image.open(image_path)
    ax1.imshow(img)
    ax1.set_title('Input Image')
    ax1.axis('off')
    
    # Create bar chart
    conditions = list(predictions.keys())
    probabilities = [pred['probability'] for pred in predictions.values()]
    risk_levels = [pred['risk_level'] for pred in predictions.values()]
    
    # Create bars with different colors based on risk level
    colors = []
    for risk in risk_levels:
        if risk == 'Low':
            colors.append('green')
        elif risk == 'Medium':
            colors.append('orange')
        else:
            colors.append('red')
    
    bars = ax2.bar(conditions, probabilities, color=colors)
    ax2.set_ylim(0, 1)
    ax2.set_title('Prediction Results')
    ax2.set_ylabel('Probability')
    plt.xticks(rotation=45)
    
    # Add probability and risk level labels on top of bars
    for bar, prob, risk in zip(bars, probabilities, risk_levels):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.2%}\n({risk})',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('prediction_results.png')
    plt.close()

def main():
    # Load the model
    print("Loading model...")
    model = load_model(MODEL_PATH)
    
    # Get list of test images
    test_images = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not test_images:
        print("No test images found in the test directory.")
        return
    
    print(f"\nFound {len(test_images)} test images.")
    
    # Process first 5 images
    for i, image_name in enumerate(test_images[:5]):
        image_path = os.path.join(TEST_DIR, image_name)
        print(f"\nProcessing image {i+1}: {image_name}")
        
        # Make predictions
        predictions = predict_image(model, image_path)
        
        # Display results
        display_results(image_path, predictions)
        
        # Print predictions
        print("\nPrediction Results:")
        for condition, pred in predictions.items():
            print(f"{condition}: {pred['probability']:.2%} ({pred['risk_level']} Risk)")

if __name__ == "__main__":
    main() 