# type: ignore
"""
Retinal Image Analysis Model Training
-----------------------------------
This script implements the training of a deep learning model for heart disease detection
from retinal images. The model analyzes retinal vessel patterns to predict various
health metrics and assess heart disease risk.

Training Process Overview:
1. Data Loading and Preprocessing
2. Model Architecture Definition
3. Training Configuration
4. Model Training with Advanced Techniques
5. Model Evaluation and Saving
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define paths for data organization
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'retinal_disease_classification_data', 'Training_Set', 'Training_Set')
TRAIN_DIR = os.path.join(DATA_DIR, 'Training')
LABELS_FILE = os.path.join(DATA_DIR, 'RFMiD_Training_Labels.csv')

def load_custom_dataset(data_dir, labels_file, image_size=(224, 224)):
    """
    STEP 1: DATA PREPARATION
    -----------------------
    This function loads and preprocesses the retinal image dataset.
    
    Key Steps:
    1. Load image labels from CSV
    2. Process each image:
       - Resize to standard size (224x224)
       - Normalize pixel values (0-1)
    3. Extract corresponding health metrics
    4. Perform data validation
    
    Parameters:
    - data_dir: Directory containing retinal images
    - labels_file: CSV file with health metrics
    - image_size: Target image dimensions
    """
    # Read labels from CSV file
    labels_df = pd.read_csv(labels_file)
    
    # Initialize lists for data storage
    images = []
    targets = []
    
    # Process each image and its labels
    for idx, row in labels_df.iterrows():
        img_id = row.get('ID', row.get('image_name', f"{idx}"))
        img_filename = f"{str(img_id)}.png"
        img_path = os.path.join(data_dir, img_filename)
        
        if os.path.exists(img_path):
            try:
                # Image preprocessing steps
                img = tf.keras.preprocessing.image.load_img(
                    img_path, target_size=image_size
                )
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = img_array / 255.0  # Normalize pixel values
                
                # Extract health metrics (target values)
                target = [
                    row.get('Disease_Risk', 0),    # Overall disease risk
                    row.get('DR', 0),              # Diabetic Retinopathy
                    row.get('ARMD', 0),            # Age-related Macular Degeneration
                    row.get('MH', 0),              # Macular Hole
                    row.get('DN', 0)               # Diabetic Nephropathy
                ]
                
                images.append(img_array)
                targets.append(target)
            except Exception as e:
                print(f"Error loading image {img_path}: {str(e)}")
                continue
    
    # Convert to numpy arrays for training
    images = np.array(images)
    targets = np.array(targets)
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Total images loaded: {len(images)}")
    print("\nTarget Distribution:")
    conditions = ['Disease Risk', 'DR', 'ARMD', 'MH', 'DN']
    for condition, count in zip(conditions, np.sum(targets, axis=0)):
        print(f"{condition}: {count} samples")
    
    return images, targets

def create_improved_model():
    """
    STEP 2: MODEL ARCHITECTURE
    -------------------------
    Defines the deep learning model architecture using CNN.
    
    Architecture Overview:
    1. Input Layer (224x224x3)
    2. Convolutional Blocks
    3. Feature Extraction
    4. Dense Layers
    5. Output Layer (5 health metrics)
    
    Key Features:
    - Deep CNN architecture
    - Dropout for regularization
    - Batch normalization
    - Skip connections
    """
    # Input layer
    inputs = layers.Input(shape=(224, 224, 3))
    
    # First Convolutional Block
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    
    # Second Convolutional Block
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    
    # Third Convolutional Block
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    
    # Flatten and Dense Layers
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)  # Prevent overfitting
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output Layer (5 predictions)
    outputs = layers.Dense(5, activation='sigmoid')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def train_improved_model(data_dir, labels_file, output_dir='./'):
    """
    STEP 3: MODEL TRAINING
    
    Implements the complete training pipeline with advanced techniques.
    
    Training Process:
    1. Data Loading and Splitting
    2. Data Augmentation Setup
    3. Model Compilation
    4. Training with Callbacks
    5. Model Evaluation and Saving
    
    Advanced Features:
    - Learning rate scheduling
    - Early stopping
    - Model checkpointing
    - Data augmentation
    - Class weight balancing
    """
    # Load and prepare data
    print("Loading data...")
    X, y = load_custom_dataset(data_dir, labels_file)
    
    if len(X) == 0:
        print("No images found! Please check the data directory and file paths.")
        return
    
    print(f"Loaded {len(X)} images")
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Calculate class weights for imbalanced data
    class_weights = {}
    for i in range(y.shape[1]):
        class_weights[i] = len(y) / (2 * np.sum(y[:, i]))
    
    print("\nClass Weights:")
    conditions = ['Disease Risk', 'DR', 'ARMD', 'MH', 'DN']
    for condition, weight in zip(conditions, class_weights.values()):
        print(f"{condition}: {weight:.2f}")
    
    # Create and compile model
    print("\nCreating improved model...")
    model = create_improved_model()
    
    # Configure optimizer with learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    # Setup data augmentation for training
    datagen = ImageDataGenerator(
        rotation_range=30,          # Random rotation
        width_shift_range=0.2,      # Random horizontal shift
        height_shift_range=0.2,     # Random vertical shift
        shear_range=0.2,           # Random shear
        zoom_range=0.2,            # Random zoom
        horizontal_flip=True,       # Random horizontal flip
        vertical_flip=True,         # Random vertical flip
        fill_mode='nearest',        # Fill strategy
        brightness_range=[0.8, 1.2] # Random brightness
    )
    
    # Learning rate reduction on plateau
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Save best model during training
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(output_dir, 'best_model.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Train the model
    print("\nTraining improved model...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_val, y_val),
        epochs=100,
        class_weight=class_weights,
        callbacks=[reduce_lr, early_stopping, checkpoint]
    )
    
    # Save the final model
    print("\nSaving model...")
    model.save(os.path.join(output_dir, 'retinal_heart_risk_model.h5'))
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

if __name__ == "__main__":
    print(tf.__version__)
    # Use the correct paths defined above
    train_improved_model(TRAIN_DIR, LABELS_FILE, output_dir="./") 