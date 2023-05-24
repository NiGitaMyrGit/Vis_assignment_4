#!/usr/bin/env python3
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from argparse import ArgumentParser
import argparse

#ignore warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Weather Classification')
parser.add_argument('-r','--report_path', type=str, default='out/classification_report.txt',
                    help='Path to save the classification report')
args = parser.parse_args()

# Define the weather conditions (classes)
def load_data():
    weather_conditions = ['dew', 'fogsmog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'rainbow', 'rime', 'sandstorm', 'snow']
    dataset_path = "/work/cds-viz/vis_assignment4/dataset"
    # Prepare the dataset
    images = []
    labels = []
    for condition_idx, condition in enumerate(weather_conditions):
        condition_path = os.path.join(dataset_path, condition)
        for image_file in os.listdir(condition_path):
            image_path = os.path.join(condition_path, image_file)
            try:
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to read image: {image_path}")
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
                image = cv2.resize(image, (224, 224))  # Resize to a consistent shape
                images.append(image)
                labels.append(condition_idx)
            except Exception as e:
                print(f"Error processing image: {image_path}")
                print(f"Error details: {str(e)}")

    # Convert lists to arrays
    images = np.array(images)
    labels = np.array(labels)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Create an image data generator for data augmentation
    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    return X_train, X_test, y_train, y_test, datagen

def train_model(X_train, X_test, y_train, y_test, datagen):
    # Build the CNN model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(len(weather_conditions), activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')

    # Generate the classification report
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    report = classification_report(y_test, y_pred, target_names=weather_conditions)
    return report

   
def main():
     # Load data
    X_train, X_test, y_train, y_test, datagen = load_data()

    # Train model
    report = train_model(X_train, X_test, y_train, y_test, datagen)

     # Save the classification report to a text file
    with open(args.report_path, "w") as file:
        file.write(report)

# Call main function
if __name__ == "__main__":
    main()