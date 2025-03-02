import pytest
import numpy as np
import os
import cv2
from unittest.mock import MagicMock
from predict import load_and_preprocess_image, predict_breed, display_prediction
import matplotlib.pyplot as plt

# Test for loading and preprocessing an image with a real image
def test_load_and_preprocess_image_valid():
    # Assuming a real image is located in the test folder
    image_path = "test_image.jpg"
    
    # Check if the image exists at the provided path
    assert os.path.exists(image_path), f"Image not found at {image_path}"
    
    # Load and preprocess the image
    img, img_processed = load_and_preprocess_image(image_path)
    
    # Check if the output is a tuple (original image, processed image)
    assert isinstance(img, np.ndarray)
    assert isinstance(img_processed, np.ndarray)
    
    # Check the shape of the original image (this will depend on the real image's size)
    assert img.shape[2] == 3  # Check if it's a color image (3 channels)
    
    # The processed image should be resized to (224, 224)
    assert img_processed.shape == (224, 224, 3)

# Test for invalid image path (file not found)
def test_load_and_preprocess_image_invalid():
    # Test invalid path (file not found)
    invalid_image_path = "invalid_image.jpg"
    
    with pytest.raises(FileNotFoundError):
        load_and_preprocess_image(invalid_image_path)

# Test for breed prediction
def test_predict_breed():
    # Mock the model and breed name mapping
    model = MagicMock()
    breed_name_mapping = {1: "havanese", 2: "leonberger", 3: "samoyed"}

    # Mock the model's prediction output
    model.predict = MagicMock(return_value=np.array([[0, 0, 1]]))  # Mock prediction for "samoyed"
    
    # Mock image processing output
    img_processed = np.ones((224, 224, 3))  # Dummy processed image
    
    predicted_breed = predict_breed(model, img_processed, breed_name_mapping)
    
    # Check if the predicted breed is correct
    assert predicted_breed == "samoyed"

# Test for displaying the prediction
def test_display_prediction():
    # Mock the image and predicted breed
    img = np.ones((224, 224, 3))  # Dummy image
    predicted_breed = "leonberger"

    # Mock plt.show() to avoid displaying during tests
    plt.show = MagicMock()
    
    # Call the function
    display_prediction(img, predicted_breed)
    
    # Check if plt.show() was called
    plt.show.assert_called_once()

