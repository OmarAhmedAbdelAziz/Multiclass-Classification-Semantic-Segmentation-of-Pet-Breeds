import pytest
import numpy as np
import cv2
from preprocess import preprocessing_images, split_dataset, create_data_generator

@pytest.fixture
def mock_dataframe():
    """Mock dataframe with two image samples"""
    import pandas as pd
    return pd.DataFrame({
        'image_name': ['test_image_1', 'test_image_2'],
        'class_id': [1, 2]
    })

@pytest.fixture
def mock_images_dir(tmp_path):
    """Create a temporary directory with fake images"""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    for img_name in ["test_image_1.jpg", "test_image_2.jpg"]:
        img_path = images_dir / img_name
        img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

    return str(images_dir)

def test_preprocessing_images(mock_dataframe, mock_images_dir):
    x, y = preprocessing_images(mock_dataframe, mock_images_dir, target_size=(224, 224))

    assert x.shape == (2, 224, 224, 3)  # Check if images are resized correctly
    assert y.shape == (2,)  # Check label shape
    assert y[0] == 0 and y[1] == 1  # Ensure class_id is mapped correctly

def test_split_dataset():
    x = np.random.rand(100, 224, 224, 3)
    y = np.random.randint(0, 37, 100)

    x_train, x_test, y_train, y_test = split_dataset(x, y, test_size=0.2)

    assert len(x_train) == 80 and len(x_test) == 20  # Ensure correct split
    assert len(y_train) == 80 and len(y_test) == 20  # Ensure labels match

def test_create_data_generator():
    data_gen = create_data_generator()
    assert hasattr(data_gen, 'flow')  # Check if it's a valid ImageDataGenerator
