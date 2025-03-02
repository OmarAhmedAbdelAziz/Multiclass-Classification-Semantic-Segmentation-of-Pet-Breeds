import pytest
import pandas as pd
from data_loader import read_annotation_file, load_pet_dataset, extracting_breed_name

# Mock annotation data for testing
@pytest.fixture
def mock_annotation_file(tmp_path):
    file_content = """# Comment line
    Abyssinian_1 1 1 1
    bengal_5 2 2 2
    Birman_3 3 1 3
    """
    file_path = tmp_path / "mock_list.txt"
    file_path.write_text(file_content)
    return str(file_path)

# Test read_annotation_file
def test_read_annotation_file(mock_annotation_file):
    df = read_annotation_file(mock_annotation_file)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3  # Should read 3 lines (excluding comments)
    assert list(df.columns) == ["image_name", "class_id", "species", "breed_id"]

# Test extracting_breed_name
@pytest.mark.parametrize("image_name, expected", [
    ("Abyssinian_1", "Abyssinian"),
    ("bengal_5", "bengal"),
    ("RandomBreed_100", "RandomBreed"),
    ("NoNumber", "NoNumber")
])
def test_extracting_breed_name(image_name, expected):
    assert extracting_breed_name(image_name) == expected

# Mock dataset directory for load_pet_dataset
@pytest.fixture
def mock_dataset_dir(tmp_path):
    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir()

    trainval_txt = annotations_dir / "trainval.txt"
    trainval_txt.write_text("Abyssinian_1 1 1 1\nbengal_5 2 2 2")

    test_txt = annotations_dir / "test.txt"
    test_txt.write_text("Birman_3 3 1 3\npersian_10 4 1 4")

    list_txt = annotations_dir / "list.txt"
    list_txt.write_text("Abyssinian_1 1 1 1\nbengal_5 2 2 2\nBirman_3 3 1 3")

    return str(tmp_path)

# Test load_pet_dataset
def test_load_pet_dataset(mock_dataset_dir):
    trainval_df, test_df, list_df, images_dir = load_pet_dataset(mock_dataset_dir)

    assert isinstance(trainval_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert isinstance(list_df, pd.DataFrame)
    assert trainval_df.shape[0] > 0 
    assert test_df.shape[0] <= 2  
    assert list_df.shape[0] == 3  
