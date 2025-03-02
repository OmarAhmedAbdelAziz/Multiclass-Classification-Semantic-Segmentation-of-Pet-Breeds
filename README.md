# Oxford-IIIT Pet Classification and Segmentation

## Project Overview
The goal of this project is to develop a machine learning model that classify pet breeds and performing semantic segmentation to outline the pets in the images. The dataset used is the Oxford-IIIT Pet Dataset, which contains images of pets labeled by species (cat or dog) and breed. The project implements both classification and segmentation tasks using pretrained ResNet50 and MobileNetV2 model.


## Folders Descriptions:

1. data:

    This folder contains the Oxford-IIIT Pet Dataset, including images and their corresponding annotations. The images are organized into categories based on the pet species and breed, while the annotations are used for semantic segmentation tasks.

    - Images: Contains over 7,000 images of cats and dogs.
    - Annotations: Contains segmentation masks outlining the pets in the images.

2. models:

    This folder stores the trained models, including saved weights and model configurations. The models are used for both classification and segmentation tasks. They are saved in formats like TensorFlow’s .h5.

3. notebooks:

    Contains Jupyter notebooks used for exploratory data analysis, training, and model evaluation. These notebooks are helpful for visualizing results, tuning hyperparameters, and understanding the dataset.

4. outputs:

    This folder contains the outputs generated from running the models. These include predicted breed labels for classification tasks and segmentation masks for the pets, along with visualizations and performance metrics.

5. src:

    This folder contains the source code for both the classification and segmentation tasks. It is organized into two subfolders:

    - Classification: Contains scripts for loading data, preprocessing it, training the model, and making predictions for breed classification.
    - Segmentation: Contains scripts for loading data, preprocessing it, training the model, and making predictions for segmentation masks of the pets.

    Each folder includes the following scripts:
    - data_loader.py: Loads the dataset and prepares it for training and testing.
    - preprocess.py: Handles preprocessing tasks like resizing, normalization, and augmentation.
    - train.py: Contains the model architecture and the training loop.
    - predict.py: Loads the trained model and performs inference on new images.

6. tests:

    Contains unit tests for the scripts in the src folder. These tests ensure that the data loading, preprocessing, training, and prediction functions work correctly. The tests are written using the pytest framework.

7. requirements.txt

    Lists the required Python libraries and dependencies to run the project.