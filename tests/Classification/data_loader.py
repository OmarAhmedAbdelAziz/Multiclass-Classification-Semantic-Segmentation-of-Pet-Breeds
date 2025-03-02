import os
import re
import pandas as pd

def read_annotation_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Remove any comment line in the file
    data_lines = [line.strip() for line in lines if not line.startswith('#')]
    
    # making the data dataframe to bmake it easy to be understood
    data = []
    for line in data_lines:
        parts = line.split()
        if len(parts) >= 4:  
            image_name = parts[0]
            class_id = int(parts[1])
            species = int(parts[2])  
            breed_id = int(parts[3])
            data.append({
                'image_name': image_name,
                'class_id': class_id,
                'species': species,
                'breed_id': breed_id
            })
    
    return pd.DataFrame(data)

def load_pet_dataset(base_path):
    images_dir = os.path.join(base_path, "images")
    annotations_dir = os.path.join(base_path, "annotations")

    # Reading the annotation files used for classification
    trainval_file = os.path.join(annotations_dir, "trainval.txt")
    test_file = os.path.join(annotations_dir, "test.txt")
    list_file = os.path.join(annotations_dir, "list.txt")
    
    # Loading the annotations
    trainval_df = pd.DataFrame(read_annotation_file(trainval_file))
    test_df = pd.DataFrame(read_annotation_file(test_file))
    list_df = pd.DataFrame(read_annotation_file(list_file))
    
    num_of_test_samples = min(1000, len(test_df))  # Ensure valid sample size

    # Randomly selecting 1000 test samples without losing order
    test_samples = test_df.sample(n=num_of_test_samples, random_state=42)

    # Getting the remaining images
    taken_samples = test_df.drop(test_samples.index)  

    # Appending taken samples to trainval dataset withouy losing order
    trainval_df = pd.concat([trainval_df, taken_samples], ignore_index=True)

    # Making new test dataset
    test_df = test_samples
    
    # Extracting the name without the number and avoid many different underscores in ther name
    list_df['breed_name'] = list_df['image_name'].apply(extracting_breed_name)
    
    return trainval_df, test_df, list_df, images_dir

def extracting_breed_name(image_name):
    match = re.match(r"(.+)_\d+$", image_name)
    return match.group(1) if match else image_name