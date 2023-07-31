import os
import shutil
import random

def split_data(source_folder, train_folder, test_folder, val_folder, split_ratio=(0.8, 0.1, 0.1)):
    """
    Split data from the source folder into train, test, and validation folders.

    Args:
        source_folder (str): Path to the source folder containing the data.
        train_folder (str): Path to the train folder for storing train data.
        test_folder (str): Path to the test folder for storing test data.
        val_folder (str): Path to the validation folder for storing validation data.
        split_ratio (tuple): Split ratio for train, test, and validation datasets.
                             Default is (0.8, 0.1, 0.1).
    """
    assert sum(split_ratio) == 1.0, "Split ratio should sum up to 1.0"

    # Create destination folders if they don't exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    # Get all files in the source folder
    files = os.listdir(source_folder)
    random.shuffle(files)

    # Split files based on split ratio
    train_split = int(len(files) * split_ratio[0])
    test_split = int(len(files) * (split_ratio[0] + split_ratio[1]))

    train_files = files[:train_split]
    test_files = files[train_split:test_split]
    val_files = files[test_split:]

    # Move files to their respective folders
    move_files(train_files, source_folder, train_folder)
    move_files(test_files, source_folder, test_folder)
    move_files(val_files, source_folder, val_folder)

def move_files(files, source_folder, destination_folder):
    """Move files from the source folder to the destination folder."""
    for file in files:
        source_path = os.path.join(source_folder, file)
        destination_path = os.path.join(destination_folder, file)
        shutil.move(source_path, destination_path)

# Example usage
source_folder = '/Users/m294161/Library/CloudStorage/OneDrive-MayoClinic/tRNAsformer/tRNAsformer/Clear cell adenocarcinoma, NOS'
train_folder = '/Users/m294161/Library/CloudStorage/OneDrive-MayoClinic/tRNAsformer/tRNAsformer/train'
test_folder = '/Users/m294161/Library/CloudStorage/OneDrive-MayoClinic/tRNAsformer/tRNAsformer/test'
val_folder = '/Users/m294161/Library/CloudStorage/OneDrive-MayoClinic/tRNAsformer/tRNAsformer/validation'
split_ratio = (0.8, 0.1, 0.1)

split_data(source_folder, train_folder, test_folder, val_folder, split_ratio)