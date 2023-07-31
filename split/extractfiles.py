import os
import csv

def create_csv_from_folders(root_folder_path, csv_file_path):
    """
    Create a CSV file with image file paths, gene file paths, and labels.

    Args:
        root_folder_path (str): Path to the root folder containing the nested subfolders.
        csv_file_path (str): Path to the CSV file for storing the data.
    """
    

    subfolders = os.listdir(root_folder_path)
    data = []
    label_value = 0

    for root, dirs, files in os.walk(root_folder_path):
        if len(dirs) == 2:
            folder1_path = os.path.join(root, dirs[0])
            folder2_path = os.path.join(root, dirs[1])

            folder1_files = sorted(os.listdir(folder1_path))
            folder2_files = sorted(os.listdir(folder2_path))

            for folder1_file, folder2_file in zip(folder1_files, folder2_files):
                folder1_file_path = os.path.join(folder1_path, folder1_file)
                folder2_file_path = os.path.join(folder2_path, folder2_file)

                data.append({"image_files": folder1_file_path, "gene_files": folder2_file_path, "labels": label_value})
    for subfolder in subfolders:
        subfolder_path = os.path.join(root_folder_path, subfolder)
        for dirs in subfolder_path:
            if len(dirs) == 2:
                folder1_path = os.path.join(subfolder_path, dirs[0])
                folder2_path = os.path.join(subfolder_path, dirs[1])

                folder1_files = sorted(os.listdir(folder1_path))
                folder2_files = sorted(os.listdir(folder2_path))

                for folder1_file, folder2_file in zip(folder1_files, folder2_files):
                    folder1_file_path = os.path.join(folder1_path, folder1_file)
                    folder2_file_path = os.path.join(folder2_path, folder2_file)

                data.append({"image_files": folder1_file_path, "gene_files": folder2_file_path, "labels": label_value})
        
    # Write the data to a CSV file
    with open(csv_file_path, "w", newline="") as csvfile:
        fieldnames = ["image_files", "gene_files", "labels"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(data)

# Example usage
root_folder_path = 'tRNAsformer/validation'
csv_file_path = 'validmh.csv'


create_csv_from_folders(root_folder_path, csv_file_path)






