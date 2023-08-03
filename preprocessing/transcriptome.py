import csv
import math

def apply_log_transformation(input_filename, output_filename, column_name):
    # Open the input CSV file
    with open(input_filename, 'r') as input_file:
        reader = csv.DictReader(input_file)
        data = list(reader)

    # Apply the transformation to the specified column
    for row in data:
        try:
            a = float(row[column_name])
            transformed_a = math.log10(1 + a)
            row[column_name] = transformed_a
        except ValueError:
            # In case the column value cannot be converted to float, skip the transformation for that row
            pass

    # Write the updated data to the output CSV file
    with open(output_filename, 'w', newline='') as output_file:
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

# Example usage
input_filename = 'input.csv'  # Replace with the path to your input CSV file
output_filename = 'output.csv'  # Replace with the path where you want to save the output CSV file
column_name = 'column_name'  # Replace 'column_name' with the name of the column you want to transform
apply_log_transformation(input_filename, output_filename, column_name)
