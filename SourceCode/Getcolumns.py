import pandas as pd

# Your file CSV
file_path = "first_200_rows.csv" 

# Read file CSV
data = pd.read_csv(file_path)

# Display column names in CSV file (for reference)
print("Name of the Columns in file CSV:")
print(data.columns)

# Select the columns you want
columns_to_select = ['id', 'Gender', 'Age', 'Height', 'Weight']  # Columns you need
selected_columns = data[columns_to_select]

# Display selected data
print("Selected Data:")
print(selected_columns)

# New file CSV
output_path = "selected_columns.csv"
selected_columns.to_csv(output_path, index=False)

print("Saved in new file:", output_path)
