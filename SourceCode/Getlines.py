import pandas as pd

# Read file CSV
file_path = "test.csv"
data = pd.read_csv(file_path)

# Get first n lines
first_200_rows = data.head(200)

# New file
output_path = "first_200_rows.csv"
first_200_rows.to_csv(output_path, index=False)

print("Save 200 lines rows in file:", output_path)
