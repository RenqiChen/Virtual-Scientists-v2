import csv

# File path to the CSV
file_path = "rankings.csv"

# Load CSV as a dictionary
rankings_dict = {}
with open(file_path, mode="r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        rankings_dict[row["Name"]] = int(row["Rank"])  # Convert 'Rank' to integer

# Print the dictionary
print(rankings_dict)
