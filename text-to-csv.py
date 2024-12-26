import pandas as pd

input_file = "data/MachineLearningRating_v3.txt"  
output_file = "data/MachineLearningRating_v3.csv"  

data = pd.read_csv(input_file, delimiter='|') # Delimeter


data.to_csv(output_file, index=False)

print(f"File successfully converted and saved as {output_file}")