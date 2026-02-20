import pandas as pd
from sklearn.datasets import fetch_openml

# Load the dataset
boston = fetch_openml(name='housing', version=1, as_frame=True)
df = boston.frame

# Display the DataFrame
print(df)