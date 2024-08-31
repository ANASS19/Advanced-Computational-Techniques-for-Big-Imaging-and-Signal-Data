import pandas as pd

# Load the train dataset to get the feature columns
train_df = pd.read_csv('test.csv')

# Extract feature columns
feature_columns = train_df.columns[:-2]  # Assuming last two columns are 'Activity' and 'subject'
#sample_data = train_df[feature_columns].head(100)  # Select a few rows to create a sample
sample_data = train_df[feature_columns]  # Select a few rows to create a sample

# Save to CSV
sample_data.to_csv('datainput.csv', index=False)
