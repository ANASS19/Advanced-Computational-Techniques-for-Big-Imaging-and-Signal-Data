import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

# Define the neural network model
class HARModel(nn.Module):
    def __init__(self):
        super(HARModel, self).__init__()
        self.fc1 = nn.Linear(561, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 6)  # Adjust based on the number of classes

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the model and normalization parameters
checkpoint = torch.load('har_model.pth')
model = HARModel()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
mean = checkpoint['mean']
std = checkpoint['std']
label_to_index = checkpoint['label_to_index']
index_to_label = {v: k for k, v in label_to_index.items()}

# Define prediction function
def predict_activity(model, input_data):
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    input_tensor = (input_tensor - mean) / std
    with torch.no_grad():
        output = model(input_tensor.unsqueeze(0))
        _, predicted = torch.max(output, 1)
        return index_to_label[predicted.item()]

# Streamlit UI
st.title("Human Activity Recognition")

uploaded_file = st.file_uploader("Choose a file...")
if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    st.write("### Input Data", input_data)
    # Visualize the input data
    st.write("### Input Data Visualization")
    for column in input_data.columns[:5]:  # Limiting to first 5 columns for better visualization
        plt.figure(figsize=(10, 4))
        sns.histplot(input_data[column], bins=50, kde=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        st.pyplot(plt)
        
    if st.button('Predict Activity'):
        predictions = []
        for i in range(len(input_data)):
            prediction = predict_activity(model, input_data.values[i])
            predictions.append(prediction)
        
        input_data['Predicted Activity'] = predictions
        st.write("### Predictions", input_data)
        
        # Show prediction summary
        prediction_counts = input_data['Predicted Activity'].value_counts()
        st.write("### Prediction Summary", prediction_counts)
        
        # Plot prediction distribution
        st.write("### Prediction Distribution")
        plt.figure(figsize=(10, 5))
        sns.countplot(x='Predicted Activity', data=input_data, palette='viridis')
        plt.title('Distribution of Predicted Activities')
        plt.xlabel('Activity')
        plt.ylabel('Count')
        st.pyplot(plt)
        
        # Plot activity distribution pie chart
        st.write("### Prediction Distribution Pie Chart")
        plt.figure(figsize=(10, 5))
        colors = sns.color_palette('viridis', n_colors=len(prediction_counts))
        input_data['Predicted Activity'].value_counts().plot.pie(autopct='%1.1f%%', colors=colors)
        plt.ylabel('')
        plt.title('Distribution of Predicted Activities')
        st.pyplot(plt)
